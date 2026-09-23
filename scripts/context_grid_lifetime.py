"""Prepare, train and evaluate the first context-grid lifetime model.

Run --help or read docs/context_grid_lifetime_v1.md. Existing pipelines,
checkpoints, caches and handover documents are not modified.
"""
import argparse
import ast
import hashlib
import json
import pickle
import random
import subprocess
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.feature.lifetime_curves import LifetimeCurveExtractor
from src.models.rul_predictors.context_grid_lifetime import ContextGridLifetime


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def revision(path=ROOT):
    return subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()


def save_new(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(obj, stream)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def subset(data, indices, device=None):
    return {k: v[indices].to(device) if device else v[indices] for k, v in data.items()}


def stack_features(items):
    return {key: torch.stack([x[key] for x in items]) for key in items[0]}


def read_official_splits(path):
    """Read literal official lists and list additions without executing code."""
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'split_recorder')
    values = {}
    def evaluate(node):
        if isinstance(node, (ast.List, ast.Tuple, ast.Constant)):
            return ast.literal_eval(node)
        if isinstance(node, ast.Name):
            return values[node.id]
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return evaluate(node.left) + evaluate(node.right)
        raise ValueError('Unsupported official split expression; inspect pinned source.')
    for item in cls.body:
        if isinstance(item, ast.Assign) and isinstance(item.targets[0], ast.Name):
            values[item.targets[0].id] = evaluate(item.value)
    return {part: values['MIX_large_' + part + '_files'] for part in ['train', 'val', 'test']}


def validate_disjoint(parts):
    seen = set()
    for part, ids in parts.items():
        if len(set(ids)) != len(ids):
            raise ValueError(f'Duplicate cell IDs in {part}.')
        if seen.intersection(ids):
            raise ValueError(f'Cell overlap involving {part}.')
        seen.update(ids)


def prepare(args):
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    root = Path(args.data_root).resolve()
    extraction = dict(cycles=args.cycles, phase_points=args.phase_points,
                      drop_cycles=[10] if args.dataset == 'mix20' else [])
    extractor = LifetimeCurveExtractor(**extraction)
    metadata = dict(schema_version=1, dataset=args.dataset, extraction=extraction,
                    code_commit=revision(), source_files=[], excluded=[],
                    protocol='fixed-window lifetime; not official variable-window BatteryLife aggregate')
    records = {}
    if args.dataset.startswith('mix'):
        from src.data.battery_data import BatteryData
        from src.label.rul import RULLabelAnnotator
        from src.train_test_split.MIX20_split import MIX20TrainTestSplitter
        from src.train_test_split.MIX100_split import MIX100TrainTestSplitter
        expected = 20 if args.dataset == 'mix20' else 100
        if args.cycles != expected:
            raise ValueError('MIX task and observation window must agree.')
        names = ['CALCE', 'RWTH', 'UL_PUR', 'HNEI', 'MATR', 'HUST']
        if expected == 20:
            names.append('SNL')
        splitter = (MIX20TrainTestSplitter if expected == 20 else MIX100TrainTestSplitter)(
            [str(root / name) for name in names])
        labeler = RULLabelAnnotator(eol_soh=.9 if expected == 20 else .8)
        metadata['label_policy'] = dict(eol_soh=labeler.eol_soh, min_rul_limit=labeler.min_rul_limit,
                                        pad_eol=labeler.pad_eol, implementation='existing RULLabelAnnotator')
        for part, paths in zip(['train', 'test'], splitter.split()):
            records[part] = []
            for path in sorted(paths):
                cell = BatteryData.load(path)
                value = float(labeler.process_cell(cell))
                if not np.isfinite(value):
                    metadata['excluded'].append(dict(cell=cell.cell_id, reason='existing label filter'))
                    continue
                records[part].append((cell.cell_id, path, None, value))
        # Fixed cell-level validation, independent of model/training seed.
        rng = np.random.RandomState(args.split_seed)
        order = rng.permutation(len(records['train']))
        nval = max(1, round(len(order) * args.val_fraction))
        if nval >= len(order) - 1:
            raise ValueError('Not enough training cells after label filtering.')
        val = set(order[:nval].tolist())
        records['val'] = [r for i, r in enumerate(records['train']) if i in val]
        records['train'] = [r for i, r in enumerate(records['train']) if i not in val]
        metadata['split_seed'] = args.split_seed
    else:
        if not args.official_repo or not args.data_version:
            raise ValueError('BatteryLife requires --official-repo and --data-version (snapshot revision).')
        repo = Path(args.official_repo)
        snapshot_manifest = root / 'context_grid_snapshot.json'
        if snapshot_manifest.exists():
            recorded = json.loads(snapshot_manifest.read_text(encoding='utf-8'))
            if recorded['revision'] != args.data_version:
                raise ValueError('--data-version differs from the downloaded snapshot manifest.')
        split_path = repo / 'data_provider/data_split_recorder.py'
        splits = read_official_splits(split_path)
        validate_disjoint(splits)
        metadata.update(official_commit=revision(repo), split_sha256=digest(split_path),
                        data_version=args.data_version, label_policy='official Life labels JSON; EOL > 100')
        for part, names in splits.items():
            records[part] = []
            for name in names:
                prefix = name.split('_')[0]
                folder = {'UL-PUR': 'UL_PUR', 'ISU-ILCC': 'ISU_ILCC', 'MICH': 'total_MICH',
                          'SMICH': 'MICH_EXP'}.get(prefix, prefix)
                if prefix.startswith('Tongji'):
                    folder = 'Tongji'
                actual = name[1:] if prefix == 'SMICH' else name
                path = root / folder / actual
                # Some released snapshots retain MICH cells in their source folders.
                if not path.exists() and prefix == 'MICH':
                    matches = list(root.glob('MICH*/' + actual))
                    if len(matches) == 1:
                        path = matches[0]
                label_prefix = 'total_MICH' if prefix == 'MICH' else ('Tongji' if prefix.startswith('Tongji') else prefix)
                label_path = root / 'Life labels' / (label_prefix + '_labels.json')
                if not label_path.exists() and prefix in ('MICH', 'SMICH'):
                    label_path = root / 'Life labels' / (path.parent.name + '_labels.json')
                labels = json.loads(label_path.read_text(encoding='utf-8'))
                key = name.replace('--', '-#') if prefix.startswith('Tongji') else name
                value = labels.get(key)
                if value is None or not np.isfinite(float(value)) or float(value) <= 100:
                    metadata['excluded'].append(dict(cell=name, reason='official missing/EOL<=100 filter'))
                    continue
                # Match nominal-capacity corrections in the official loader.
                nominal_override = None
                if prefix == 'RWTH':
                    nominal_override = 1.85
                elif name.startswith('SNL_18650_NCA_25C_20-80'):
                    nominal_override = 3.2
                metadata.setdefault('label_files', {})[str(label_path.relative_to(root))] = digest(label_path)
                records[part].append((name.removesuffix('.pkl'), path, nominal_override, float(value)))
    validate_disjoint({part: [r[0] for r in rows] for part, rows in records.items()})
    payload = dict(metadata=metadata)
    content_splits = {}
    for part, rows in records.items():
        if not rows:
            raise ValueError(f'Empty {part} partition.')
        features = []
        for i, (cell_id, path, nominal_override, label) in enumerate(rows):
            with path.open('rb') as stream:
                cell = pickle.load(stream)
            if nominal_override is not None:
                cell['nominal_capacity_in_Ah'] = nominal_override
            features.append(extractor(cell))
            checksum = digest(path)
            if checksum in content_splits and content_splits[checksum] != part:
                raise ValueError(f'Identical source file content appears in different splits: {cell_id}')
            content_splits[checksum] = part
            metadata['source_files'].append(dict(cell_id=cell_id, split=part,
                path=str(path.relative_to(root)), sha256=checksum))
            if (i + 1) % 25 == 0:
                print(f'{part}: {i + 1}/{len(rows)}', flush=True)
        payload[part] = dict(features=stack_features(features),
                             labels=torch.tensor([r[3] for r in rows], dtype=torch.float32),
                             ids=[r[0] for r in rows])
    save_new(payload, out)
    out.with_suffix('.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({part: len(payload[part]['ids']) for part in ['train', 'val', 'test']}))


def training_partition(data, refit):
    if not refit:
        return data['train']
    return dict(features={k: torch.cat((data['train']['features'][k], data['val']['features'][k]))
                          for k in data['train']['features']},
                labels=torch.cat((data['train']['labels'], data['val']['labels'])),
                ids=data['train']['ids'] + data['val']['ids'])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def make_indices(nquery, nsupport, seed, support_size=32):
    return torch.randint(nsupport, (nquery, support_size), generator=torch.Generator().manual_seed(seed))


def scores(pred, labels):
    if not torch.isfinite(pred).all():
        raise FloatingPointError('Non-finite inverse-transformed predictions.')
    error = pred.double() - labels.double()
    return dict(RMSE=error.square().mean().sqrt().item(), MAE=error.abs().mean().item(),
                MAPE=(error.abs() / labels).mean().item(),
                ACC15=(error.abs() / labels <= .15).double().mean().item())


@torch.no_grad()
def evaluate(model, train, query, indices, mean, scale, device, encode_batch=8, pair_chunk=8):
    if indices.shape != (len(query['ids']), 32) or indices.min() < 0 or indices.max() >= len(train['ids']):
        raise ValueError('Invalid fixed reference indices.')
    model.eval()
    tokens, masks = [], []
    for start in range(0, len(train['ids']), encode_batch):
        t, m = model.encoder(subset(train['features'], slice(start, start + encode_batch), device))
        tokens.append(t)
        masks.append(m)
    tokens, masks = torch.cat(tokens), torch.cat(masks)
    labels = ((train['labels'].log() - mean) / scale).to(device)
    ori, support = [], []
    for i in range(len(query['ids'])):
        t, m = model.encoder(subset(query['features'], slice(i, i + 1), device))
        parts = []
        for start in range(0, 32, pair_chunk):
            ix = indices[i:i + 1, start:start + pair_chunk].to(device)
            result = model.components_from_tokens(t, m, tokens[ix], masks[ix], labels[ix])
            parts.append(result['y_sup'])
        ori.append(result['y_ori'].cpu())
        support.append(torch.cat(parts, 1).cpu())
    ori, support = torch.cat(ori), torch.cat(support)
    aggregate = support.median(1).values
    final = (1 - model.alpha) * ori + model.alpha * aggregate
    inv = lambda x: (x * scale + mean).exp()
    return dict(prediction=inv(final), labels=query['labels'], scores=scores(inv(final), query['labels']),
                diagnostics=dict(y_ori=inv(ori), y_sup=inv(support), y_sup_agg=inv(aggregate),
                                 support_index=indices),
                branch_scores=dict(ori=scores(inv(ori), query['labels']),
                                   support=scores(inv(aggregate), query['labels'])),
                train_ids=train['ids'], test_ids=query['ids'], diagnostic_units='cycles after inverse transform')


def train(args):
    seed_all(args.seed)
    workspace = Path(args.workspace)
    workspace.mkdir(parents=True, exist_ok=False)
    data = load(args.data)
    training = training_partition(data, args.refit)
    config = dict(cycles=data['metadata']['extraction']['cycles'],
                  phase_points=data['metadata']['extraction']['phase_points'],
                  channels=args.channels, bins=32, heads=4, dropout=args.dropout,
                  use_context=not args.no_context, alpha=.5)
    model = ContextGridLifetime(**config)
    model.encoder.statistics.fit(training['features'])
    device = torch.device(args.device)
    model.to(device)
    y = training['labels'].log()
    mean, scale = y.mean(), y.std(unbiased=False).clamp_min(1e-6)
    y = (y - mean) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    amp = args.amp and device.type == 'cuda'
    if amp and not torch.cuda.is_bf16_supported():
        raise ValueError('This AMP path requires CUDA bfloat16 support; omit --amp.')
    fixed = make_indices(len(data['val']['ids']), len(training['ids']), args.seed + 100000)
    info = dict(arguments={k: v for k, v in vars(args).items() if k != 'func'}, model=config, code_commit=revision(), data_sha256=digest(args.data),
                dataset_metadata=data['metadata'], train_ids=training['ids'],
                parameters=sum(p.numel() for p in model.parameters()), selection_metric=args.selection_metric)
    (workspace / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        started = time.perf_counter()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        order = torch.randperm(len(y))
        loss_sum = 0.
        batches = list(order.split(args.batch_size))
        for batch_no, ix in enumerate(batches):
            si = torch.randint(len(y), (len(ix), 2))
            target = subset(training['features'], ix, device)
            reference = subset(training['features'], si, device)
            # Match a full effective batch even when the final microbatch is short.
            group_start = (batch_no // args.accumulation) * args.accumulation
            group_count = sum(len(b) for b in batches[group_start:group_start + args.accumulation])
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                result = model(target, reference, y[si].to(device))
                label = y[ix].to(device)
                loss = .5 * (result['y_ori'].float() - label).square().mean() + .5 * (result['y_sup_agg'].float() - label).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}.')
            (loss * len(ix) / group_count).backward()
            loss_sum += loss.item() * len(ix)
            if (batch_no + 1) % args.accumulation == 0 or batch_no + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        record = dict(epoch=epoch, train_loss=loss_sum / len(y),
                      train_seconds=time.perf_counter() - started,
                      cuda_peak_GiB=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == 'cuda' else 0)
        checkpoint = None
        if not args.refit and (epoch % args.evaluate_every == 0 or epoch == args.epochs):
            result = evaluate(model, training, data['val'], fixed, mean, scale, device,
                              args.encode_batch, args.pair_chunk)
            record['validation'] = result['scores']
            record['validation_branches'] = result['branch_scores']
            value = result['scores'][args.selection_metric]
            if value < best:
                best = value
                checkpoint = 'best.pt'
        if args.refit and epoch == args.epochs:
            checkpoint = 'final.pt'
        if checkpoint:
            torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, config=config,
                            epoch=epoch, label_mean=mean, label_scale=scale, seed=args.seed,
                            refit=args.refit, data_sha256=info['data_sha256'], train_ids=training['ids'],
                            selection_metric=args.selection_metric), workspace / checkpoint)
        with (workspace / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    print('Training complete. Test set was not evaluated.', flush=True)


def run_eval(args):
    data, checkpoint = load(args.data), load(args.checkpoint)
    if digest(args.data) != checkpoint['data_sha256']:
        raise ValueError('Prepared data differ from the checkpoint data fingerprint.')
    training = training_partition(data, checkpoint['refit'])
    if training['ids'] != checkpoint['train_ids']:
        raise ValueError('Training reference order mismatch.')
    model = ContextGridLifetime(**checkpoint['config'])
    model.load_state_dict(checkpoint['state'], strict=True)
    model.to(args.device)
    query = data[args.partition]
    if args.partition == 'val' and checkpoint['refit']:
        raise ValueError('Validation cells were used in refit; do not evaluate them as held-out data.')
    if args.protocol:
        protocol = load(args.protocol)
        if protocol['seed'] != checkpoint['seed']:
            raise ValueError('Protocol seed does not match the checkpoint.')
        if protocol['train_ids'] != training['ids'] or protocol['query_ids'] != query['ids']:
            raise ValueError('Protocol cell IDs/order do not match this experiment.')
        indices = protocol['indices']
    else:
        indices = make_indices(len(query['ids']), len(training['ids']), checkpoint['seed'] + 200000)
    result = evaluate(model, training, query, indices, checkpoint['label_mean'], checkpoint['label_scale'],
                      args.device, args.encode_batch, args.pair_chunk)
    result.update(seed=checkpoint['seed'], epoch=checkpoint['epoch'], config=checkpoint['config'],
                  data_sha256=checkpoint['data_sha256'], checkpoint_sha256=digest(args.checkpoint),
                  partition=args.partition, refit=checkpoint['refit'])
    save_new(result, args.output)
    print(json.dumps(dict(scores=result['scores'], branches=result['branch_scores']), indent=2))


def protocol(args):
    data = load(args.data)
    training = training_partition(data, args.refit)
    query = data[args.partition]
    indices = make_indices(len(query['ids']), len(training['ids']), args.seed + 200000)
    if args.legacy_predictions:
        # Trusted local experiment pickle, including possible CUDA tensors.
        import io
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)
                return super().find_class(module, name)
        with open(args.legacy_predictions, 'rb') as stream:
            old = CPUUnpickler(stream).load()
        if old.get('seed') != args.seed:
            raise ValueError('Legacy prediction seed differs from --seed.')
        bundle = old['data']
        old_train = [m['cell_id'] for m in bundle.train_data.metadata]
        old_query = [m['cell_id'] for m in bundle.test_data.metadata]
        if set(old_train) != set(training['ids']) or set(old_query) != set(query['ids']):
            raise ValueError('Legacy cell sets differ; full training requires --refit.')
        old_labels = bundle.label_transformation.inverse_transform(bundle.test_data.label)
        lookup = {cell: i for i, cell in enumerate(old_query)}
        rows = torch.tensor([lookup[cell] for cell in query['ids']])
        if not torch.allclose(old_labels.cpu()[rows], query['labels'], atol=1e-2, rtol=1e-5):
            raise ValueError('Legacy test labels differ.')
        new_train = {cell: i for i, cell in enumerate(training['ids'])}
        mapping = torch.tensor([new_train[cell] for cell in old_train])
        train_labels = bundle.label_transformation.inverse_transform(bundle.train_data.label).cpu()
        if not torch.allclose(train_labels, training['labels'][mapping], atol=1e-2, rtol=1e-5):
            raise ValueError('Legacy training labels differ.')
        indices = mapping[old['diagnostics']['support_index'].cpu().long()[rows]]
    save_new(dict(indices=indices, train_ids=training['ids'], query_ids=query['ids'], seed=args.seed,
                  legacy_predictions=args.legacy_predictions), args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('--dataset', choices=['mix20', 'mix100', 'batterylife'], required=True)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--cycles', type=int, default=20)
    p.add_argument('--phase-points', type=int, default=256)
    p.add_argument('--split-seed', type=int, default=20260924)
    p.add_argument('--val-fraction', type=float, default=.2)
    p.add_argument('--official-repo')
    p.add_argument('--data-version')
    p.set_defaults(func=prepare)
    p = commands.add_parser('train')
    p.add_argument('--data', required=True)
    p.add_argument('--workspace', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=16)
    p.add_argument('--lr', type=float, default=.001)
    p.add_argument('--channels', type=int, default=64)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--evaluate-every', type=int, default=50)
    p.add_argument('--selection-metric', choices=['RMSE', 'MAE', 'MAPE'], default='RMSE')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-context', action='store_true')
    p.add_argument('--refit', action='store_true', help='Train+val, fixed epochs; save final.pt, no validation.')
    p.add_argument('--encode-batch', type=int, default=8)
    p.add_argument('--pair-chunk', type=int, default=8)
    p.set_defaults(func=train)
    p = commands.add_parser('evaluate')
    p.add_argument('--data', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--partition', choices=['val', 'test'], default='test')
    p.add_argument('--protocol')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--encode-batch', type=int, default=8)
    p.add_argument('--pair-chunk', type=int, default=8)
    p.set_defaults(func=run_eval)
    p = commands.add_parser('protocol')
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--partition', choices=['val', 'test'], default='test')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--refit', action='store_true')
    p.add_argument('--legacy-predictions')
    p.set_defaults(func=protocol)
    args = parser.parse_args()
    for name in ['batch_size', 'accumulation', 'epochs', 'evaluate_every', 'encode_batch', 'pair_chunk']:
        if hasattr(args, name) and getattr(args, name) < 1:
            parser.error(f'{name} must be positive')
    if hasattr(args, 'val_fraction') and not 0 < args.val_fraction < 1:
        parser.error('val-fraction must lie between 0 and 1')
    torch.set_num_threads(8)
    args.func(args)


if __name__ == '__main__':
    main()
