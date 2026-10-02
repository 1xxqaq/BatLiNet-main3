"""Matched MIX-20 baselines; never call the original fit/test-during-fit loops."""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import context_grid_lifetime as common
from src.data.battery_data import BatteryData
from src.feature.batlinet import BatLiNetFeatureExtractor
from src.label.rul import RULLabelAnnotator
from src.models.rul_predictors.matched_baselines import MatchedBaseline


def check_alignment(context, baseline, context_path):
    if baseline['metadata']['context_sha256'] != common.digest(context_path):
        raise ValueError('Baseline cache belongs to different context data.')
    common.validate_disjoint({p: context[p]['ids'] for p in ('train', 'val', 'test')})
    for p in ('train', 'val', 'test'):
        if context[p]['ids'] != baseline[p]['ids'] or not torch.equal(
                context[p]['labels'], baseline[p]['labels']):
            raise ValueError(f'Cell IDs/order or labels differ: {p}')


def prepare(args):
    context = common.load(args.context_data)
    if context['metadata']['dataset'] != 'mix20':
        raise ValueError('This matched experiment currently supports MIX-20 only.')
    if Path(args.output).exists():
        existing = common.load(args.output)
        check_alignment(context, existing, args.context_data)
        print(f'Validated existing baseline cache: {args.output}')
        return
    root = Path(args.data_root).resolve()
    common.validate_disjoint({p: context[p]['ids'] for p in ('train', 'val', 'test')})
    from src.train_test_split.MIX20_split import MIX20TrainTestSplitter
    names = ['CALCE', 'RWTH', 'UL_PUR', 'SNL', 'MATR', 'HUST', 'HNEI']
    split = MIX20TrainTestSplitter([str(root / n) for n in names]).split()
    original = [set(Path(p).resolve() for p in paths) for paths in split]
    entries = context['metadata']['source_files']
    lookup = {r['cell_id']: r for r in entries}
    if len(lookup) != len(entries):
        raise ValueError('Duplicate cell IDs in source manifest.')
    selected = [set(), set()]
    extractor = BatLiNetFeatureExtractor(max_cycle_index=19, cycle_to_drop=10,
                                        smooth_features=False)
    labeler = RULLabelAnnotator(eol_soh=.9)
    payload = {'metadata': dict(schema_version=1, dataset='mix20',
        context_sha256=common.digest(args.context_data), code_commit=common.revision(),
        feature='Original BatLiNetFeatureExtractor: 6x20x1000, drop cycle index 10',
        feature_normalization='none (original baseline input policy)',
        label_normalization='log + training population standard deviation (matched v1)',
        source_files=entries)}
    for part in ('train', 'val', 'test'):
        values = []
        for i, cell_id in enumerate(context[part]['ids']):
            record = lookup[cell_id]
            path = (root / record['path']).resolve()
            if not path.is_relative_to(root) or record['split'] != part:
                raise ValueError(f'Invalid manifest entry: {cell_id}')
            group = 1 if part == 'test' else 0
            if path not in original[group]:
                raise ValueError(f'Cell belongs to wrong original split: {cell_id}')
            selected[group].add(path)
            if common.digest(path) != record['sha256']:
                raise ValueError(f'Source changed since context preparation: {cell_id}')
            cell = BatteryData.load(path)
            if cell.cell_id != cell_id:
                raise ValueError(f'Cell identity mismatch: {path}')
            label = torch.tensor(float(labeler.process_cell(cell)), dtype=torch.float32)
            if not torch.isclose(label, context[part]['labels'][i], atol=.01, rtol=1e-5):
                raise ValueError(f'Lifetime label mismatch: {cell_id}')
            feature = extractor.process_cell(cell).float().contiguous()
            if feature.shape != (6, 20, 1000) or not torch.isfinite(feature).all():
                raise ValueError(f'Invalid legacy features: {cell_id}')
            values.append(feature)
            if (i + 1) % 25 == 0:
                print(f'{part}: {i+1}/{len(context[part]["ids"])}', flush=True)
        payload[part] = dict(features={'raw': torch.stack(values)},
                             ids=context[part]['ids'], labels=context[part]['labels'])
    # Any cells omitted by the label filter must really have invalid labels.
    for group in (0, 1):
        for path in sorted(original[group] - selected[group]):
            value = torch.tensor(float(labeler.process_cell(BatteryData.load(path))))
            if torch.isfinite(value):
                raise ValueError(f'Valid original-split cell missing from context cache: {path}')
    check_alignment(context, payload, args.context_data)
    common.save_new(payload, args.output)
    Path(args.output).with_suffix('.json').write_text(
        json.dumps(payload['metadata'], ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({p: len(payload[p]['ids']) for p in ('train', 'val', 'test')}))


def fixed_protocol(context, seed, partition, path):
    """Validation exactly reproduces the already-run context training sampler."""
    offset = 100000 if partition == 'val' else 200000
    expected = dict(indices=common.make_indices(len(context[partition]['ids']),
        len(context['train']['ids']), seed + offset), train_ids=list(context['train']['ids']),
        query_ids=list(context[partition]['ids']), seed=seed, partition=partition)
    path = Path(path)
    if path.exists():
        actual = common.load(path)
        if any(actual.get(k) != expected[k] for k in ('train_ids', 'query_ids', 'seed', 'partition')):
            raise ValueError(f'Existing protocol identity differs: {path}')
        if not torch.equal(actual['indices'], expected['indices']):
            raise ValueError(f'Existing protocol references differ: {path}')
    else:
        common.save_new(expected, path)
    return expected


def protocols(args):
    context = common.load(args.context_data)
    for seed in range(8):
        for part in ('val', 'test'):
            fixed_protocol(context, seed, part, Path(args.output_dir) / f'{part}_seed{seed}.pt')
    print('Validated/saved protocols for seeds 0-7; no model or test metric evaluated.')


@torch.no_grad()
def evaluate(model, training, query, indices, mean, scale, device, pair_chunk=8):
    if indices.shape != (len(query['ids']), 32) or indices.min() < 0 or indices.max() >= len(training['ids']):
        raise ValueError('Invalid reference indices.')
    model.eval()
    y = (training['labels'].log() - mean) / scale
    own, support = [], []
    for i in range(len(query['ids'])):
        target = common.subset(query['features'], slice(i, i+1), device)
        pieces = []
        for start in range(0, 32, pair_chunk):
            ix = indices[i:i+1, start:start+pair_chunk]
            result = model(target, common.subset(training['features'], ix, device), y[ix].to(device))
            pieces.append(result['y_sup'].float().cpu())
        own.append(result['y_ori'].float().cpu())
        support.append(torch.cat(pieces, 1))
    own, support = torch.cat(own), torch.cat(support)
    aggregate = support.median(1).values  # One median over all 32, never medians of chunks.
    inverse = lambda z: (z * scale + mean).exp()
    prediction = inverse(.5 * own + .5 * aggregate)
    return dict(prediction=prediction, labels=query['labels'], scores=common.scores(prediction, query['labels']),
        branch_scores=dict(ori=common.scores(inverse(own), query['labels']),
                           support=common.scores(inverse(aggregate), query['labels'])),
        diagnostics=dict(y_ori=inverse(own), y_sup=inverse(support),
                         y_sup_agg=inverse(aggregate), support_index=indices),
        train_ids=training['ids'], test_ids=query['ids'], diagnostic_units='cycles after inverse transform')


def completed_run(workspace, expected, data_sha):
    info = json.loads((workspace / 'run.json').read_text(encoding='utf-8'))
    if info['data_sha256'] != data_sha or info['arguments'] != expected:
        raise ValueError(f'Existing run configuration/data differ: {workspace}')
    records = [json.loads(s) for s in (workspace / 'train.jsonl').read_text().splitlines() if s.strip()]
    if [r['epoch'] for r in records] != list(range(1, expected['epochs'] + 1)):
        raise ValueError(f'Incomplete run; no automatic overwrite/resume: {workspace}')
    best = min((r for r in records if 'validation' in r), key=lambda r: r['validation']['RMSE'])
    checkpoint = common.load(workspace / 'best.pt')
    if checkpoint['epoch'] != best['epoch'] or checkpoint['data_sha256'] != data_sha:
        raise ValueError(f'Checkpoint/log mismatch: {workspace}')


def train(args):
    context, data = common.load(args.context_data), common.load(args.data)
    check_alignment(context, data, args.context_data)
    sha = common.digest(args.data)
    arguments = {k: v for k, v in vars(args).items() if k not in ('func', 'skip_complete')}
    workspace = Path(args.workspace)
    if workspace.exists():
        if not args.skip_complete:
            raise FileExistsError(workspace)
        completed_run(workspace, arguments, sha)
        print(f'Skipping verified completed run: {workspace}')
        return
    fixed = fixed_protocol(context, args.seed, 'val', Path(args.protocol_dir) / f'val_seed{args.seed}.pt')['indices']
    common.seed_all(args.seed)
    config = dict(architecture=args.model, cycles=20, width=1000)
    if args.model in ('latent_cycle_mixer', 'latent_cycle_conv'):
        config.update(cycle_mixer_hidden=16, cycle_conv_kernel=3)
    model = MatchedBaseline(**config).to(args.device)
    training = data['train']
    y = training['labels'].log()
    mean, scale = y.mean(), y.std(unbiased=False).clamp_min(1e-6)
    y = (y - mean) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    amp = args.amp and torch.device(args.device).type == 'cuda'
    if amp and not torch.cuda.is_bf16_supported():
        raise ValueError('bfloat16 AMP unsupported; do not silently change matched settings.')
    workspace.mkdir(parents=True, exist_ok=False)
    info = dict(arguments=arguments, model=config, code_commit=common.revision(),
        data_sha256=sha, context_sha256=common.digest(args.context_data),
        train_ids=training['ids'], selection_metric='RMSE',
        validation_protocol_sha256=common.digest(Path(args.protocol_dir) / f'val_seed{args.seed}.pt'),
        parameters=sum(p.numel() for p in model.parameters()))
    (workspace / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    best = float('inf')
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        batches = list(torch.randperm(len(y)).split(args.batch_size))
        loss_sum = 0.
        for number, ix in enumerate(batches):
            si = torch.randint(len(y), (len(ix), 2))
            target = common.subset(training['features'], ix, args.device)
            references = common.subset(training['features'], si, args.device)
            group_start = number // args.accumulation * args.accumulation
            count = sum(len(b) for b in batches[group_start:group_start+args.accumulation])
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                out = model(target, references, y[si].to(args.device))
                label = y[ix].to(args.device)
                loss = .5 * (out['y_ori'].float() - label).square().mean() + .5 * (out['y_sup_agg'].float() - label).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite training loss: epoch {epoch}')
            (loss * len(ix) / count).backward()
            loss_sum += loss.item() * len(ix)
            if (number + 1) % args.accumulation == 0 or number + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
        record = dict(epoch=epoch, train_loss=loss_sum/len(y), train_seconds=time.perf_counter()-start,
            cuda_peak_GiB=torch.cuda.max_memory_allocated(args.device)/2**30 if torch.device(args.device).type == 'cuda' else 0)
        if epoch % args.evaluate_every == 0 or epoch == args.epochs:
            result = evaluate(model, training, data['val'], fixed, mean, scale, args.device, args.pair_chunk)
            record.update(validation=result['scores'], validation_branches=result['branch_scores'])
            if result['scores']['RMSE'] < best:
                best = result['scores']['RMSE']
                torch.save(dict(state={k: v.cpu() for k, v in model.state_dict().items()}, config=config,
                    epoch=epoch, label_mean=mean, label_scale=scale, seed=args.seed, refit=False,
                    data_sha256=sha, context_sha256=info['context_sha256'], train_ids=training['ids'],
                    selection_metric='RMSE'), workspace / 'best.pt')
        with (workspace / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        print(json.dumps(record), flush=True)
    print('Training complete. Test set was not evaluated.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    q = commands.add_parser('prepare')
    q.add_argument('--context-data', required=True)
    q.add_argument('--data-root', required=True)
    q.add_argument('--output', required=True)
    q.set_defaults(func=prepare)
    q = commands.add_parser('protocols')
    q.add_argument('--context-data', required=True)
    q.add_argument('--output-dir', required=True)
    q.set_defaults(func=protocols)
    q = commands.add_parser('train')
    q.add_argument('--context-data', required=True)
    q.add_argument('--data', required=True)
    q.add_argument('--protocol-dir', required=True)
    q.add_argument('--workspace', required=True)
    q.add_argument('--model', choices=['batlinet', 'latent_cross_attention',
                                     'latent_cycle_mixer', 'latent_cycle_conv'], required=True)
    q.add_argument('--seed', type=int, required=True)
    q.add_argument('--epochs', type=int, default=1000)
    q.add_argument('--batch-size', type=int, default=8)
    q.add_argument('--accumulation', type=int, default=16)
    q.add_argument('--evaluate-every', type=int, default=25)
    q.add_argument('--lr', type=float, default=.001)
    q.add_argument('--pair-chunk', type=int, default=8)
    q.add_argument('--device', default='cuda:0')
    q.add_argument('--amp', action='store_true')
    q.add_argument('--skip-complete', action='store_true')
    q.set_defaults(func=train)
    args = p.parse_args()
    for name in ('epochs', 'batch_size', 'accumulation', 'evaluate_every', 'pair_chunk'):
        if hasattr(args, name) and getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    torch.set_num_threads(8)
    args.func(args)


if __name__ == '__main__':
    main()
