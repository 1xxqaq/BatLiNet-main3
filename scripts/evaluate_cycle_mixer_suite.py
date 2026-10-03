"""Audit matched MIX-20 runs and optionally evaluate the fixed eight-seed suite."""
import argparse
import json
import math
import statistics
from pathlib import Path

import torch

import context_grid_lifetime as common
import matched_baselines as runner


ARCHITECTURES = ('latent_cross_attention', 'latent_cycle_mixer', 'latent_cycle_conv')


def model_config(architecture):
    config = dict(architecture=architecture, cycles=20, width=1000)
    if architecture != 'latent_cross_attention':
        config.update(cycle_mixer_hidden=16, cycle_conv_kernel=3)
    return config


def preflight(folder, data, data_sha, context_sha, seed, architecture, protocol_path):
    """Check completion, selection policy and the exact data/reference identity."""
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (folder / 'train.jsonl').read_text().splitlines()
               if line.strip()]
    if [r['epoch'] for r in records] != list(range(1, 1001)):
        raise ValueError(f'Incomplete 1000-epoch run: {folder}')
    validations = [r for r in records if 'validation' in r]
    if [r['epoch'] for r in validations] != list(range(25, 1001, 25)):
        raise ValueError(f'Validation schedule differs: {folder}')
    if any(not math.isfinite(r['validation']['RMSE']) for r in validations):
        raise ValueError(f'Nonfinite validation RMSE: {folder}')
    expected = dict(model=architecture, seed=seed, epochs=1000, batch_size=8,
                    accumulation=16, evaluate_every=25, lr=.001, amp=True, pair_chunk=8)
    for key, value in expected.items():
        if info['arguments'].get(key) != value:
            raise ValueError(f'Unmatched setting {key}: {folder}')
    checkpoint = common.load(folder / 'best.pt')
    best = min(validations, key=lambda record: record['validation']['RMSE'])
    if checkpoint['epoch'] != best['epoch'] or checkpoint['seed'] != seed:
        raise ValueError(f'Best checkpoint/log mismatch: {folder}')
    if checkpoint['refit'] or checkpoint['selection_metric'] != 'RMSE':
        raise ValueError(f'Unexpected checkpoint policy: {folder}')
    for name, value in (('data_sha256', data_sha), ('context_sha256', context_sha),
                        ('train_ids', data['train']['ids'])):
        if checkpoint[name] != value or info[name] != value:
            raise ValueError(f'Checkpoint/run identity differs ({name}): {folder}')
    if not protocol_path.is_file() or info['validation_protocol_sha256'] != common.digest(protocol_path):
        raise ValueError(f'Validation references changed: {folder}')
    config = model_config(architecture)
    if checkpoint['config'] != config or info['model'] != config:
        raise ValueError(f'Unexpected encoder configuration: {folder}')
    label = data['train']['labels'].log()
    torch.testing.assert_close(checkpoint['label_mean'], label.mean())
    torch.testing.assert_close(checkpoint['label_scale'], label.std(unbiased=False).clamp_min(1e-6))
    return checkpoint, best


def print_summary(rows, partition):
    print(f'{partition}指标汇总；MAPE、ACC15 均为小数比例。', flush=True)
    for architecture in ARCHITECTURES:
        selected = [r for r in rows if r['model'] == architecture]
        if not selected:
            continue
        print(architecture, flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [row[key] for row in selected]
            deviation = statistics.stdev(values) if len(values) > 1 else 0.
            print(f'  {key}: {statistics.mean(values):.6f} ± {deviation:.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    existing = Path('/root/autodl-tmp/BatLiNet-context-grid-v1')
    parser.add_argument('--context-data', default=str(existing / 'artifacts/context_grid_v1/mix20.pt'))
    parser.add_argument('--data', default=str(existing / 'artifacts/mix20_matched_baselines_v1/mix20.pt'))
    parser.add_argument('--baseline-runs', default=str(existing / 'workspaces/mix20_matched_baselines_v1'))
    parser.add_argument('--runs', default='workspaces/mix20_cycle_mixer_v1')
    parser.add_argument('--protocol-dir', default=str(existing / 'protocols/mix20_matched_v1'))
    parser.add_argument('--output-dir', default='evaluations/mix20_cycle_mixer_v1')
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--run-test', action='store_true', help='Evaluate only after all eight seeds pass the audit.')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in range(8) for seed in args.seeds):
        parser.error('Seeds must be distinct integers in 0-7.')
    args.seeds.sort()
    if args.run_test and (args.baseline_only or args.seeds != list(range(8))):
        parser.error('Test evaluation requires all three models and seeds 0-7.')
    torch.set_num_threads(8)
    context, data = common.load(args.context_data), common.load(args.data)
    runner.check_alignment(context, data, args.context_data)
    if context['metadata']['dataset'] != 'mix20':
        raise ValueError('This runner is for the existing MIX-20 experiment only.')
    sizes = tuple(len(data[part]['ids']) for part in ('train', 'val', 'test'))
    if sizes != (166, 41, 147):
        raise ValueError(f'The existing MIX-20 split changed: {sizes}')
    data_sha, context_sha = common.digest(args.data), common.digest(args.context_data)
    jobs, validation = [], []
    names = ARCHITECTURES[:1] if args.baseline_only else ARCHITECTURES
    # Audit the complete requested suite before computing any test prediction.
    for name in names:
        root = Path(args.baseline_runs if name == 'latent_cross_attention' else args.runs)
        for seed in args.seeds:
            folder = root / f'{name}_seed{seed}'
            protocol_path = Path(args.protocol_dir) / f'val_seed{seed}.pt'
            if not protocol_path.is_file():
                raise FileNotFoundError(protocol_path)
            runner.fixed_protocol(context, seed, 'val', protocol_path)
            checkpoint, best = preflight(folder, data, data_sha, context_sha, seed, name, protocol_path)
            validation.append(dict(model=name, seed=seed, epoch=checkpoint['epoch'], **best['validation']))
            jobs.append((name, seed, folder, checkpoint))
    print(f'已核对 {len(jobs)} 次训练及其验证参考名单。', flush=True)
    print_summary(validation, '验证集')
    if args.baseline_only:
        return
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation_summary.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.run_test:
        print('本次只汇总验证集；没有计算测试集指标。', flush=True)
        return
    report = []
    for name, seed, folder, checkpoint in jobs:
        protocol_path = Path(args.protocol_dir) / f'test_seed{seed}.pt'
        protocol = runner.fixed_protocol(context, seed, 'test', protocol_path)
        destination = output / name / f'seed{seed}.pt'
        provenance = dict(seed=seed, architecture=name, epoch=checkpoint['epoch'],
            checkpoint_sha256=common.digest(folder / 'best.pt'),
            protocol_sha256=common.digest(protocol_path), data_sha256=data_sha,
            context_sha256=context_sha, partition='test', refit=False)
        if destination.exists():
            result = common.load(destination)
            if any(result.get(key) != value for key, value in provenance.items()):
                raise ValueError(f'Existing result belongs to a different run: {destination}')
        else:
            common.seed_all(seed)
            model = runner.MatchedBaseline(**checkpoint['config']).to(args.device)
            model.load_state_dict(checkpoint['state'], strict=True)
            result = runner.evaluate(model, data['train'], data['test'], protocol['indices'],
                checkpoint['label_mean'], checkpoint['label_scale'], args.device, pair_chunk=8)
            result.update(provenance)
            common.save_new(result, destination)
            del model
        row = dict(model=name, seed=seed, epoch=checkpoint['epoch'], **result['scores'])
        report.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(report, '测试集')


if __name__ == '__main__':
    main()
