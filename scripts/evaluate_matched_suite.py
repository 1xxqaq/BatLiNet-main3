"""Explicit final evaluation of all four models, after all 32 runs finish."""
import argparse
import json
import statistics
from pathlib import Path

import torch

import context_grid_lifetime as common
import matched_baselines as baseline
from src.models.rul_predictors.context_grid_lifetime import ContextGridLifetime


def preflight(folder, data, sha, seed, architecture):
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(s) for s in (folder / 'train.jsonl').read_text().splitlines() if s.strip()]
    if [r['epoch'] for r in records] != list(range(1, 1001)):
        raise ValueError(f'Incomplete 1000-epoch run: {folder}')
    if [r['epoch'] for r in records if 'validation' in r] != list(range(25, 1001, 25)):
        raise ValueError(f'Validation schedule differs: {folder}')
    args = info['arguments']
    for key, value in dict(seed=seed, epochs=1000, batch_size=8, accumulation=16,
                           evaluate_every=25, lr=.001, amp=True).items():
        if args.get(key) != value:
            raise ValueError(f'Unmatched setting {key}: {folder}')
    checkpoint = common.load(folder / 'best.pt')
    best = min((r for r in records if 'validation' in r), key=lambda r: r['validation']['RMSE'])
    if checkpoint['epoch'] != best['epoch'] or checkpoint['seed'] != seed:
        raise ValueError(f'Best checkpoint/log mismatch: {folder}')
    if checkpoint['refit'] or checkpoint['selection_metric'] != 'RMSE':
        raise ValueError(f'Unexpected checkpoint policy: {folder}')
    if checkpoint['data_sha256'] != sha or info['data_sha256'] != sha:
        raise ValueError(f'Cache fingerprint differs: {folder}')
    if checkpoint['train_ids'] != data['train']['ids'] or info['train_ids'] != data['train']['ids']:
        raise ValueError(f'Training cell order differs: {folder}')
    config = checkpoint['config']
    if architecture in ('context', 'local_only'):
        if config != dict(cycles=20, phase_points=256, channels=64, bins=32,
                          heads=4, dropout=.1, use_context=architecture == 'context', alpha=.5):
            raise ValueError(f'Unexpected context architecture: {folder}')
    elif config != dict(architecture=architecture, cycles=20, width=1000):
        raise ValueError(f'Unexpected baseline architecture: {folder}')
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--context-data', default='artifacts/context_grid_v1/mix20.pt')
    parser.add_argument('--baseline-data', default='artifacts/mix20_matched_baselines_v1/mix20.pt')
    parser.add_argument('--context-runs', default='workspaces/context_grid_v1')
    parser.add_argument('--baseline-runs', default='workspaces/mix20_matched_baselines_v1')
    parser.add_argument('--protocol-dir', default='protocols/mix20_matched_v1')
    parser.add_argument('--output-dir', default='evaluations/mix20_matched_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--run-test', action='store_true', help='Without this flag, only audit all runs.')
    args = parser.parse_args()
    torch.set_num_threads(8)
    context, old = common.load(args.context_data), common.load(args.baseline_data)
    baseline.check_alignment(context, old, args.context_data)
    context_sha, baseline_sha = common.digest(args.context_data), common.digest(args.baseline_data)
    jobs, validation = [], []
    # Audit every run before allowing even the first test prediction.
    for name in ('context', 'local_only', 'batlinet', 'latent_cross_attention'):
        modern = name in ('context', 'local_only')
        data, sha = (context, context_sha) if modern else (old, baseline_sha)
        root = Path(args.context_runs if modern else args.baseline_runs)
        for seed in range(8):
            folder = root / (f'mix20_{name}_seed{seed}' if modern else f'{name}_seed{seed}')
            checkpoint = preflight(folder, data, sha, seed, name)
            records = [json.loads(s) for s in (folder / 'train.jsonl').read_text().splitlines() if s.strip()]
            best = next(r for r in records if r['epoch'] == checkpoint['epoch'])
            validation.append(dict(model=name, seed=seed, epoch=checkpoint['epoch'], **best['validation']))
            valpath = Path(args.protocol_dir) / f'val_seed{seed}.pt'
            baseline.fixed_protocol(context, seed, 'val', valpath)
            if not modern:
                info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
                if info['validation_protocol_sha256'] != common.digest(valpath):
                    raise ValueError(f'Validation protocol fingerprint changed: {folder}')
            jobs.append((name, seed, folder, data, checkpoint, modern))
    print('All 32 runs passed preflight. No test metrics computed yet.', flush=True)
    for name in ('context', 'local_only', 'batlinet', 'latent_cross_attention'):
        print(f'\nValidation: {name}')
        rows = [r for r in validation if r['model'] == name]
        for row in rows:
            print(json.dumps(row))
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [r[key] for r in rows]
            print(f'{key}: {statistics.mean(values):.6f} +/- {statistics.stdev(values):.6f}')
    if not args.run_test:
        return
    report = []
    for name, seed, folder, data, checkpoint, modern in jobs:
        protocol_path = Path(args.protocol_dir) / f'test_seed{seed}.pt'
        protocol = baseline.fixed_protocol(context, seed, 'test', protocol_path)
        destination = Path(args.output_dir) / name / f'seed{seed}.pt'
        provenance = dict(seed=seed, architecture=name, epoch=checkpoint['epoch'],
            checkpoint_sha256=common.digest(folder / 'best.pt'),
            protocol_sha256=common.digest(protocol_path), data_sha256=checkpoint['data_sha256'],
            context_sha256=context_sha, partition='test', refit=False)
        if destination.exists():
            result = common.load(destination)
            if any(result.get(k) != v for k, v in provenance.items()):
                raise ValueError(f'Existing test result provenance mismatch: {destination}')
        else:
            common.seed_all(seed)
            cls = ContextGridLifetime if modern else baseline.MatchedBaseline
            model = cls(**checkpoint['config']).to(args.device)
            model.load_state_dict(checkpoint['state'])
            evaluator = common.evaluate if modern else baseline.evaluate
            result = evaluator(model, data['train'], data['test'], protocol['indices'],
                checkpoint['label_mean'], checkpoint['label_scale'], args.device)
            result.update(provenance)
            common.save_new(result, destination)
            del model
        row = dict(model=name, seed=seed, epoch=checkpoint['epoch'], **result['scores'])
        report.append(row)
        print(json.dumps(row), flush=True)
    output = Path(args.output_dir) / 'summary.json'
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
