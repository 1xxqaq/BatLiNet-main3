"""Four independent soft-difference controls on MIX-20 and MIX-100."""
import argparse
import copy
import csv
import json
import os
import statistics
import subprocess
import traceback
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix_cycle_soft_difference as base
from src.models.rul_predictors import soft_difference_refinements  # registration
import torch
from torch.utils.data import DataLoader

common = base.common
TASKS = base.TASKS
VARIANTS = ('soft_baseline', 'content_values', 'no_matching_dropout', 'pair_supervision')
METRICS = ('RMSE', 'MAE', 'MAPE', 'ACC15', 'own_RMSE', 'reference_RMSE', 'pair_MSE_z')


def model_config(task, variant):
    if task not in TASKS or variant not in VARIANTS:
        raise ValueError('Unknown task or variant.')
    cfg = copy.deepcopy(base.model_config(task))
    model = cfg['model']
    model.pop('cycle_mixer_hidden')
    model['name'] = 'SoftAlignedDifferenceBatLiNetRULPredictor'
    if variant != 'soft_baseline':
        model['name'] = 'SoftDifferenceRefinementBatLiNetRULPredictor'
        model.update(content_only_values=variant == 'content_values',
                     matching_dropout=0. if variant == 'no_matching_dropout' else None,
                     pair_loss_weight=.1 if variant == 'pair_supervision' else 0.)
    cfg['runner']['train_reference_rng'] = 'independent_cpu_generator_seed_plus_1000003_times_epoch'
    return cfg


def sources():
    hashes = base.sources()
    for name in ('scripts/run_mix_soft_difference_refinements.py',
                 'scripts/start_mix_soft_difference_refinements_nohup.sh'):
        hashes[name] = common.digest(ROOT/name)
    return hashes


def reference_indices(size, support_size, seed, epoch):
    # Entire epoch sampled before model computation: dropout cannot change pairs.
    generator = torch.Generator(device='cpu').manual_seed(seed + 1000003*epoch)
    return torch.randint(size, (size, support_size), generator=generator)


def train_epoch(model, optimizer, ds, micro, monitor_every, epoch, num_test):
    model.train()
    indices = reference_indices(len(ds), model.train_support_size, model.seed, epoch).to(ds.device)
    total, steps, micros, offset = 0., 0, 0, 0
    for batch in DataLoader(ds, model.train_batch_size, shuffle=False):
        x, y = batch['feature'], batch['label']
        batch_indices = indices[offset:offset+len(x)]
        offset += len(x)
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(x), micro):
            end = min(start+micro, len(x))
            sx, sy = model.get_support_set(x[start:end], ds.feature, ds.label,
                fixed_indices=batch_indices[start:end], support_is_prepared=True)
            loss = model(x[start:end], y[start:end], sx, sy, return_loss=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失非有限。')
            (loss*((end-start)/len(x))).backward()
            total += loss.detach().item()*(end-start)
            micros += 1
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('梯度非有限。')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('参数非有限。')
        steps += 1
    if epoch % monitor_every == 0:
        model.eval()
        base.historical20.advance_old_monitor_rng(
            len(ds), num_test, model.test_support_size, model.test_batch_size, ds.device)
    return dict(train_loss=total/len(ds), optimizer_steps=steps, micro_batches=micros)


def evaluate_run(data, cfg, seed, folder, binding, indices, device):
    row = base.evaluate_run(data, cfg, seed, folder, binding, indices, device)
    result = common.load(folder/'test.pt')
    diag = result['diagnostics']
    for key, output in (('own_RMSE', 'y_ori'), ('reference_RMSE', 'y_sup_agg')):
        cycles = (diag[output]*data['scale']+data['mean']).exp()
        row[key] = common.scores(cycles, data['test']['label'])['RMSE']
    row['pair_MSE_z'] = (diag['y_sup']-data['test']['label_z'][:, None]).square().mean().item()
    return row


def write_summary(rows, workspace):
    common.atomic_json(rows, workspace/'per_seed.json')
    path = workspace/'per_seed.csv'
    with path.with_suffix('.csv.tmp').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'model', 'seed', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(path.with_suffix('.csv.tmp'), path)
    summary, comparisons = [], []
    for task in TASKS:
        control = {r['seed']: r for r in rows if r['task'] == task and r['model'] == VARIANTS[0]}
        for variant in VARIANTS:
            selected = [r for r in rows if r['task'] == task and r['model'] == variant]
            if len(selected) != 8 or {r['seed'] for r in selected} != set(range(8)):
                continue
            summary.append(dict(task=task, model=variant, seeds=8,
                comparison='concurrent_control', metrics={k: dict(
                    mean=statistics.mean(r[k] for r in selected),
                    sample_std=statistics.stdev(r[k] for r in selected)) for k in METRICS}))
            if variant != VARIANTS[0] and set(control) == set(range(8)):
                comparisons.append(dict(task=task, model=variant, baseline=VARIANTS[0],
                    metrics={k: dict(mean_delta=statistics.mean(r[k]-control[r['seed']][k] for r in selected),
                                    wins=sum((r[k]>control[r['seed']][k]) if k=='ACC15' else
                                             (r[k]<control[r['seed']][k]) for r in selected)) for k in METRICS}))
    common.atomic_json(summary, workspace/'summary.json')
    common.atomic_json(comparisons, workspace/'paired_comparisons.json')


def execute_queue(workspace, configs, identity, shas, device):
    rows = []
    for task in TASKS:
        data = common.load(workspace/'datasets'/task/'data.pt')
        bindings = {}
        # No test-set prediction until all four variants finish training this task.
        for variant in VARIANTS:
            cfg = configs[task][variant]
            for seed in range(8):
                binding = dict(task=task, model=variant, seed=seed, config=cfg,
                    data_sha256=shas[task], queue=identity,
                    protocol_sha256=common.digest(workspace/'protocols'/task/f'seed_{seed}.pt'))
                bindings[variant, seed] = binding
                if device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(device)
                base.train_run(data, cfg, seed, workspace/task/f'{variant}_seed{seed}',
                               binding, device, epoch_runner=train_epoch)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        for variant in VARIANTS:
            for seed in range(8):
                indices = common.load(workspace/'protocols'/task/f'seed_{seed}.pt')['indices']
                row = evaluate_run(data, configs[task][variant], seed,
                    workspace/task/f'{variant}_seed{seed}', bindings[variant, seed], indices, device)
                rows.append(row)
                write_summary(rows, workspace)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        del data
    if len(rows) != 64:
        raise ValueError('结果数不完整。')
    common.atomic_json(dict(status='complete', training_runs=64, predictions=64), workspace/'status.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--workspace', type=Path, default=ROOT/'workspaces/mix_soft_difference_refinements_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    workspace = args.workspace.resolve()
    with common.queue_lock(workspace):
        try:
            if not args.device.startswith('cuda') or not torch.cuda.is_available():
                raise RuntimeError('正式队列要求 CUDA，不自动切换 CPU。')
            base.historical20.set_seed(0)
            configs = {t: {v: model_config(t, v) for v in VARIANTS} for t in TASKS}
            revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            identity = dict(git_commit=revision, source_code=sources(), runtime=base.runtime(args.device),
                configs=configs, history_root=str(args.history_root.resolve()),
                original_root=str(args.original_root.resolve()), runs=64)
            queue_path = workspace/'queue.json'
            if queue_path.exists() and json.loads(queue_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('已有队列代码、提交、配置、环境或来源变化，拒绝混用。')
            common.atomic_json(identity, queue_path)
            common.atomic_json(dict(status='preparing', pid=os.getpid(), runs=64), workspace/'status.json')
            gates, shas = {}, {}
            for task in TASKS:
                data, _, shas[task] = base.prepare_history(task, args.history_root.resolve(),
                                                          args.original_root.resolve(), workspace)
                gates[task] = {}
                for variant in VARIANTS:
                    torch.cuda.reset_peak_memory_stats(args.device)
                    gates[task][variant] = base.preflight(data, configs[task][variant], args.device)
                    torch.cuda.empty_cache()
                del data
            common.atomic_json(gates, workspace/'preflight.json')
            if args.prepare_only:
                common.atomic_json(dict(status='prepared', formal_training_started=False), workspace/'status.json')
                return
            print('预检通过：四组独立对照，两任务八种子各1000轮，共64次训练。', flush=True)
            execute_queue(workspace, configs, identity, shas, args.device)
            print('已完成。汇总：'+str(workspace/'summary.json'), flush=True)
        except BaseException as error:
            common.atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()),
                               workspace/'status.json')
            raise


if __name__ == '__main__':
    main()
