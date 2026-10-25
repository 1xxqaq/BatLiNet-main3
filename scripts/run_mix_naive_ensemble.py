"""Synchronous dual-encoder naive ensemble; two MIX tasks, eight seeds each."""
import argparse
import contextlib
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix_branch_ablation as branch
from src.models.rul_predictors.naive_branch_ensemble_batlinet import (
    BranchRandomStreams, build_naive_model,
)
import torch

common, base = branch.common, branch.base
TASKS, SEEDS, METRICS = branch.TASKS, branch.SEEDS, branch.METRICS
MODEL = 'naive_ensemble'
reference_generator = branch.reference_generator
prepared = branch.prepared
predict = branch.predict
validate_result = branch.validate_result


def model_config(task):
    cfg = copy.deepcopy(branch.model_config(task, 'full'))
    cfg['model']['branch_mode'] = MODEL
    cfg['runner'].update(
        output_mode=MODEL, initialization='same_seed_full_copied_to_two_independent_branches',
        branch_rng_policy='isolated_self_and_reference_cpu_cuda_states_same_seed',
        optimizer_scope='one_AdamW_for_both_disjoint_parameter_sets',
        test_gate='all_16_trainings_complete')
    return cfg


def sources():
    hashes = branch.sources()
    for name in ('scripts/run_mix_naive_ensemble.py',
                 'scripts/start_mix_naive_ensemble_nohup.sh'):
        hashes[name] = common.digest(ROOT/name)
    return hashes


def train_epoch(model, optimizer, ds, micro, generator, streams):
    model.train()
    if micro < 1:
        raise ValueError('物理批量必须大于零。')
    total, own_total, relation_total, steps, micros = 0., 0., 0., 0, 0
    fingerprint = hashlib.sha256()
    for start in range(0, len(ds), model.train_batch_size):
        end = min(start+model.train_batch_size, len(ds))
        indices = torch.randint(len(ds), (end-start, model.train_support_size),
                                generator=generator, device='cpu')
        fingerprint.update(indices.numpy().tobytes())
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, end, micro):
            stop = min(offset+micro, end)
            sx, sy = model.get_support_set(ds.feature[offset:stop], ds.feature, ds.label,
                fixed_indices=indices[offset-start:stop-start], support_is_prepared=True)
            parts = model.predict_components(ds.feature[offset:stop], sx, sy,
                                              prepared=True, streams=streams)
            own_loss = .5*(parts['y_ori']-ds.label[offset:stop]).square().mean()
            reference_loss = .5*(parts['y_sup_agg']-ds.label[offset:stop]).square().mean()
            loss = own_loss+reference_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失非有限。')
            (loss*((stop-offset)/(end-start))).backward()
            total += loss.detach().item()*(stop-offset)
            own_total += own_loss.detach().item()*(stop-offset)
            relation_total += reference_loss.detach().item()*(stop-offset)
            micros += 1
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('训练梯度非有限。')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('优化器更新后参数非有限。')
        steps += 1
    return dict(train_loss=total/len(ds), self_loss=own_total/len(ds),
                reference_loss=relation_total/len(ds), optimizer_steps=steps,
                micro_batches=micros, reference_fingerprint=fingerprint.hexdigest())


def train_run(data, cfg, seed, folder, binding, device):
    folder.mkdir(parents=True, exist_ok=True)
    common.seed_all(seed)
    model = build_naive_model(cfg['model'], seed).to(device)
    if (folder/'final.pt').exists():
        repair_final_receipt(folder, binding, model)
        cp = branch.validate_complete(folder, binding)
        model.load_state_dict(cp['state'], strict=True)
        print(f'跳过已核验训练：{folder.parent.name}/{folder.name}', flush=True)
        return
    ds = prepared(model, data['train'], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'],
                                 weight_decay=cfg['runner']['weight_decay'])
    generator, streams = reference_generator(seed), BranchRandomStreams(seed, device)
    progress, start = folder/'progress.pt', 0
    if progress.exists():
        cp = common.checked_checkpoint(progress, binding)
        model.load_state_dict(cp['state'], strict=True)
        optimizer.load_state_dict(cp['optimizer'])
        common.restore_rng(cp['rng'], generator, device)
        streams.load_state_dict(cp['branch_rng'])
        start = cp['epoch']
    else:
        if (folder/'run.json').exists():
            old = json.loads((folder/'run.json').read_text(encoding='utf-8'))
            if old['binding'] != binding:
                raise ValueError('已有目录属于不同设置，拒绝覆盖。')
        common.save(dict(epoch=0, binding=binding, state=model.state_dict(),
            optimizer=optimizer.state_dict(), rng=common.rng_state(generator, device),
            branch_rng=streams.state_dict()), progress)
    common.align_log(folder, start)
    parameters = sum(p.numel() for p in model.parameters())
    common.atomic_json(dict(status='training', binding=binding, parameters=parameters), folder/'run.json')
    elapsed = 0.
    with (folder/'train.jsonl').open('a', encoding='utf-8') as log:
        for epoch in range(start+1, cfg['model']['epochs']+1):
            tick = time.monotonic()
            row = train_epoch(model, optimizer, ds, cfg['runner']['micro_batch_size'], generator, streams)
            row.update(epoch=epoch, seconds=time.monotonic()-tick,
                cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.)
            elapsed += row['seconds']
            log.write(json.dumps(row)+'\n')
            log.flush()
            if epoch % cfg['runner']['checkpoint_every'] == 0 or epoch == cfg['model']['epochs']:
                os.fsync(log.fileno())
                common.save(dict(epoch=epoch, binding=binding, state=model.state_dict(),
                    optimizer=optimizer.state_dict(), rng=common.rng_state(generator, device),
                    branch_rng=streams.state_dict()), progress)
                remaining = elapsed/(epoch-start)*(cfg['model']['epochs']-epoch)
                common.atomic_json(dict(status='training', task=binding['task'], seed=seed,
                    epoch=epoch, current_run_remaining_seconds=remaining), folder.parent.parent/'status.json')
                print(f'{binding["task"]}/{MODEL} 种子{seed} [{epoch}/{cfg["model"]["epochs"]}] '
                      f'损失 {row["train_loss"]:.6f}，当前训练预计剩余 {remaining/60:.1f} 分钟', flush=True)
    common.save(dict(epoch=cfg['model']['epochs'], binding=binding, state=model.state_dict()), folder/'final.pt')
    common.atomic_json(dict(status='trained', binding=binding, parameters=parameters,
        final_sha256=common.digest(folder/'final.pt')), folder/'run.json')


def repair_final_receipt(folder, binding, model):
    """Recover only a crash after final.pt save but before its receipt save."""
    info = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if info.get('status') != 'training' or 'final_sha256' in info:
        return
    if info['binding'] != binding:
        raise ValueError('未完成的最终权重回执属于不同设置。')
    final = common.checked_checkpoint(folder/'final.pt', binding)
    progress = common.checked_checkpoint(folder/'progress.pt', binding)
    expected = binding['config']['model']['epochs']
    if (final['epoch'] != expected or progress['epoch'] != expected
            or final['state'].keys() != progress['state'].keys()
            or any(not torch.equal(v, progress['state'][k]) or not torch.isfinite(v).all()
                   for k,v in final['state'].items())):
        raise ValueError('最终权重与最终恢复检查点不同，拒绝补写完成回执。')
    model.load_state_dict(final['state'], strict=True)
    records = [json.loads(s) for s in (folder/'train.jsonl').read_text(encoding='utf-8').splitlines()]
    if ([r['epoch'] for r in records] != list(range(1,expected+1))
            or any(not math.isfinite(r['train_loss'])
                   or len(r.get('reference_fingerprint','')) != 64 for r in records)):
        raise ValueError('最终训练日志不完整，拒绝补写完成回执。')
    common.atomic_json(dict(status='trained', binding=binding,
        parameters=sum(p.numel() for p in model.parameters()),
        final_sha256=common.digest(folder/'final.pt'),
        receipt_recovered_from_verified_final_progress=True), folder/'run.json')


def evaluate_run(data, cfg, seed, folder, binding, indices, device):
    cp = branch.validate_complete(folder, binding)
    sha, path = common.digest(folder/'final.pt'), folder/'test.pt'
    if path.exists():
        result = common.load(path)
    else:
        common.seed_all(seed)
        model = build_naive_model(cfg['model'], seed).to(device)
        model.load_state_dict(cp['state'], strict=True)
        result = predict(model, data, indices, device, cfg['runner']['pair_chunk'])
        result.update(binding=binding, checkpoint_sha256=sha)
        validate_result(result, MODEL, data, indices)
        common.save(result, path)
    if result['binding'] != binding or result['checkpoint_sha256'] != sha:
        raise ValueError('测试结果与训练来源或最终权重不同。')
    measured = validate_result(result, MODEL, data, indices)
    common.atomic_json(dict(status='complete', binding=binding, parameters=result['parameters'],
        final_sha256=sha, test_sha256=common.digest(path), scores=measured), folder/'run.json')
    return dict(task=binding['task'], model=MODEL, seed=seed, **measured)


def write_summary(rows, workspace):
    keys = [(r['task'], r['model'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or any(t not in TASKS or m != MODEL or s not in SEEDS for t,m,s in keys):
        raise ValueError('汇总存在重复或未知任务、模型、种子。')
    summary = summarize(rows)
    common.atomic_json(rows, workspace/'per_seed.json')
    temp = workspace/'per_seed.csv.tmp'
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task','model','seed',*METRICS])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, workspace/'per_seed.csv')
    common.atomic_json(summary, workspace/'summary.json')


def summarize(rows):
    summary = []
    for task in TASKS:
        group = [r for r in rows if r['task']==task]
        if len(group) == len(SEEDS) and {r['seed'] for r in group} == set(SEEDS):
            summary.append(dict(task=task, model=MODEL, seeds=len(group),
                metrics={k:dict(mean=statistics.mean(r[k] for r in group),
                    sample_std=statistics.stdev(r[k] for r in group) if len(group)>1 else 0.) for k in METRICS}))
    return summary


def make_binding(task, seed, cfg, identity, sha, workspace):
    return dict(task=task, model=MODEL, seed=seed, config=cfg, queue=identity,
        data_sha256=sha, protocol_sha256=common.digest(workspace/'protocols'/task/f'seed_{seed}.pt'))


def check_all_trainings(workspace, configs, identity, shas):
    bindings = {}
    for task in TASKS:
        for seed in SEEDS:
            binding = make_binding(task, seed, configs[task], identity, shas[task], workspace)
            branch.validate_complete(workspace/task/f'{MODEL}_seed{seed}', binding)
            bindings[task,seed] = binding
    return bindings


def compare_previous(data, task, seed, folder, branch_workspace):
    """Optional read-only comparison; missing/rejected history never becomes data."""
    receipt = dict(task=task, seed=seed, status='unavailable', scope='optional_same_protocol_reference')
    if branch_workspace is None:
        return receipt
    try:
        result = common.load(folder/'test.pt')
        current = common.load(folder/'final.pt')['state']
        old_receipt = json.loads((branch_workspace/'datasets'/task/'receipt.json').read_text(encoding='utf-8'))
        if old_receipt['historical_binding']['dataset'] != data['historical_binding']['dataset']:
            raise ValueError('旧分支消融的实际输入或标签不同。')
        comparisons = {}
        old_predictions, old_scores = {}, {}
        new_cfg = copy.deepcopy(result['binding']['config']['model'])
        new_cfg.pop('branch_mode')
        for mode, prefix in (('self_only','self_branch.'), ('reference_only','reference_branch.')):
            source = branch_workspace/task/f'{mode}_seed{seed}'
            old = common.load(source/'test.pt')
            old_cfg = copy.deepcopy(old['binding']['config']['model'])
            old_cfg.pop('branch_mode')
            if old_cfg != new_cfg or old['binding']['seed'] != seed or old['binding']['task'] != task:
                raise ValueError('旧单分支的模型配置、任务或种子不同。')
            if old['binding']['queue']['runtime'] != result['binding']['queue']['runtime']:
                raise ValueError('运行环境不同，不作为逐位等价复核。')
            old_sources = old['binding']['queue'].get('source_code')
            if old_sources is not None:
                new_sources = result['binding']['queue'].get('source_code', {})
                if any(new_sources.get(name) != sha for name,sha in old_sources.items()):
                    raise ValueError('原单分支依赖代码发生变化，不作为逐位等价复核。')
            for key in ('micro_batch_size','train_reference_rng','checkpoint_every','weight_decay'):
                if old['binding']['config']['runner'][key] != result['binding']['config']['runner'][key]:
                    raise ValueError('旧单分支训练设置不同。')
            branch.validate_complete(source, old['binding'])
            if old['checkpoint_sha256'] != common.digest(source/'final.pt'):
                raise ValueError('旧单分支权重指纹错误。')
            validate_result(old, mode, data, result['diagnostics']['support_index'])
            state = common.load(source/'final.pt')['state']
            target = {k[len(prefix):]:v for k,v in current.items() if k.startswith(prefix)}
            if target.keys() != state.keys():
                raise ValueError('旧单分支参数键不同。')
            new_log = [json.loads(s) for s in (folder/'train.jsonl').read_text(encoding='utf-8').splitlines()]
            old_log = [json.loads(s) for s in (source/'train.jsonl').read_text(encoding='utf-8').splitlines()]
            if [r['reference_fingerprint'] for r in old_log] != [r['reference_fingerprint'] for r in new_log]:
                raise ValueError('训练参考抽样序列不同。')
            comparisons[mode] = dict(
                parameters_bitwise_equal=all(torch.equal(target[k], state[k]) for k in state),
                maximum_parameter_absolute_difference=max((target[k]-state[k]).abs().max().item() for k in state),
                source_weight_sha256=common.digest(source/'final.pt'),
                source_prediction_sha256=common.digest(source/'test.pt'))
            old_predictions[mode] = old['prediction']
            old_scores[mode] = old['scores']
        expected = .5*(old_predictions['self_only']+old_predictions['reference_only'])
        old_scores['independent_ensemble'] = common.scores(
            (expected*data['scale']+data['mean']).exp(), data['test']['label'])
        receipt.update(status='verified_comparison', branches=comparisons,
            current_scores=result['scores'], previous_scores=old_scores,
            prediction_max_absolute_difference_z=(expected-result['prediction']).abs().max().item(),
            interpretation='numerical_equivalence_check_not_new_cooperation_mechanism')
        try:
            source = branch_workspace/task/f'full_seed{seed}'
            full = common.load(source/'test.pt')
            config = copy.deepcopy(full['binding']['config']['model'])
            config.pop('branch_mode')
            if (config != new_cfg or full['binding']['task'] != task or full['binding']['seed'] != seed
                    or full['binding']['queue'] != old['binding']['queue']):
                raise ValueError('完整模型与已核验的旧单分支队列不同。')
            branch.validate_complete(source, full['binding'])
            if full['checkpoint_sha256'] != common.digest(source/'final.pt'):
                raise ValueError('完整模型权重指纹不同。')
            validate_result(full, 'full', data, result['diagnostics']['support_index'])
            records = [json.loads(s) for s in (source/'train.jsonl').read_text(encoding='utf-8').splitlines()]
            if [r['reference_fingerprint'] for r in records] != [r['reference_fingerprint'] for r in new_log]:
                raise ValueError('完整模型训练参考序列不同。')
            old_scores['full'] = full['scores']
            receipt['full_source_prediction_sha256'] = common.digest(source/'test.pt')
        except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
            receipt['full_unavailable_reason'] = str(error)
    except (OSError, ValueError, KeyError, AssertionError, RuntimeError) as error:
        receipt.update(status='unavailable_or_rejected', reason=str(error))
    return receipt


def write_previous_comparisons(receipts, workspace):
    rows = []
    for task in TASKS:
        for name in ('full','self_only','reference_only','independent_ensemble'):
            group = [r for r in receipts if r.get('status')=='verified_comparison' and r.get('task')==task
                     and name in r['previous_scores']]
            if len(group) != len(SEEDS) or {r['seed'] for r in group} != set(SEEDS):
                continue
            rows.append(dict(task=task, model=MODEL, baseline=name, seeds=len(group),
                delta_definition='naive_ensemble_minus_previous_model',
                metrics={key:dict(
                    baseline_mean=statistics.mean(r['previous_scores'][name][key] for r in group),
                    mean_delta=statistics.mean(r['current_scores'][key]-r['previous_scores'][name][key] for r in group),
                    wins=sum(r['current_scores'][key]>r['previous_scores'][name][key] if key=='ACC15' else
                             r['current_scores'][key]<r['previous_scores'][name][key] for r in group)) for key in METRICS}))
    common.atomic_json(rows, workspace/'paired_previous_comparisons.json')


def execute_queue(workspace, configs, identity, shas, device, branch_workspace=None):
    total, current = len(TASKS)*len(SEEDS), 0
    for task in TASKS:
        data = common.load(workspace/'datasets'/task/'data.pt')
        for seed in SEEDS:
            current += 1
            print(f'同步朴素集成训练 {current}/{total}：{task}/seed{seed}', flush=True)
            binding = make_binding(task, seed, configs[task], identity, shas[task], workspace)
            if device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(device)
            train_run(data, configs[task], seed, workspace/task/f'{MODEL}_seed{seed}', binding, device)
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        del data
    bindings = check_all_trainings(workspace, configs, identity, shas)
    common.atomic_json(dict(status='evaluating', training_runs=total, complete_training_gate=True), workspace/'status.json')
    rows, references = [], []
    for task in TASKS:
        data = common.load(workspace/'datasets'/task/'data.pt')
        for seed in SEEDS:
            fixed = common.load(workspace/'protocols'/task/f'seed_{seed}.pt')['indices']
            folder = workspace/task/f'{MODEL}_seed{seed}'
            row = evaluate_run(data, configs[task], seed, folder, bindings[task,seed], fixed, device)
            rows.append(row)
            write_summary(rows, workspace)
            references.append(compare_previous(data, task, seed, folder, branch_workspace))
            common.atomic_json(references, workspace/'independent_equivalence.json')
            write_previous_comparisons(references, workspace)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
        del data
    common.atomic_json(dict(status='complete', training_runs=total, predictions=len(rows)), workspace/'status.json')


def preflight(data, cfg, device):
    common.seed_all(0)
    model = build_naive_model(cfg['model'], 0).to(device)
    ds = prepared(model, data['train'], device)
    n = min(len(ds), cfg['runner']['micro_batch_size'])
    ix = torch.randint(len(ds), (n,2), generator=reference_generator(0))
    sx, sy = model.get_support_set(ds.feature[:n], ds.feature, ds.label,
                                 fixed_indices=ix, support_is_prepared=True)
    loss = model(ds.feature[:n], label=ds.label[:n], support_feature=sx, support_label=sy,
                 return_loss=True, prepared=True, streams=BranchRandomStreams(0, device))
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                     for p in model.parameters()):
        raise ValueError('朴素集成预检前向或梯度失败。')
    toy = dict(data, test={k:v[:1] for k,v in data['train'].items()})
    fixed = torch.arange(32).reshape(1,32)%len(ds)
    output = predict(model, toy, fixed, device, cfg['runner']['pair_chunk'])
    validate_result(output, MODEL, toy, fixed)
    return dict(parameters=output['parameters'], finite=True, loss=float(loss.detach()),
        train_count=len(ds), test_count=len(data['test']['ids']), actual_test_inference=False,
        cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.)


def report_workspace(workspace):
    path = workspace/'per_seed.json'
    if not path.exists():
        status = workspace/'status.json'
        print(status.read_text(encoding='utf-8') if status.exists() else '队列尚未启动。')
        return []
    rows = json.loads(path.read_text(encoding='utf-8'))
    if len({(r['task'],r['seed']) for r in rows}) != len(rows):
        raise ValueError('结果汇总存在重复。')
    queue_path = workspace/'queue.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8')) if queue_path.exists() else None
    for row in rows:
        if row['task'] not in TASKS or row['model'] != MODEL or row['seed'] not in SEEDS:
            raise ValueError('结果含未知任务、模型或种子。')
        folder = workspace/row['task']/f'{MODEL}_seed{row["seed"]}'
        result = common.load(folder/'test.pt')
        binding = result['binding']
        branch.validate_complete(folder, binding)
        if any(binding[k] != row[k] for k in ('task','model','seed')):
            raise ValueError('结果任务或种子与指标表不同。')
        if common.digest(folder/'final.pt') != result['checkpoint_sha256']:
            raise ValueError('最终权重与预测指纹不同。')
        fixed_path = workspace/'protocols'/row['task']/f'seed_{row["seed"]}.pt'
        if queue is not None:
            receipt = json.loads((workspace/'datasets'/row['task']/'receipt.json').read_text(encoding='utf-8'))
            if (binding['queue'] != queue or binding['data_sha256'] != receipt['sha256']
                    or binding['protocol_sha256'] != common.digest(fixed_path)):
                raise ValueError('队列、数据或参考来源绑定不同。')
        scores = validate_result(result, MODEL, indices=common.load(fixed_path)['indices'])
        if any(row[k] != scores[k] for k in METRICS):
            raise ValueError('汇总不能从逐电池结果复算。')
    summary = summarize(rows)
    print(f'已核对 {len(rows)}/16 份评价；以下仅显示完整八种子组。')
    print('均值 ± 样本标准差；RMSE/MAE为循环，MAPE/ACC15显示百分数。')
    for item in summary:
        print(f'{item["task"]}/{MODEL}')
        for key, value in item['metrics'].items():
            factor = 100 if key in ('MAPE','ACC15') else 1
            print(f'  {key}: {value["mean"]*factor:.4f} ± {value["sample_std"]*factor:.4f}'
                  + ('%' if factor==100 else ''))
    return summary


@contextlib.contextmanager
def gpu_slot(workspace, branch_workspace, device):
    """Use the original branch queue's lock, including across Git worktrees."""
    if os.name != 'posix':
        raise RuntimeError('正式队列需在Linux服务器运行。')
    import fcntl
    lock_dir = branch_workspace.parent/'gpu_locks'/device.replace(':','_')
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir/'.queue.lock').open('a') as stream:
        elapsed = 0
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if elapsed % 60 == 0:
                    common.atomic_json(dict(status='waiting_gpu', device=device,
                        waited_seconds=elapsed, lock_path=str(lock_dir/'.queue.lock')), workspace/'status.json')
                    print(f'等待原分支消融释放 {device}，已等待{elapsed//60}分钟；不会终止其他进程。', flush=True)
                time.sleep(10)
                elapsed += 10
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--workspace', type=Path, default=ROOT/'workspaces/mix_naive_ensemble_v1')
    parser.add_argument('--branch-workspace', type=Path, default=Path('/root/autodl-tmp/BatLiNet-cycle-mixer/workspaces/mix_branch_ablation_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    workspace, prior = args.workspace.resolve(), args.branch_workspace.resolve()
    if args.report:
        report_workspace(workspace)
        return
    if workspace == prior:
        raise ValueError('新旧实验必须使用不同输出目录。')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式队列要求CUDA；不自动切换CPU。')
    torch.set_num_threads(8)
    with common.queue_lock(workspace):
        claimed = False
        try:
            common.seed_all(0)
            configs = {t:model_config(t) for t in TASKS}
            identity = dict(schema=1, git_commit=subprocess.check_output(
                ['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
                source_code=sources(), runtime=base.runtime(args.device), configs=configs,
                history_root=str(args.history_root.resolve()), original_root=str(args.original_root.resolve()),
                branch_workspace=str(prior), training_runs=16, predictions=16)
            path = workspace/'queue.json'
            if path.exists() and json.loads(path.read_text(encoding='utf-8')) != identity:
                raise ValueError('已有队列源码、配置、环境或来源变化，拒绝混用。')
            if not path.exists() and any((workspace/p).exists() for p in ('datasets','per_seed.json','summary.json')):
                raise ValueError('目录有缺少队列来源记录的旧结果，拒绝覆盖。')
            common.atomic_json(identity, path)
            claimed = True
            with gpu_slot(workspace, prior, args.device):
                common.atomic_json(dict(status='preparing', pid=os.getpid(), training_runs=16), workspace/'status.json')
                shas, gates = {}, {}
                for task in TASKS:
                    data, _, shas[task] = base.prepare_history(task, args.history_root.resolve(),
                        args.original_root.resolve(), workspace)
                    torch.cuda.reset_peak_memory_stats(args.device)
                    gates[task] = preflight(data, configs[task], args.device)
                    del data
                    torch.cuda.empty_cache()
                common.atomic_json(gates, workspace/'preflight.json')
                if args.prepare_only:
                    common.atomic_json(dict(status='prepared', formal_training_started=False), workspace/'status.json')
                    return
                print('预检通过：同步朴素集成，两任务各八种子固定1000轮，共16次训练。', flush=True)
                execute_queue(workspace, configs, identity, shas, args.device, prior)
                report_workspace(workspace)
        except BaseException as error:
            if claimed:
                common.atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()), workspace/'status.json')
            raise


if __name__ == '__main__':
    main()
