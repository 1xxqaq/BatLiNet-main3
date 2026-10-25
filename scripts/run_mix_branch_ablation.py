"""Frozen MIX branch ablation: 48 trainings and 64 final evaluations."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix_cycle_soft_difference as base
from src.data.databundle import Dataset
from src.models.rul_predictors.branch_ablation_batlinet import build_branch_model
import torch

common = base.common
TASKS = ('mix_20', 'mix_100')
TRAIN_MODES = ('full', 'self_only', 'reference_only')
RESULT_MODES = TRAIN_MODES + ('independent_ensemble',)
SEEDS = tuple(range(8))
METRICS = ('RMSE', 'MAE', 'MAPE', 'ACC15')
REFERENCE_SEED_OFFSET = 310000


def model_config(task, mode):
    if task not in TASKS or mode not in TRAIN_MODES:
        raise ValueError('未知任务或分支设置。')
    cfg = copy.deepcopy(base.model_config(task))
    cfg['model']['branch_mode'] = mode
    cfg['runner'].update(
        schema=2, train_reference_rng='independent_cpu_generator_seed_plus_310000',
        periodic_rng_compensation=False, periodic_rng_every=None,
        initialization='same_seed_untrained_full_then_prune',
        self_loss_weight=.5, reference_loss_weight=.5,
        output_mode=mode, test_gate='all_48_trainings_complete',
        ensemble_space='standardized_natural_log', ensemble_weight=.5)
    return cfg


def sources():
    hashes = base.sources()
    for name in ('scripts/run_mix_branch_ablation.py',
                 'scripts/start_mix_branch_ablation_nohup.sh'):
        hashes[name] = common.digest(ROOT / name)
    return hashes


def reference_generator(seed):
    return torch.Generator(device='cpu').manual_seed(seed + REFERENCE_SEED_OFFSET)


def prepared(model, part, device):
    return Dataset(model._prepare_feature(part['feature'].to(device)),
                   part['label_z'].to(device))


def train_epoch(model, optimizer, ds, micro, generator):
    """One optimizer step per logical batch; references have their own RNG."""
    if micro < 1:
        raise ValueError('物理批量必须为正整数。')
    model.train()
    total, steps, micros = 0., 0, 0
    reference_hash = hashlib.sha256()
    for start in range(0, len(ds), model.train_batch_size):
        end = min(start + model.train_batch_size, len(ds))
        indices = torch.randint(len(ds), (end-start, model.train_support_size),
                                generator=generator, device='cpu')
        reference_hash.update(indices.contiguous().numpy().tobytes())
        # Also advance this dedicated stream in self_only, for auditable pairing.
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, end, micro):
            stop = min(offset + micro, end)
            sx = sy = None
            if model.requires_reference:
                sx, sy = model.get_support_set(
                    ds.feature[offset:stop], ds.feature, ds.label,
                    fixed_indices=indices[offset-start:stop-start],
                    support_is_prepared=True)
            loss = model(ds.feature[offset:stop], label=ds.label[offset:stop],
                         support_feature=sx, support_label=sy,
                         return_loss=True, prepared=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失非有限。')
            (loss * ((stop-offset)/(end-start))).backward()
            total += loss.detach().item() * (stop-offset)
            micros += 1
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for p in model.parameters()):
            raise FloatingPointError('训练梯度非有限。')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('优化器更新后参数非有限。')
        steps += 1
    return dict(train_loss=total/len(ds), optimizer_steps=steps,
                micro_batches=micros, reference_fingerprint=reference_hash.hexdigest())


def validate_complete(folder, binding):
    final = folder / 'final.pt'
    cp = common.checked_checkpoint(final, binding)
    if cp['epoch'] != binding['config']['model']['epochs']:
        raise ValueError(f'训练未达到固定最终轮次：{folder}')
    records = [json.loads(line) for line in (folder/'train.jsonl').read_text(
        encoding='utf-8').splitlines()]
    if ([r['epoch'] for r in records] != list(range(1, cp['epoch']+1))
            or any(not math.isfinite(r['train_loss']) for r in records)
            or any(not isinstance(r.get('reference_fingerprint'), str)
                   or len(r['reference_fingerprint']) != 64 for r in records)):
        raise ValueError(f'训练日志不完整或参考抽样记录缺失：{folder}')
    info = json.loads((folder/'run.json').read_text(encoding='utf-8'))
    if (info['binding'] != binding or info.get('final_sha256') != common.digest(final)
            or info.get('status') not in ('trained', 'complete')):
        raise ValueError(f'最终权重与训练回执不一致：{folder}')
    if 'test_sha256' in info and (not (folder/'test.pt').is_file()
            or info['test_sha256'] != common.digest(folder/'test.pt')):
        raise ValueError(f'已完成的测试文件与原回执指纹不同：{folder}')
    if any(not torch.isfinite(v).all() for v in cp['state'].values()):
        raise ValueError(f'最终权重存在非有限值：{folder}')
    return cp


def train_run(data, cfg, seed, folder, binding, device):
    folder.mkdir(parents=True, exist_ok=True)
    common.seed_all(seed)
    model = build_branch_model(cfg['model'], seed, cfg['model']['branch_mode']).to(device)
    if (folder/'final.pt').exists():
        cp = validate_complete(folder, binding)
        model.load_state_dict(cp['state'], strict=True)
        print(f'跳过已核验训练：{folder.parent.name}/{folder.name}', flush=True)
        return
    ds = prepared(model, data['train'], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'],
                                 weight_decay=cfg['runner']['weight_decay'])
    generator = reference_generator(seed)
    progress = folder/'progress.pt'
    start = 0
    if progress.exists():
        saved = common.checked_checkpoint(progress, binding)
        start = saved['epoch']
        model.load_state_dict(saved['state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        common.restore_rng(saved['rng'], generator, device)
    else:
        if (folder/'run.json').exists():
            prior = json.loads((folder/'run.json').read_text(encoding='utf-8'))
            if prior['binding'] != binding:
                raise ValueError('已有目录属于不同的分支消融设置。')
        common.save(dict(epoch=0, binding=binding, state=model.state_dict(),
                         optimizer=optimizer.state_dict(),
                         rng=common.rng_state(generator, device)), progress)
    common.align_log(folder, start)
    parameters = sum(p.numel() for p in model.parameters())
    common.atomic_json(dict(status='training', binding=binding, parameters=parameters),
                       folder/'run.json')
    elapsed = 0.
    with (folder/'train.jsonl').open('a', encoding='utf-8') as stream:
        for epoch in range(start+1, cfg['model']['epochs']+1):
            tick = time.monotonic()
            record = train_epoch(model, optimizer, ds, cfg['runner']['micro_batch_size'], generator)
            seconds = time.monotonic()-tick
            elapsed += seconds
            record.update(epoch=epoch, seconds=seconds,
                          cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30
                          if device.startswith('cuda') else 0.)
            stream.write(json.dumps(record)+'\n')
            stream.flush()
            if epoch % cfg['runner']['checkpoint_every'] == 0 or epoch == cfg['model']['epochs']:
                os.fsync(stream.fileno())
                common.save(dict(epoch=epoch, binding=binding, state=model.state_dict(),
                                 optimizer=optimizer.state_dict(),
                                 rng=common.rng_state(generator, device)), progress)
                remaining = elapsed/(epoch-start)*(cfg['model']['epochs']-epoch)
                common.atomic_json(dict(status='training', task=binding['task'],
                    model=binding['model'], seed=seed, epoch=epoch,
                    current_run_remaining_seconds=remaining), folder.parent.parent/'status.json')
                print(f'{binding["task"]}/{binding["model"]} 种子{seed} '
                      f'[{epoch}/{cfg["model"]["epochs"]}] 损失 {record["train_loss"]:.6f}，'
                      f'当前训练预计剩余 {remaining/60:.1f} 分钟', flush=True)
    common.save(dict(epoch=cfg['model']['epochs'], binding=binding,
                     state=model.state_dict()), folder/'final.pt')
    common.atomic_json(dict(status='trained', binding=binding, parameters=parameters,
                           final_sha256=common.digest(folder/'final.pt')), folder/'run.json')


def validate_indices(indices, data):
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
            or tuple(indices.shape) != (len(data['test']['ids']), 32)
            or indices.min().item() < 0 or indices.max().item() >= len(data['train']['ids'])):
        raise ValueError('固定32参考的形状、类型或范围错误。')


@torch.no_grad()
def predict(model, data, indices, device, chunk):
    validate_indices(indices, data)
    if not 1 <= chunk <= 32:
        raise ValueError('参考分块必须在1到32之间。')
    model.eval()
    query = prepared(model, data['test'], device)
    training = prepared(model, data['train'], device) if model.requires_reference else None
    own, references = [], []
    for i in range(len(query)):
        x = query.feature[i:i+1]
        if model.requires_reference:
            pairs = []
            for start in range(0, 32, chunk):
                sx, sy = model.get_support_set(x, training.feature, training.label,
                    fixed_indices=indices[i:i+1, start:start+chunk], support_is_prepared=True)
                value = model.predict_components(x, sx, sy, prepared=True)
                pairs.append(value['y_sup'].cpu())
            references.append(torch.cat(pairs, dim=1))
        else:
            value = model.predict_components(x, prepared=True)
        if value['y_ori'] is not None:
            own.append(value['y_ori'].cpu())
    ori = torch.cat(own) if own else None
    sup = torch.cat(references) if references else None
    agg = sup.median(dim=1).values if sup is not None else None
    z = ori if agg is None else agg if ori is None else .5*(ori+agg)
    cycles = (z*data['scale']+data['mean']).exp()
    return dict(branch_mode=model.branch_mode, prediction=z, prediction_cycles=cycles,
        truth_cycles=data['test']['label'], label_mean=data['mean'],
        label_sample_std=data['scale'], train_ids=data['train']['ids'],
        test_ids=data['test']['ids'], parameters=sum(p.numel() for p in model.parameters()),
        diagnostics=dict(y_ori=ori, y_sup=sup, y_sup_agg=agg, support_index=indices.cpu()),
        reference_indices_used=model.requires_reference,
        diagnostic_units='standardized_natural_log',
        scores=common.scores(cycles, data['test']['label']))


def validate_result(result, mode, data=None, indices=None):
    if result['branch_mode'] != mode:
        raise ValueError('预测文件的分支设置错误。')
    if 'binding' in result and result['binding']['model'] != mode:
        raise ValueError('预测与来源绑定中的分支名称不同。')
    diag = result['diagnostics']
    n = len(result['test_ids'])
    ori, sup, agg = (diag[k] for k in ('y_ori', 'y_sup', 'y_sup_agg'))
    fixed = diag['support_index']
    if (fixed.dtype != torch.int64 or tuple(fixed.shape) != (n, 32)
            or fixed.min().item() < 0 or fixed.max().item() >= len(result['train_ids'])):
        raise ValueError('预测保存的固定参考索引不合法。')
    if (not torch.isfinite(result['label_mean']).all()
            or not torch.isfinite(result['label_sample_std']).all()
            or (result['label_sample_std'] <= 0).any()):
        raise ValueError('预测保存的标签变换不合法。')
    if mode == 'self_only':
        if ori is None or sup is not None or agg is not None or result['reference_indices_used']:
            raise ValueError('仅自身结果混入参考分支。')
        z = ori
    else:
        if not result['reference_indices_used']:
            raise ValueError('参考分支错误地标记为未使用参考。')
        if sup is None or tuple(sup.shape) != (n, 32) or agg is None:
            raise ValueError('参考结果没有保存完整32份单参考预测。')
        torch.testing.assert_close(agg, sup.median(1).values)
        if mode == 'reference_only':
            if ori is not None:
                raise ValueError('仅参考结果混入自身分支。')
            z = agg
        else:
            if ori is None:
                raise ValueError('融合结果缺少自身分支。')
            z = .5*(ori+agg)
    for value in (z, result['prediction'], result['prediction_cycles'], result['truth_cycles']):
        if tuple(value.shape) != (n,) or not torch.isfinite(value).all():
            raise ValueError('预测或真值形状错误或含非有限值。')
    torch.testing.assert_close(result['prediction'], z)
    torch.testing.assert_close(result['prediction_cycles'],
        (z*result['label_sample_std']+result['label_mean']).exp())
    if data is not None:
        if any(result[k+'_ids'] != data[k]['ids'] for k in ('train', 'test')):
            raise ValueError('预测对应的电池或顺序变化。')
        torch.testing.assert_close(result['truth_cycles'], data['test']['label'], rtol=0, atol=0)
        for key, source in (('label_mean', 'mean'), ('label_sample_std', 'scale')):
            torch.testing.assert_close(result[key], data[source], rtol=0, atol=0)
    if indices is not None and not torch.equal(diag['support_index'], indices):
        raise ValueError('预测实际参考索引与冻结协议不同。')
    measured = common.scores(result['prediction_cycles'], result['truth_cycles'])
    if measured != result['scores'] or any(not math.isfinite(v) for v in measured.values()):
        raise ValueError('保存的指标不能从逐电池预测复算。')
    return measured


def evaluate_run(data, cfg, seed, folder, binding, indices, device):
    cp = validate_complete(folder, binding)
    sha = common.digest(folder/'final.pt')
    path = folder/'test.pt'
    mode = cfg['model']['branch_mode']
    if path.exists():
        result = common.load(path)
    else:
        common.seed_all(seed)
        model = build_branch_model(cfg['model'], seed, mode).to(device)
        model.load_state_dict(cp['state'], strict=True)
        result = predict(model, data, indices, device, cfg['runner']['pair_chunk'])
        result.update(binding=binding, checkpoint_sha256=sha)
        validate_result(result, mode, data, indices)
        common.save(result, path)
    if result['binding'] != binding or result['checkpoint_sha256'] != sha:
        raise ValueError('预测与训练权重、来源信息不匹配。')
    measured = validate_result(result, mode, data, indices)
    common.atomic_json(dict(status='complete', binding=binding, parameters=result['parameters'],
        final_sha256=sha, test_sha256=common.digest(path), scores=measured), folder/'run.json')
    return dict(task=binding['task'], model=mode, seed=seed, **measured)


def ensemble_result(own_result, reference_result):
    """No optimizer or learned fusion; independently trained single branches."""
    validate_result(own_result, 'self_only')
    validate_result(reference_result, 'reference_only')
    a, b = own_result['binding'], reference_result['binding']
    for key in ('task', 'seed', 'data_sha256', 'queue', 'protocol_sha256'):
        if a[key] != b[key]:
            raise ValueError(f'独立集成的两个来源不匹配：{key}')
    for key in ('train_ids', 'test_ids'):
        if own_result[key] != reference_result[key]:
            raise ValueError('独立集成的电池顺序不同。')
    for key in ('truth_cycles', 'label_mean', 'label_sample_std'):
        torch.testing.assert_close(own_result[key], reference_result[key], rtol=0, atol=0)
    if not torch.equal(own_result['diagnostics']['support_index'],
                       reference_result['diagnostics']['support_index']):
        raise ValueError('独立集成的参考协议上下文不同。')
    result = copy.deepcopy(reference_result)
    result.update(branch_mode='independent_ensemble', reference_indices_used=True,
        parameters=own_result['parameters']+reference_result['parameters'],
        binding=dict(task=a['task'], model='independent_ensemble', seed=a['seed'],
                     data_sha256=a['data_sha256'], queue=a['queue'],
                     protocol_sha256=a['protocol_sha256'], fusion_space='standardized_natural_log',
                     self_weight=.5, reference_weight=.5),
        source_checkpoint_sha256={m:r['checkpoint_sha256'] for m,r in
                                 (('self_only',own_result),('reference_only',reference_result))},
        source_test_sha256={m:r['test_sha256'] for m,r in
                           (('self_only',own_result),('reference_only',reference_result))})
    result.pop('checkpoint_sha256', None)
    result.pop('test_sha256', None)
    result['diagnostics']['y_ori'] = own_result['prediction'].clone()
    result['prediction'] = .5*(own_result['prediction']+reference_result['prediction'])
    result['prediction_cycles'] = (result['prediction']*result['label_sample_std']
                                   +result['label_mean']).exp()
    result['scores'] = common.scores(result['prediction_cycles'], result['truth_cycles'])
    validate_result(result, 'independent_ensemble')
    return result


def evaluate_ensemble(data, seed, workspace, task, indices):
    inputs = []
    for mode in ('self_only', 'reference_only'):
        path = workspace/task/f'{mode}_seed{seed}'/'test.pt'
        result = common.load(path)
        result['test_sha256'] = common.digest(path)
        inputs.append(result)
    expected = ensemble_result(*inputs)
    folder = workspace/task/f'independent_ensemble_seed{seed}'
    path = folder/'test.pt'
    if path.exists():
        receipt = folder/'run.json'
        if receipt.exists():
            info = json.loads(receipt.read_text(encoding='utf-8'))
            if info.get('test_sha256', common.digest(path)) != common.digest(path):
                raise ValueError('已有独立集成文件与原回执指纹不同。')
        actual = common.load(path)
        for key in ('binding', 'source_checkpoint_sha256', 'source_test_sha256', 'parameters'):
            if actual[key] != expected[key]:
                raise ValueError('已有独立集成结果的来源变化。')
        torch.testing.assert_close(actual['prediction'], expected['prediction'], rtol=0, atol=0)
    else:
        actual = expected
        common.save(actual, path)
    measured = validate_result(actual, 'independent_ensemble', data, indices)
    common.atomic_json(dict(status='complete', binding=actual['binding'],
        parameters=actual['parameters'], test_sha256=common.digest(path),
        source_checkpoint_sha256=actual['source_checkpoint_sha256'],
        source_test_sha256=actual['source_test_sha256'], scores=measured), folder/'run.json')
    return dict(task=task, model='independent_ensemble', seed=seed, **measured)


def summarize(rows):
    keys = [(r['task'], r['model'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)) or any(t not in TASKS or m not in RESULT_MODES
                                        or s not in SEEDS for t,m,s in keys):
        raise ValueError('指标表存在重复或未知任务、模型、种子。')
    summary, comparisons = [], []
    for task in TASKS:
        full = {r['seed']:r for r in rows if r['task']==task and r['model']=='full'}
        for mode in RESULT_MODES:
            group = [r for r in rows if r['task']==task and r['model']==mode]
            if len(group) != 8:
                continue
            summary.append(dict(task=task, model=mode, seeds=8, comparison='matched_branch_ablation',
                metrics={k:dict(mean=statistics.mean(r[k] for r in group),
                                sample_std=statistics.stdev(r[k] for r in group)) for k in METRICS}))
            if mode != 'full' and len(full)==8:
                comparisons.append(dict(task=task, model=mode, baseline='full',
                    delta_definition='model_minus_full',
                    metrics={k:dict(mean_delta=statistics.mean(r[k]-full[r['seed']][k] for r in group),
                                   wins=sum(r[k]>full[r['seed']][k] if k=='ACC15' else
                                            r[k]<full[r['seed']][k] for r in group)) for k in METRICS}))
    return summary, comparisons


def write_summary(rows, workspace):
    summary, comparisons = summarize(rows)
    common.atomic_json(rows, workspace/'per_seed.json')
    temp = workspace/'per_seed.csv.tmp'
    with temp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['task', 'model', 'seed', *METRICS])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, workspace/'per_seed.csv')
    common.atomic_json(summary, workspace/'summary.json')
    common.atomic_json(comparisons, workspace/'paired_comparisons.json')


def make_binding(task, mode, seed, cfg, identity, sha, workspace):
    return dict(task=task, model=mode, seed=seed, config=cfg, data_sha256=sha,
                queue=identity,
                protocol_sha256=common.digest(workspace/'protocols'/task/f'seed_{seed}.pt'))


def check_all_trainings(workspace, configs, identity, shas):
    bindings = {}
    for task in TASKS:
        logs = {}
        for mode in TRAIN_MODES:
            for seed in SEEDS:
                folder = workspace/task/f'{mode}_seed{seed}'
                binding = make_binding(task, mode, seed, configs[task][mode], identity, shas[task], workspace)
                validate_complete(folder, binding)
                records = [json.loads(s) for s in (folder/'train.jsonl').read_text(encoding='utf-8').splitlines()]
                sequence = [r['reference_fingerprint'] for r in records]
                if seed in logs and logs[seed] != sequence:
                    raise ValueError(f'{task}种子{seed}三种模式训练参考序列不同。')
                logs[seed] = sequence
                bindings[task, mode, seed] = binding
    return bindings


def execute_queue(workspace, configs, identity, shas, device):
    total = len(TASKS)*len(TRAIN_MODES)*len(SEEDS)
    current = 0
    for task in TASKS:
        data = common.load(workspace/'datasets'/task/'data.pt')
        for mode in TRAIN_MODES:
            for seed in SEEDS:
                current += 1
                print(f'训练队列 {current}/{total}：{task}/{mode}/seed{seed}', flush=True)
                binding = make_binding(task, mode, seed, configs[task][mode], identity, shas[task], workspace)
                if device.startswith('cuda'):
                    torch.cuda.reset_peak_memory_stats(device)
                train_run(data, configs[task][mode], seed, workspace/task/f'{mode}_seed{seed}',
                          binding, device)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
        del data
    # No test-set inference before every task and every training mode is complete.
    bindings = check_all_trainings(workspace, configs, identity, shas)
    common.atomic_json(dict(status='evaluating', training_runs=total, complete_training_gate=True),
                       workspace/'status.json')
    rows = []
    for task in TASKS:
        data = common.load(workspace/'datasets'/task/'data.pt')
        for seed in SEEDS:
            indices = common.load(workspace/'protocols'/task/f'seed_{seed}.pt')['indices']
            for mode in TRAIN_MODES:
                row = evaluate_run(data, configs[task][mode], seed,
                    workspace/task/f'{mode}_seed{seed}', bindings[task,mode,seed], indices, device)
                rows.append(row)
                write_summary(rows, workspace)
                print(json.dumps(row, ensure_ascii=False), flush=True)
                if device.startswith('cuda'):
                    torch.cuda.empty_cache()
            row = evaluate_ensemble(data, seed, workspace, task, indices)
            rows.append(row)
            write_summary(rows, workspace)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        del data
    expected = len(TASKS)*len(RESULT_MODES)*len(SEEDS)
    if len(rows) != expected:
        raise ValueError('最终评价条数不完整。')
    common.atomic_json(dict(status='complete', training_runs=total, predictions=expected,
                           independent_ensemble_trained=False), workspace/'status.json')


def preflight(data, cfg, device):
    common.seed_all(0)
    model = build_branch_model(cfg['model'], 0, cfg['model']['branch_mode']).to(device)
    ds = prepared(model, data['train'], device)
    count = min(len(ds), cfg['runner']['micro_batch_size'])
    sx = sy = None
    if model.requires_reference:
        indices = torch.randint(len(ds), (count, 2), generator=reference_generator(0))
        sx, sy = model.get_support_set(ds.feature[:count], ds.feature, ds.label,
                                      fixed_indices=indices, support_is_prepared=True)
    loss = model(ds.feature[:count], label=ds.label[:count], support_feature=sx,
                 support_label=sy, return_loss=True, prepared=True)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                     for p in model.parameters()):
        raise ValueError('预检训练前向或梯度不合法。')
    toy = dict(data, test={key:value[:1] for key,value in data['train'].items()})
    fixed = torch.arange(32).reshape(1,32)%len(ds)
    result = predict(model, toy, fixed, device, cfg['runner']['pair_chunk'])
    validate_result(result, model.branch_mode, toy, fixed)
    return dict(parameters=result['parameters'], train_count=len(ds),
                test_count=len(data['test']['ids']), finite=True,
                loss=float(loss.detach()), actual_test_inference=False)


def report_workspace(workspace):
    """Read existing per-cell results; no training, inference, or file writes."""
    path = workspace/'per_seed.json'
    if not path.is_file():
        print('尚无最终评价结果；训练进度：', flush=True)
        status = workspace/'status.json'
        print(status.read_text(encoding='utf-8') if status.exists() else '队列尚未启动。')
        return []
    rows = json.loads(path.read_text(encoding='utf-8'))
    summarize(rows)  # Reject duplicate or unexpected keys before resolving paths.
    queue_path = workspace/'queue.json'
    queue = json.loads(queue_path.read_text(encoding='utf-8')) if queue_path.exists() else None
    for row in rows:
        folder = workspace/row['task']/f'{row["model"]}_seed{row["seed"]}'
        receipt = json.loads((folder/'run.json').read_text(encoding='utf-8'))
        if receipt['test_sha256'] != common.digest(folder/'test.pt'):
            raise ValueError('测试文件与保存的完成回执指纹不同。')
        result = common.load(folder/'test.pt')
        if receipt['binding'] != result['binding']:
            raise ValueError('测试文件与完成回执的来源不同。')
        binding = result['binding']
        if any(binding[key] != row[key] for key in ('task', 'model', 'seed')):
            raise ValueError('指标表与逐电池结果的任务、模型或种子不同。')
        if queue is not None:
            data_receipt = json.loads((workspace/'datasets'/row['task']/'receipt.json').read_text(encoding='utf-8'))
            if (binding['queue'] != queue or binding['data_sha256'] != data_receipt['sha256']
                    or binding['protocol_sha256'] != common.digest(
                        workspace/'protocols'/row['task']/f'seed_{row["seed"]}.pt')):
                raise ValueError('结果与队列、数据或固定参考来源不一致。')
        indices = common.load(workspace/'protocols'/row['task']/f'seed_{row["seed"]}.pt')['indices']
        measured = validate_result(result, row['model'], indices=indices)
        if any(row[k] != measured[k] for k in METRICS):
            raise ValueError('汇总与逐电池指标不同。')
        if row['model'] == 'independent_ensemble':
            inputs = []
            for mode in ('self_only', 'reference_only'):
                source = workspace/row['task']/f'{mode}_seed{row["seed"]}'/'test.pt'
                value = common.load(source)
                if value['checkpoint_sha256'] != common.digest(source.parent/'final.pt'):
                    raise ValueError('独立集成的单分支来源权重已变化。')
                inputs.append(dict(value, test_sha256=common.digest(source)))
            expected = ensemble_result(*inputs)
            for key in ('binding', 'source_checkpoint_sha256', 'source_test_sha256'):
                if result[key] != expected[key]:
                    raise ValueError('独立集成来源指纹错误。')
            torch.testing.assert_close(result['prediction'], expected['prediction'], rtol=0, atol=0)
        else:
            validate_complete(folder, binding)
            if common.digest(folder/'final.pt') != result['checkpoint_sha256']:
                raise ValueError('结果对应的训练权重已变化。')
    summary, _ = summarize(rows)
    print(f'已核对 {len(rows)}/64 份逐电池评价；以下仅显示完整八种子组。')
    print('均值 ± 样本标准差；RMSE/MAE单位为循环，MAPE/ACC15显示为百分数。')
    for group in summary:
        print(f'{group["task"]} / {group["model"]}')
        for key, value in group['metrics'].items():
            factor = 100 if key in ('MAPE', 'ACC15') else 1
            suffix = '%' if factor == 100 else ''
            print(f'  {key}: {factor*value["mean"]:.4f} ± {factor*value["sample_std"]:.4f}{suffix}')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--workspace', type=Path, default=ROOT/'workspaces/mix_branch_ablation_v1')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.report:
        report_workspace(workspace)
        return
    torch.set_num_threads(8)
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式队列要求CUDA；本地仅运行合成数据检查。')
    gpu_lock = ROOT/'workspaces/gpu_locks'/args.device.replace(':', '_')
    with common.queue_lock(workspace), common.queue_lock(gpu_lock):
        claimed = False
        try:
            common.seed_all(0)
            configs = {task:{mode:model_config(task,mode) for mode in TRAIN_MODES} for task in TASKS}
            identity = dict(schema=1, git_commit=subprocess.check_output(
                ['git','-C',str(ROOT),'rev-parse','HEAD'], text=True).strip(),
                source_code=sources(), runtime=base.runtime(args.device), configs=configs,
                history_root=str(args.history_root.resolve()), original_root=str(args.original_root.resolve()),
                training_runs=48, predictions=64)
            queue_path = workspace/'queue.json'
            if queue_path.exists() and json.loads(queue_path.read_text(encoding='utf-8')) != identity:
                raise ValueError('已有队列提交、源代码、环境、配置或历史来源不同，拒绝混用。')
            if not queue_path.exists() and any((workspace/name).exists() for name in
                                               ('per_seed.json', 'summary.json', 'datasets')):
                raise ValueError('输出目录存在缺少队列来源记录的旧材料，拒绝覆盖。')
            common.atomic_json(identity, queue_path)
            claimed = True
            common.atomic_json(dict(status='preparing', pid=os.getpid(), training_runs=48), workspace/'status.json')
            shas, gates = {}, {}
            for task in TASKS:
                data, _, shas[task] = base.prepare_history(task, args.history_root.resolve(),
                                                          args.original_root.resolve(), workspace)
                gates[task] = {}
                for mode in TRAIN_MODES:
                    torch.cuda.reset_peak_memory_stats(args.device)
                    gates[task][mode] = preflight(data, configs[task][mode], args.device)
                    torch.cuda.empty_cache()
                del data
            common.atomic_json(gates, workspace/'preflight.json')
            if args.prepare_only:
                common.atomic_json(dict(status='prepared', formal_training_started=False), workspace/'status.json')
                return
            print('预检通过：两任务各三种模式、八种子，共48次固定1000轮训练；'
                  '全部完成后评价64份结果，独立集成不另训。', flush=True)
            execute_queue(workspace, configs, identity, shas, args.device)
            report_workspace(workspace)
        except BaseException as error:
            if claimed:
                common.atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()),
                                   workspace/'status.json')
            raise


if __name__ == '__main__':
    main()
