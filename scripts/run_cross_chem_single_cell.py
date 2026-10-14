"""Same encoder: target-only and supervised LFP pretraining/fine-tuning."""
import argparse
import contextlib
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_three_tasks as common
from scripts import run_cross_chem_cycle_soft_difference as combo
from scripts import transfer_cycle_soft_data as data
from src.models.rul_predictors.cycle_residual_single_cell import CycleResidualSingleCell

DIRECT = 'cycle_single_direct'
FINETUNE = 'cycle_single_LFP_finetune'
PRETRAIN = 'cycle_single_LFP_pretrain'
MODELS = (DIRECT, FINETUNE)
LABELS = {DIRECT: '单电池版直接目标训练', FINETUNE: '单电池版LFP预训练后微调', PRETRAIN: 'LFP来源预训练', combo.MODEL: '已有完整组合版（参考比较）'}
CONFIG = ROOT / 'configs/cross_chem_single_cell_v1/protocol.yaml'
DEFAULT_WORKSPACE = ROOT / 'workspaces/cross_chem_single_cell_v1'
DEFAULT_BASE = ROOT / 'workspaces/cross_chem_two_models_v2'
CHINA = timezone(timedelta(hours=8))


def read_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(schema=1, network_from='configs/cross_chem_cycle_soft_difference_v1/combination.yaml',
        training=dict(pretrain_epochs=100, target_epochs=100, logical_batch=32, micro_batch=8,
            inference_chunk=8, lr=.001, weight_decay=.01, shuffle=False, precision='float32',
            checkpoint_every=1, selection='fixed_final_epoch', test_during_training=False,
            finetune_parameters='all_encoder_and_own_head', finetune_optimizer='fresh_AdamW',
            pretrain_reuse='one_LFP_checkpoint_per_seed_independent_target_restarts'),
        labels=dict(transform='natural_log_cycles_divided_by_fixed_1000', fixed_cycle_unit=1000., empirical_statistics=False),
        input=dict(raw_six_channels=True, clip_absolute_over=10., cycle_difference=False),
        data=dict(protocol='reuse_verified_existing_two_model_protocols', source_chemistry='LFP', source_count=275, seeds=list(range(8))))
    if cfg != expected:
        raise ValueError('单电池配置偏离冻结协议，请建立独立实验版本。')
    return dict(cfg, combination=combo.read_config())


def make_model(cfg, seed, device):
    return CycleResidualSingleCell(combo.make_model(cfg['combination'], seed, device))


def encode_labels(labels, cfg):
    if not torch.isfinite(labels).all() or (labels <= 0).any():
        raise ValueError('寿命标签必须为有限正数。')
    return (labels.double() / cfg['labels']['fixed_cycle_unit']).log().float()


def decode_labels(prediction, cfg):
    value = prediction.double().exp() * cfg['labels']['fixed_cycle_unit']
    if not torch.isfinite(value).all() or (value <= 0).any():
        raise FloatingPointError('寿命预测溢出或非正；不截断、不删除种子。')
    return value


def plan(targets, counts=None):
    return [(chem, n, seed, method) for chem, n, seed in combo.plan(targets, counts) for method in MODELS]


def code_hashes():
    result = combo.sources()
    for path in (Path(__file__), CONFIG, ROOT / 'scripts/start_cross_chem_single_cell_nohup.sh'):
        result[path.relative_to(ROOT).as_posix()] = common.digest(path)
    return result


def verified_inputs(base, jobs):
    """Read old artifacts without preparing, overwriting, or reinterpreting them."""
    base = Path(base).resolve()
    qpath = base / 'queue.json'
    old_queue = json.loads(qpath.read_text(encoding='utf-8'))
    if old_queue.get('schema') != 2 or combo.MODEL not in old_queue.get('models', []):
        raise ValueError('输入必须是已有双模型迁移工作区。')
    mpath = base / 'dataset/manifest.json'
    manifest = json.loads(mpath.read_text(encoding='utf-8'))
    data.validate_counts(manifest['records'])
    if manifest['identity']['policy'] != old_queue['data_policy'] or manifest['identity']['code_hashes'] != old_queue['code_hashes']:
        raise ValueError('旧数据收据与旧队列不一致。')
    policy = old_queue['data_policy']
    if policy.get('stored_features') != 'raw_unclipped_author_features' or policy.get('source_count') != 275:
        raise ValueError('旧缓存不是本实验所需的原六通道特征和275个LFP来源。')
    for record in manifest['records']:
        path = base / 'dataset' / record['feature_file']
        if path != data.feature_path(base / 'dataset', record['cell_id']) or common.digest(path) != record['feature_sha256']:
            raise ValueError('旧特征文件路径或指纹变化。')
    protocols = {}
    receipts = {}
    for target, n, seed, _ in jobs:
        key = (target, n, seed)
        if key in protocols:
            continue
        path = base / 'protocols' / target / f'train_{n}' / f'seed_{seed}.pt'
        p = common.load(path)
        data.validate_protocol(p, manifest['records'], policy)
        if (p['target_chemistry'], p['target_train_count'], p['seed']) != key:
            raise ValueError('旧协议文件与任务不一致。')
        protocols[key] = p
        receipts[path.relative_to(base).as_posix()] = common.digest(path)
    receipt = dict(base_workspace=str(base), old_queue_sha256=common.digest(qpath),
        manifest_sha256=common.digest(mpath), protocols_sha256=receipts,
        observed_counts=manifest['observed_counts'], reuse_features_without_copy=True,
        reuse_old_protocols_without_new_splits=True)
    return manifest, protocols, receipt


def stages(workspace, jobs, cfg):
    seeds = sorted({job[2] for job in jobs})
    result = [dict(model=PRETRAIN, target='LFP', train_count=275, seed=seed,
        epochs=cfg['training']['pretrain_epochs'], folder=str(Path(workspace) / 'pretraining' / f'seed_{seed}')) for seed in seeds]
    result += [dict(model=model, target=target, train_count=n, seed=seed,
        epochs=cfg['training']['target_epochs'], folder=str(Path(workspace) / target / f'train_{n}' / f'{model}_seed{seed}'))
        for target, n, seed, model in jobs]
    return result


def duration_text(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours}小时{minutes:02}分{seconds:02}秒'


class Progress:
    """ETA extrapolates epoch time by training-set size; pauses are excluded."""
    def __init__(self, workspace, schedule):
        self.workspace = Path(workspace)
        self.schedule = schedule
        self.done = {}
        self.rates = {}
        for stage in schedule:
            folder = Path(stage['folder'])
            state_path = folder / 'final.pt'
            if not state_path.exists():
                state_path = folder / 'progress.pt'
            epoch = common.load(state_path)['epoch'] if state_path.exists() else 0
            self.done[stage['folder']] = epoch
            log = folder / 'train.jsonl'
            if log.exists():
                for line in log.read_text(encoding='utf-8').splitlines()[:epoch]:
                    record = json.loads(line)
                    self.sample(stage['train_count'], record['seconds'])

    def sample(self, n, seconds):
        if math.isfinite(seconds) and seconds > 0:
            self.rates.setdefault(n, []).append(seconds)
            self.rates[n] = self.rates[n][-20:]

    def rate(self, n):
        if n in self.rates:
            return statistics.mean(self.rates[n])
        if not self.rates:
            return None
        nearest = min(self.rates, key=lambda k: abs(math.log(k/n)))
        # Micro-batch work scales more faithfully than optimizer steps for n<=32.
        ratio = math.ceil(n/8) / math.ceil(nearest/8)
        return statistics.mean(self.rates[nearest]) * ratio

    def publish(self, stage=None, epoch=None, seconds=None, status='training'):
        if stage is not None and epoch is not None:
            self.done[stage['folder']] = epoch
            if seconds is not None:
                self.sample(stage['train_count'], seconds)
        remaining = 0.
        available = True
        for item in self.schedule:
            left = item['epochs'] - self.done[item['folder']]
            rate = self.rate(item['train_count'])
            if left and rate is None:
                available = False
            if left and rate is not None:
                remaining += left * rate
        now = datetime.now(CHINA)
        complete_targets = sum(self.done[s['folder']] == s['epochs'] for s in self.schedule if s['model'] != PRETRAIN)
        complete_source = sum(self.done[s['folder']] == s['epochs'] for s in self.schedule if s['model'] == PRETRAIN)
        value = dict(status=status, updated_at_beijing=now.isoformat(timespec='seconds'),
            target_training_finished=complete_targets, target_runs=sum(s['model'] != PRETRAIN for s in self.schedule),
            pretraining_finished=complete_source, pretraining_runs=sum(s['model'] == PRETRAIN for s in self.schedule),
            total_epochs=sum(s['epochs'] for s in self.schedule), completed_epochs=sum(self.done.values()),
            estimated_remaining_seconds=remaining if available else None,
            estimated_end_beijing=(now + timedelta(seconds=remaining)).isoformat(timespec='seconds') if available else None,
            eta_note='按已测每轮耗时估算；不含未来停机、启动检查或异常重试，测试和汇总耗时可能造成偏差。')
        if stage:
            value.update(current_model=stage['model'], current_target=stage['target'], current_train_count=stage['train_count'],
                current_seed=stage['seed'], current_epoch=self.done[stage['folder']], current_max_epochs=stage['epochs'])
        common.atomic_json(value, self.workspace / 'status.json')
        common.atomic_json(value, self.workspace / 'progress.json')
        return value


def epoch_train(model, part, cfg, optimizer, device):
    model.train()
    logical = cfg['training']['logical_batch']
    micro = cfg['training']['micro_batch']
    y = encode_labels(part['label'], cfg)
    total = 0.
    steps = 0
    for start in range(0, len(y), logical):
        end = min(start+logical, len(y))
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, end, micro):
            stop = min(offset+micro, end)
            prediction = model(part['feature'][offset:stop].to(device))
            loss = (prediction-y[offset:stop].to(device)).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('单电池训练损失非有限。')
            (loss * (stop-offset)/(end-start)).backward()
            total += float(loss.detach()) * (stop-offset)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('单电池训练梯度非有限。')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('单电池训练参数非有限。')
        steps += 1
    return dict(train_loss=total/len(y), optimizer_steps=steps)


def checked(path, binding, maximum, final=False):
    saved = common.load(path)
    if saved['binding'] != binding or not 0 <= saved['epoch'] <= maximum:
        raise ValueError('单电池检查点绑定或轮次不一致。')
    if final and (saved['epoch'] != maximum or saved['selection'] != 'fixed_final_epoch'):
        raise ValueError('单电池最终权重不是固定最后一轮。')
    return saved


def train_stage(part, cfg, stage, binding, device, progress=None, initial=None):
    folder = Path(stage['folder'])
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / 'final.pt'
    if final.exists():
        state = checked(final, binding, stage['epochs'], final=True)
        common.align_log(folder, state['epoch'])
        if progress:
            progress.publish(stage, state['epoch'])
        return state
    model = make_model(cfg, stage['seed'], device)
    if initial is not None:
        model.load_state_dict(initial, strict=True)
    # Fine-tuning creates a new optimizer and updates every retained parameter.
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['training']['lr'], weight_decay=cfg['training']['weight_decay'])
    generator = torch.Generator().manual_seed(stage['seed'])
    start = 0
    path = folder / 'progress.pt'
    if path.exists():
        saved = checked(path, binding, stage['epochs'])
        start = saved['epoch']
        model.load_state_dict(saved['state'])
        optimizer.load_state_dict(saved['optimizer'])
        common.restore_rng(saved['rng'], generator, device)
    common.align_log(folder, start)
    def checkpoint(epoch):
        common.save(dict(binding=binding, epoch=epoch, state=model.state_dict(), optimizer=optimizer.state_dict(),
            rng=common.rng_state(generator, device)), path)
    if start == 0:
        checkpoint(0)
    for epoch in range(start+1, stage['epochs']+1):
        begin = time.monotonic()
        measured = epoch_train(model, part, cfg, optimizer, device)
        record = dict(epoch=epoch, seconds=time.monotonic()-begin, **measured)
        with (folder / 'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n')
            handle.flush()
            os.fsync(handle.fileno())
        checkpoint(epoch)
        receipt = progress.publish(stage, epoch, time.monotonic()-begin) if progress else {}
        eta = receipt.get('estimated_end_beijing') or '测量中'
        print(f"{LABELS[stage['model']]} {stage['target']} 训练{stage['train_count']}块 种子{stage['seed']} "
            f"轮次{epoch}/{stage['epochs']} 损失{measured['train_loss']:.6f}；预计结束（北京时间）{eta}", flush=True)
    saved = dict(binding=binding, epoch=stage['epochs'], state=common.cpu(model.state_dict()), selection='fixed_final_epoch')
    common.save(saved, final)
    return saved


@torch.no_grad()
def predict(model, part, cfg, device):
    model.eval()
    chunks = []
    width = cfg['training']['inference_chunk']
    for offset in range(0, len(part['ids']), width):
        chunks.append(model(part['feature'][offset:offset+width].to(device)).cpu())
    return decode_labels(torch.cat(chunks), cfg)


def target_binding(identity, stage, p, source_checkpoint=None):
    return dict(queue=identity, model=stage['model'], target=stage['target'], train_count=stage['train_count'],
        seed=stage['seed'], train_ids=p['target_train_ids'], test_ids=p['target_test_ids'],
        source_checkpoint_sha256=source_checkpoint, source_used_in_training=stage['model'] == FINETUNE,
        empirical_label_statistics=False)


def evaluate(cfg, stage, binding, p, testing, device):
    folder = Path(stage['folder'])
    saved = checked(folder / 'final.pt', binding, stage['epochs'], final=True)
    model = make_model(cfg, stage['seed'], device)
    model.load_state_dict(saved['state'])
    if testing['ids'] != p['target_test_ids']:
        raise ValueError('测试输入顺序与冻结名单不同。')
    pred = predict(model, testing, cfg, device)
    output = dict(binding=binding, checkpoint_sha256=common.digest(folder/'final.pt'),
        target_test_ids=list(testing['ids']), prediction_cycles=pred, truth_cycles=testing['label'].double(),
        selected_epoch=saved['epoch'], trained_epochs=saved['epoch'])
    common.save(output, folder/'test.pt')
    measured = common.scores(output['prediction_cycles'], output['truth_cycles'])
    common.atomic_json(dict(status='complete', binding=binding, final_sha256=output['checkpoint_sha256'],
        test_sha256=common.digest(folder/'test.pt'), scores=measured), folder/'run.json')
    return dict(target=stage['target'], train_count=stage['train_count'], model=stage['model'], seed=stage['seed'],
        selected_epoch=saved['epoch'], trained_epochs=saved['epoch'], **measured)


def saved_result(stage, binding):
    folder = Path(stage['folder'])
    path = folder/'run.json'
    if not path.exists():
        return None
    r = json.loads(path.read_text(encoding='utf-8'))
    if r['status'] != 'complete' or r['binding'] != binding:
        raise ValueError('单电池完成收据绑定不一致。')
    if common.digest(folder/'test.pt') != r['test_sha256'] or common.digest(folder/'final.pt') != r['final_sha256']:
        raise ValueError('单电池预测或权重文件指纹变化。')
    state = checked(folder/'final.pt', binding, stage['epochs'], final=True)
    output = common.load(folder/'test.pt')
    if output['binding'] != binding or output['checkpoint_sha256'] != r['final_sha256'] or output['target_test_ids'] != binding['test_ids']:
        raise ValueError('单电池预测绑定或电池顺序不同。')
    if output['selected_epoch'] != state['epoch'] or output['trained_epochs'] != state['epoch']:
        raise ValueError('单电池测试不是最终轮次。')
    metrics = common.scores(output['prediction_cycles'], output['truth_cycles'])
    if any(not math.isclose(metrics[k], r['scores'][k], rel_tol=1e-10, abs_tol=1e-10) for k in metrics):
        raise ValueError('单电池指标不能从预测复算。')
    return dict(target=stage['target'], train_count=stage['train_count'], model=stage['model'], seed=stage['seed'],
        selected_epoch=state['epoch'], trained_epochs=state['epoch'], **metrics)


def reference_rows(base, protocols, receipt):
    """Compare saved combination predictions, never the historical original CNN."""
    base = Path(base)
    old_queue = json.loads((base/'queue.json').read_text(encoding='utf-8'))
    rows = []
    hashes = {}
    for (target, n, seed), p in protocols.items():
        folder = base/target/f'train_{n}'/f'{combo.MODEL}_seed{seed}'
        path = folder/'run.json'
        if not path.exists():
            raise FileNotFoundError(f'已有组合结果未完成：{path}')
        r = json.loads(path.read_text(encoding='utf-8'))
        protocol_key = f'protocols/{target}/train_{n}/seed_{seed}.pt'
        binding = r['binding']
        if (r['status'] != 'complete' or binding['queue'] != old_queue or binding['model'] != combo.MODEL
                or binding['manifest_sha256'] != receipt['manifest_sha256']
                or binding['protocol_sha256'] != receipt['protocols_sha256'][protocol_key]
                or (binding['target'], binding['train_count'], binding['seed']) != (target,n,seed)):
            raise ValueError('已有组合结果与冻结数据或协议不同。')
        if common.digest(folder/'test.pt') != r['test_sha256'] or common.digest(folder/'final.pt') != r['final_sha256']:
            raise ValueError('已有组合预测或权重指纹变化。')
        out = common.load(folder/'test.pt')
        if out['binding'] != binding or out['checkpoint_sha256'] != r['final_sha256'] or out['target_test_ids'] != p['target_test_ids'] or out['source_ids'] != p['source_ids']:
            raise ValueError('已有组合预测绑定或电池顺序不同。')
        scores = common.scores(out['prediction_cycles'], out['truth_cycles'])
        if any(not math.isclose(scores[k], r['scores'][k], rel_tol=1e-10, abs_tol=1e-10) for k in scores):
            raise ValueError('已有组合指标不能复算。')
        rows.append(dict(target=target, train_count=n, seed=seed, model=combo.MODEL,
            selected_epoch=out['selected_epoch'], trained_epochs=out['trained_epochs'], **scores))
        hashes[folder.relative_to(base).as_posix()] = dict(run=common.digest(path), test=r['test_sha256'], final=r['final_sha256'])
    return rows, hashes


def summarize(rows):
    groups = []
    for target in data.TEST_COUNTS:
        for n in data.TRAIN_COUNTS[target]:
            for model in (*MODELS, combo.MODEL):
                group = [r for r in rows if (r['target'],r['train_count'],r['model']) == (target,n,model)]
                if len(group) == 8 and {r['seed'] for r in group} == set(range(8)):
                    groups.append(dict(target=target, train_count=n, model=model, seeds=8,
                        metrics={key:dict(mean=statistics.mean(r[key] for r in group), sample_std=statistics.stdev(r[key] for r in group))
                            for key in ('RMSE','MAE','MAPE','ACC15')}))
    pairs = []
    lookup = {(r['target'],r['train_count'],r['seed'],r['model']):r for r in rows}
    for target in data.TEST_COUNTS:
        for n in data.TRAIN_COUNTS[target]:
            for baseline in MODELS:
                paired = [(lookup.get((target,n,seed,baseline)),lookup.get((target,n,seed,combo.MODEL))) for seed in range(8)]
                if any(a is None or b is None for a,b in paired):
                    continue
                delta = [100*(a['MAPE']-b['MAPE']) for a,b in paired]
                pairs.append(dict(target=target,train_count=n,baseline=baseline,positive_means_combination_better=True,
                    MAPE_mean_advantage_percentage_points=statistics.mean(delta),better_seeds=sum(x>0 for x in delta)))
    return dict(groups=groups, combination_reference_differences=pairs,
        caveat='同电池名单、同编码器结构；已有组合采用来源+目标标签标准化和关系验证早停，新单电池对照采用固定对数变换与100轮最终权重。不是只有迁移策略不同的严格消融，不据此宣称显著性。')


def write_summary(workspace, rows, references, expected):
    workspace = Path(workspace)
    common.atomic_json(rows, workspace/'per_seed.json')
    combined = rows+references
    value = summarize(combined)
    value.update(completed_runs=len(rows),expected_runs=expected)
    common.atomic_json(value, workspace/'summary.json')
    for name, records in [('per_seed.csv', rows),('comparison_per_seed.csv',combined)]:
        if not records:
            continue
        tmp = workspace/(name+'.tmp')
        with tmp.open('w',encoding='utf-8-sig',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        os.replace(tmp,workspace/name)
    return value


def recover_rows(workspace, identity, cfg):
    schedule = stages(workspace, [tuple(job) for job in identity['runs']], cfg)
    rows = []
    for stage in schedule:
        if stage['model'] == PRETRAIN:
            continue
        p = common.load(Path(workspace)/'protocols'/stage['target']/f"train_{stage['train_count']}"/f"seed_{stage['seed']}.pt")
        source_hash = None
        if stage['model'] == FINETUNE:
            source_path = Path(workspace)/'pretraining'/f"seed_{stage['seed']}"/'final.pt'
            if source_path.exists():
                source_hash = common.digest(source_path)
        r = saved_result(stage, target_binding(identity,stage,p,source_hash))
        if r is not None:
            rows.append(r)
    return rows


def report(workspace):
    workspace = Path(workspace)
    identity = json.loads((workspace/'queue.json').read_text(encoding='utf-8'))
    rows = recover_rows(workspace,identity,identity['config'])
    refs = json.loads((workspace/'combination_reference.json').read_text(encoding='utf-8'))['rows']
    value = summarize(rows+refs)
    print(f"已完成单电池目标结果：{len(rows)}/{len(identity['runs'])}；均值 ± 样本标准差。")
    for group in value['groups']:
        print(f"\n{group['target']} 目标训练{group['train_count']}块：{LABELS[group['model']]}")
        for key,name,mult,unit in [('RMSE','均方根误差',1,'循环'),('MAE','平均绝对误差',1,'循环'),
                ('MAPE','平均绝对百分比误差',100,'%'),('ACC15','15%以内准确率',100,'%')]:
            m = group['metrics'][key]
            print(f"  {name}：{m['mean']*mult:.4f} ± {m['sample_std']*mult:.4f} {unit}")
    print('\n'+value['caveat'])


def show_progress(workspace):
    path = Path(workspace)/'progress.json'
    status_path = Path(workspace)/'status.json'
    if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8')).get('status') == 'verifying_inputs':
        print('正在核验已有数据与预测；核验结束后显示预计结束时间。')
        return
    if not path.exists():
        print('正在启动或检查数据，暂未测得每轮耗时；查看最新日志。')
        return
    value = json.loads(path.read_text(encoding='utf-8'))
    running = False
    lock = Path(workspace)/'.queue.lock'
    if os.name == 'posix' and lock.exists():
        import fcntl
        with lock.open('r') as handle:
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                fcntl.flock(handle,fcntl.LOCK_UN)
            except BlockingIOError:
                running = True
    print(f"状态：{value['status']}；进程{'正在运行' if running else '当前未运行'}")
    print(f"目标训练完成：{value['target_training_finished']}/{value['target_runs']}；"
        f"LFP预训练：{value['pretraining_finished']}/{value['pretraining_runs']}")
    if 'current_model' in value:
        print(f"当前：{LABELS[value['current_model']]}，{value['current_target']}，训练{value['current_train_count']}块，"
            f"种子{value['current_seed']}，轮次{value['current_epoch']}/{value['current_max_epochs']}")
    if value['status'] == 'complete':
        print('所有目标测试结果已保存，可使用 --report 查看。')
    elif not running:
        print('记录来自上一次运行；恢复后会重新估算结束时间。')
    elif value['estimated_end_beijing'] is not None:
        print(f"预计剩余：{duration_text(value['estimated_remaining_seconds'])}")
        print(f"预计结束（北京时间）：{value['estimated_end_beijing']}")
    else:
        print('预计结束时间：测量中。')
    print(f"最近更新（北京时间）：{value['updated_at_beijing']}")
    print(value['eta_note'])


def preflight(cfg, source, target, device, progress):
    if device.startswith('cuda'):
        torch.cuda.reset_peak_memory_stats(device)
    observations = {}
    # Disposable models calibrate both large-source and one-/sixteen-cell epochs.
    for part in (source,{k:v[:1] for k,v in target.items()},target):
        model = make_model(cfg,0,device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=cfg['training']['lr'],weight_decay=cfg['training']['weight_decay'])
        begin = time.monotonic()
        measurements = epoch_train(model,part,cfg,optimizer,device)
        if device.startswith('cuda'):
            torch.cuda.synchronize(device)
        elapsed = time.monotonic()-begin
        n = len(part['ids'])
        progress.sample(n,elapsed)
        pred = predict(model,{k:v[:1] for k,v in part.items()},cfg,device)
        observations[str(n)] = dict(**measurements,seconds=elapsed,parameters=sum(p.numel() for p in model.parameters()),positive_prediction=bool((pred>0).all()))
        del model,optimizer
    if device.startswith('cuda'):
        observations['cuda_peak_GiB'] = torch.cuda.max_memory_allocated(device)/2**30
        torch.cuda.empty_cache()
    return observations


def main():
    parser = argparse.ArgumentParser(description='现有编码器单电池版：直接目标训练、LFP预训练后微调，支持恢复与预计结束时间')
    parser.add_argument('--base-workspace',type=Path,default=DEFAULT_BASE)
    parser.add_argument('--workspace',type=Path,default=DEFAULT_WORKSPACE)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--targets',nargs='+',choices=list(data.TEST_COUNTS),default=list(data.TEST_COUNTS))
    parser.add_argument('--counts',nargs='+',type=int)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--progress',action='store_true')
    parser.add_argument('--report',action='store_true')
    args = parser.parse_args()
    workspace,base = args.workspace.resolve(),args.base_workspace.resolve()
    if args.progress:
        show_progress(workspace)
        return
    if args.report:
        report(workspace)
        return
    if os.name != 'posix' or not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式训练只在Linux CUDA服务器执行；本地仅做合成检查。')
    if workspace == base or workspace in base.parents or base in workspace.parents:
        raise ValueError('新输出与旧工作区必须独立，不能嵌套或覆盖。')
    torch.set_num_threads(8)
    cfg = read_config()
    jobs = plan(args.targets,args.counts)
    with contextlib.ExitStack() as stack:
        stack.enter_context(common.queue_lock(workspace))
        others = {base,*(ROOT/'workspaces'/name for name in ('cross_chem_cycle_soft_difference_v1','mix_cycle_soft_difference_v1','three_tasks_three_models_v1'))}
        for other in sorted(others):
            if other.exists():
                stack.enter_context(common.queue_lock(other))
        try:
            common.atomic_json(dict(status='verifying_inputs'),workspace/'status.json')
            print('正在核验已有特征、固定名单与组合预测；不重新抽样、不复制特征。',flush=True)
            manifest,protocols,receipt = verified_inputs(base,jobs)
            refs,reference_hashes = reference_rows(base,protocols,receipt)
            identity = dict(schema=1,models=list(MODELS),config=cfg,runs=[list(j) for j in jobs],
                inputs=receipt,combination_reference_sha256=reference_hashes,code_hashes=code_hashes(),runtime=common.runtime(args.device))
            queue = workspace/'queue.json'
            if queue.exists() and json.loads(queue.read_text(encoding='utf-8')) != identity:
                raise ValueError('单电池队列的数据、配置、代码或环境变化，拒绝混合恢复。')
            common.atomic_json(identity,queue)
            common.atomic_json(dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                note='恢复按运行代码指纹绑定；仅文档提交不会导致拒绝恢复。'),workspace/'code_receipt.json')
            common.atomic_json(dict(rows=refs,source=receipt,hashes=reference_hashes),workspace/'combination_reference.json')
            for (target,n,seed),p in protocols.items():
                path = workspace/'protocols'/target/f'train_{n}'/f'seed_{seed}.pt'
                if not path.exists():
                    common.save(p,path)
                old = common.load(path)
                data.validate_protocol(old,manifest['records'],p['policy'])
            schedule = stages(workspace,jobs,cfg)
            progress = Progress(workspace,schedule)
            rows = recover_rows(workspace,identity,cfg)
            write_summary(workspace,rows,refs,len(jobs))
            all_complete = len(rows) == len(jobs)
            if all_complete:
                progress.publish(status='complete')
                print('所有结果已完成并核验，无需重复训练。',flush=True)
                return
            source_ids = next(iter(protocols.values()))['source_ids']
            source = data.load_part(source_ids,manifest,base)
            biggest = max(protocols.values(),key=lambda p:p['target_train_count'])
            target = data.load_part(biggest['target_train_ids'],manifest,base)
            observations = preflight(cfg,source,target,args.device,progress)
            common.atomic_json(dict(observed_counts=manifest['observed_counts'],checks=observations,
                target_runs=len(jobs),source_pretraining_runs=8,formal_training_started=False),workspace/'preflight.json')
            del target
            progress.publish(status='prepared' if args.prepare_only else 'training')
            if args.prepare_only:
                print('预检通过；仅运行临时模型检查，未开始正式队列。',flush=True)
                return
            print(f'预检通过：{len(jobs)}组目标训练＋8次可复用LFP预训练。',flush=True)
            for stage in schedule:
                if stage['model'] == PRETRAIN:
                    binding = dict(queue=identity,model=PRETRAIN,seed=stage['seed'],source_ids=source_ids,
                        empirical_label_statistics=False)
                    train_stage(source,cfg,stage,binding,args.device,progress)
            del source
            for stage in schedule:
                if stage['model'] == PRETRAIN:
                    continue
                key = (stage['target'],stage['train_count'],stage['seed'])
                p = protocols[key]
                source_hash = None
                initial = None
                if stage['model'] == FINETUNE:
                    path = workspace/'pretraining'/f"seed_{stage['seed']}"/'final.pt'
                    source_hash = common.digest(path)
                    source_binding = dict(queue=identity,model=PRETRAIN,seed=stage['seed'],source_ids=source_ids,
                        empirical_label_statistics=False)
                    pretrained = checked(path,source_binding,cfg['training']['pretrain_epochs'],final=True)
                    initial = pretrained['state']
                binding = target_binding(identity,stage,p,source_hash)
                row = saved_result(stage,binding)
                if row is None:
                    training = data.load_part(p['target_train_ids'],manifest,base)
                    train_stage(training,cfg,stage,binding,args.device,progress,initial)
                    del training
                    progress.publish(stage,stage['epochs'],status='evaluating')
                    testing = data.load_part(p['target_test_ids'],manifest,base)
                    row = evaluate(cfg,stage,binding,p,testing,args.device)
                    del testing
                rows = [r for r in rows if (r['target'],r['train_count'],r['seed'],r['model']) != (*key,stage['model'])]+[row]
                write_summary(workspace,rows,refs,len(jobs))
                print(json.dumps(row,ensure_ascii=False),flush=True)
                initial = None
                if args.device.startswith('cuda'):
                    torch.cuda.empty_cache()
            progress.publish(status='complete')
            print('单电池对照全部完成，正式测试结果已保存。使用 --report 查看。',flush=True)
        except BaseException as exc:
            state_path = workspace/'status.json'
            state = json.loads(state_path.read_text(encoding='utf-8')) if state_path.exists() else {}
            state.update(status='failed',error=f'{type(exc).__name__}: {exc}')
            common.atomic_json(state,state_path)
            if (workspace/'progress.json').exists():
                previous = json.loads((workspace/'progress.json').read_text(encoding='utf-8'))
                previous.update(status='failed',error=state['error'])
                common.atomic_json(previous,workspace/'progress.json')
            traceback.print_exc()
            raise


if __name__ == '__main__':
    main()
