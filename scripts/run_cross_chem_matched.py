"""Source-only development, then 344 fixed-list matched transfer training runs."""
import argparse
import contextlib
import csv
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_three_tasks as common
from scripts import cross_chem_matched_data as data
from scripts import cross_chem_matched_training as training
import torch
import yaml

CONFIG = ROOT / 'configs/cross_chem_matched_v3/protocol.yaml'
KINDS = ('direct', 'finetune', 'combination')
LABELS = {'direct': '目标直接训练', 'finetune': 'LFP预训练后微调',
          'combination': '循环轴残差与软匹配差分组合'}


def read_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if (cfg['schema'] != 3 or cfg['models'] != list(KINDS)
            or cfg['seeds'] != list(range(8)) or cfg['data'] != dict(
                source_chemistry='LFP', source_count=275, input_shape=[6, 100, 1000],
                clip_absolute_over=10., reuse_actual_saved_ids=True,
                regenerate_missing_protocols=False, use_all_training_pairs=True, eol_soh=.8)
            or cfg['label'] != dict(transformation='natural_log_fixed_scale', lifetime_scale=1000.)
            or cfg['training'] != dict(
                optimizer='AdamW', weight_decay=.01, logical_batch=128, micro_batch=8,
                precision='float32', shuffle=False, checkpoint_every=25, report_every=25,
                selection='fixed_final_optimizer_step', early_stopping=False,
                test_after_all_formal_training=True, finetune_all_parameters=True,
                finetune_new_optimizer=True, finetune_start_from_original_source_checkpoint=True,
                reference_count=275, inference_chunk=8)
            or cfg['pilot'] != dict(
                source_only=True, split_seeds=[0, 1], source_count=219,
                training_pool_count=16, validation_count=40, target_counts=[1, 16],
                learning_rates=[.001, .0003], candidate_steps=[100, 300, 1000],
                metric='mean_squared_fixed_log_lifetime_error',
                aggregation='equal_weight_split_count_method',
                learning_rate_shared_by_all_stages=True,
                ties='learning_rate_list_order_then_smallest_step')):
        raise ValueError('配置偏离本次预定协议；变更应使用独立版本。')
    return cfg


def formal_plan():
    jobs = [dict(kind='pretrain', seed=seed) for seed in range(8)]
    jobs += [dict(kind=kind, seed=seed, target=target, train_count=count)
             for target, counts in data.TRAIN_COUNTS.items() for count in counts
             for seed in range(8) for kind in KINDS]
    return jobs


def pilot_split(source_ids, seed):
    if len(source_ids) != 275 or len(set(source_ids)) != 275:
        raise ValueError('来源开发划分需要完整且不重复的275个LFP编号。')
    values = list(source_ids)
    random.Random(240000 + seed).shuffle(values)
    return dict(source_ids=values[:219], training_pool_ids=values[219:235],
                validation_ids=values[235:])


def select_budget(records, cfg):
    """Select from complete LFP-only development scores; never target test scores."""
    pilot = cfg['pilot']
    rates, steps = pilot['learning_rates'], sorted(pilot['candidate_steps'])
    expected = {(lr, seed, kind, n, step) for lr in rates
                for seed in pilot['split_seeds'] for step in steps
                for kind, n in [('pretrain', 0)] + [
                    (kind, n) for n in pilot['target_counts'] for kind in KINDS]}
    indexed = {}
    for record in records:
        key = tuple(record[k] for k in ('lr', 'split_seed', 'kind', 'train_count', 'step'))
        score = record['validation_log_mse']
        if (key in indexed or key not in expected or not isinstance(score, (int, float))
                or not math.isfinite(score) or score < 0):
            raise ValueError('来源开发记录重复、越界或包含无效指标。')
        indexed[key] = score
    if set(indexed) != expected:
        raise ValueError('来源开发候选记录不完整，不能选择正式训练预算。')
    candidates = []
    for lr in rates:
        pre = {step: statistics.mean(indexed[(lr, seed, 'pretrain', 0, step)]
                                     for seed in pilot['split_seeds']) for step in steps}
        pre_steps = min(steps, key=lambda step: (pre[step], step))
        target = {step: statistics.mean(indexed[(lr, seed, kind, n, step)]
                    for seed in pilot['split_seeds'] for n in pilot['target_counts']
                    for kind in KINDS) for step in steps}
        target_steps = min(steps, key=lambda step: (target[step], step))
        candidates.append(dict(learning_rate=lr, pretrain_steps=pre_steps,
            target_steps=target_steps, development_log_mse=target[target_steps],
            pretrain_scores=pre, target_scores=target))
    chosen = min(enumerate(candidates), key=lambda item: (
        item[1]['development_log_mse'], item[0]))[1]
    result = dict(schema=3, learning_rate=chosen['learning_rate'],
                pretrain_steps=chosen['pretrain_steps'], target_steps=chosen['target_steps'],
                criterion=pilot['aggregation'], source_only=True,
                formal_target_scores_used=False, candidates=candidates)
    # JSON object keys are strings; keep checkpoint bindings identical on restart.
    return json.loads(json.dumps(result, allow_nan=False))


def part_from_ids(part, ids):
    if part['ids'] == list(ids):
        return part
    lookup = {cell_id: i for i, cell_id in enumerate(part['ids'])}
    try:
        index = torch.tensor([lookup[cell_id] for cell_id in ids], dtype=torch.long)
    except KeyError as error:
        raise ValueError('请求的开发电池不在LFP来源缓存中。') from error
    return dict(feature=part['feature'][index], label=part['label'][index], ids=list(ids))


def job_folder(workspace, job):
    workspace = Path(workspace)
    if job['kind'] == 'pretrain':
        return workspace / 'pretrain' / f"seed_{job['seed']}"
    return workspace / job['target'] / f"train_{job['train_count']}" / f"{job['kind']}_seed{job['seed']}"


def job_binding(identity, job, selection, workspace):
    value = dict(queue=identity, job=job, selection=selection)
    if job['kind'] == 'finetune':
        pre = job_folder(workspace, dict(kind='pretrain', seed=job['seed'])) / 'final.pt'
        value['initial_source_checkpoint_sha256'] = common.digest(pre)
    return value


def validate_training_job(workspace, identity, job):
    selection = json.loads((Path(workspace) / 'selection.json').read_text(encoding='utf-8'))
    folder = job_folder(workspace, job)
    expected = selection['pretrain_steps'] if job['kind'] == 'pretrain' else selection['target_steps']
    return training.load_completed(folder, job_binding(identity, job, selection, workspace),
                                   job['kind'], expected)


def all_training_complete(workspace, identity, jobs):
    if jobs != formal_plan():
        raise ValueError('正式测试要求完整344次训练计划。')
    for job in jobs:
        if not (job_folder(workspace, job) / 'final.pt').is_file():
            return False
        validate_training_job(workspace, identity, job)
    return True


def code_hashes():
    paths = [Path(__file__), CONFIG, ROOT / 'scripts/cross_chem_matched_data.py',
             ROOT / 'scripts/cross_chem_matched_training.py', ROOT / 'scripts/run_three_tasks.py']
    paths += sorted((ROOT / 'src').rglob('*.py'))
    return {p.relative_to(ROOT).as_posix(): common.digest(p) for p in paths}


def training_kwargs(cfg, lr):
    tr = cfg['training']
    return dict(lr=lr, weight_decay=tr['weight_decay'], checkpoint_every=tr['checkpoint_every'],
                micro_batch=tr['micro_batch'], logical_batch=tr['logical_batch'],
                report_every=tr['report_every'])


def run_pilot(source, workspace, identity, cfg, device):
    """Only source tensors enter this function; chemistry target caches are absent."""
    root = Path(workspace) / 'pilot'
    records = []
    steps = cfg['pilot']['candidate_steps']
    maximum = max(steps)
    for lr in cfg['pilot']['learning_rates']:
        for split_seed in cfg['pilot']['split_seeds']:
            ids = pilot_split(source['ids'], split_seed)
            common.atomic_json(ids, root / f'split_{split_seed}.json')
            src = part_from_ids(source, ids['source_ids'])
            pool = part_from_ids(source, ids['training_pool_ids'])
            validation = part_from_ids(source, ids['validation_ids'])
            base = root / f'lr_{lr:g}' / f'split_{split_seed}'

            def fit(kind, n, target, initial=None, dependency=None):
                folder = base / ('pretrain' if kind == 'pretrain' else f'train_{n}/{kind}')
                binding = dict(queue=identity, stage='source_only_development', kind=kind,
                    train_count=n, split_seed=split_seed, lr=lr, ids=ids,
                    initial_checkpoint_sha256=dependency)

                def candidate(step, model):
                    if step not in steps:
                        return
                    output = training.predict(model, src, validation, kind, device,
                                              chunk=cfg['training']['inference_chunk'])
                    residual = (output['prediction_cycles'] / 1000.).log() - (validation['label'] / 1000.).log()
                    loss = float(residual.double().square().mean())
                    if not math.isfinite(loss):
                        raise FloatingPointError('LFP开发验证指标非有限，不排除候选或种子。')
                    record = dict(lr=lr, split_seed=split_seed, kind=kind,
                                  train_count=n, step=step, validation_log_mse=loss)
                    common.save(dict(binding=binding, record=record,
                        state={k: v.detach().cpu().clone() for k, v in model.state_dict().items()}),
                        folder / f'candidate_{step}.pt')
                    print(f'来源开发 {kind} n={n} seed={split_seed} lr={lr:g} '
                          f'更新{step} 验证对数均方误差={loss:.6f}', flush=True)

                training.train_run(src, target, kind, split_seed, maximum, folder, binding, device,
                    initial_state=initial, step_callback=candidate, **training_kwargs(cfg, lr))
                values = []
                for step in steps:
                    saved = common.load(folder / f'candidate_{step}.pt')
                    if saved['binding'] != binding or saved['record']['step'] != step:
                        raise ValueError('来源开发候选权重来源不一致。')
                    values.append(saved['record'])
                records.extend(values)
                common.atomic_json(records, root / 'scores.json')
                return folder, values

            fit('pretrain', 0, src)
            # Use one common pretrain budget for both source splits for this learning rate.
            # Candidates for split 0 and split 1 must both exist before downstream fits.
            del src, pool, validation
        pre_by_step = {step: statistics.mean(row['validation_log_mse'] for row in records
            if row['lr'] == lr and row['kind'] == 'pretrain' and row['step'] == step) for step in steps}
        pre_step = min(steps, key=lambda step: (pre_by_step[step], step))
        for split_seed in cfg['pilot']['split_seeds']:
            ids = pilot_split(source['ids'], split_seed)
            src = part_from_ids(source, ids['source_ids'])
            pool = part_from_ids(source, ids['training_pool_ids'])
            validation = part_from_ids(source, ids['validation_ids'])
            base = root / f'lr_{lr:g}' / f'split_{split_seed}'
            pre = base / 'pretrain' / f'candidate_{pre_step}.pt'
            initial = common.load(pre)['state']
            for n in cfg['pilot']['target_counts']:
                target = {key: value[:n] for key, value in pool.items()}
                for kind in KINDS:
                    fit(kind, n, target, initial if kind == 'finetune' else None,
                        common.digest(pre) if kind == 'finetune' else None)
            del src, pool, validation
    selection = select_budget(records, cfg)
    common.atomic_json(selection, Path(workspace) / 'selection.json')
    common.atomic_json(dict(status='pilot_complete',training_runs=28,
        candidate_records=len(records),selection=selection), root / 'receipt.json')
    return selection


def preflight(source, target, cfg, device):
    results = {}
    for kind in ('direct', 'combination'):
        model = training.make_model(kind, 0, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
        n = min(cfg['training']['micro_batch'], len(target['label']))
        x = target['feature'][:n].to(device)
        y = (target['label'][:n].to(device) / 1000.).log()
        model.train()
        if kind == 'combination':
            sx = source['feature'][:n].to(device).unsqueeze(1)
            sy = (source['label'][:n].to(device) / 1000.).log().unsqueeze(1)
            loss = model(x, y, sx, sy, return_loss=True)
        else:
            loss = (model(x) - y).square().mean()
        loss.backward()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all()
                                         for p in model.parameters()):
            raise FloatingPointError('正式尺寸预检查损失或梯度非有限。')
        optimizer.step()
        small = {k: v[:1] for k, v in target.items()}
        output = training.predict(model, source, small, kind, device,
                                  chunk=cfg['training']['inference_chunk'])
        results[kind] = dict(parameters=sum(p.numel() for p in model.parameters()),
            loss=float(loss.detach()), finite=bool(torch.isfinite(output['prediction_cycles']).all()))
        del model, optimizer, x, y, output, loss
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    return results


def verify_test(output, state, job, binding, weight_sha, protocol, manifest):
    if (output['binding'] != binding or output['checkpoint_sha256'] != weight_sha
            or output['step'] != state['step'] or output['target_test_ids'] != protocol['target_test_ids']):
        raise ValueError('测试预测与正式权重或目标名单不同。')
    lookup = {row['cell_id']: row for row in manifest['records']}
    truth = torch.tensor([lookup[cell_id]['label_cycles'] for cell_id in protocol['target_test_ids']],
                         dtype=torch.float64)
    torch.testing.assert_close(output['truth_cycles'], truth, rtol=0, atol=0)
    measured = common.scores(output['prediction_cycles'], truth)
    for key, value in measured.items():
        if not math.isfinite(value) or not math.isclose(value, output['scores'][key], rel_tol=1e-10, abs_tol=1e-10):
            raise ValueError('保存测试指标不能从逐电池预测复算。')
    if job['kind'] == 'combination':
        if output['source_ids'] != protocol['source_ids']:
            raise ValueError('测试参考名单或顺序不同。')
        diag = output['diagnostics']
        expected = torch.arange(len(protocol['source_ids'])).expand(len(truth), -1)
        if not torch.equal(diag['support_index'], expected):
            raise ValueError('未使用保存名单中的全部275参考。')
        torch.testing.assert_close(diag['y_sup_agg'], diag['y_sup'].median(1).values)
        log_value = .5 * diag['y_ori'] + .5 * diag['y_sup_agg']
    else:
        log_value = output['diagnostics']['y_ori']
    torch.testing.assert_close(output['prediction_cycles'], training.decode_labels(log_value))
    return measured


def evaluate_job(workspace, identity, job, source, testing, protocol, manifest, cfg, device):
    state = validate_training_job(workspace, identity, job)
    folder = job_folder(workspace, job)
    weight_sha = common.digest(folder / 'final.pt')
    binding = state['binding']
    path = folder / 'test.pt'
    receipt = folder / 'evaluation.json'
    if path.exists():
        output = common.load(path)
        if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))['test_sha256'] != common.digest(path):
            raise ValueError('测试预测文件指纹变化。')
    else:
        model = training.make_model(job['kind'], job['seed'], device)
        model.load_state_dict(state['state'], strict=True)
        output = training.predict(model, source, testing, job['kind'], device,
                                  chunk=cfg['training']['inference_chunk'])
        output.update(binding=binding, checkpoint_sha256=weight_sha, step=state['step'],
                      target_test_ids=testing['ids'])
        common.save(output, path)
        del model
    measured = verify_test(output, state, job, binding, weight_sha, protocol, manifest)
    common.atomic_json(dict(test_sha256=common.digest(path), final_sha256=weight_sha,
                            scores=measured), receipt)
    return dict(**job, optimizer_steps=state['step'], **measured)


def summarize(rows):
    indexed = {(r['target'], r['train_count'], r['seed'], r['kind']): r for r in rows}
    if len(indexed) != len(rows):
        raise ValueError('汇总存在重复种子结果。')
    groups, differences = [], []
    for target, counts in data.TRAIN_COUNTS.items():
        for count in counts:
            for kind in KINDS:
                group = [indexed.get((target, count, seed, kind)) for seed in range(8)]
                if any(row is None for row in group):
                    continue
                groups.append(dict(target=target, train_count=count, kind=kind, seeds=8,
                    metrics={key: dict(mean=statistics.mean(r[key] for r in group),
                        sample_std=statistics.stdev(r[key] for r in group))
                        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15')}))
            for baseline in ('direct', 'finetune'):
                pairs = [(indexed.get((target, count, seed, baseline)),
                          indexed.get((target, count, seed, 'combination'))) for seed in range(8)]
                if any(a is None or b is None for a, b in pairs):
                    continue
                differences.append(dict(target=target, train_count=count, baseline=baseline,
                    positive_means_combination_better=True, metrics={key: dict(
                        mean_advantage=statistics.mean((a[key] - b[key]) * (-1 if key == 'ACC15' else 1)
                                                       for a, b in pairs),
                        better_seeds=sum(b[key] > a[key] if key == 'ACC15' else b[key] < a[key]
                                         for a, b in pairs)) for key in ('RMSE', 'MAE', 'MAPE', 'ACC15')}))
    return dict(expected_predictions=336, completed_predictions=len(rows),
                groups=groups, paired_differences=differences,
                metric_units=dict(RMSE='cycles', MAE='cycles', MAPE='fraction', ACC15='fraction'))


def write_summary(rows, workspace):
    common.atomic_json(rows, Path(workspace) / 'per_seed.json')
    common.atomic_json(summarize(rows), Path(workspace) / 'summary.json')
    if rows:
        path = Path(workspace) / 'per_seed.csv'
        temp = path.with_suffix('.csv.tmp')
        with temp.open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)


def completed_rows(workspace, identity, protocols, manifest):
    rows = []
    for job in formal_plan():
        if job['kind'] == 'pretrain':
            continue
        folder = job_folder(workspace, job)
        if not (folder / 'evaluation.json').is_file():
            continue
        state = validate_training_job(workspace, identity, job)
        receipt = json.loads((folder / 'evaluation.json').read_text(encoding='utf-8'))
        if (common.digest(folder / 'final.pt') != receipt['final_sha256']
                or common.digest(folder / 'test.pt') != receipt['test_sha256']):
            raise ValueError('汇总权重或测试文件指纹不一致。')
        p = protocols[(job['target'], job['train_count'], job['seed'])]
        output = common.load(folder / 'test.pt')
        scores = verify_test(output, state, job, state['binding'], receipt['final_sha256'], p, manifest)
        rows.append(dict(**job, optimizer_steps=state['step'], **scores))
    return rows


def report(workspace):
    workspace = Path(workspace)
    if not (workspace / 'queue.json').is_file():
        print('新迁移队列尚未开始；没有可报告的结果。')
        return
    identity = json.loads((workspace / 'queue.json').read_text(encoding='utf-8'))
    # Pure read-only report; deliberately does not import or recreate protocols.
    existing = Path(identity['existing_workspace'])
    manifest_path = existing / 'dataset/manifest.json'
    imported = identity['imported_data']
    if (common.digest(manifest_path) != imported['existing_manifest_sha256']
            or common.digest(existing / 'queue.json') != imported['existing_queue_sha256']):
        raise ValueError('旧数据收据或队列指纹已变化，不能报告为同一次实验。')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    protocols = {}
    for target, counts in data.TRAIN_COUNTS.items():
        for count in counts:
            for seed in range(8):
                relative = f'{target}/train_{count}/seed_{seed}.pt'
                old_path = existing / 'protocols' / relative
                if common.digest(old_path) != imported['protocols'][relative]['sha256']:
                    raise ValueError('旧实际名单指纹已变化。')
                old = common.load(old_path)
                copied = json.loads((workspace / 'protocols' / target /
                    f'train_{count}' / f'seed_{seed}.json').read_text(encoding='utf-8'))
                for key in ('source_ids', 'target_train_ids', 'target_test_ids', 'unused_target_ids'):
                    if copied[key] != old[key]:
                        raise ValueError('新工作区名单与旧实际名单不同。')
                protocols[(target, count, seed)] = copied
    if not (workspace / 'selection.json').is_file():
        status_path = workspace / 'status.json'
        status = json.loads(status_path.read_text(encoding='utf-8')) if status_path.exists() else {}
        print(f'来源开发尚未完成；尚未锁定正式预算。当前状态：{status.get("status", "preparing")}')
        return
    value = summarize(completed_rows(workspace, identity, protocols, manifest))
    print(f"已保存测试结果：{value['completed_predictions']}/336；比例项以下显示百分数。")
    selection = json.loads((workspace / 'selection.json').read_text(encoding='utf-8'))
    print(f"学习率 {selection['learning_rate']:g}；预训练更新{selection['pretrain_steps']}；"
          f"目标阶段更新{selection['target_steps']}；预算只由LFP开发试验选择。")
    for group in value['groups']:
        print(f"\n{group['target']} 训练{group['train_count']}块：{LABELS[group['kind']]}")
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            metric = group['metrics'][key]
            factor = 100 if key in ('MAPE', 'ACC15') else 1
            print(f"  {key}：{factor * metric['mean']:.4f} ± {factor * metric['sample_std']:.4f}")


@contextlib.contextmanager
def legacy_read_lock(existing):
    """Prevent an old writer from changing evidence, without writing its lock."""
    import fcntl
    path = Path(existing) / '.queue.lock'
    if not path.is_file():
        raise FileNotFoundError(f'缺少旧队列锁文件，无法保证只读证据不被并发更新：{path}')
    with path.open('rb') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('旧迁移队列仍在运行，等待其结束后启动新实验。') from None
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-workspace', type=Path, default=ROOT / 'workspaces/cross_chem_two_models_v2')
    parser.add_argument('--workspace', type=Path, default=ROOT / 'workspaces/cross_chem_matched_v3')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    workspace, existing = args.workspace.resolve(), args.existing_workspace.resolve()
    if args.report:
        report(workspace)
        return
    if workspace == existing or workspace.is_relative_to(existing) or existing.is_relative_to(workspace):
        raise ValueError('新旧工作区必须独立，不能互相覆盖或嵌套。')
    if os.name != 'posix' or not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式流程要求Linux CUDA服务器；本地只运行合成检查。')
    torch.set_num_threads(8)
    common.seed_all(0)
    cfg = read_config()
    jobs = formal_plan()
    with contextlib.ExitStack() as stack:
        stack.enter_context(common.queue_lock(workspace))
        stack.enter_context(legacy_read_lock(existing))
        try:
            manifest, protocols, audit = data.import_protocols(existing, workspace)
            identity = dict(schema=3, config=cfg, existing_workspace=str(existing),
                imported_data=audit, code_hashes=code_hashes(), runtime=common.runtime(args.device),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                jobs=jobs, formal_training_runs=344, formal_predictions=336)
            queue = workspace / 'queue.json'
            if queue.exists() and json.loads(queue.read_text(encoding='utf-8')) != identity:
                raise ValueError('队列代码、环境或数据绑定变化，拒绝混合恢复。')
            common.atomic_json(identity, queue)
            common.atomic_json(dict(status='preflight', formal_training_runs=344,
                                    pilot_training_runs=28), workspace / 'status.json')
            p = protocols[('LCO', 16, 0)]
            source = data.load_part(p['source_ids'], manifest, existing)
            target = data.load_part(p['target_train_ids'], manifest, existing)
            gate = preflight(source, target, cfg, args.device)
            common.atomic_json(gate, workspace / 'preflight.json')
            del target
            if args.prepare_only:
                common.atomic_json(dict(status='prepared', formal_training_started=False), workspace / 'status.json')
                return
            common.atomic_json(dict(status='source_only_development', formal_training_started=False), workspace / 'status.json')
            selection = run_pilot(source, workspace, identity, cfg, args.device)
            print(f'来源开发已选定：学习率{selection["learning_rate"]:g}，'
                  f'预训练{selection["pretrain_steps"]}次更新，目标阶段{selection["target_steps"]}次更新。', flush=True)
            if args.pilot_only:
                common.atomic_json(dict(status='pilot_complete', selection=selection), workspace / 'status.json')
                return
            durations = []
            for index, job in enumerate(jobs):
                begin = time.monotonic()
                folder = job_folder(workspace, job)
                source_ids = protocols[('LCO', 1, job['seed'])]['source_ids'] if job['kind'] == 'pretrain' else protocols[
                    (job['target'], job['train_count'], job['seed'])]['source_ids']
                src = part_from_ids(source, source_ids)
                target = src if job['kind'] == 'pretrain' else data.load_part(protocols[
                    (job['target'], job['train_count'], job['seed'])]['target_train_ids'], manifest, existing)
                steps = selection['pretrain_steps'] if job['kind'] == 'pretrain' else selection['target_steps']
                binding = job_binding(identity, job, selection, workspace)
                initial = None
                if job['kind'] == 'finetune':
                    initial = validate_training_job(workspace, identity, dict(kind='pretrain', seed=job['seed']))['state']
                common.atomic_json(dict(status='formal_training', completed_training=index,
                    total_training=344, current_job=job, selection=selection), workspace / 'status.json')
                training.train_run(src, target, job['kind'], job['seed'], steps, folder, binding, args.device,
                    initial_state=initial, **training_kwargs(cfg, selection['learning_rate']))
                elapsed = time.monotonic() - begin
                if elapsed > 2:
                    durations.append(elapsed / steps)
                remaining_steps = sum(selection['pretrain_steps'] if rest['kind'] == 'pretrain'
                    else selection['target_steps'] for rest in jobs[index + 1:])
                eta = remaining_steps * statistics.mean(durations) / 3600 if durations else None
                print(f'正式训练完成{index + 1}/344；剩余{len(jobs)-index-1}组；'
                      f'粗略剩余小时：{eta:.2f}' if eta is not None else f'正式训练完成{index + 1}/344', flush=True)
                del target, src, initial
                torch.cuda.empty_cache()
            if not all_training_complete(workspace, identity, jobs):
                raise ValueError('全部344次正式训练完成之前禁止计算目标测试成绩。')
            common.atomic_json(dict(status='testing', completed_training=344), workspace / 'status.json')
            rows = completed_rows(workspace, identity, protocols, manifest)
            for job in jobs:
                if job['kind'] == 'pretrain':
                    continue
                p = protocols[(job['target'], job['train_count'], job['seed'])]
                testing = data.load_part(p['target_test_ids'], manifest, existing)
                src = part_from_ids(source, p['source_ids'])
                row = evaluate_job(workspace, identity, job, src, testing, p, manifest, cfg, args.device)
                rows = [r for r in rows if any(r[k] != job[k] for k in ('kind', 'target', 'train_count', 'seed'))] + [row]
                write_summary(rows, workspace)
                common.atomic_json(dict(status='testing', completed_training=344,
                    completed_predictions=len(rows), total_predictions=336), workspace / 'status.json')
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del testing, src
                torch.cuda.empty_cache()
            common.atomic_json(dict(status='complete', training_runs=344, predictions=len(rows),
                                    selection=selection), workspace / 'status.json')
            report(workspace)
        except BaseException as error:
            common.atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()), workspace / 'status.json')
            raise


if __name__ == '__main__':
    main()
