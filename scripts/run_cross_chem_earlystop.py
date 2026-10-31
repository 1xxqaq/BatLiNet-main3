"""Source-only per-method early stopping, then fixed-budget target refits (v4)."""
import argparse
import contextlib
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
from scripts import run_three_tasks as common
from scripts import run_cross_chem_matched as matched
from scripts import cross_chem_matched_training as fixed
from scripts import cross_chem_matched_data as data
from scripts import cross_chem_earlystop_data as inputs
from scripts import cross_chem_earlystop_training as early
import torch
import yaml

CONFIG = ROOT / 'configs/cross_chem_earlystop_v4/protocol.yaml'
KINDS = matched.KINDS
prepare_inputs = inputs.prepare_inputs
KNOWN_QUEUES = ('cross_chem_matched_v3', 'cross_chem_two_models_v2',
    'cross_chem_cycle_soft_difference_v1', 'mix_cycle_soft_difference_v1',
    'three_tasks_three_models_v1', 'mix20_matched_baselines_v1',
    'mix_branch_ablation_v1', 'mix_naive_ensemble_v1', 'mix_soft_difference_refinements_v1')


def read_config():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    expected = dict(schema=4, experiment='cross_chem_earlystop_v4', models=list(KINDS),
        seeds=list(range(8)), label='ln_cycles_divided_by_fixed_1000',
        learning_rate='reuse_previous_source_selected_value',
        development=dict(reuse_saved_lfp_splits=True, split_seeds=[0, 1], source_count=219,
            training_pool_count=16, validation_count=40, target_counts=[1, 2, 4, 8, 16],
            metric='final_prediction_fixed_log_mse',
            pretrain_initialization='split_specific_best_source_weights', expected_runs=32),
        early_stopping=dict(max_steps=1000, min_stop=100, validate_every=50, patience=5),
        budget=dict(aggregation='median_best_steps_ceil_to_validation_interval',
            group_by='method_and_target_train_count', selection='source_only_no_target_test_scores'),
        training=dict(optimizer='AdamW', weight_decay=.01, logical_batch=128, micro_batch=8,
            inference_chunk=8, checkpoint_every=25, report_every=25, precision='float32',
            shuffle=False, final_weight='fixed_selected_optimizer_step', expected_formal_training=344,
            expected_formal_predictions=336, test_after_all_formal_training=True))
    if cfg != expected:
        raise ValueError('v4配置与本轮约定不一致，不能混合修改恢复。')
    return cfg


def formal_plan():
    return matched.formal_plan()


def select_budgets(records, cfg, learning_rate):
    """Keep each method/count separate; never average across the three methods."""
    dev = cfg['development']
    settings = cfg['early_stopping']
    interval = settings['validate_every']
    keys = [('pretrain', 0)] + [(kind, n) for n in dev['target_counts'] for kind in KINDS]
    expected = {(seed, kind, n) for seed in dev['split_seeds'] for kind, n in keys}
    indexed = {}
    for row in records:
        key = row['seed'], row['kind'], row['train_count']
        best, trained = row['best_step'], row['trained_steps']
        loss = row['best_validation_log_mse']
        if (key not in expected or key in indexed or not isinstance(best, int)
                or not isinstance(trained, int) or not interval <= best <= trained <= settings['max_steps']
                or best % interval or trained % interval or trained < settings['min_stop']
                or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0
                or row['stop_reason'] not in ('early_stopping', 'max_steps')
                or (row['stop_reason'] == 'max_steps') != (trained == settings['max_steps'])):
            raise ValueError('来源早停结果重复、缺损、非有限或与训练规则不符。')
        indexed[key] = row
    if set(indexed) != expected or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError('完整32次来源开发完成前不能冻结正式预算。')
    selected, groups = {}, []
    for kind, n in keys:
        group = [indexed[(seed, kind, n)] for seed in dev['split_seeds']]
        bests = [row['best_step'] for row in group]
        median = statistics.median(bests)
        chosen = int(math.ceil(median / interval) * interval)
        selected[(kind, n)] = chosen
        groups.append(dict(kind=kind, train_count=n, best_steps=bests,
            trained_steps=[row['trained_steps'] for row in group], median_best_step=median,
            selected_steps=chosen, hit_max_runs=sum(row['trained_steps'] == settings['max_steps'] for row in group),
            validation_log_mse=[row['best_validation_log_mse'] for row in group]))
    result = dict(schema=4, learning_rate=learning_rate, source_only=True,
        pretrain_steps=selected[('pretrain', 0)],
        target_steps={kind: {str(n): selected[(kind, n)] for n in dev['target_counts']} for kind in KINDS},
        budget_aggregation=cfg['budget']['aggregation'], groups=groups,
        formal_target_scores_used=False, development_runs=len(records),
        development_records=records, formal_selection='fixed_method_and_count_budget_refit')
    return json.loads(json.dumps(result, allow_nan=False))


def steps_for(job, selection):
    if job['kind'] == 'pretrain':
        return selection['pretrain_steps']
    return selection['target_steps'][job['kind']][str(job['train_count'])]


def job_folder(workspace, job):
    return matched.job_folder(workspace, job)


def job_binding(identity, job, selection, workspace):
    value = dict(queue=identity, job=job, selection=selection, selected_steps=steps_for(job, selection))
    if job['kind'] == 'finetune':
        path = job_folder(workspace, dict(kind='pretrain', seed=job['seed'])) / 'final.pt'
        value['initial_source_checkpoint_sha256'] = common.digest(path)
    return value


def validate_training_job(workspace, identity, job):
    selection = inputs.read_json(Path(workspace) / 'selection.json')
    return fixed.load_completed(job_folder(workspace, job), job_binding(identity, job, selection, workspace),
                                job['kind'], steps_for(job, selection))


def all_training_complete(workspace, identity, jobs):
    if jobs != formal_plan():
        raise ValueError('正式测试要求完整344次训练计划。')
    for job in jobs:
        if not (job_folder(workspace, job) / 'final.pt').is_file():
            return False
        validate_training_job(workspace, identity, job)
    return True


def _kwargs(cfg, lr):
    tr = cfg['training']
    return dict(lr=lr, weight_decay=tr['weight_decay'], checkpoint_every=tr['checkpoint_every'],
        micro_batch=tr['micro_batch'], logical_batch=tr['logical_batch'], report_every=tr['report_every'])


def code_hashes():
    paths = [Path(__file__), CONFIG, ROOT / 'scripts/cross_chem_earlystop_data.py',
        ROOT / 'scripts/cross_chem_earlystop_training.py', ROOT / 'scripts/cross_chem_matched_data.py',
        ROOT / 'scripts/cross_chem_matched_training.py', ROOT / 'scripts/run_cross_chem_matched.py',
        ROOT / 'scripts/run_three_tasks.py']
    paths += sorted((ROOT / 'src').rglob('*.py'))
    return {path.relative_to(ROOT).as_posix(): common.digest(path) for path in paths}


def development_binding(workspace, identity, receipt, seed, kind, count):
    dependency = None
    if kind == 'finetune':
        dependency = common.digest(Path(workspace) / 'development' / f'split_{seed}' / 'pretrain/final.pt')
    return dict(queue=identity, stage='source_only_early_stopping', seed=seed,
        kind=kind, train_count=count, split=receipt['development_splits'][str(seed)],
        learning_rate=receipt['learning_rate'], initial_source_checkpoint_sha256=dependency)


def run_development(source, workspace, identity, cfg, receipt, device):
    """Only LFP tensors and saved LFP splits enter this selection procedure."""
    workspace = Path(workspace)
    root = workspace / 'development'
    records = []
    settings = cfg['early_stopping']
    lr = receipt['learning_rate']
    for seed in cfg['development']['split_seeds']:
        split = receipt['development_splits'][str(seed)]
        inputs.validate_split(split, source['ids'])
        src = matched.part_from_ids(source, split['source_ids'])
        pool = matched.part_from_ids(source, split['training_pool_ids'])
        validation = matched.part_from_ids(source, split['validation_ids'])
        base = root / f'split_{seed}'

        def fit(kind, n, target, initial=None):
            folder = base / ('pretrain' if kind == 'pretrain' else f'train_{n}/{kind}')
            binding = development_binding(workspace, identity, receipt, seed, kind, n)
            common.atomic_json(dict(status='source_only_development', completed_development=len(records),
                total_development=32, current=dict(seed=seed, kind=kind, train_count=n)), workspace / 'status.json')
            state = early.train_run(src, target, validation, kind, seed, folder, binding, device,
                initial_state=initial, settings=settings, inference_chunk=cfg['training']['inference_chunk'],
                **_kwargs(cfg, lr))
            record = dict(seed=seed, kind=kind, train_count=n, best_step=state['best_step'],
                trained_steps=state['trained_steps'], best_validation_log_mse=state['best_loss'],
                stop_reason=state['stop_reason'], hit_max_steps=state['hit_max_steps'],
                checkpoint=folder.relative_to(workspace).as_posix() + '/final.pt',
                checkpoint_sha256=common.digest(folder / 'final.pt'))
            records.append(record)
            common.atomic_json(records, root / 'per_run.json')
            print(f'来源开发完成{len(records)}/32：{kind} n={n} seed={seed} '
                f'最佳步{state["best_step"]}，实际停止{state["trained_steps"]}，'
                f'原因{state["stop_reason"]}', flush=True)
            return state

        pretrained = fit('pretrain', 0, src)
        for n in cfg['development']['target_counts']:
            target = {key: value[:n] for key, value in pool.items()}
            for kind in KINDS:
                fit(kind, n, target, pretrained['state'] if kind == 'finetune' else None)
        del pretrained, src, pool, validation
        if str(device).startswith('cuda'):
            torch.cuda.empty_cache()
    selection = select_budgets(records, cfg, lr)
    selected_path = workspace / 'selection.json'
    if selected_path.exists() and inputs.read_json(selected_path) != selection:
        raise ValueError('来源开发已保存预算与本次复算结果不同。')
    if not selected_path.exists():
        common.atomic_json(selection, selected_path)
    common.atomic_json(dict(status='complete', training_runs=32, selection=selection), root / 'receipt.json')
    return selection


def _verify_selection(workspace, identity):
    selection = inputs.read_json(Path(workspace) / 'selection.json')
    if select_budgets(selection['development_records'], identity['config'],
                      identity['source_receipt']['learning_rate']) != selection:
        raise ValueError('正式训练预算不能从完整来源开发结果复算。')
    for row in selection['development_records']:
        path = (Path(workspace) / row['checkpoint']).resolve()
        if not path.is_relative_to(Path(workspace).resolve()) or common.digest(path) != row['checkpoint_sha256']:
            raise ValueError('用于冻结预算的来源开发权重指纹变化。')
        checked = early.load_completed(path.parent, development_binding(workspace, identity,
            identity['source_receipt'], row['seed'], row['kind'], row['train_count']),
            row['kind'], identity['config']['early_stopping'])
        expected = dict(best_step=checked['best_step'], trained_steps=checked['trained_steps'],
            best_validation_log_mse=checked['best_loss'], stop_reason=checked['stop_reason'],
            hit_max_steps=checked['hit_max_steps'])
        if any(row[key] != value for key, value in expected.items()):
            raise ValueError('已选预算中的最佳步数或停止原因与实际开发权重不同。')
    return selection


def evaluate_job(workspace, identity, job, source, testing, protocol, manifest, cfg, device):
    state = validate_training_job(workspace, identity, job)
    folder = job_folder(workspace, job)
    path, receipt = folder / 'test.pt', folder / 'evaluation.json'
    weight_sha = common.digest(folder / 'final.pt')
    if path.exists():
        output = common.load(path)
        if receipt.exists() and inputs.read_json(receipt)['test_sha256'] != common.digest(path):
            raise ValueError('正式测试预测文件指纹变化。')
    else:
        model = fixed.make_model(job['kind'], job['seed'], device)
        model.load_state_dict(state['state'], strict=True)
        output = fixed.predict(model, source, testing, job['kind'], device, cfg['training']['inference_chunk'])
        output.update(binding=state['binding'], checkpoint_sha256=weight_sha, step=state['step'])
        common.save(output, path)
        del model
    scores = matched.verify_test(output, state, job, state['binding'], weight_sha, protocol, manifest)
    common.atomic_json(dict(test_sha256=common.digest(path), final_sha256=weight_sha, scores=scores), receipt)
    return dict(**job, optimizer_steps=state['step'], **scores)


def completed_rows(workspace, identity, protocols, manifest):
    rows = []
    for job in formal_plan():
        if job['kind'] == 'pretrain':
            continue
        folder = job_folder(workspace, job)
        if not (folder / 'evaluation.json').is_file():
            continue
        receipt = inputs.read_json(folder / 'evaluation.json')
        if (common.digest(folder / 'final.pt') != receipt['final_sha256']
                or common.digest(folder / 'test.pt') != receipt['test_sha256']):
            raise ValueError('已完成的正式权重或预测文件指纹变化。')
        state = validate_training_job(workspace, identity, job)
        p = protocols[(job['target'], job['train_count'], job['seed'])]
        scores = matched.verify_test(common.load(folder / 'test.pt'), state, job, state['binding'],
                                    receipt['final_sha256'], p, manifest)
        rows.append(dict(**job, optimizer_steps=state['step'], **scores))
    return rows


def report(workspace):
    workspace = Path(workspace)
    if not (workspace / 'queue.json').is_file():
        print('v4尚未开始。')
        return
    identity = inputs.read_json(workspace / 'queue.json')
    receipt = identity['source_receipt']
    inputs.verify_receipt(receipt)
    if not (workspace / 'selection.json').is_file():
        status = inputs.read_json(workspace / 'status.json') if (workspace / 'status.json').exists() else {}
        print(f'来源早停开发尚未完成；当前状态：{status.get("status", "preparing")}')
        return
    selection = _verify_selection(workspace, identity)
    print(f'v4来源早停开发32次；沿用学习率{selection["learning_rate"]:g}。')
    for group in selection['groups']:
        name = 'LFP预训练' if group['kind'] == 'pretrain' else matched.LABELS[group['kind']]
        print(f'{name} n={group["train_count"]}：两个开发最佳步{group["best_steps"]}，'
              f'正式预算{group["selected_steps"]}，触及上限{group["hit_max_runs"]}/2')
    manifest = inputs.read_json(Path(receipt['data_workspace']) / 'dataset/manifest.json')
    protocols = {}
    for target, counts in data.TRAIN_COUNTS.items():
        for count in counts:
            for seed in range(8):
                path = workspace / 'protocols' / target / f'train_{count}' / f'seed_{seed}.json'
                value = inputs.read_json(path)
                old = inputs.read_json(Path(receipt['previous_workspace']) / 'protocols' / target /
                                       f'train_{count}' / f'seed_{seed}.json')
                if any(value[key] != old[key] for key in inputs.ID_FIELDS):
                    raise ValueError('v4报告所用电池名单与旧实际名单不同。')
                protocols[(target, count, seed)] = value
    value = matched.summarize(completed_rows(workspace, identity, protocols, manifest))
    print(f'已保存正式测试：{value["completed_predictions"]}/336；八种子均值±样本标准差。')
    for group in value['groups']:
        print(f'\n{group["target"]} n={group["train_count"]} {matched.LABELS[group["kind"]]}')
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            factor = 100 if key in ('MAPE', 'ACC15') else 1
            metric = group['metrics'][key]
            print(f'  {key}: {factor*metric["mean"]:.4f} ± {factor*metric["sample_std"]:.4f}')


def old_lock_folders(previous_workspace, old_data):
    previous_workspace, old_data = Path(previous_workspace).resolve(), Path(old_data).resolve()
    folders = [previous_workspace, old_data]
    for parent in {ROOT / 'workspaces', previous_workspace.parent}:
        folders.extend(parent / name for name in KNOWN_QUEUES if (parent / name / '.queue.lock').is_file())
    return list(dict.fromkeys(path.resolve() for path in folders))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-workspace', type=Path, default=ROOT / 'workspaces/cross_chem_matched_v3')
    parser.add_argument('--workspace', type=Path, default=ROOT / 'workspaces/cross_chem_earlystop_v4')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--pilot-only', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    workspace, previous_workspace = args.workspace.resolve(), args.previous_workspace.resolve()
    if args.report:
        report(workspace)
        return
    if args.prepare_only and args.pilot_only:
        parser.error('准备模式和来源开发模式不能同时指定。')
    cfg = read_config()
    old_queue = inputs.read_json(previous_workspace / 'queue.json')
    old_data = Path(old_queue['existing_workspace']).resolve()
    protected = old_lock_folders(previous_workspace, old_data)
    inputs.separate_output(workspace, *protected)
    if os.name != 'posix' or not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式流程需要Linux CUDA；本地仅执行合成检查。')
    torch.set_num_threads(8)
    common.seed_all(0)
    jobs = formal_plan()
    with contextlib.ExitStack() as stack:
        stack.enter_context(common.queue_lock(workspace))
        for folder in protected:
            stack.enter_context(matched.legacy_read_lock(folder))
        try:
            manifest, protocols, receipt = prepare_inputs(previous_workspace, workspace)
            identity = dict(schema=4, config=cfg, source_receipt=receipt, jobs=jobs,
                formal_training_runs=344, formal_predictions=336, development_runs=32,
                code_hashes=code_hashes(), runtime=common.runtime(args.device),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
            queue = workspace / 'queue.json'
            if queue.exists() and inputs.read_json(queue) != identity:
                raise ValueError('v4队列代码、数据、学习率或运行环境变化，拒绝混合恢复。')
            if not queue.exists():
                common.atomic_json(identity, queue)
            common.atomic_json(dict(status='preflight', formal_training_runs=344,
                                    development_runs=32), workspace / 'status.json')
            source = data.load_part(receipt['imported_data']['source_ids'], manifest, old_data)
            sample = {key: value[:16] for key, value in source.items()}
            gate = matched.preflight(source, sample, cfg, args.device)
            common.atomic_json(gate, workspace / 'preflight.json')
            del sample
            if args.prepare_only:
                common.atomic_json(dict(status='prepared', formal_training_started=False), workspace / 'status.json')
                return
            selection = run_development(source, workspace, identity, cfg, receipt, args.device)
            _verify_selection(workspace, identity)
            if args.pilot_only:
                common.atomic_json(dict(status='development_complete', selection=selection), workspace / 'status.json')
                return
            durations = {}
            for index, job in enumerate(jobs):
                begin = time.monotonic()
                folder = job_folder(workspace, job)
                target = source if job['kind'] == 'pretrain' else data.load_part(protocols[
                    (job['target'], job['train_count'], job['seed'])]['target_train_ids'], manifest, old_data)
                steps = steps_for(job, selection)
                binding = job_binding(identity, job, selection, workspace)
                initial = None
                if job['kind'] == 'finetune':
                    initial = validate_training_job(workspace, identity, dict(kind='pretrain', seed=job['seed']))['state']
                common.atomic_json(dict(status='formal_refit', completed_training=index,
                    total_training=344, current_job=job, selected_steps=steps), workspace / 'status.json')
                fixed.train_run(source, target, job['kind'], job['seed'], steps, folder, binding, args.device,
                    initial_state=initial, **_kwargs(cfg, selection['learning_rate']))
                seconds = time.monotonic() - begin
                if seconds > 2:
                    durations.setdefault(job['kind'], []).append(seconds / steps)
                remaining = jobs[index + 1:]
                fallback = statistics.mean(value for values in durations.values() for value in values) if durations else 0.
                eta = sum(steps_for(rest, selection) * statistics.mean(durations.get(rest['kind'], [fallback]))
                          for rest in remaining) / 3600
                print(f'正式重训完成{index+1}/344，当前{job}，预算{steps}次更新；'
                      f'粗略剩余训练{eta:.2f}小时', flush=True)
                del initial, target
                torch.cuda.empty_cache()
            if not all_training_complete(workspace, identity, jobs):
                raise ValueError('344次正式重训未完整完成，禁止计算目标测试成绩。')
            inputs.verify_receipt(receipt)
            rows = completed_rows(workspace, identity, protocols, manifest)
            for job in jobs:
                if job['kind'] == 'pretrain':
                    continue
                p = protocols[(job['target'], job['train_count'], job['seed'])]
                testing = data.load_part(p['target_test_ids'], manifest, old_data)
                row = evaluate_job(workspace, identity, job, source, testing, p, manifest, cfg, args.device)
                rows = [r for r in rows if any(r[key] != job[key]
                    for key in ('kind', 'target', 'train_count', 'seed'))] + [row]
                matched.write_summary(rows, workspace)
                common.atomic_json(dict(status='testing', completed_training=344,
                    completed_predictions=len(rows), total_predictions=336), workspace / 'status.json')
                print(json.dumps(row, ensure_ascii=False), flush=True)
                del testing
                torch.cuda.empty_cache()
            common.atomic_json(dict(status='complete', development_runs=32, formal_training_runs=344,
                predictions=len(rows), selection=selection), workspace / 'status.json')
            report(workspace)
        except BaseException as error:
            common.atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()), workspace / 'status.json')
            raise


if __name__ == '__main__':
    main()
