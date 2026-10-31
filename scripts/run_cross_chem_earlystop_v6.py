"""V6: monitor every 10 updates; keep patience 5, minimum 100 and maximum 1000."""
import argparse
import contextlib
import json
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
from scripts import cross_chem_earlystop_v6_data as inputs
from scripts import cross_chem_earlystop_v5_training as early
import torch
import yaml

CONFIG = ROOT / 'configs/cross_chem_earlystop_v6/protocol.yaml'
V2_CONFIG = ROOT / 'configs/cross_chem_earlystop_v6/from_v2.yaml'
KINDS = matched.KINDS
prepare_inputs = inputs.prepare_inputs


def protected_folders(previous_workspace, old_data, workspace):
    # Preserve all older queues, including a live v5 in its separate worktree.
    previous_workspace, workspace = Path(previous_workspace).resolve(), Path(workspace).resolve()
    names = (*inputs.KNOWN_QUEUES, 'cross_chem_earlystop_v6', 'cross_chem_earlystop_v6_v2')
    trees = (*inputs.KNOWN_WORKTREES, 'BatLiNet-cross-chem-earlystop-v6')
    parents = {ROOT / 'workspaces', previous_workspace.parent}
    for parent in {ROOT.parent, previous_workspace.parent.parent.parent}:
        parents.update(parent / tree / 'workspaces' for tree in trees)
    folders = [previous_workspace, Path(old_data).resolve()]
    for parent in sorted(parents):
        for name in names:
            path = (parent / name).resolve()
            if name not in ('cross_chem_earlystop_v6', 'cross_chem_earlystop_v6_v2') or path != workspace:
                folders.append(path)
    return list(dict.fromkeys(path.resolve() for path in folders))


def read_config(input_mode=inputs.V3_MODE):
    if input_mode not in (inputs.V3_MODE, inputs.V2_MODE):
        raise ValueError('未知的数据和学习率来源模式。')
    path = V2_CONFIG if input_mode == inputs.V2_MODE else CONFIG
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    expected = dict(schema=6, experiment='cross_chem_earlystop_v6', models=list(KINDS),
        seeds=list(range(8)), label='ln_cycles_divided_by_fixed_1000',
        learning_rate=(inputs.FIXED_RATE_SOURCE if input_mode == inputs.V2_MODE
                       else 'reuse_previous_source_selected_value'),
        data=dict(source_count=275, reuse_actual_saved_ids=True, reuse_saved_pair_split=True,
            pair_index_convention=inputs.PAIR_CONVENTION, combination_training_pairs='saved_80_percent',
            combination_validation_pairs='saved_20_percent', test_reference_count=275),
        early_stopping=dict(max_steps=1000, min_stop=100, validate_every=10, patience=5),
        monitoring={kind: early.monitor_policy(kind) for kind in ('pretrain', *KINDS)},
        training=dict(optimizer='AdamW', weight_decay=.01, logical_batch=128, micro_batch=8,
            inference_chunk=8, checkpoint_every=25, report_every=25, precision='float32',
            shuffle=False, final_weight='first_strictly_lowest_monitor_log_mse',
            expected_formal_training=344, expected_formal_predictions=336, test_after_all_formal_training=True))
    if cfg != expected:
        raise ValueError('v6配置与约定不同，不能混合修改恢复。')
    return cfg


def resolve_source(args, parser):
    """Resolve the explicit source route before opening any new output files."""
    if args.existing_workspace is not None:
        if args.previous_workspace is not None:
            parser.error('--existing-workspace与--previous-workspace不能同时指定。')
        try:
            rate = inputs.fixed_rate(args.learning_rate)
        except ValueError as error:
            parser.error(str(error))
        existing = args.existing_workspace.resolve()
        return dict(input_mode=inputs.V2_MODE, source_workspace=existing, old_data=existing,
            workspace=(args.workspace or ROOT / 'workspaces/cross_chem_earlystop_v6_v2').resolve(),
            learning_rate=rate)
    if args.learning_rate is not None:
        parser.error('--learning-rate仅用于--existing-workspace模式；不能覆盖v3已保存的选择。')
    previous = (args.previous_workspace or ROOT / 'workspaces/cross_chem_matched_v3').resolve()
    old = Path(inputs.read_json(previous / 'queue.json')['existing_workspace']).resolve()
    return dict(input_mode=inputs.V3_MODE, source_workspace=previous, old_data=old,
        workspace=(args.workspace or ROOT / 'workspaces/cross_chem_earlystop_v6').resolve(),
        learning_rate=None)


def formal_plan():
    return matched.formal_plan()


def job_folder(workspace, job):
    return matched.job_folder(workspace, job)


def job_binding(identity, job, workspace):
    value = dict(queue=identity, job=job, monitoring=early.monitor_policy(job['kind']))
    if job['kind'] != 'pretrain':
        relative = f'protocols/{job["target"]}/train_{job["train_count"]}/seed_{job["seed"]}.json'
        value['protocol_sha256'] = identity['source_receipt']['protocol_fingerprints'][relative]
    if job['kind'] == 'finetune':
        path = job_folder(workspace, dict(kind='pretrain', seed=job['seed'])) / 'final.pt'
        value['initial_source_checkpoint_sha256'] = common.digest(path)
    return value


def validate_training_job(workspace, identity, job):
    value = early.load_completed(job_folder(workspace, job), job_binding(identity, job, workspace),
                                 job['kind'], identity['config']['early_stopping'])
    if value['monitor_policy'] != identity['config']['monitoring'][job['kind']]:
        raise ValueError('实际停止指标与本次训练约定不同。')
    return value


def all_training_complete(workspace, identity, jobs):
    if jobs != formal_plan():
        raise ValueError('正式测试要求完整344次训练计划。')
    for job in jobs:
        if not (job_folder(workspace, job) / 'final.pt').is_file():
            return False
        validate_training_job(workspace, identity, job)
    return True


def _kwargs(cfg, learning_rate):
    tr = cfg['training']
    return dict(lr=learning_rate, weight_decay=tr['weight_decay'], settings=cfg['early_stopping'],
        checkpoint_every=tr['checkpoint_every'], micro_batch=tr['micro_batch'],
        logical_batch=tr['logical_batch'], report_every=tr['report_every'],
        inference_chunk=tr['inference_chunk'])


def code_hashes():
    paths = [Path(__file__), CONFIG, V2_CONFIG, ROOT / 'scripts/cross_chem_earlystop_v6_data.py',
        ROOT / 'scripts/cross_chem_earlystop_v5_data.py',
        ROOT / 'scripts/cross_chem_earlystop_v5_training.py', ROOT / 'scripts/cross_chem_matched_data.py',
        ROOT / 'scripts/cross_chem_matched_training.py', ROOT / 'scripts/run_cross_chem_matched.py',
        ROOT / 'scripts/run_three_tasks.py', ROOT / 'configs/cross_chem_matched_v3/protocol.yaml',
        ROOT / 'scripts/run_cross_chem_two_models.py', ROOT / 'scripts/run_cross_chem_cycle_soft_difference.py',
        ROOT / 'scripts/original_cross_chem_transfer.py', ROOT / 'scripts/transfer_cycle_soft_data.py']
    paths += sorted((ROOT / 'src').rglob('*.py'))
    return {path.relative_to(ROOT).as_posix(): common.digest(path) for path in paths}


def result_metadata(state):
    return dict(optimizer_steps=state['trained_steps'], trained_steps=state['trained_steps'],
        best_step=state['best_step'], best_monitor_log_mse=state['best_loss'],
        monitor_policy=state['monitor_policy'], stop_reason=state['stop_reason'],
        hit_max_steps=state['hit_max_steps'])


def _verify_prediction(output, state, job, weight_sha, protocol, manifest):
    if output.get('training_metadata') != result_metadata(state):
        raise ValueError('测试所用最佳步数、停止步数或监控指标与权重不同。')
    return matched.verify_test(output, state, job, state['binding'], weight_sha, protocol, manifest)


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
        output.update(binding=state['binding'], checkpoint_sha256=weight_sha, step=state['step'],
                      training_metadata=result_metadata(state))
        common.save(output, path)
        del model
    scores = _verify_prediction(output, state, job, weight_sha, protocol, manifest)
    common.atomic_json(dict(test_sha256=common.digest(path), final_sha256=weight_sha, scores=scores,
                           training_metadata=result_metadata(state)), receipt)
    return dict(**job, **result_metadata(state), **scores)


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
            raise ValueError('已完成正式权重或预测文件指纹变化。')
        state = validate_training_job(workspace, identity, job)
        if receipt['training_metadata'] != result_metadata(state):
            raise ValueError('测试收据中的训练停止信息变化。')
        p = protocols[(job['target'], job['train_count'], job['seed'])]
        scores = _verify_prediction(common.load(folder / 'test.pt'), state, job,
                                    receipt['final_sha256'], p, manifest)
        rows.append(dict(**job, **result_metadata(state), **scores))
    return rows


def training_row(workspace, job, state):
    return dict(**job, **result_metadata(state), samples_seen=state['samples_seen'],
        training_seconds=state['training_seconds'], monitor_seconds=state['monitor_seconds'],
        final_sha256=common.digest(job_folder(workspace, job) / 'final.pt'))


def training_summary(rows):
    groups = []
    for kind in ('pretrain', *KINDS):
        group = [row for row in rows if row['kind'] == kind]
        if not group:
            continue
        groups.append(dict(kind=kind, runs=len(group), monitor_policy=early.monitor_policy(kind),
            total_trained_steps=sum(row['trained_steps'] for row in group),
            mean_best_step=statistics.mean(row['best_step'] for row in group),
            mean_trained_steps=statistics.mean(row['trained_steps'] for row in group),
            hit_max_runs=sum(row['hit_max_steps'] for row in group),
            training_seconds=sum(row['training_seconds'] for row in group),
            monitor_seconds=sum(row['monitor_seconds'] for row in group)))
    return dict(schema=6, completed_training=len(rows), expected_training=344, groups=groups,
        fresh_training=True, development_runs=0, independent_target_validation=False,
        baseline_stopping='training_error_plateau', combination_stopping='heldout_pair_relation_validation')


def load_protocols(workspace, receipt):
    inputs.verify_receipt(receipt, workspace)
    return {(target, count, seed): inputs.read_json(Path(workspace) / 'protocols' / target /
            f'train_{count}' / f'seed_{seed}.json') for target, counts in data.TRAIN_COUNTS.items()
            for count in counts for seed in range(8)}


def report(workspace):
    workspace = Path(workspace)
    if not (workspace / 'queue.json').is_file():
        print('v6尚未开始。')
        return
    identity = inputs.read_json(workspace / 'queue.json')
    mode = identity.get('source_receipt', {}).get('input_mode', inputs.V3_MODE)
    if identity.get('schema') != 6 or identity.get('config') != read_config(mode):
        raise ValueError('v6报告只接受每10次评估的本版队列，不能混用旧版结果。')
    receipt = identity['source_receipt']
    protocols = load_protocols(workspace, receipt)
    rows = []
    for job in formal_plan():
        folder = job_folder(workspace, job)
        if (folder / 'final.pt').is_file() and (folder / 'run.json').is_file():
            state = validate_training_job(workspace, identity, job)
            rows.append(training_row(workspace, job, state))
    summary = training_summary(rows)
    print(f'v6完成训练{len(rows)}/344；学习率{receipt["learning_rate"]:g}；无新增开发阶段。')
    if mode == inputs.V2_MODE:
        print('数据来自已有v2；学习率由本轮显式固定，不是v3开发选择值。')
    print('单电池方法按训练误差停滞停止；组合按留出电池对的关系误差停止。')
    for group in summary['groups']:
        name = 'LFP预训练' if group['kind'] == 'pretrain' else matched.LABELS[group['kind']]
        print(f'{name} {group["runs"]}次：平均最佳步{group["mean_best_step"]:.1f}，'
              f'平均实际更新{group["mean_trained_steps"]:.1f}，触及上限{group["hit_max_runs"]}次')
    if len(rows) != 344:
        return
    manifest = inputs.read_json(Path(receipt['data_workspace']) / 'dataset/manifest.json')
    value = matched.summarize(completed_rows(workspace, identity, protocols, manifest))
    print(f'已保存正式测试{value["completed_predictions"]}/336；八种子均值±样本标准差。')
    for group in value['groups']:
        print(f'\n{group["target"]} n={group["train_count"]} {matched.LABELS[group["kind"]]}')
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            factor = 100 if key in ('MAPE', 'ACC15') else 1
            metric = group['metrics'][key]
            print(f'  {key}: {factor*metric["mean"]:.4f} ± {factor*metric["sample_std"]:.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument('--previous-workspace', type=Path, help='复用v3实际名单和已选学习率')
    source_group.add_argument('--existing-workspace', type=Path, help='直接复用已有v2缓存和实际电池对')
    parser.add_argument('--learning-rate', type=float, help='v2模式必须显式指定的固定学习率')
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if args.report:
        default = 'cross_chem_earlystop_v6_v2' if args.existing_workspace else 'cross_chem_earlystop_v6'
        workspace = (args.workspace or ROOT / 'workspaces' / default).resolve()
        report(workspace)
        return
    source_route = resolve_source(args, parser)
    workspace, previous_workspace = source_route['workspace'], source_route['source_workspace']
    cfg = read_config(source_route['input_mode'])
    old_data = source_route['old_data']
    protected = protected_folders(previous_workspace, old_data, workspace)
    inputs.separate_output(workspace, *protected)
    if os.name != 'posix' or not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('正式流程需要Linux CUDA；本地只执行合成检查。')
    torch.set_num_threads(8)
    common.seed_all(0)
    jobs = formal_plan()
    with contextlib.ExitStack() as stack:
        stack.enter_context(common.queue_lock(workspace))
        for folder in protected:
            if folder in (previous_workspace, old_data) or (folder / '.queue.lock').is_file():
                stack.enter_context(matched.legacy_read_lock(folder))
        try:
            if source_route['input_mode'] == inputs.V2_MODE:
                manifest, protocols, receipt = inputs.prepare_existing(previous_workspace, workspace,
                                                                       source_route['learning_rate'])
            else:
                manifest, protocols, receipt = prepare_inputs(previous_workspace, workspace)
            identity = dict(schema=6, config=cfg, source_receipt=receipt, jobs=jobs,
                input_mode=source_route['input_mode'],
                formal_training_runs=344, formal_predictions=336, development_runs=0,
                lfp_validation_holdout=0, reuse_previous_weights=False,
                code_hashes=code_hashes(), runtime=common.runtime(args.device),
                git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
            queue = workspace / 'queue.json'
            if queue.exists() and inputs.read_json(queue) != identity:
                raise ValueError('v6代码、实际名单、停止设置或环境变化，拒绝混合恢复。')
            if not queue.exists():
                common.atomic_json(identity, queue)
            common.atomic_json(dict(status='preflight', formal_training_runs=344,
                                    development_runs=0), workspace / 'status.json')
            source = data.load_part(receipt['imported_data']['source_ids'], manifest, old_data)
            sample = {key: value[:16] for key, value in source.items()}
            gate = matched.preflight(source, sample, cfg, args.device)
            common.atomic_json(gate, workspace / 'preflight.json')
            del sample
            if args.prepare_only:
                common.atomic_json(dict(status='prepared', formal_training_started=False), workspace / 'status.json')
                return
            durations, training_rows = [], []
            for index, job in enumerate(jobs):
                begin = time.monotonic()
                p = None if job['kind'] == 'pretrain' else protocols[
                    (job['target'], job['train_count'], job['seed'])]
                target = source if p is None else data.load_part(p['target_train_ids'], manifest, old_data)
                train_pairs, val_pairs = inputs.native_pairs(p) if job['kind'] == 'combination' else (None, None)
                initial = None
                if job['kind'] == 'finetune':
                    initial = validate_training_job(workspace, identity, dict(kind='pretrain', seed=job['seed']))['state']
                common.atomic_json(dict(status='formal_training', completed_training=index,
                    total_training=344, current_job=job, monitor_policy=early.monitor_policy(job['kind'])),
                    workspace / 'status.json')
                state = early.train_run(source, target, job['kind'], job['seed'], job_folder(workspace, job),
                    job_binding(identity, job, workspace), args.device, initial_state=initial,
                    train_pairs=train_pairs, val_pairs=val_pairs, **_kwargs(cfg, receipt['learning_rate']))
                training_rows.append(training_row(workspace, job, state))
                common.atomic_json(training_rows, workspace / 'training_runs.json')
                common.atomic_json(training_summary(training_rows), workspace / 'training_summary.json')
                seconds = time.monotonic() - begin
                if seconds > 2:
                    durations.append(seconds)
                eta = (len(jobs) - index - 1) * statistics.mean(durations) / 3600 if durations else 0.
                print(f'正式训练完成{index+1}/344：{job}；最佳步{state["best_step"]}，'
                    f'实际更新{state["trained_steps"]}，停止原因{state["stop_reason"]}；'
                    f'粗略剩余{eta:.2f}小时', flush=True)
                del initial, target, state, train_pairs, val_pairs
                torch.cuda.empty_cache()
            if not all_training_complete(workspace, identity, jobs):
                raise ValueError('344次正式训练未完整完成，禁止计算目标测试成绩。')
            inputs.verify_receipt(receipt, workspace)
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
            common.atomic_json(dict(status='complete', development_runs=0, formal_training_runs=344,
                predictions=len(rows), learning_rate=receipt['learning_rate']), workspace / 'status.json')
            report(workspace)
        except BaseException as error:
            common.atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()), workspace / 'status.json')
            raise


if __name__ == '__main__':
    main()
