"""Evaluate a completed chemistry in a separate directory while training continues."""
import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

# Also works as a standalone download to /tmp when invoked from the repository.
ROOT = Path.cwd() if (Path.cwd() / 'scripts/run_cross_chem_matched.py').is_file() else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_cross_chem_matched as suite
from scripts import cross_chem_matched_training as training
from scripts import cross_chem_matched_data as data
from scripts import run_three_tasks as common
import torch


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def preview_plan(identity, target, counts=None):
    jobs = identity.get('jobs')
    if identity.get('schema') != 3 or jobs != suite.formal_plan():
        raise ValueError('只支持已有v3完整三方法队列，不重新生成训练名单。')
    if target not in data.TRAIN_COUNTS:
        raise ValueError('目标体系不在正式协议中。')
    counts = list(data.TRAIN_COUNTS[target]) if counts is None else list(counts)
    if not counts or len(set(counts)) != len(counts) or any(n not in data.TRAIN_COUNTS[target] for n in counts):
        raise ValueError('训练档位缺失、重复或越界。')
    return [job for job in jobs if job['kind'] != 'pretrain'
            and job['target'] == target and job['train_count'] in counts]


def check_ready(workspace, identity, jobs):
    """Validate every selected final and every dependent source final before testing."""
    workspace = Path(workspace)
    if not jobs or any(job not in identity['jobs'] for job in jobs):
        raise ValueError('预览任务不属于已保存的正式队列。')
    dependencies = [dict(kind='pretrain', seed=seed) for seed in sorted({job['seed'] for job in jobs})]
    hashes = {}
    for job in dependencies + jobs:
        folder = suite.job_folder(workspace, job)
        if not (folder / 'final.pt').is_file() or not (folder / 'run.json').is_file():
            raise ValueError(f'所选体系尚未全部训练完成，暂不预览：{folder}')
        suite.validate_training_job(workspace, identity, job)
        for name in ('final.pt', 'run.json'):
            path = folder / name
            hashes[path.relative_to(workspace).as_posix()] = common.digest(path)
    return hashes


def read_protocols(workspace, identity, target, counts=None):
    """Read existing manifests and actual IDs without touching the training queue."""
    workspace = Path(workspace)
    existing = Path(identity['existing_workspace']).resolve()
    imported = identity['imported_data']
    manifest_path = existing / 'dataset/manifest.json'
    if (common.digest(manifest_path) != imported['existing_manifest_sha256']
            or common.digest(existing / 'queue.json') != imported['existing_queue_sha256']):
        raise ValueError('旧数据或队列指纹与本次正式训练不同。')
    # An added preview script is outside the original protected file set.
    for relative, fingerprint in identity['code_hashes'].items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT.resolve()) or common.digest(path) != fingerprint:
            raise ValueError(f'正式训练代码已经变化，拒绝用其他版本推理：{relative}')
    jobs = preview_plan(identity, target, counts)
    protocols = {}
    for job in jobs:
        key = (target, job['train_count'], job['seed'])
        if key in protocols:
            continue
        relative = f'{target}/train_{job["train_count"]}/seed_{job["seed"]}.pt'
        old_path = existing / 'protocols' / relative
        if common.digest(old_path) != imported['protocols'][relative]['sha256']:
            raise ValueError('保存的实际名单指纹发生变化。')
        old = common.load(old_path)
        copied = _json(workspace / 'protocols' / target / f'train_{job["train_count"]}' / f'seed_{job["seed"]}.json')
        for field in ('source_ids', 'target_train_ids', 'target_test_ids', 'unused_target_ids'):
            if copied[field] != old[field]:
                raise ValueError('预览所用名单与正式训练实际名单不同。')
        protocols[key] = copied
    return _json(manifest_path), protocols


def validate_output_path(workspace, existing, output):
    output = Path(output).resolve()
    for protected in (Path(workspace).resolve(), Path(existing).resolve()):
        if output == protected or output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('预览输出必须独立于正式及旧工作区，不能覆盖或嵌套。')
    return output


def _output_file(output, job, name):
    path = output / f'train_{job["train_count"]}' / f'{job["kind"]}_seed{job["seed"]}' / name
    if not path.resolve().is_relative_to(output.resolve()):
        raise ValueError('预览输出中的符号链接指向目录外。')
    return path


def _unchanged(workspace, identity, snapshots):
    for relative, fingerprint in snapshots.items():
        if common.digest(Path(workspace) / relative) != fingerprint:
            raise ValueError('预览过程中正式权重或训练收据变化，停止保存结果。')
    if _json(Path(workspace) / 'queue.json') != identity:
        raise ValueError('预览过程中正式队列发生变化。')


def _print_summary(summary):
    print(f'独立预览：{summary["completed_predictions"]}/{summary["expected_predictions"]}；'
          '八种子均值±样本标准差，MAPE和ACC15显示百分数。', flush=True)
    for group in summary['groups']:
        print(f'\n{group["target"]} 训练{group["train_count"]}块：{suite.LABELS[group["kind"]]}', flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            factor = 100 if key in ('MAPE', 'ACC15') else 1
            metric = group['metrics'][key]
            print(f'  {key}: {factor * metric["mean"]:.4f} ± {factor * metric["sample_std"]:.4f}', flush=True)


def write_preview_summary(rows, output, expected, target):
    summary = suite.summarize(rows)
    summary.update(expected_predictions=expected, independent_preview=True, target=target)
    common.atomic_json(rows, output / 'per_seed.json')
    common.atomic_json(summary, output / 'summary.json')
    path = output / 'per_seed.csv'
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)
    return summary


def run_preview(workspace, output, target='LCO', device='cpu', chunk=1, counts=None):
    workspace = Path(workspace).resolve()
    identity = _json(workspace / 'queue.json')
    output = validate_output_path(workspace, identity['existing_workspace'], output)
    jobs = preview_plan(identity, target, counts)
    snapshots = check_ready(workspace, identity, jobs)
    manifest, protocols = read_protocols(workspace, identity, target, counts)
    cfg = suite.read_config()
    if cfg != identity['config']:
        raise ValueError('推理配置与正式训练不一致。')
    selection = _json(workspace / 'selection.json')
    binding = dict(schema=1, purpose='independent_fixed_checkpoint_preview',
        preview_script_sha256=common.digest(__file__),
        formal_workspace=str(workspace), formal_queue=identity, selection=selection,
        target=target, jobs=jobs, final_and_receipt_sha256=snapshots,
        device=str(device), inference_chunk=chunk, formal_outputs_modified=False,
        preview_scores_used_for_training_selection=False)
    receipt = output / 'preview.json'
    if receipt.exists() and _json(receipt) != binding:
        raise ValueError('已有预览绑定不同，请使用另一独立预览目录。')
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('所选CUDA设备不可用；默认CPU可以与主训练并行。')
    if chunk < 1:
        raise ValueError('推理分块必须为正数。')
    # Lock only the preview directory. The active formal queue stays untouched.
    with common.queue_lock(output):
        common.atomic_json(binding, receipt)
        rows = []
        source = None
        for index, job in enumerate(jobs):
            state = suite.validate_training_job(workspace, identity, job)
            folder = suite.job_folder(workspace, job)
            weight_sha = common.digest(folder / 'final.pt')
            protocol = protocols[(target, job['train_count'], job['seed'])]
            test_path = _output_file(output, job, 'test.pt')
            test_receipt = _output_file(output, job, 'evaluation.json')
            if test_path.exists():
                result = common.load(test_path)
                if test_receipt.exists():
                    saved = _json(test_receipt)
                    if common.digest(test_path) != saved['test_sha256'] or saved['preview_binding'] != binding:
                        raise ValueError('已有预览预测或来源指纹变化。')
            else:
                testing = data.load_part(protocol['target_test_ids'], manifest, identity['existing_workspace'])
                if job['kind'] == 'combination':
                    if source is None:
                        source = data.load_part(protocol['source_ids'], manifest, identity['existing_workspace'])
                    elif source['ids'] != protocol['source_ids']:
                        raise ValueError('所选任务的LFP参考顺序不一致。')
                model = training.make_model(job['kind'], job['seed'], device)
                model.load_state_dict(state['state'], strict=True)
                result = training.predict(model, source, testing, job['kind'], device, chunk=chunk)
                result.update(binding=state['binding'], checkpoint_sha256=weight_sha, step=state['step'],
                              preview_binding=binding)
                del model, testing
                _unchanged(workspace, identity, snapshots)
                common.save(result, test_path)
            if result.get('preview_binding') != binding:
                raise ValueError('预览预测与独立预览来源绑定不同。')
            measured = suite.verify_test(result, state, job, state['binding'], weight_sha, protocol, manifest)
            if any(not math.isfinite(value) for value in measured.values()):
                raise FloatingPointError('预览指标非有限，不剔除失败种子。')
            # Recover an interruption between atomic prediction and receipt saves.
            common.atomic_json(dict(test_sha256=common.digest(test_path), preview_binding=binding), test_receipt)
            rows.append(dict(**job, optimizer_steps=state['step'], **measured))
            # Only preview files are written; formal test files and summaries are untouched.
            summary = write_preview_summary(rows, output, len(jobs), target)
            common.atomic_json(dict(status='running', completed_predictions=len(rows),
                                    expected_predictions=len(jobs)), output / 'status.json')
            print(f'预览 {index + 1}/{len(jobs)} {target} n={job["train_count"]} '
                  f'seed={job["seed"]} {suite.LABELS[job["kind"]]} '
                  f'RMSE={measured["RMSE"]:.4f} MAPE={100 * measured["MAPE"]:.4f}%', flush=True)
            if str(device).startswith('cuda'):
                torch.cuda.empty_cache()
        _unchanged(workspace, identity, snapshots)
        common.atomic_json(dict(status='complete', completed_predictions=len(rows),
                                expected_predictions=len(jobs)), output / 'status.json')
        _print_summary(summary)
        print(f'独立预览目录：{output}', flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=ROOT / 'workspaces/cross_chem_matched_v3')
    parser.add_argument('--target', choices=list(data.TRAIN_COUNTS), default='LCO')
    parser.add_argument('--counts', type=int, nargs='+')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--chunk', type=int, default=1)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args()
    if args.threads < 1:
        parser.error('--threads必须为正数。')
    suffix = '' if args.counts is None else '_n' + '_'.join(map(str, sorted(args.counts)))
    output = args.output or ROOT / f'workspaces/cross_chem_matched_v3_preview/{args.target}{suffix}'
    if args.report:
        _print_summary(_json(output / 'summary.json'))
        return
    if args.background:
        if os.name != 'posix':
            raise RuntimeError('后台启动需要Linux；本地只执行合成检查。')
        workspace = args.workspace.resolve()
        identity = _json(workspace / 'queue.json')
        output = validate_output_path(workspace, identity['existing_workspace'], output)
        check_ready(workspace, identity, preview_plan(identity, args.target, args.counts))
        output.mkdir(parents=True, exist_ok=True)
        command = [value for value in sys.argv[1:] if value != '--background']
        with (output / 'preview.log').open('ab') as log:
            process = subprocess.Popen(['nohup', sys.executable, '-u', '-B', str(Path(__file__).resolve()), *command],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        print(f'预览已后台启动，PID={process.pid}；默认CPU，不占主训练GPU。')
        print(f'日志：{output / "preview.log"}')
        return
    torch.set_num_threads(args.threads)
    run_preview(args.workspace, output, args.target, args.device, args.chunk, args.counts)


if __name__ == '__main__':
    main()
