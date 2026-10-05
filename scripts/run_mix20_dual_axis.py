"""MIX-20 166/41: reuse cycle-only runs, train capacity-only and dual-axis (8 seeds)."""
import argparse
import json
import math
import platform
import statistics
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import context_grid_lifetime as common
import matched_baselines as matched
import evaluate_cycle_mixer_suite as cycle_suite
from src.models.rul_predictors.dual_axis_matched_baselines import DualAxisMatchedBaseline


NEW_MODELS = ('latent_capacity_mixer', 'latent_dual_axis_mixer')
MODELS = ('latent_cycle_mixer',) + NEW_MODELS
SEEDS = tuple(range(8))
SETTINGS = dict(epochs=1000, batch_size=8, accumulation=16, evaluate_every=25,
                lr=.001, amp=True, pair_chunk=8)
SOURCES = (
    'scripts/run_mix20_dual_axis.py', 'scripts/matched_baselines.py',
    'scripts/context_grid_lifetime.py', 'scripts/evaluate_cycle_mixer_suite.py',
    'src/models/rul_predictors/dual_axis_matched_baselines.py',
    'src/models/rul_predictors/dual_axis_latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/matched_baselines.py',
    'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/batlinet.py', 'src/models/nn_model.py',
)


def model_config(name):
    if name == MODELS[0]:
        return cycle_suite.model_config(name)
    if name not in NEW_MODELS:
        raise ValueError(f'Unknown model: {name}')
    return dict(architecture=name, cycles=20, width=1000,
                cycle_mixer_hidden=16, capacity_mixer_hidden=16)


def build_model(config):
    if config['architecture'] == MODELS[0]:
        return matched.MatchedBaseline(**config)
    return DualAxisMatchedBaseline(**config)


def source_hashes():
    return {name: common.digest(ROOT / name) for name in SOURCES}


def runtime_info(device):
    result = dict(python=platform.python_version(), torch=str(torch.__version__),
                  cuda=torch.version.cuda, threads=torch.get_num_threads(),
                  cudnn=torch.backends.cudnn.version(), device_type=torch.device(device).type)
    if result['device_type'] == 'cuda':
        result['gpu'] = torch.cuda.get_device_name(device)
        result['capability'] = list(torch.cuda.get_device_capability(device))
    return result


def read_protocol(context, seed, partition, root):
    path = Path(root) / f'{partition}_seed{seed}.pt'
    # Never create or replace historical reference lists in this experiment.
    if not path.is_file():
        raise FileNotFoundError(f'缺少已有固定参考名单：{path}')
    return matched.fixed_protocol(context, seed, partition, path)


def read_records(folder):
    return [json.loads(line) for line in (folder / 'train.jsonl').read_text(
        encoding='utf-8').splitlines() if line.strip()]


def audit_new(folder, name, seed, data, identity, protocol_path, sources, settings):
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    config = model_config(name)
    expected = dict(model=name, seed=seed, **settings)
    if info['arguments'] != expected or info['model'] != config:
        raise ValueError(f'已有训练配置不同：{folder}')
    if info['source_sha256'] != sources:
        raise ValueError(f'已有训练的源代码指纹不同：{folder}')
    records = read_records(folder)
    if [r['epoch'] for r in records] != list(range(1, settings['epochs'] + 1)):
        raise ValueError(f'已有训练未完成；不会覆盖或自动续训：{folder}')
    validations = [r for r in records if 'validation' in r]
    schedule = sorted(set(range(settings['evaluate_every'], settings['epochs'] + 1,
                                settings['evaluate_every'])) | {settings['epochs']})
    if [r['epoch'] for r in validations] != schedule:
        raise ValueError(f'验证轮次不一致：{folder}')
    if any(not math.isfinite(r['train_loss']) for r in records) or any(
            not math.isfinite(r['validation'][key])
            for r in validations for key in ('RMSE', 'MAE', 'MAPE', 'ACC15')):
        raise ValueError(f'训练或验证指标非有限值：{folder}')
    checkpoint = common.load(folder / 'best.pt')
    best = min(validations, key=lambda r: r['validation']['RMSE'])
    if checkpoint['epoch'] != best['epoch'] or checkpoint['seed'] != seed:
        raise ValueError(f'最佳检查点与验证日志不一致：{folder}')
    if checkpoint['config'] != config or checkpoint['refit'] or checkpoint['selection_metric'] != 'RMSE':
        raise ValueError(f'检查点模型或选择规则不同：{folder}')
    expected_identity = dict(**identity, train_ids=data['train']['ids'],
        validation_protocol_sha256=common.digest(protocol_path), source_sha256=sources)
    for key, value in expected_identity.items():
        if info.get(key) != value or checkpoint.get(key) != value:
            raise ValueError(f'已有训练身份不一致（{key}）：{folder}')
    label = data['train']['labels'].log()
    torch.testing.assert_close(checkpoint['label_mean'], label.mean())
    torch.testing.assert_close(checkpoint['label_scale'], label.std(unbiased=False).clamp_min(1e-6))
    return checkpoint, best, info


def train_one(name, seed, folder, data, fixed, identity, protocol_path,
              sources, runtime, device, settings, jobs_left):
    """Same loss, RNG order, accumulation weighting and optimizer as matched v1."""
    common.seed_all(seed)
    config = model_config(name)
    model = build_model(config).to(device)
    training = data['train']
    label = training['labels'].log()
    mean, scale = label.mean(), label.std(unbiased=False).clamp_min(1e-6)
    y = (label - mean) / scale
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=.01)
    amp = settings['amp'] and torch.device(device).type == 'cuda'
    if settings['amp'] and not amp:
        raise ValueError('正式协议要求 CUDA bfloat16；不会自动改为 CPU/float32。')
    if amp and not torch.cuda.is_bf16_supported():
        raise ValueError('当前显卡不支持 bfloat16；不会自动改变训练精度。')
    folder.mkdir(parents=True, exist_ok=False)
    info = dict(arguments=dict(model=name, seed=seed, **settings), model=config,
        code_commit=common.revision(), source_sha256=sources, runtime=runtime,
        **identity, train_ids=training['ids'], selection_metric='RMSE',
        validation_protocol_sha256=common.digest(protocol_path),
        parameters=sum(p.numel() for p in model.parameters()))
    (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    best, begun = float('inf'), time.perf_counter()
    print(f'开始 {name}，种子 {seed}，参数 {info["parameters"]}，从头训练。', flush=True)
    for epoch in range(1, settings['epochs'] + 1):
        start = time.perf_counter()
        model.train()
        optimizer.zero_grad()
        batches = list(torch.randperm(len(y)).split(settings['batch_size']))
        loss_sum = 0.
        for number, ix in enumerate(batches):
            si = torch.randint(len(y), (len(ix), 2))
            target = common.subset(training['features'], ix, device)
            references = common.subset(training['features'], si, device)
            group_start = number // settings['accumulation'] * settings['accumulation']
            count = sum(len(b) for b in batches[group_start:group_start + settings['accumulation']])
            with torch.autocast('cuda', dtype=torch.bfloat16) if amp else nullcontext():
                out = model(target, references, y[si].to(device))
                truth = y[ix].to(device)
                loss = (.5 * (out['y_ori'].float() - truth).square().mean()
                        + .5 * (out['y_sup_agg'].float() - truth).square().mean())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'训练损失非有限值：{name} seed{seed} epoch{epoch}')
            (loss * len(ix) / count).backward()
            loss_sum += loss.item() * len(ix)
            if (number + 1) % settings['accumulation'] == 0 or number + 1 == len(batches):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
                optimizer.step()
                optimizer.zero_grad()
        record = dict(epoch=epoch, train_loss=loss_sum / len(y),
            train_seconds=time.perf_counter() - start,
            cuda_peak_GiB=torch.cuda.max_memory_allocated(device) / 2**30 if amp else 0.)
        if epoch % settings['evaluate_every'] == 0 or epoch == settings['epochs']:
            result = matched.evaluate(model, training, data['val'], fixed,
                                      mean, scale, device, settings['pair_chunk'])
            record.update(validation=result['scores'], validation_branches=result['branch_scores'])
            if result['scores']['RMSE'] < best:
                best = result['scores']['RMSE']
                checkpoint = dict(state={k: v.cpu() for k, v in model.state_dict().items()},
                    config=config, epoch=epoch, label_mean=mean, label_scale=scale,
                    seed=seed, refit=False, selection_metric='RMSE', **identity,
                    train_ids=training['ids'], source_sha256=sources,
                    validation_protocol_sha256=info['validation_protocol_sha256'])
                torch.save(checkpoint, folder / 'best.pt')
        record['epoch_seconds'] = time.perf_counter() - start
        with (folder / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record) + '\n')
        if epoch % 20 == 0 or 'validation' in record or epoch == settings['epochs']:
            average = (time.perf_counter() - begun) / epoch
            remaining = average * (settings['epochs'] - epoch)
            total = remaining + average * settings['epochs'] * jobs_left
            validation = f'；验证RMSE {record["validation"]["RMSE"]:.3f}' if 'validation' in record else ''
            print(f'{name} 种子{seed} [{epoch}/{settings["epochs"]}] 损失 {record["train_loss"]:.5f}'
                  f'{validation}；本次训练预计剩余 {remaining / 60:.1f} 分钟；'
                  f'后续 {jobs_left} 次训练；整批粗估剩余 {total / 60:.1f} 分钟。', flush=True)
    del model, optimizer


def print_summary(rows, partition):
    print(f'{partition}八种子均值 ± 样本标准差；MAPE、ACC15 均为小数比例。', flush=True)
    for name in MODELS:
        selected = [r for r in rows if r['model'] == name]
        print(name, flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [r[key] for r in selected]
            print(f'  {key}: {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}', flush=True)
    cycle = {r['seed']: r for r in rows if r['model'] == MODELS[0]}
    for name in NEW_MODELS:
        print(f'{name} 相对已有循环轴模型的优势：正数表示新模型更好。', flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            differences = [(r[key] - cycle[r['seed']][key]) * (1 if key == 'ACC15' else -1)
                           for r in rows if r['model'] == name]
            print(f'  {key}: 平均优势 {statistics.mean(differences):.6f}；'
                  f'更好的种子数 {sum(v > 0 for v in differences)}/8', flush=True)


def evaluate_tests(jobs, args, context, data, identity, sources):
    report = []
    output = Path(args.output_dir)
    for name, seed, folder, checkpoint, _ in jobs:
        protocol_path = Path(args.protocol_dir) / f'test_seed{seed}.pt'
        protocol = read_protocol(context, seed, 'test', args.protocol_dir)
        destination = output / name / f'seed{seed}.pt'
        provenance = dict(model=name, seed=seed, partition='test', refit=False,
            epoch=checkpoint['epoch'], checkpoint_sha256=common.digest(folder / 'best.pt'),
            protocol_sha256=common.digest(protocol_path), evaluation_source_sha256=sources, **identity)
        if destination.exists():
            result = common.load(destination)
            if any(result.get(k) != v for k, v in provenance.items()):
                raise ValueError(f'已有测试结果属于不同训练或协议：{destination}')
            if result['train_ids'] != data['train']['ids'] or result['test_ids'] != data['test']['ids']:
                raise ValueError(f'已有测试结果电池名单不同：{destination}')
            if not torch.equal(result['diagnostics']['support_index'], protocol['indices']):
                raise ValueError(f'已有测试结果实际参考名单不同：{destination}')
            torch.testing.assert_close(result['labels'], data['test']['labels'], rtol=0, atol=0)
            recalculated = common.scores(result['prediction'], result['labels'])
            if result['scores'] != recalculated:
                raise ValueError(f'已有测试结果指标与预测不同：{destination}')
        else:
            common.seed_all(seed)
            model = build_model(checkpoint['config']).to(args.device)
            model.load_state_dict(checkpoint['state'], strict=True)
            result = matched.evaluate(model, data['train'], data['test'], protocol['indices'],
                checkpoint['label_mean'], checkpoint['label_scale'], args.device, SETTINGS['pair_chunk'])
            result.update(provenance)
            common.save_new(result, destination)
            del model
        row = dict(model=name, seed=seed, epoch=checkpoint['epoch'], **result['scores'])
        report.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (output / 'test_summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(report, '测试集')


def execute(args, context, data, settings=None):
    settings = SETTINGS if settings is None else settings
    sources = source_hashes()
    identity = dict(data_sha256=common.digest(args.data), context_sha256=common.digest(args.context_data))
    # Every baseline and reference list is checked before creating new outputs.
    jobs, pending, runtime = [], [], None
    for seed in SEEDS:
        read_protocol(context, seed, 'val', args.protocol_dir)
        read_protocol(context, seed, 'test', args.protocol_dir)
        folder = Path(args.cycle_runs) / f'{MODELS[0]}_seed{seed}'
        checkpoint, best = cycle_suite.preflight(folder, data, identity['data_sha256'],
            identity['context_sha256'], seed, MODELS[0], Path(args.protocol_dir) / f'val_seed{seed}.pt')
        # Strict restoration also catches missing or incompatible model tensors.
        model = build_model(checkpoint['config'])
        model.load_state_dict(checkpoint['state'], strict=True)
        del model
        jobs.append((MODELS[0], seed, folder, checkpoint, best))
    for seed in SEEDS:
        for name in NEW_MODELS:
            folder = Path(args.workspace_root) / f'{name}_seed{seed}'
            if folder.exists():
                checkpoint, best, info = audit_new(folder, name, seed, data, identity,
                    Path(args.protocol_dir) / f'val_seed{seed}.pt', sources, settings)
                model = build_model(checkpoint['config'])
                model.load_state_dict(checkpoint['state'], strict=True)
                del model
                if not args.audit_only:
                    runtime = runtime or runtime_info(args.device)
                    if info['runtime'] != runtime:
                        raise ValueError(f'已有新模型训练环境不同：{folder}')
                jobs.append((name, seed, folder, checkpoint, best))
            else:
                pending.append((name, seed, folder))
    print(f'已核对八次已有循环轴训练和全部固定参考名单；'
          f'新模型已完成 {16 - len(pending)}/16 次。', flush=True)
    if args.run_test and pending:
        raise ValueError('测试评价要求全部16次新训练完成；此入口不会代为训练。')
    if args.audit_only:
        print('只读核对完成，没有训练、写文件或计算测试指标。', flush=True)
        return
    runtime = runtime or runtime_info(args.device)
    for number, (name, seed, folder) in enumerate(pending):
        fixed = read_protocol(context, seed, 'val', args.protocol_dir)['indices']
        train_one(name, seed, folder, data, fixed, identity,
            Path(args.protocol_dir) / f'val_seed{seed}.pt', sources, runtime,
            args.device, settings, len(pending) - number - 1)
        checkpoint, best, _ = audit_new(folder, name, seed, data, identity,
            Path(args.protocol_dir) / f'val_seed{seed}.pt', sources, settings)
        jobs.append((name, seed, folder, checkpoint, best))
    if len(jobs) != 24:
        raise ValueError('三组八种子训练没有全部完成。')
    jobs.sort(key=lambda item: (MODELS.index(item[0]), item[1]))
    validation = [dict(model=name, seed=seed, epoch=cp['epoch'], **best['validation'])
                  for name, seed, _, cp, best in jobs]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'validation_summary.json').write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(validation, '验证集')
    if args.run_test:
        evaluate_tests(jobs, args, context, data, identity, sources)
    else:
        print('三组八种子对照完成；本次只汇总验证集，未计算测试集指标。', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--existing-root', default='/root/autodl-tmp/BatLiNet-context-grid-v1')
    parser.add_argument('--cycle-runs', default=str(ROOT / 'workspaces/mix20_cycle_mixer_v1'))
    parser.add_argument('--workspace-root', default=str(ROOT / 'workspaces/mix20_dual_axis_v1'))
    parser.add_argument('--output-dir', default=str(ROOT / 'evaluations/mix20_dual_axis_v1'))
    parser.add_argument('--device', default='cuda:0')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--audit-only', action='store_true', help='Read-only preflight, no training or outputs.')
    mode.add_argument('--run-test', action='store_true', help='Only evaluate after all 24 runs pass the audit.')
    args = parser.parse_args()
    existing = Path(args.existing_root)
    args.context_data = str(existing / 'artifacts/context_grid_v1/mix20.pt')
    args.data = str(existing / 'artifacts/mix20_matched_baselines_v1/mix20.pt')
    args.protocol_dir = str(existing / 'protocols/mix20_matched_v1')
    torch.set_num_threads(8)
    context, data = common.load(args.context_data), common.load(args.data)
    matched.check_alignment(context, data, args.context_data)
    sizes = tuple(len(data[p]['ids']) for p in ('train', 'val', 'test'))
    if context['metadata']['dataset'] != 'mix20' or sizes != (166, 41, 147):
        raise ValueError(f'要求已有 MIX-20 的166/41/147划分，实际为 {sizes}。')
    for part in ('train', 'val'):
        expected = (len(data[part]['ids']), 6, 20, 1000)
        if tuple(data[part]['features']['raw'].shape) != expected:
            raise ValueError(f'六通道特征尺寸不同：{part}')
    if not args.audit_only and torch.device(args.device).type != 'cuda':
        parser.error('正式训练和评价要求 CUDA；只读核对可使用 --device cpu。')
    execute(args, context, data)


if __name__ == '__main__':
    main()
