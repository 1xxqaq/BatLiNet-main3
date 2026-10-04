"""Retrain the original latent encoder with the completed 207-cell mixer protocol."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix20_cycle_mixer_207 as shared


MODEL = 'latent_cross_attention_current'


def model_config():
    return shared.import_config(
        ROOT / 'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_20.yaml',
        ['model'])['model']


def enhanced_provenance(history):
    # Keep the completed c276979 experiment's original provenance contract.
    expected = dict(history)
    expected['source_sha256'] = {name: shared.digest(ROOT / name) for name in (
        'scripts/run_mix20_cycle_mixer_207.py',
        'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py',
        'src/models/rul_predictors/latent_cross_attention_batlinet.py')}
    return expected


def checked_scores(result, data, protocols, seed, architecture, provenance, checkpoint_sha):
    expected = dict(model=architecture, seed=seed, epoch=1000,
                    provenance=provenance, checkpoint_sha256=checkpoint_sha)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f'测试结果与训练、数据或代码不一致：{architecture}，种子 {seed}。')
    indices = result.get('diagnostics', {}).get('support_index')
    if indices is None or not torch.equal(indices.cpu(), protocols[seed]):
        raise ValueError(f'实际测试参考名单不同：{architecture}，种子 {seed}。')
    measured = shared.scores(data, result['prediction'])
    for key, value in measured.items():
        if not math.isfinite(value) or not math.isclose(
                value, result['scores'][key], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f'保存的测试指标不能重算：{architecture}，种子 {seed}，{key}。')
    return dict(model=architecture, seed=seed, **measured)


def audit_enhanced(root, data, protocols, history):
    provenance, config = enhanced_provenance(history), shared.model_config()
    rows, inputs = [], {}
    for seed in range(8):
        folder = root / f'{shared.MODEL}_seed{seed}'
        checkpoint = shared.validate_complete(folder, provenance, config, seed)
        # Check that the saved state actually belongs to the enhanced encoder.
        model = shared.MODELS.build(config, seed=seed)
        model.load_state_dict(checkpoint['state'], strict=True)
        checkpoint_sha = shared.digest(folder / 'epoch1000.pt')
        path = folder / 'test.pt'
        result = shared.load(path)
        rows.append(checked_scores(result, data, protocols, seed, shared.MODEL,
                                   provenance, checkpoint_sha))
        inputs[str(seed)] = dict(checkpoint_sha256=checkpoint_sha,
                                test_sha256=shared.digest(path))
        print(f'已核对已有增强版：种子 {seed}，1000 轮权重及固定测试结果。', flush=True)
    return rows, inputs


def runtime_info(device):
    is_cuda = torch.device(device).type == 'cuda'
    return dict(python=sys.version, pytorch=str(torch.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if is_cuda else None,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def validate_complete(folder, provenance, config, seed):
    if any(not (folder / name).is_file() for name in ('run.json', 'train.jsonl', 'epoch1000.pt')):
        raise ValueError(f'训练目录不完整：{folder}；不支持中途续训，也不会覆盖。')
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (folder / 'train.jsonl').read_text().splitlines()]
    checkpoint = shared.load(folder / 'epoch1000.pt')
    expected = dict(provenance=provenance, config=config, seed=seed, model=MODEL)
    for source in (info, checkpoint):
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f'已有训练与本次协议、环境或模型不一致：{folder}')
    if (info.get('status') != 'complete' or checkpoint.get('epoch') != 1000
            or [r['epoch'] for r in records] != list(range(1, 1001))
            or any(not math.isfinite(r['train_loss']) for r in records)):
        raise ValueError(f'训练尚未完整完成：{folder}；不支持中途续训。')
    model = shared.MODELS.build(config, seed=seed)
    model.load_state_dict(checkpoint['state'], strict=True)
    return checkpoint


def print_summary(rows):
    labels = [('batlinet', '原始 BatLiNet（历史）'),
              ('latent_cross_attention', 'latent_cross_attention（历史）'),
              (MODEL, 'latent_cross_attention（本次重训）'),
              (shared.MODEL, '循环轴残差增强版（已有）')]
    print('测试集八种子均值 ± 样本标准差；MAPE、ACC15 均为小数比例。', flush=True)
    for architecture, label in labels:
        selected = [row for row in rows if row['model'] == architecture]
        if not selected:
            continue
        print(label, flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [row[key] for row in selected]
            print(f'  {key}: {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}', flush=True)
    current = sorted((row for row in rows if row['model'] == MODEL), key=lambda row: row['seed'])
    enhanced = sorted((row for row in rows if row['model'] == shared.MODEL), key=lambda row: row['seed'])
    if current:
        print('增强版相对本次重训原模型的优势：正数表示增强版更好。', flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            delta = [(a[key] - b[key]) * (-1 if key == 'ACC15' else 1)
                     for a, b in zip(current, enhanced)]
            print(f'  {key}: 平均优势 {statistics.mean(delta):.6f}；'
                  f'增强版更好的种子数 {sum(value > 0 for value in delta)}/8', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--enhanced-runs', type=Path, default=Path('workspaces/mix20_cycle_mixer_207_v1'))
    parser.add_argument('--workspace', type=Path, default=Path('workspaces/mix20_latent_current_207_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    data, protocols, historical, history = shared.audit_history(
        args.history_root, args.original_root, args.protocol_dir)
    enhanced, inputs = audit_enhanced(args.enhanced_runs, data, protocols, history)
    if args.audit_only:
        print_summary(historical + enhanced)
        return
    config = model_config()
    shared.set_seed(0)
    provenance = dict(history, enhanced_results=inputs, runtime=runtime_info(args.device))
    provenance['source_sha256'] = {name: shared.digest(ROOT / name) for name in (
        'scripts/run_mix20_latent_current_207.py', 'scripts/run_mix20_cycle_mixer_207.py',
        'src/models/rul_predictors/latent_cross_attention_batlinet.py',
        'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_20.yaml')}
    for seed in range(8):
        folder = args.workspace / f'{MODEL}_seed{seed}'
        if folder.exists():
            validate_complete(folder, provenance, config, seed)
    args.workspace.mkdir(parents=True, exist_ok=True)
    for seed in range(8):
        folder = args.workspace / f'{MODEL}_seed{seed}'
        if folder.exists():
            print(f'原模型种子 {seed} 已完成并通过核对，跳过训练。', flush=True)
            continue
        shared.set_seed(seed)
        model = shared.MODELS.build(config, seed=seed).to(args.device)
        training = data.train_data.to(args.device)
        folder.mkdir()
        info = dict(provenance=provenance, config=config, seed=seed, model=MODEL, status='training')
        (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        with (folder / 'train.jsonl').open('x', encoding='utf-8') as stream:
            def progress(record):
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                if record['epoch'] % 20 == 0:
                    remaining = record['elapsed_seconds'] / record['epoch'] * (1000 - record['epoch'])
                    print(f"原模型种子 {seed} [{record['epoch']}/1000] 损失 {record['train_loss']:.5f}；"
                          f'本种子预计剩余 {remaining / 60:.1f} 分钟；后续尚有 {7 - seed} 个种子。', flush=True)
            # Exactly the same training function and monitor RNG policy as the
            # completed enhanced experiment; only the original model is built.
            shared.legacy_train(model, training, 1000, 100, 147, progress)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        shared.save_new(dict(state=state, epoch=1000, **{k: v for k, v in info.items() if k != 'status'}),
                        folder / 'epoch1000.pt')
        info['status'] = 'complete'
        (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        training.to('cpu')
        del model, state
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    checkpoints = [validate_complete(args.workspace / f'{MODEL}_seed{seed}', provenance, config, seed)
                   for seed in range(8)]
    rows = historical + enhanced
    # All eight new runs have completed before any new test prediction.
    for seed, checkpoint in enumerate(checkpoints):
        folder = args.workspace / f'{MODEL}_seed{seed}'
        path = folder / 'test.pt'
        checkpoint_sha = shared.digest(folder / 'epoch1000.pt')
        if path.exists():
            result = shared.load(path)
        else:
            shared.set_seed(seed)
            model = shared.MODELS.build(config, seed=seed).to(args.device)
            model.load_state_dict(checkpoint['state'], strict=True)
            model._fixed_test_support_index = protocols[seed]
            model.fixed_test_support_index_path = history['protocols_paths'][seed]
            data.to(args.device)
            prediction, diagnostics = model.predict(data, return_diagnostics=True)
            result = dict(model=MODEL, seed=seed, epoch=1000,
                          prediction=prediction.detach().cpu(), scores=shared.scores(data, prediction),
                          prediction_cycles=data.label_transformation.inverse_transform(prediction).detach().cpu(),
                          truth_cycles=data.label_transformation.inverse_transform(data.test_data.label).detach().cpu(),
                          label_mean=data.label_transformation.transformations[1]._mean.detach().cpu(),
                          label_scale=data.label_transformation.transformations[1]._std.detach().cpu(),
                          diagnostics={key: value.detach().cpu() if value is not None else None
                                       for key, value in diagnostics.items()},
                          provenance=provenance, checkpoint_sha256=checkpoint_sha)
            shared.save_new(result, path)
            data.to('cpu')
            del model
        row = checked_scores(result, data, protocols, seed, MODEL, provenance, checkpoint_sha)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.workspace / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(rows)


if __name__ == '__main__':
    main()
