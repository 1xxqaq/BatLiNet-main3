"""Compare original and soft-aligned-difference predictors on the frozen 207-cell protocol."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix20_latent_current_207 as current

shared = current.shared
# Explicit registration keeps all historical entry points and package files unchanged.
from src.models.rul_predictors import soft_aligned_difference_batlinet

NEW_MODELS = ('latent_original_matched', 'latent_soft_aligned_difference')
LABELS = {
    'batlinet': '原始BatLiNet（历史）',
    'latent_cross_attention': '潜在交叉注意力（历史）',
    NEW_MODELS[0]: '潜在交叉注意力（本次同条件重训）',
    NEW_MODELS[1]: '软匹配潜在差分（本次训练）',
}
SOURCE_FILES = (
    'scripts/run_mix20_soft_aligned_difference_207.py',
    'scripts/run_mix20_latent_current_207.py',
    'scripts/run_mix20_cycle_mixer_207.py',
    'scripts/pipeline.py',
    'src/models/rul_predictors/soft_aligned_difference_batlinet.py',
    'src/models/rul_predictors/latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/batlinet.py',
    'src/models/nn_model.py',
    'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_20.yaml',
)


def model_config(architecture):
    config = current.model_config()
    if architecture == NEW_MODELS[1]:
        config['name'] = 'SoftAlignedDifferenceBatLiNetRULPredictor'
    elif architecture != NEW_MODELS[0]:
        raise ValueError(f'未知模型：{architecture}')
    return config


def validate_complete(folder, provenance, config, architecture, seed):
    if any(not (folder / name).is_file() for name in ('run.json', 'train.jsonl', 'epoch1000.pt')):
        raise ValueError(f'训练目录不完整：{folder}；不支持中途续训，也不会覆盖。')
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (folder / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    checkpoint = shared.load(folder / 'epoch1000.pt')
    expected = dict(provenance=provenance, config=config, seed=seed, model=architecture)
    for source in (info, checkpoint):
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f'已有训练与当前协议、代码、环境或模型不一致：{folder}')
    if (info.get('status') != 'complete' or checkpoint.get('epoch') != 1000
            or [record.get('epoch') for record in records] != list(range(1, 1001))
            or any(not math.isfinite(record['train_loss']) for record in records)):
        raise ValueError(f'训练未完整完成1000轮：{folder}；不支持中途续训。')
    model = shared.MODELS.build(config, seed=seed)
    model.load_state_dict(checkpoint['state'], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if info.get('parameters') != parameters or checkpoint.get('parameters') != parameters:
        raise ValueError(f'保存的参数量与实际模型不一致：{folder}')
    return checkpoint


def checked_test(folder, data, protocols, provenance, architecture, seed):
    result = shared.load(folder / 'test.pt')
    prediction = result.get('prediction')
    if (not isinstance(prediction, torch.Tensor)
            or prediction.shape != data.test_data.label.shape
            or not torch.isfinite(prediction).all()):
        raise ValueError(f'测试预测形状或数值不合法：{folder}')
    return current.checked_scores(result, data, protocols, seed, architecture,
                                  provenance, shared.digest(folder / 'epoch1000.pt'))


def print_summary(rows):
    print('207块全训练；固定第1000轮测试；八种子均值 ± 样本标准差。', flush=True)
    print('MAPE、ACC15 均为小数比例；不使用验证集或测试指标挑选轮次。', flush=True)
    for architecture, label in LABELS.items():
        selected = [row for row in rows if row['model'] == architecture]
        if not selected:
            continue
        print(label, flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [row[key] for row in selected]
            print(f'  {key}: {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}', flush=True)
    lookup = {(row['model'], row['seed']): row for row in rows}
    control, candidate = NEW_MODELS
    if all((name, seed) in lookup for name in NEW_MODELS for seed in range(8)):
        print('潜在差分相对本次原模型：正数表示差分版更好。', flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            delta = [(lookup[control, seed][key] - lookup[candidate, seed][key])
                     * (-1 if key == 'ACC15' else 1) for seed in range(8)]
            print(f'  {key}: 平均优势 {statistics.mean(delta):.6f}；'
                  f'更好的种子数 {sum(value > 0 for value in delta)}/8', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('workspaces/mix20_soft_aligned_difference_207_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    data, protocols, historical, history = shared.audit_history(
        args.history_root, args.original_root, args.protocol_dir)
    shared.set_seed(0)
    runtime = current.runtime_info(args.device)
    provenance = dict(history, runtime=runtime,
                      source_sha256={name: shared.digest(ROOT / name) for name in SOURCE_FILES})
    configs = {name: model_config(name) for name in NEW_MODELS}
    # Reject incomplete, changed or corrupted existing runs before new training.
    for architecture in NEW_MODELS:
        for seed in range(8):
            folder = args.workspace / f'{architecture}_seed{seed}'
            if folder.exists():
                validate_complete(folder, provenance, configs[architecture], architecture, seed)
                if (folder / 'test.pt').exists():
                    checked_test(folder, data, protocols, provenance, architecture, seed)
    if args.audit_only:
        print_summary(historical)
        print('只读核对结束，未训练或写入实验文件。', flush=True)
        return
    args.workspace.mkdir(parents=True, exist_ok=True)
    for group, architecture in enumerate(NEW_MODELS):
        config = configs[architecture]
        for seed in range(8):
            folder = args.workspace / f'{architecture}_seed{seed}'
            if folder.exists():
                print(f'{LABELS[architecture]}种子 {seed} 已完成并通过核对，跳过训练。', flush=True)
                continue
            shared.set_seed(seed)
            model = shared.MODELS.build(config, seed=seed).to(args.device)
            training = data.train_data.to(args.device)
            folder.mkdir()
            info = dict(provenance=provenance, config=config, seed=seed, model=architecture,
                        parameters=sum(p.numel() for p in model.parameters()), status='training')
            print(f'开始{LABELS[architecture]}，种子 {seed}，参数量 {info["parameters"]}。', flush=True)
            (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            with (folder / 'train.jsonl').open('x', encoding='utf-8') as stream:
                def progress(record):
                    stream.write(json.dumps(record) + '\n')
                    stream.flush()
                    if record['epoch'] % 20 == 0:
                        remaining = record['elapsed_seconds'] / record['epoch'] * (1000 - record['epoch'])
                        later = sum(not (args.workspace / f'{name}_seed{s}').exists()
                                    for name in NEW_MODELS[group:] for s in range(8)
                                    if name != architecture or s > seed)
                        print(f'{LABELS[architecture]}种子 {seed} [{record["epoch"]}/1000] '
                              f'损失 {record["train_loss"]:.5f}；本种子预计剩余 {remaining / 60:.1f} 分钟；'
                              f'后续尚有 {later} 次训练。', flush=True)
                # This function accepts no validation data or test features/labels.
                shared.legacy_train(model, training, 1000, 100, 147, progress)
            state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            shared.save_new(dict(state=state, epoch=1000,
                                 **{key: value for key, value in info.items() if key != 'status'}),
                            folder / 'epoch1000.pt')
            info['status'] = 'complete'
            (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
            training.to('cpu')
            del model, state
            if torch.device(args.device).type == 'cuda':
                torch.cuda.empty_cache()
    # Complete and strictly audit every one of the 16 runs before new testing.
    for architecture in NEW_MODELS:
        for seed in range(8):
            validate_complete(args.workspace / f'{architecture}_seed{seed}', provenance,
                              configs[architecture], architecture, seed)
    rows = list(historical)
    for architecture in NEW_MODELS:
        for seed in range(8):
            folder = args.workspace / f'{architecture}_seed{seed}'
            if not (folder / 'test.pt').exists():
                checkpoint = shared.load(folder / 'epoch1000.pt')
                shared.set_seed(seed)
                model = shared.MODELS.build(configs[architecture], seed=seed).to(args.device)
                model.load_state_dict(checkpoint['state'], strict=True)
                model._fixed_test_support_index = protocols[seed]
                model.fixed_test_support_index_path = history['protocols_paths'][seed]
                data.to(args.device)
                prediction, diagnostics = model.predict(data, return_diagnostics=True)
                result = dict(model=architecture, seed=seed, epoch=1000,
                    prediction=prediction.detach().cpu(), scores=shared.scores(data, prediction),
                    prediction_cycles=data.label_transformation.inverse_transform(prediction).detach().cpu(),
                    truth_cycles=data.label_transformation.inverse_transform(data.test_data.label).detach().cpu(),
                    label_mean=data.label_transformation.transformations[1]._mean.detach().cpu(),
                    label_scale=data.label_transformation.transformations[1]._std.detach().cpu(),
                    diagnostics={key: value.detach().cpu() if value is not None else None
                                 for key, value in diagnostics.items()},
                    provenance=provenance, checkpoint_sha256=shared.digest(folder / 'epoch1000.pt'))
                shared.save_new(result, folder / 'test.pt')
                data.to('cpu')
                del model, checkpoint
                if torch.device(args.device).type == 'cuda':
                    torch.cuda.empty_cache()
            row = checked_test(folder, data, protocols, provenance, architecture, seed)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.workspace / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(rows)


if __name__ == '__main__':
    main()
