"""Compare raw and self-cycle-difference inputs, 207 training cells, eight seeds."""
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
from src.models.rul_predictors.cycle_difference_batlinet import CycleDifferenceBatLiNetRULPredictor

shared = current.shared
NEW_MODELS = ('cycle_raw_current', 'cycle_self_difference')
LABELS = {'batlinet': '原始BatLiNet（历史）',
          'latent_cross_attention': '基础潜在交叉注意力（历史）',
          NEW_MODELS[0]: '循环轴版：原始输入（本次重训）',
          NEW_MODELS[1]: '循环轴版：自身循环差分（本次重训）'}
SOURCE_FILES = (
    'scripts/run_mix20_cycle_difference_207.py',
    'scripts/run_mix20_latent_current_207.py',
    'scripts/run_mix20_cycle_mixer_207.py', 'scripts/pipeline.py',
    'src/models/rul_predictors/cycle_difference_batlinet.py',
    'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/batlinet.py', 'src/models/nn_model.py',
    'src/data/databundle.py',
    'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_20.yaml',
)


def model_config(architecture):
    if architecture not in NEW_MODELS:
        raise ValueError(f'未知输入对照：{architecture}')
    config = shared.model_config()
    config.update(name='CycleDifferenceBatLiNetRULPredictor', difference_base=0, alpha=.5,
                  input_mode='raw' if architecture == NEW_MODELS[0] else 'self_difference')
    return config


def build_model(config, seed):
    return shared.MODELS.build(config, seed=seed)


def validate_complete(folder, provenance, config, architecture, seed):
    required = ('run.json', 'train.jsonl', 'epoch1000.pt')
    if any(not (folder / name).is_file() for name in required):
        raise ValueError(f'训练目录不完整：{folder}；不覆盖或从中途续训。')
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (folder / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    checkpoint = shared.load(folder / 'epoch1000.pt')
    expected = dict(provenance=provenance, config=config, seed=seed, model=architecture)
    for source in (info, checkpoint):
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f'已有训练与当前输入、协议、环境或代码不一致：{folder}')
    if (info.get('status') != 'complete' or checkpoint.get('epoch') != 1000
            or [r.get('epoch') for r in records] != list(range(1, 1001))
            or any(not math.isfinite(r['train_loss']) for r in records)):
        raise ValueError(f'训练没有完整完成1000轮：{folder}')
    model = build_model(config, seed)
    model.load_state_dict(checkpoint['state'], strict=True)
    parameters = sum(p.numel() for p in model.parameters())
    if any(source.get('parameters') != parameters for source in (info, checkpoint)):
        raise ValueError(f'参数量不一致：{folder}')
    return checkpoint


def checked_test(folder, data, protocols, provenance, architecture, seed):
    result = shared.load(folder / 'test.pt')
    prediction = result.get('prediction')
    if (not isinstance(prediction, torch.Tensor) or prediction.shape != data.test_data.label.shape
            or not torch.isfinite(prediction).all()):
        raise ValueError(f'测试预测形状或数值不合法：{folder}')
    row = current.checked_scores(result, data, protocols, seed, architecture,
                                 provenance, shared.digest(folder / 'epoch1000.pt'))
    diagnostics = result['diagnostics']
    ori, sup, aggregate = (diagnostics[key] for key in ('y_ori', 'y_sup', 'y_sup_agg'))
    if ori.shape != prediction.shape or aggregate.shape != prediction.shape or sup.shape != protocols[seed].shape:
        raise ValueError(f'测试分支诊断形状不一致：{folder}')
    for value in (ori, sup, aggregate):
        if not torch.isfinite(value).all():
            raise ValueError(f'测试分支诊断包含非有限值：{folder}')
    torch.testing.assert_close(aggregate, sup.median(1)[0])
    torch.testing.assert_close(prediction, .5 * ori + .5 * aggregate)
    transform = data.label_transformation
    torch.testing.assert_close(result['prediction_cycles'], transform.inverse_transform(prediction))
    torch.testing.assert_close(result['truth_cycles'], transform.inverse_transform(data.test_data.label))
    torch.testing.assert_close(result['label_mean'], transform.transformations[1]._mean)
    torch.testing.assert_close(result['label_scale'], transform.transformations[1]._std)
    for key in ('y_ori', 'y_sup_agg'):
        measured = shared.scores(data, diagnostics[key])
        for metric, value in measured.items():
            if not math.isclose(value, result['branch_scores'][key][metric], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f'分支指标不能复算：{folder}，{key}，{metric}')
        row[key + '_RMSE'] = measured['RMSE']
    return row


def print_summary(rows):
    print('207训练／147测试；无验证集，固定第1000轮；八种子均值 ± 样本标准差。', flush=True)
    print('寿命误差单位为循环；MAPE、ACC15为小数比例。', flush=True)
    for architecture, label in LABELS.items():
        selected = [row for row in rows if row['model'] == architecture]
        if not selected:
            continue
        print(label, flush=True)
        keys = ('RMSE', 'MAE', 'MAPE', 'ACC15')
        if architecture in NEW_MODELS:
            keys += ('y_ori_RMSE', 'y_sup_agg_RMSE')
        for key in keys:
            values = [row[key] for row in selected]
            print(f'  {key}: {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}', flush=True)
    lookup = {(row['model'], row['seed']): row for row in rows}
    if not all((name, seed) in lookup for name in NEW_MODELS for seed in range(8)):
        return
    print('自身差分相对本次原始输入的优势：正数表示差分更好。', flush=True)
    for key in ('RMSE', 'MAE', 'MAPE', 'ACC15', 'y_ori_RMSE', 'y_sup_agg_RMSE'):
        delta = [(lookup[NEW_MODELS[0], seed][key] - lookup[NEW_MODELS[1], seed][key])
                 * (-1 if key == 'ACC15' else 1) for seed in range(8)]
        print(f'  {key}: 平均优势 {statistics.mean(delta):.6f}；'
              f'差分更好的种子数 {sum(value > 0 for value in delta)}/8', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('workspaces/mix20_cycle_difference_207_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    data, protocols, historical, history = shared.audit_history(
        args.history_root, args.original_root, args.protocol_dir)
    shared.set_seed(0)
    provenance = dict(history, runtime=current.runtime_info(args.device),
                      input_policy=dict(clean_before_difference=True, difference_base=0,
                                        preserve_zero_cycles=True, independent_cell_difference=True),
                      source_sha256={name: shared.digest(ROOT / name) for name in SOURCE_FILES})
    configs = {name: model_config(name) for name in NEW_MODELS}
    # Reject invalid first cycles before creating any training directory.
    preflight = build_model(configs[NEW_MODELS[1]], 0)
    for part in (data.train_data, data.test_data):
        preflight._prepare_feature(part.feature)
    del preflight
    for architecture in NEW_MODELS:
        for seed in range(8):
            folder = args.workspace / f'{architecture}_seed{seed}'
            if folder.exists():
                validate_complete(folder, provenance, configs[architecture], architecture, seed)
                if (folder / 'test.pt').exists():
                    checked_test(folder, data, protocols, provenance, architecture, seed)
    if args.audit_only:
        print('数据、固定参考及差分基准只读核对通过；未训练或写入结果。', flush=True)
        return
    args.workspace.mkdir(parents=True, exist_ok=True)
    for group, architecture in enumerate(NEW_MODELS):
        config = configs[architecture]
        for seed in range(8):
            folder = args.workspace / f'{architecture}_seed{seed}'
            if folder.exists():
                print(f'{LABELS[architecture]}种子 {seed} 已完成，跳过训练。', flush=True)
                continue
            shared.set_seed(seed)
            model = build_model(config, seed).to(args.device)
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
                              f'损失 {record["train_loss"]:.5f}；预计剩余 {remaining / 60:.1f} 分钟；'
                              f'后续 {later} 次训练。', flush=True)
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
    # All 16 trainings must finish before any new test prediction.
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
                model = build_model(configs[architecture], seed).to(args.device)
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
                    branch_scores={key: shared.scores(data, diagnostics[key]) for key in ('y_ori', 'y_sup_agg')},
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
