"""Retrain both frozen encoders on the historical MIX-100 data and references."""
import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix20_cycle_mixer_207 as shared

BASELINE = 'latent_cross_attention_current'
ENHANCED = 'latent_cycle_mixer'
ARCHITECTURES = (BASELINE, ENHANCED)
TRAIN_COUNT, TEST_COUNT = 205, 137
CONFIG_PATH = 'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_100.yaml'


def model_configs():
    baseline = shared.import_config(ROOT / CONFIG_PATH, ['model'])['model']
    enhanced = dict(baseline, name='CycleMixerLatentCrossAttentionBatLiNetRULPredictor',
                    cycle_mixer_hidden=16, cycle_mixer_type='mlp', cycle_conv_kernel=3)
    return {BASELINE: dict(baseline), ENHANCED: enhanced}


def dataset_identity(data):
    if (len(data.train_data), len(data.test_data)) != (TRAIN_COUNT, TEST_COUNT):
        raise ValueError('MIX-100 历史数据必须为 205 个训练电池、137 个测试电池。')
    if data.feature_transformation is not None:
        raise ValueError('历史六通道输入不应有额外特征变换。')
    transform = data.label_transformation
    if (type(transform).__name__ != 'SequentialDataTransformation'
            or [type(t).__name__ for t in transform.transformations] != [
                'LogScaleDataTransformation', 'ZScoreDataTransformation']
            or transform.transformations[0].base != math.e):
        raise ValueError('历史标签变换应为自然对数及标准化。')
    standardizer = transform.transformations[1]
    identity = dict(label_mean=shared.tensor_digest(standardizer._mean),
                    label_std=shared.tensor_digest(standardizer._std))
    all_ids = []
    for name in ('train', 'test'):
        part = getattr(data, name + '_data')
        if tuple(part.feature.shape[1:]) != (6, 100, 1000):
            raise ValueError('历史输入必须为六通道、100 循环、1000 容量位置。')
        if not torch.isfinite(part.feature).all() or not torch.isfinite(part.label).all():
            raise ValueError('历史输入或标签含非有限值。')
        if part.metadata is None or len(part.metadata) != len(part):
            raise ValueError('缺少历史电池编号，无法核对参考索引对应的电池。')
        ids = [row['cell_id'] for row in part.metadata]
        all_ids.extend(ids)
        identity[name] = dict(ids=ids, feature=shared.tensor_digest(part.feature),
                              label=shared.tensor_digest(part.label))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('历史训练、测试电池编号重复或重叠。')
    logarithms = transform.inverse_transform(data.train_data.label).log()
    torch.testing.assert_close(logarithms.mean(0, keepdim=True), standardizer._mean)
    torch.testing.assert_close(logarithms.std(0, keepdim=True), standardizer._std)
    return identity


def validate_protocol(payload, seed):
    expected = dict(seed=seed, num_train_samples=TRAIN_COUNT,
                    num_test_samples=TEST_COUNT, test_support_size=32)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f'MIX-100 固定协议元数据不一致：种子 {seed}。')
    indices = payload.get('indices')
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
            or tuple(indices.shape) != (TEST_COUNT, 32)
            or indices.min().item() < 0 or indices.max().item() >= TRAIN_COUNT):
        raise ValueError(f'MIX-100 固定参考索引不合法：种子 {seed}。')
    return indices


def audit_history(history_root, protocol_dir=None):
    folder = history_root / 'workspaces/fair_eval/batlinet_latent_cross_attention/mix_100/protocol_v1_formal'
    paths = [shared.one_prediction(folder, seed) for seed in range(8)]
    if protocol_dir is None:
        candidate = history_root / 'artifacts/fixed_test_support_indices/mix_100/protocol_v1'
        if candidate.exists():
            protocol_dir = candidate
    # An existing directory must contain all eight protocols. If the directory
    # was not copied, recover the EXACT recorded arrays; never sample new ones.
    external = None
    if protocol_dir is not None:
        external = [validate_protocol(shared.load(protocol_dir / f'seed_{s}.pt'), s)
                    for s in range(8)]
    data, identity = None, None
    protocols, rows, files, protocol_sources = [], [], {}, []
    for seed, path in enumerate(paths):
        result = shared.read_pickle(path)
        current = dataset_identity(result['data'])
        if identity is None:
            identity, data = current, result['data']
        if current != identity or result.get('seed') != seed:
            raise ValueError(f'历史数据、顺序或标签变换不一致：{path}')
        if result.get('evaluation_protocol', {}).get('test_support_size') != 32:
            raise ValueError(f'历史预测未记录 32 参考固定协议：{path}')
        indices = result.get('diagnostics', {}).get('support_index')
        payload = dict(seed=seed, num_train_samples=TRAIN_COUNT,
                       num_test_samples=TEST_COUNT, test_support_size=32, indices=indices)
        indices = validate_protocol(payload, seed).cpu().clone()
        if external is not None and not torch.equal(indices, external[seed]):
            raise ValueError(f'历史预测与固定协议的实际参考名单不同：{path}')
        measured = shared.scores(result['data'], result['prediction'])
        for key in ('RMSE', 'MAE', 'MAPE'):
            if not math.isfinite(measured[key]) or not math.isclose(
                    measured[key], result['scores'][key], rel_tol=1e-5, abs_tol=1e-5):
                raise ValueError(f'历史预测指标不能重算：{path}')
        rows.append(dict(model='latent_cross_attention_history', seed=seed, **measured))
        files[str(seed)] = dict(path=str(path.resolve()), sha256=shared.digest(path))
        protocol_sources.append(dict(
            source='fixed_protocol_file' if external is not None else 'historical_actual_indices',
            path=str((protocol_dir / f'seed_{seed}.pt').resolve()) if external is not None else str(path.resolve()),
            indices_sha256=shared.tensor_digest(indices)))
        protocols.append(indices)
        print(f'已核对 MIX-100 历史数据及固定参考：种子 {seed}。', flush=True)
        del result
    provenance = dict(dataset=identity, historical_predictions=files,
                      protocol_sources=protocol_sources)
    return data, protocols, rows, provenance


def runtime_info(device):
    is_cuda = torch.device(device).type == 'cuda'
    return dict(python=sys.version, pytorch=str(torch.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device=str(device), gpu=torch.cuda.get_device_name(device) if is_cuda else None,
                cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_benchmark=torch.backends.cudnn.benchmark,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def train(model, training, epochs, micro_batch_size, num_test, progress=None):
    """Two logical updates per MIX-100 epoch, with sample-weighted microbatches.

    Physical batches change dropout RNG scheduling compared with historical
    full-batch training. BOTH new models use this same loop and physical size;
    no bitwise equivalence to historical training is claimed.
    """
    if model.grad_accum_steps != 1 or micro_batch_size < 1:
        raise ValueError('本入口要求原配置累积步数为 1，并使用正整数分批大小。')
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=.01)
    prepared = model.build_cell_dataset(training)
    loader = DataLoader(prepared, model.train_batch_size, shuffle=False)
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total, steps, micro_batches = 0., 0, 0
        for batch in loader:
            x, y = batch.values()
            indices = torch.randint(len(prepared), (len(x) * model.train_support_size,),
                                    device=training.device).view(len(x), -1)
            optimizer.zero_grad()
            for offset in range(0, len(x), micro_batch_size):
                end = min(offset + micro_batch_size, len(x))
                query = x[offset:end]
                support_x, support_y = model.get_support_set(
                    query, prepared.feature, training.label,
                    fixed_indices=indices[offset:end], support_is_prepared=True)
                loss = model.forward(query, y[offset:end], support_x, support_y, return_loss=True)
                if not torch.isfinite(loss):
                    raise ValueError(f'第 {epoch} 轮训练损失非有限。')
                # Also correct for the short final chunk and the 77-cell batch.
                (loss * (len(query) / len(x))).backward()
                total += float(loss.detach()) * len(query)
                micro_batches += 1
            optimizer.step()
            steps += 1
        if epoch % model.evaluate_freq == 0:
            model.eval()
            shared.advance_old_monitor_rng(len(training), num_test, model.test_support_size,
                                          model.test_batch_size, training.device)
        if progress is not None:
            progress(dict(epoch=epoch, train_loss=total / len(training),
                          optimizer_steps=steps, micro_batches=micro_batches,
                          elapsed_seconds=time.monotonic() - start))


def validate_complete(folder, architecture, seed, config, provenance):
    if any(not (folder / name).is_file() for name in ('run.json', 'train.jsonl', 'epoch1000.pt')):
        raise ValueError(f'训练目录不完整：{folder}；不支持中途续训，也不会覆盖。')
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (folder / 'train.jsonl').read_text(encoding='utf-8').splitlines()]
    checkpoint = shared.load(folder / 'epoch1000.pt')
    expected = dict(model=architecture, seed=seed, config=config, provenance=provenance)
    for source in (info, checkpoint):
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f'已有训练与本次模型、协议或环境不一致：{folder}')
    count = len(provenance['dataset']['train']['ids'])
    batches = [min(config['train_batch_size'], count - start)
               for start in range(0, count, config['train_batch_size'])]
    micros = sum(math.ceil(n / provenance['policy']['micro_batch_size']) for n in batches)
    if (info.get('status') != 'complete' or checkpoint.get('epoch') != 1000
            or [r['epoch'] for r in records] != list(range(1, 1001))
            or any(not math.isfinite(r['train_loss']) or r['optimizer_steps'] != len(batches)
                   or r['micro_batches'] != micros for r in records)):
        raise ValueError(f'训练未完整完成或更新次数不符：{folder}')
    model = shared.MODELS.build(config, seed=seed)
    model.load_state_dict(checkpoint['state'], strict=True)
    return checkpoint


def checked_scores(result, data, indices, architecture, seed, provenance, checkpoint_sha):
    expected = dict(model=architecture, seed=seed, epoch=1000,
                    provenance=provenance, checkpoint_sha256=checkpoint_sha)
    if any(result.get(key) != value for key, value in expected.items()):
        raise ValueError(f'测试结果与本次训练不一致：{architecture}，种子 {seed}。')
    actual_indices = result.get('diagnostics', {}).get('support_index')
    if actual_indices is None or not torch.equal(actual_indices.cpu(), indices):
        raise ValueError(f'实际测试参考名单不同：{architecture}，种子 {seed}。')
    measured = shared.scores(data, result['prediction'])
    for key, value in measured.items():
        if not math.isfinite(value) or not math.isclose(value, result['scores'][key], rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f'保存的测试指标不能重算：{architecture}，种子 {seed}，{key}。')
    return dict(model=architecture, seed=seed, **measured)


def print_summary(rows):
    print('MIX-100 测试集八种子均值 ± 样本标准差；MAPE、ACC15 均为小数比例。', flush=True)
    for architecture, label in (('latent_cross_attention_history', 'latent_cross_attention（历史）'),
                                (BASELINE, 'latent_cross_attention（本次重训）'),
                                (ENHANCED, '循环轴残差增强版（本次重训）')):
        selected = [row for row in rows if row['model'] == architecture]
        if not selected:
            continue
        print(label, flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [row[key] for row in selected]
            print(f'  {key}: {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}', flush=True)
    paired = {name: {row['seed']: row for row in rows if row['model'] == name}
              for name in ARCHITECTURES}
    if all(set(paired[name]) == set(range(8)) for name in ARCHITECTURES):
        print('增强版相对本次重训原模型的优势：正数表示增强版更好。', flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            delta = [(paired[BASELINE][s][key] - paired[ENHANCED][s][key])
                     * (-1 if key == 'ACC15' else 1) for s in range(8)]
            print(f'  {key}: 平均优势 {statistics.mean(delta):.6f}；'
                  f'增强版更好的种子数 {sum(value > 0 for value in delta)}/8', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('workspaces/mix100_cycle_mixer_v1'))
    parser.add_argument('--micro-batch-size', type=int, default=16)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.micro_batch_size < 1:
        parser.error('--micro-batch-size 必须为正整数。')
    torch.set_num_threads(8)
    data, protocols, historical, provenance = audit_history(args.history_root, args.protocol_dir)
    if args.audit_only:
        print_summary(historical)
        return
    configs = model_configs()
    shared.set_seed(0)
    provenance['runtime'] = runtime_info(args.device)
    provenance['policy'] = dict(epochs=1000, selection='fixed_epoch_1000',
        train_batch_size=128, micro_batch_size=args.micro_batch_size,
        micro_loss_weight='micro_samples/logical_batch_samples',
        shuffle=False, precision='float32', optimizer='AdamW', lr=.001, weight_decay=.01,
        train_support_size=2, test_support_size=32, periodic_rng_every=100,
        label_std='historical_sample_std', test_during_training=False)
    provenance['source_sha256'] = {name: shared.digest(ROOT / name) for name in (
        'scripts/run_mix100_cycle_mixer.py', 'scripts/run_mix20_cycle_mixer_207.py',
        'scripts/pipeline.py', 'src/models/rul_predictors/latent_cross_attention_batlinet.py',
        'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py', CONFIG_PATH)}
    jobs = [(name, seed) for seed in range(8) for name in ARCHITECTURES]
    for architecture, seed in jobs:
        folder = args.workspace / f'{architecture}_seed{seed}'
        if folder.exists():
            validate_complete(folder, architecture, seed, configs[architecture], provenance)
    args.workspace.mkdir(parents=True, exist_ok=True)
    for seed, indices in enumerate(protocols):
        path = args.workspace / 'protocols' / f'seed_{seed}.pt'
        payload = dict(seed=seed, num_train_samples=len(data.train_data),
                       num_test_samples=len(data.test_data), test_support_size=32, indices=indices)
        if path.exists():
            if not torch.equal(shared.load(path)['indices'], indices):
                raise ValueError(f'已有参考名单不同：{path}')
        else:
            shared.save_new(payload, path)
    print('共 16 次训练：两个模型各八种子，每次 1000 轮；全部训练完成后统一固定测试。', flush=True)
    for number, (architecture, seed) in enumerate(jobs):
        folder = args.workspace / f'{architecture}_seed{seed}'
        if folder.exists():
            print(f'{architecture} 种子 {seed} 已完成并通过核对，跳过训练。', flush=True)
            continue
        shared.set_seed(seed)
        model = shared.MODELS.build(configs[architecture], seed=seed).to(args.device)
        training = data.train_data.to(args.device)
        folder.mkdir()
        info = dict(provenance=provenance, config=configs[architecture], seed=seed,
                    model=architecture, status='training')
        (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        with (folder / 'train.jsonl').open('x', encoding='utf-8') as stream:
            def progress(record):
                stream.write(json.dumps(record) + '\n')
                stream.flush()
                if record['epoch'] % 20 == 0:
                    remaining = record['elapsed_seconds'] / record['epoch'] * (1000 - record['epoch'])
                    print(f"{architecture} 种子 {seed} [{record['epoch']}/1000] 损失 {record['train_loss']:.5f}；"
                          f'本次训练预计剩余 {remaining / 60:.1f} 分钟；后续尚有 {15 - number} 次训练。', flush=True)
            train(model, training, 1000, args.micro_batch_size, len(data.test_data), progress)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        shared.save_new(dict(state=state, epoch=1000, **{k: v for k, v in info.items() if k != 'status'}),
                        folder / 'epoch1000.pt')
        info['status'] = 'complete'
        (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        training.to('cpu')
        del model, state
        if torch.device(args.device).type == 'cuda':
            torch.cuda.empty_cache()
    # Audit ALL 16 runs before any newly trained model sees test queries.
    for architecture, seed in jobs:
        validate_complete(args.workspace / f'{architecture}_seed{seed}', architecture,
                          seed, configs[architecture], provenance)
    rows = list(historical)
    for architecture, seed in jobs:
        folder = args.workspace / f'{architecture}_seed{seed}'
        path = folder / 'test.pt'
        checkpoint_sha = shared.digest(folder / 'epoch1000.pt')
        if path.exists():
            result = shared.load(path)
        else:
            shared.set_seed(seed)
            model = shared.MODELS.build(configs[architecture], seed=seed).to(args.device)
            model.load_state_dict(shared.load(folder / 'epoch1000.pt')['state'], strict=True)
            model._fixed_test_support_index = protocols[seed]
            model.fixed_test_support_index_path = str(args.workspace / 'protocols' / f'seed_{seed}.pt')
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
                provenance=provenance, checkpoint_sha256=checkpoint_sha)
            shared.save_new(result, path)
            data.to('cpu')
            del model
        row = checked_scores(result, data, protocols[seed], architecture, seed, provenance, checkpoint_sha)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (args.workspace / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(rows)


if __name__ == '__main__':
    main()
