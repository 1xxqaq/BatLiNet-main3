"""Run eight cycle-mixer seeds against the frozen historical 207-cell protocol."""
import argparse
import hashlib
import io
import json
import math
import pickle
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src  # Register the original and enhanced predictors.
from src.builders import MODELS
from src.utils import import_config
from scripts.pipeline import set_seed


MODEL = 'latent_cycle_mixer'
POLICY = dict(epochs=1000, train_batch_size=128, gradient_accumulation_steps=1,
              shuffle=False, precision='float32', optimizer='AdamW', lr=.001,
              weight_decay=.01, train_support_size=2, test_support_size=32,
              periodic_rng_every=100, selection='fixed_epoch_1000',
              label_std='historical_sample_std', test_during_training=False)


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda value: torch.load(io.BytesIO(value), map_location='cpu',
                                           weights_only=False)
        return super().find_class(module, name)


def read_pickle(path):
    with Path(path).open('rb') as stream:
        return CPUUnpickler(stream).load()


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def save_new(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        torch.save(value, stream)


def tensor_digest(value):
    value = value.detach().cpu().contiguous()
    checksum = hashlib.sha256(str((value.dtype, tuple(value.shape))).encode())
    view = memoryview(value.numpy()).cast('B')
    for start in range(0, len(view), 1024 * 1024):
        checksum.update(view[start:start + 1024 * 1024])
    return checksum.hexdigest()


def dataset_identity(data):
    """Array indices are meaningful only with the historical battery ordering."""
    if (len(data.train_data), len(data.test_data)) != (207, 147):
        raise ValueError('历史数据必须为 207 个训练电池、147 个测试电池。')
    if data.feature_transformation is not None:
        raise ValueError('历史六通道输入不应有额外特征变换。')
    transform = data.label_transformation
    if type(transform).__name__ != 'SequentialDataTransformation':
        raise ValueError('缺少历史标签变换。')
    if [type(t).__name__ for t in transform.transformations] != [
            'LogScaleDataTransformation', 'ZScoreDataTransformation']:
        raise ValueError('历史标签变换应为自然对数及标准化。')
    logarithm, standardizer = transform.transformations
    if logarithm.base != math.e:
        raise ValueError('历史标签变换不是自然对数。')
    identity = dict(label_mean=tensor_digest(standardizer._mean),
                    label_std=tensor_digest(standardizer._std))
    all_ids = []
    for name in ('train', 'test'):
        part = getattr(data, name + '_data')
        if tuple(part.feature.shape[1:]) != (6, 20, 1000):
            raise ValueError('历史输入必须为六通道、20 循环、1000 容量位置。')
        if not torch.isfinite(part.feature).all() or not torch.isfinite(part.label).all():
            raise ValueError('历史输入或标签含非有限值。')
        if part.metadata is None or len(part.metadata) != len(part):
            raise ValueError('缺少历史电池编号；不能核对固定索引对应的电池。')
        ids = [row['cell_id'] for row in part.metadata]
        all_ids.extend(ids)
        identity[name] = dict(ids=ids, feature=tensor_digest(part.feature),
                              label=tensor_digest(part.label))
    if len(set(all_ids)) != len(all_ids):
        raise ValueError('历史训练、测试电池编号重复或重叠。')
    # Keep the original sample standard deviation, rather than the population
    # standard deviation used by the newer 166/41 runner.
    raw = transform.inverse_transform(data.train_data.label)
    torch.testing.assert_close(raw.log().mean(0, keepdim=True), standardizer._mean)
    torch.testing.assert_close(raw.log().std(0, keepdim=True), standardizer._std)
    return identity


def validate_protocol(payload, seed):
    expected = dict(seed=seed, num_train_samples=207,
                    num_test_samples=147, test_support_size=32)
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError(f'旧固定协议元数据不一致：种子 {seed}。')
    indices = payload.get('indices')
    if (not isinstance(indices, torch.Tensor) or indices.dtype != torch.int64
            or tuple(indices.shape) != (147, 32)
            or indices.min().item() < 0 or indices.max().item() >= 207):
        raise ValueError(f'旧固定参考索引不合法：种子 {seed}。')
    return indices


def one_prediction(folder, seed):
    files = list(Path(folder).glob(f'predictions_seed_{seed}_*.pkl'))
    if len(files) != 1:
        raise ValueError(f'{folder} 的种子 {seed} 应恰有一份历史正式预测，实际 {len(files)} 份。')
    return files[0]


def scores(data, prediction):
    result = {key: data.evaluate(prediction, key) for key in ('RMSE', 'MAE', 'MAPE')}
    transform = data.label_transformation
    actual = transform.inverse_transform(data.test_data.label)
    predicted = transform.inverse_transform(prediction)
    result['ACC15'] = float(((predicted - actual).abs() / actual <= .15).float().mean())
    return result


def audit_history(history_root, original_root, protocol_dir=None):
    latent_dir = history_root / 'workspaces/fair_eval/batlinet_latent_cross_attention/mix_20/protocol_v1_formal'
    batlinet_dir = original_root / 'workspaces/fair_eval/batlinet_original/mix_20/protocol_v1_formal'
    if protocol_dir is None:
        relative = Path('artifacts/fixed_test_support_indices/mix_20/protocol_v1')
        protocol_dir = next((root / relative for root in (history_root, original_root)
                             if all((root / relative / f'seed_{s}.pt').is_file() for s in range(8))), None)
    if protocol_dir is None:
        raise FileNotFoundError('未找到八份旧固定参考协议；不会重新生成参考名单。')
    protocols, protocol_hashes, protocol_paths = [], [], []
    for seed in range(8):
        path = protocol_dir / f'seed_{seed}.pt'
        protocols.append(validate_protocol(load(path), seed))
        protocol_hashes.append(digest(path))
        protocol_paths.append(str(path.resolve()))
    reference_data, identity = None, None
    rows, files = [], {}
    # Both historical baselines must have the same full features, labels,
    # battery order, transformation and actual 32-reference arrays.
    for model, folder in (('latent_cross_attention', latent_dir), ('batlinet', batlinet_dir)):
        for seed in range(8):
            path = one_prediction(folder, seed)
            result = read_pickle(path)
            current = dataset_identity(result['data'])
            if identity is None:
                identity, reference_data = current, result['data']
            if current != identity or result['seed'] != seed:
                raise ValueError(f'历史数据顺序、特征或标签变换不一致：{path}')
            indices = result.get('diagnostics', {}).get('support_index')
            if indices is None or not torch.equal(indices.cpu(), protocols[seed]):
                raise ValueError(f'历史预测的实际参考名单与旧协议不同：{path}')
            measured = scores(result['data'], result['prediction'])
            for key in ('RMSE', 'MAE', 'MAPE'):
                if not math.isclose(measured[key], result['scores'][key], rel_tol=1e-5, abs_tol=1e-5):
                    raise ValueError(f'历史预测指标不能重算：{path}')
            rows.append(dict(model=model, seed=seed, **measured))
            files[f'{model}_seed{seed}'] = dict(path=str(path.resolve()), sha256=digest(path))
            print(f'已核对历史结果：{model}，种子 {seed}。', flush=True)
    provenance = dict(dataset=identity, protocols_sha256=protocol_hashes,
                      protocols_paths=protocol_paths,
                      historical_predictions=files, policy=POLICY)
    return reference_data, protocols, rows, provenance


def model_config():
    config = import_config(ROOT / 'configs/ablation/diff_branch/batlinet_latent_cross_attention/mix_20.yaml', ['model'])['model']
    config.update(name='CycleMixerLatentCrossAttentionBatLiNetRULPredictor',
                  cycle_mixer_hidden=16, cycle_mixer_type='mlp', cycle_conv_kernel=3)
    return config


def advance_old_monitor_rng(num_train, num_test, support_size, batch_size, device):
    """Reproduce RNG draws from the old periodic test without a model/query."""
    # A DataLoader iterator also draws one CPU base seed, even with no workers.
    for batch in DataLoader(torch.arange(num_test), batch_size=batch_size, shuffle=False):
        torch.randint(num_train, (len(batch) * support_size,), device=device)


def legacy_train(model, training, epochs, evaluate_every, num_test, progress=None):
    """Match the old fit loop; no test features or test labels are accepted."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr)
    prepared = model.build_cell_dataset(training)
    loader = DataLoader(prepared, model.train_batch_size, shuffle=False)
    start = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.
        for number, batch in enumerate(loader):
            x, y = batch.values()
            sup_x, sup_y = model.get_support_set(
                x, prepared.feature, training.label, support_is_prepared=True)
            loss = model.forward(x, y, sup_x, sup_y, return_loss=True)
            if not torch.isfinite(loss):
                raise ValueError(f'第 {epoch} 轮训练损失非有限。')
            (loss / model.grad_accum_steps).backward()
            total += float(loss.detach()) * len(x)
            if number == len(loader) - 1 or (number + 1) % model.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        if epoch % evaluate_every == 0:
            model.eval()
            advance_old_monitor_rng(len(training), num_test, model.test_support_size,
                                    model.test_batch_size, training.device)
        if progress is not None:
            progress(dict(epoch=epoch, train_loss=total / len(training),
                          elapsed_seconds=time.monotonic() - start))


def validate_complete(folder, provenance, config, seed):
    if any(not (folder / name).is_file() for name in ('run.json', 'train.jsonl', 'epoch1000.pt')):
        raise ValueError(f'训练目录不完整：{folder}；不支持从中途续训，也不会覆盖。')
    info = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
    records = [json.loads(line) for line in (folder / 'train.jsonl').read_text().splitlines()]
    checkpoint = load(folder / 'epoch1000.pt')
    expected = dict(provenance=provenance, config=config, seed=seed, model=MODEL)
    for source in (info, checkpoint):
        if any(source.get(key) != value for key, value in expected.items()):
            raise ValueError(f'训练目录与本次历史协议不一致：{folder}')
    if (info.get('status') != 'complete' or checkpoint.get('epoch') != POLICY['epochs']
            or [r['epoch'] for r in records] != list(range(1, POLICY['epochs'] + 1))
            or any(not math.isfinite(r['train_loss']) for r in records)):
        raise ValueError(f'训练尚未完整完成：{folder}；不支持从中途续训。')
    return checkpoint


def print_summary(rows):
    print('测试集八种子均值 ± 样本标准差；MAPE、ACC15 均为小数比例。', flush=True)
    for model in ('batlinet', 'latent_cross_attention', MODEL):
        selected = [row for row in rows if row['model'] == model]
        if not selected:
            continue
        print(model, flush=True)
        for key in ('RMSE', 'MAE', 'MAPE', 'ACC15'):
            values = [row[key] for row in selected]
            print(f'  {key}: {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    parser.add_argument('--original-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main2'))
    parser.add_argument('--protocol-dir', type=Path)
    parser.add_argument('--workspace', type=Path, default=Path('workspaces/mix20_cycle_mixer_207_v1'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(8)
    data, protocols, historical, provenance = audit_history(
        args.history_root, args.original_root, args.protocol_dir)
    if args.audit_only:
        print_summary(historical)
        return
    config = model_config()
    provenance['source_sha256'] = {name: digest(ROOT / name) for name in (
        'scripts/run_mix20_cycle_mixer_207.py',
        'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py',
        'src/models/rul_predictors/latent_cross_attention_batlinet.py')}
    # Check existing directories before beginning another costly training job.
    for seed in range(8):
        folder = args.workspace / f'{MODEL}_seed{seed}'
        if folder.exists():
            validate_complete(folder, provenance, config, seed)
    args.workspace.mkdir(parents=True, exist_ok=True)
    for seed in range(8):
        folder = args.workspace / f'{MODEL}_seed{seed}'
        if folder.exists():
            print(f'种子 {seed} 已完成并通过核对，跳过训练。', flush=True)
            continue
        set_seed(seed)
        model = MODELS.build(config, seed=seed).to(args.device)
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
                    print(f"种子 {seed} [{record['epoch']}/1000] 损失 {record['train_loss']:.5f}；"
                          f'本种子预计剩余 {remaining / 60:.1f} 分钟；后续尚有 {7 - seed} 个种子。', flush=True)
            legacy_train(model, training, 1000, 100, 147, progress)
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        save_new(dict(state=state, epoch=1000, **{k: v for k, v in info.items() if k != 'status'}),
                 folder / 'epoch1000.pt')
        info['status'] = 'complete'
        (folder / 'run.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
        training.to('cpu')
        del model, state
        if str(args.device).startswith('cuda'):
            torch.cuda.empty_cache()
    # Every seed must finish and pass the audit before any new test prediction.
    checkpoints = [validate_complete(args.workspace / f'{MODEL}_seed{s}', provenance, config, s)
                   for s in range(8)]
    rows = list(historical)
    for seed, checkpoint in enumerate(checkpoints):
        folder = args.workspace / f'{MODEL}_seed{seed}'
        destination = folder / 'test.pt'
        checkpoint_sha = digest(folder / 'epoch1000.pt')
        if destination.exists():
            result = load(destination)
            if (result.get('checkpoint_sha256') != checkpoint_sha
                    or result.get('provenance') != provenance or result.get('seed') != seed
                    or not torch.equal(result['diagnostics']['support_index'], protocols[seed])):
                raise ValueError(f'已有测试结果与当前训练不一致：{destination}')
        else:
            set_seed(seed)
            model = MODELS.build(config, seed=seed).to(args.device)
            model.load_state_dict(checkpoint['state'], strict=True)
            model._fixed_test_support_index = protocols[seed]
            # The original predictor consults this field before its cached array.
            model.fixed_test_support_index_path = provenance['protocols_paths'][seed]
            data.to(args.device)
            prediction, diagnostics = model.predict(data, return_diagnostics=True)
            measured = scores(data, prediction)
            result = dict(model=MODEL, seed=seed, epoch=1000, scores=measured,
                          prediction=prediction.detach().cpu(),
                          prediction_cycles=data.label_transformation.inverse_transform(prediction).detach().cpu(),
                          truth_cycles=data.label_transformation.inverse_transform(data.test_data.label).detach().cpu(),
                          label_mean=data.label_transformation.transformations[1]._mean.detach().cpu(),
                          label_scale=data.label_transformation.transformations[1]._std.detach().cpu(),
                          diagnostics={key: value.detach().cpu() if value is not None else None
                                       for key, value in diagnostics.items()},
                          provenance=provenance, checkpoint_sha256=checkpoint_sha)
            save_new(result, destination)
            data.to('cpu')
            del model
        rows.append(dict(model=MODEL, seed=seed, **result['scores']))
        print(json.dumps(rows[-1], ensure_ascii=False), flush=True)
    (args.workspace / 'summary.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
    print_summary(rows)


if __name__ == '__main__':
    main()
