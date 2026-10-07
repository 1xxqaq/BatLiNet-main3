"""Three official tasks, three existing models; fixed final epoch, resumable queue."""
import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src
from src.builders import MODELS, TRAIN_TEST_SPLITTERS, FEATURE_EXTRACTORS, LABEL_ANNOTATORS
from src.data.battery_data import BatteryData
from src.data.databundle import Dataset

TASKS = ('matr_1', 'hust', 'matr_2')
MODELS_NAMES = ('batlinet', 'latent_cross_attention', 'latent_cycle_mixer')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def cpu(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(cpu(v) for v in value)
    return value


def save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        torch.save(cpu(value), f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False


def runtime(device):
    packages = {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'scipy', 'pandas', 'PyYAML', 'scikit-learn', 'tqdm', 'addict')}
    return dict(python=sys.version, packages=packages, torch=str(torch.__version__), numpy=str(np.__version__),
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(), device=device,
                gpu=torch.cuda.get_device_name(device) if device.startswith('cuda') else None,
                gpu_total_GiB=torch.cuda.get_device_properties(device).total_memory/2**30 if device.startswith('cuda') else None,
                gpu_capability=list(torch.cuda.get_device_capability(device)) if device.startswith('cuda') else None,
                precision='float32', tf32=False, cudnn_benchmark=False, cudnn_deterministic=True)


def source_hashes():
    paths = list((ROOT / 'src').rglob('*.py')) + [Path(__file__)]
    paths += list((ROOT / 'configs/three_tasks_v1').rglob('*.yaml'))
    return {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(paths)}


def configs(task):
    result = {}
    for name in MODELS_NAMES:
        p = ROOT / f'configs/three_tasks_v1/{task}/{name}.yaml'
        result[name] = yaml.safe_load(p.read_text(encoding='utf-8'))
    common = result['batlinet']
    policy = dict(schema=1, seeds=list(range(8)), micro_batch_size=16, pair_chunk=8, precision='float32', shuffle=False, weight_decay=.01, checkpoint_every=25, selection='fixed_epoch_1000', label_std='sample', train_reference_rng='independent_cpu_seed', test_reference_rng='independent_cpu_seed_plus_200000', test_during_training=False, compile=False)
    if common['runner'] != policy:
        raise ValueError('配置偏离本入口固定策略，需建立新的协议入口，不能只改未被执行的配置字段。')
    for cfg in result.values():
        m = cfg['model']
        if (m['epochs'] != 1000 or m['train_batch_size'] != 128 or m['test_support_size'] != 32
                or m['train_support_size'] != (1 if task=='matr_2' else 2)
                or m['alpha'] != (.2 if task=='matr_2' else .5) or m['filter_cycles'] is not False
                or m['gradient_accumulation_steps'] != 1):
            raise ValueError('正式配置偏离三任务预定训练设置。')
        for key in ('feature', 'label', 'train_test_split', 'label_transformation', 'runner'):
            if cfg[key] != common[key]:
                raise ValueError(f'任务 {task} 的三模型数据或队列策略不同：{key}')
    return result


def validate_data(data, shape=(6, 100, 1000)):
    ids = []
    for part in ('train', 'test'):
        d = data[part]
        if (not len(d['ids']) or len(d['ids']) != len(d['feature'])
                or len(d['label']) != len(d['ids']) or tuple(d['feature'].shape[1:]) != shape
                or not torch.isfinite(d['feature']).all() or not torch.isfinite(d['label']).all()
                or (d['label'] <= 100).any()):
            raise ValueError(f'数据形状、有效寿命或有限值检查失败：{part}')
        ids += d['ids']
    if len(ids) != len(set(ids)):
        raise ValueError('电池编号重复，或训练与测试集重叠。')
    logs = data['train']['label'].log()
    if len(logs) < 2 or not torch.isfinite(logs.std()) or logs.std() <= 0:
        raise ValueError('训练标签不能进行样本标准化。')
    torch.testing.assert_close(data['mean'], logs.mean())
    torch.testing.assert_close(data['scale'], logs.std(unbiased=True).clamp_min(1e-8))


def prepare(task, cfg, source_root, workspace):
    folder = workspace / 'datasets' / task
    path = folder / 'data.pt'
    split_cfg = dict(cfg['train_test_split'])
    split_cfg['cell_data_path'] = str(source_root / split_cfg['cell_data_path'])
    directory = Path(split_cfg['cell_data_path'])
    if not directory.is_dir() or not list(directory.glob('*.pkl')):
        raise FileNotFoundError(f'未找到预处理电池文件：{directory}')
    if task.startswith('matr'):
        keys = [p.stem.split('_')[1] for p in directory.glob('*.pkl')]
        if len(keys) != len(set(keys)):
            raise ValueError('MATR 文件名重复映射到同一电池，不能静默覆盖。')
    partitions = TRAIN_TEST_SPLITTERS.build(split_cfg).split()
    expected_counts = {'matr_1': (41, 42), 'matr_2': (41, 40), 'hust': (55, 22)}[task]
    if tuple(map(len, partitions)) != expected_counts:
        raise ValueError(f'{task} 原始划分名单不完整或含额外电池：{tuple(map(len, partitions))}，官方过滤前预期 {expected_counts}。')
    if task == 'hust':
        expected_test = set('1-1 1-2 2-5 3-1 4-5 5-3 6-1 6-2 6-6 6-8 7-5 7-6 8-1 8-5 8-6 8-8 9-4 9-6 10-1 10-4 10-6 10-7'.split())
        if {Path(p).stem.split('_')[1] for p in partitions[1]} != expected_test:
            raise ValueError('HUST 测试文件名单与标准77电池划分不一致。')
        keys = [Path(p).stem.split('_')[1] for paths in partitions for p in paths]
        if len(keys) != len(set(keys)):
            raise ValueError('HUST 文件名重复映射到同一电池。')
    if task == 'hust':
        partitions = [sorted(paths, key=lambda p: Path(p).name) for paths in partitions]
    records = [dict(split=part, path=str(Path(p).resolve()), sha256=digest(p))
               for part, paths in zip(('train', 'test'), partitions) for p in paths]
    preparation = {key: cfg[key] for key in ('train_test_split', 'feature', 'label', 'label_transformation')}
    binding = dict(task=task, config=preparation, source_files=records,
                   source_code=source_hashes(), order='filename' if task == 'hust' else 'official_list')
    if path.exists():
        data = load(path)
        if data['binding'] != binding:
            raise ValueError(f'已有数据缓存与源文件或代码不一致：{path}')
        validate_data(data)
        receipt_path = folder / 'receipt.json'
        sha = digest(path)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
            if receipt['binding'] != binding or receipt['sha256'] != sha:
                raise ValueError(f'缓存回执不一致：{path}')
        else:
            # Recover a crash between atomic data save and receipt save.
            dataset_receipt(data, sha, receipt_path)
        return data, sha
    feature = FEATURE_EXTRACTORS.build(cfg['feature'])
    labeler = LABEL_ANNOTATORS.build(cfg['label'])
    data = dict(binding=binding, excluded=[])
    for part, paths in zip(('train', 'test'), partitions):
        xs, ys, ids = [], [], []
        for number, p in enumerate(paths):
            cell = BatteryData.load(p)
            y = labeler.process_cell(cell).float()
            if torch.isnan(y):
                data['excluded'].append(dict(split=part, path=str(p), cell_id=cell.cell_id,
                                             reason='official_RULLabelAnnotator_nan'))
                continue
            x = feature.process_cell(cell).float().contiguous()
            xs.append(x)
            ys.append(y)
            ids.append(cell.cell_id)
            del cell
            print(f'{task} {part} 提取 {number+1}/{len(paths)}：{ids[-1]}', flush=True)
        if not xs:
            raise ValueError(f'{task} {part} 没有有效电池。')
        data[part] = dict(feature=torch.stack(xs), label=torch.stack(ys).reshape(-1), ids=ids)
    logs = data['train']['label'].log()
    data.update(mean=logs.mean(), scale=logs.std(unbiased=True).clamp_min(1e-8))
    validate_data(data)
    save(data, path)
    sha = digest(path)
    dataset_receipt(data, sha, folder / 'receipt.json')
    return data, sha


def dataset_receipt(data, sha, path):
    atomic_json(dict(binding=data['binding'], sha256=sha, excluded=data['excluded'],
                     train_ids=data['train']['ids'], test_ids=data['test']['ids'],
                     train_count=len(data['train']['ids']), test_count=len(data['test']['ids']),
                     label_mean=float(data['mean']), label_sample_std=float(data['scale'])), path)


def protocol(data, seed, path):
    expected = dict(seed=seed, train_ids=data['train']['ids'], test_ids=data['test']['ids'],
                    indices=torch.randint(len(data['train']['ids']), (len(data['test']['ids']), 32),
                                          generator=torch.Generator().manual_seed(seed + 200000)))
    if path.exists():
        actual = load(path)
        if any(actual.get(k) != expected[k] for k in ('seed', 'train_ids', 'test_ids')):
            raise ValueError(f'固定参考的电池或种子不一致：{path}')
        if not torch.equal(actual['indices'], expected['indices']):
            raise ValueError(f'固定参考索引被修改：{path}')
    else:
        save(expected, path)
    return expected['indices']


def prepared(model, part, mean, scale, device):
    ds = Dataset(part['feature'].to(device), ((part['label'].log() - mean) / scale).to(device))
    if hasattr(model, 'build_cycle_diff_dataset'):
        own = model.build_cycle_diff_dataset(ds)
        return own.feature, own.raw_feature, ds.feature, ds.label
    own = model.build_cell_dataset(ds)
    return own.feature, own.feature, own.feature, ds.label


def supports(model, query, pool, labels, indices):
    kwargs = {} if hasattr(model, 'build_cycle_diff_dataset') else dict(support_is_prepared=True)
    return model.get_support_set(query, pool, labels, fixed_indices=indices, **kwargs)


def train_epoch(model, optimizer, tensors, generator, logical, micro):
    own, query, pool, labels = tensors
    model.train()
    loss_sum, steps = 0., 0
    for start in range(0, len(labels), logical):
        end = min(start + logical, len(labels))
        indices = torch.randint(len(labels), (end-start, model.train_support_size), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        for offset in range(start, end, micro):
            stop = min(offset + micro, end)
            sx, sy = supports(model, query[offset:stop], pool, labels, indices[offset-start:stop-start])
            loss = model(own[offset:stop], labels[offset:stop], sx, sy, return_loss=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('训练损失非有限。')
            (loss * (stop-offset) / (end-start)).backward()
            loss_sum += loss.detach().item() * (stop-offset)
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError('训练梯度非有限。')
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('优化器更新后参数非有限。')
        steps += 1
    return dict(train_loss=loss_sum / len(labels), optimizer_steps=steps)


def rng_state(generator, device):
    ns = np.random.get_state()
    return dict(python=random.getstate(), numpy=[ns[0], torch.tensor(ns[1].astype(np.int64)),
                int(ns[2]), int(ns[3]), float(ns[4])], torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device).cpu() if device.startswith('cuda') else None,
                references=generator.get_state())


def restore_rng(state, generator, device):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], ns[1].numpy().astype(np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch'])
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'], device)
    generator.set_state(state['references'])


def checked_checkpoint(path, binding):
    value = load(path)
    if value['binding'] != binding or not 0 <= value['epoch'] <= binding['config']['model']['epochs']:
        raise ValueError(f'检查点与本次配置、数据、代码或环境不一致：{path}')
    return value


def align_log(folder, epoch):
    path = folder / 'train.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines() if path.exists() else []
    keep = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row.get('epoch') > epoch:
            break
        if row.get('epoch') != len(keep)+1 or not np.isfinite(row['train_loss']):
            raise ValueError(f'日志轮次或损失异常：{folder}')
        keep.append(line)
    if len(keep) != epoch:
        raise ValueError(f'日志未覆盖已保存的 {epoch} 轮：{folder}')
    if len(lines) != len(keep):
        archive = folder / f'interrupted_log_{time.time_ns()}.jsonl'
        archive.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    path.write_text(('\n'.join(keep) + '\n') if keep else '', encoding='utf-8')


def train_run(data, cfg, seed, folder, binding, device):
    folder.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    model = MODELS.build(cfg['model'], seed=seed).to(device)
    tensors = prepared(model, data['train'], data['mean'], data['scale'], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'], weight_decay=cfg['runner']['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    final, progress = folder / 'final.pt', folder / 'progress.pt'
    start = 0
    if final.exists():
        state = checked_checkpoint(final, binding)
        if state['epoch'] != cfg['model']['epochs']:
            raise ValueError('最终权重轮次不正确。')
        align_log(folder, state['epoch'])
        model.load_state_dict(state['state'], strict=True)
        if (folder / 'run.json').exists():
            receipt = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            if 'final_sha256' in receipt and receipt['final_sha256'] != digest(final):
                raise ValueError('已完成权重的文件指纹被修改。')
        print(f'跳过已核验训练：{folder.name}', flush=True)
        return
    if progress.exists():
        state = checked_checkpoint(progress, binding)
        start = state['epoch']
        model.load_state_dict(state['state'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        restore_rng(state['rng'], generator, device)
    else:
        if (folder / 'run.json').exists():
            existing = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
            if existing['binding'] != binding:
                raise ValueError('未完成目录属于不同的运行配置。')
        # An epoch-zero checkpoint makes even the first interval resumable.
        save(dict(epoch=0, binding=binding, state=model.state_dict(), optimizer=optimizer.state_dict(),
                  rng=rng_state(generator, device)), progress)
    align_log(folder, start)
    atomic_json(dict(status='training', binding=binding, parameters=sum(p.numel() for p in model.parameters())), folder / 'run.json')
    with (folder / 'train.jsonl').open('a', encoding='utf-8') as stream:
        for epoch in range(start+1, cfg['model']['epochs']+1):
            tick = time.monotonic()
            record = train_epoch(model, optimizer, tensors, generator, cfg['model']['train_batch_size'], cfg['runner']['micro_batch_size'])
            record.update(epoch=epoch, seconds=time.monotonic()-tick,
                          cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.)
            stream.write(json.dumps(record) + '\n')
            stream.flush()
            if epoch % cfg['runner']['checkpoint_every'] == 0 or epoch == cfg['model']['epochs']:
                os.fsync(stream.fileno())
                save(dict(epoch=epoch, binding=binding, state=model.state_dict(), optimizer=optimizer.state_dict(),
                          rng=rng_state(generator, device)), progress)
                atomic_json(dict(status='training', epoch=epoch, job=folder.name), folder.parent.parent / 'status.json')
                print(f'{folder.name} [{epoch}/{cfg["model"]["epochs"]}] 损失 {record["train_loss"]:.6f}，本轮 {record["seconds"]:.1f} 秒', flush=True)
    save(dict(epoch=cfg['model']['epochs'], binding=binding, state=model.state_dict()), final)
    atomic_json(dict(status='trained', binding=binding, final_sha256=digest(final)), folder / 'run.json')


def scores(prediction, truth):
    p, y = prediction.double(), truth.double()
    if p.shape != y.shape or not torch.isfinite(p).all() or (y <= 0).any():
        raise ValueError('测试预测形状或有限值异常。')
    error = p-y
    relative = error.abs()/y
    return dict(RMSE=error.square().mean().sqrt().item(), MAE=error.abs().mean().item(),
                MAPE=relative.mean().item(), ACC15=(relative <= .15).double().mean().item())


@torch.no_grad()
def predict(model, data, indices, device, chunk=8):
    model.eval()
    own, query, _, _ = prepared(model, data['test'], data['mean'], data['scale'], device)
    _, _, pool, labels = prepared(model, data['train'], data['mean'], data['scale'], device)
    origins, refs = [], []
    for i in range(len(own)):
        pieces = []
        for start in range(0, 32, chunk):
            sx, sy = supports(model, query[i:i+1], pool, labels, indices[i:i+1, start:start+chunk])
            ori, sup, *_ = model.compute_prediction_components(own[i:i+1], sx, sy)
            pieces.append(sup.cpu())
        origins.append(ori.cpu())
        refs.append(torch.cat(pieces, 1))
    ori, sup = torch.cat(origins), torch.cat(refs)
    agg = sup.median(1).values
    z = (1-model.alpha)*ori + model.alpha*agg
    prediction = (z*data['scale']+data['mean']).exp()
    return dict(prediction=prediction, truth=data['test']['label'], scores=scores(prediction, data['test']['label']),
                label_mean=data['mean'], label_sample_std=data['scale'], train_ids=data['train']['ids'], test_ids=data['test']['ids'],
                diagnostics=dict(y_ori=ori, y_sup=sup, y_sup_agg=agg, support_index=indices),
                diagnostic_units='standardized_natural_log', prediction_units='cycles')


def evaluate_run(data, cfg, seed, folder, binding, indices, device):
    final = folder / 'final.pt'
    check = checked_checkpoint(final, binding)
    if check['epoch'] != cfg['model']['epochs']:
        raise ValueError('测试必须使用固定最终轮权重。')
    path = folder / 'test.pt'
    sha = digest(final)
    if path.exists():
        result = load(path)
    else:
        seed_all(seed)
        model = MODELS.build(cfg['model'], seed=seed).to(device)
        model.load_state_dict(check['state'], strict=True)
        result = predict(model, data, indices, device, cfg['runner']['pair_chunk'])
        result.update(binding=binding, checkpoint_sha256=sha)
        save(result, path)
    if result['binding'] != binding or result['checkpoint_sha256'] != sha:
        raise ValueError(f'测试结果与权重绑定错误：{folder}')
    if result['train_ids'] != data['train']['ids'] or result['test_ids'] != data['test']['ids']:
        raise ValueError('预测电池名单不同。')
    if not torch.equal(result['truth'], data['test']['label']) or not torch.equal(result['diagnostics']['support_index'], indices):
        raise ValueError('预测真值或实际参考不一致。')
    z = (1-cfg['model']['alpha'])*result['diagnostics']['y_ori'] + cfg['model']['alpha']*result['diagnostics']['y_sup'].median(1).values
    torch.testing.assert_close(result['prediction'], (z*data['scale']+data['mean']).exp())
    torch.testing.assert_close(result['diagnostics']['y_sup_agg'], result['diagnostics']['y_sup'].median(1).values)
    measured = scores(result['prediction'], result['truth'])
    if measured != result['scores']:
        raise ValueError('保存的指标不能复算。')
    atomic_json(dict(status='complete', binding=binding, final_sha256=sha, test_sha256=digest(path), scores=measured), folder / 'run.json')
    return measured


def preflight(data, cfg, device):
    seed_all(0)
    model = MODELS.build(cfg['model'], seed=0).to(device)
    # Use the full physical batch and reference count, including target/reference self-pairs.
    tensors = prepared(model, data['train'], data['mean'], data['scale'], device)
    own, query, pool, labels = tensors
    n = min(cfg['runner']['micro_batch_size'], len(labels))
    idx = torch.randint(len(labels), (n, model.train_support_size), generator=torch.Generator().manual_seed(0))
    sx, sy = supports(model, query[:n], pool, labels, idx)
    model.train()
    loss = model(own[:n], labels[:n], sx, sy, return_loss=True)
    loss.backward()
    if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError('首跑前前向或梯度检查失败。')
    # Evaluation memory check uses TRAINING targets only; no test metric is read.
    toy = dict(train=data['train'], test={k: v[:1] for k, v in data['train'].items()}, mean=data['mean'], scale=data['scale'])
    test_indices = torch.randint(len(labels), (1, 32), generator=torch.Generator().manual_seed(200000))
    pred = predict(model, toy, test_indices, device, cfg['runner']['pair_chunk'])
    return dict(parameters=sum(p.numel() for p in model.parameters()), loss=float(loss.detach()),
                prediction_finite=bool(torch.isfinite(pred['prediction']).all()),
                cuda_peak_GiB=torch.cuda.max_memory_allocated(device)/2**30 if device.startswith('cuda') else 0.)


@contextlib.contextmanager
def queue_lock(workspace):
    if os.name != 'posix':
        raise RuntimeError('正式队列需在 Linux 服务器运行；本地只运行合成数据检查。')
    import fcntl
    workspace.mkdir(parents=True, exist_ok=True)
    with (workspace / '.queue.lock').open('a') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('同一输出目录已有队列运行，拒绝重复启动。') from None
        yield


def write_summary(rows, workspace):
    atomic_json(rows, workspace / 'per_seed.json')
    path = workspace / 'per_seed.csv'
    temp = path.with_suffix('.csv.tmp')
    with temp.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['task', 'model', 'seed', 'RMSE', 'MAE', 'MAPE', 'ACC15'])
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)
    summary = []
    for task in TASKS:
        for name in MODELS_NAMES:
            selected = [r for r in rows if r['task']==task and r['model']==name]
            if len(selected) == 8 and {r['seed'] for r in selected} == set(range(8)):
                summary.append(dict(task=task, model=name, seeds=8, metrics={k:dict(mean=statistics.mean(r[k] for r in selected), sample_std=statistics.stdev(r[k] for r in selected)) for k in ('RMSE','MAE','MAPE','ACC15')}))
    atomic_json(summary, workspace / 'summary.json')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', type=Path, default=Path('/root/autodl-tmp/BatLiNet-main3'))
    p.add_argument('--workspace', type=Path, default=ROOT / 'workspaces/three_tasks_three_models_v1')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(8)
    workspace = args.workspace.resolve()
    with queue_lock(workspace):
        try:
            seed_all(0)
            if not args.device.startswith('cuda') or not torch.cuda.is_available():
                raise RuntimeError('正式队列要求可用 CUDA 显卡，不自动改为 CPU 训练。')
            all_configs = {task: configs(task) for task in TASKS}
            revision = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
            provenance = dict(git_commit=revision, source_code=source_hashes(), runtime=runtime(args.device),
                              configs=all_configs, source_root=str(args.source_root.resolve()))
            queue_path = workspace / 'queue.json'
            if queue_path.exists() and json.loads(queue_path.read_text(encoding='utf-8')) != provenance:
                raise ValueError('队列目录属于不同代码、配置、环境或数据路径，请使用新的协议目录。')
            atomic_json(provenance, queue_path)
            atomic_json(dict(status='preparing', pid=os.getpid(), formal_training_started=False), workspace / 'status.json')
            gates, shas = {}, {}
            train_identity = None
            # All three data and all nine memory/gradient gates must pass before ANY formal training.
            for task in TASKS:
                data, shas[task] = prepare(task, all_configs[task]['batlinet'], args.source_root.resolve(), workspace)
                if task == 'matr_1':
                    train_identity = {k: cpu(v).clone() if torch.is_tensor(v) else list(v) for k,v in data['train'].items()}
                if task == 'matr_2':
                    if train_identity['ids'] != data['train']['ids'] or any(not torch.equal(train_identity[k], data['train'][k]) for k in ('feature', 'label')):
                        raise ValueError('MATR-1 与 MATR-2 的训练电池、输入或寿命不同。')
                    train_identity = None
                gates[task] = dict(train_count=len(data['train']['ids']), test_count=len(data['test']['ids']), models={})
                for seed in range(8):
                    protocol(data, seed, workspace / 'protocols' / task / f'seed_{seed}.pt')
                for name in MODELS_NAMES:
                    print(f'首跑前检查：{task}/{name}', flush=True)
                    torch.cuda.reset_peak_memory_stats(args.device)
                    gates[task]['models'][name] = preflight(data, all_configs[task][name], args.device)
                    torch.cuda.empty_cache()
                del data
            atomic_json(gates, workspace / 'preflight.json')
            if args.prepare_only:
                atomic_json(dict(status='prepared', formal_training_started=False), workspace / 'status.json')
                print('三任务数据、八种子参考和九配置显存/梯度检查通过，未启动正式训练。', flush=True)
                return
            rows = []
            print('首跑前检查通过：开始 MATR-1、HUST、MATR-2，三模型各八种子，共72次训练。', flush=True)
            for task in TASKS:
                data = load(workspace / 'datasets' / task / 'data.pt')
                bindings = {}
                for seed in range(8):
                    for name in MODELS_NAMES:
                        cfg = all_configs[task][name]
                        folder = workspace / task / f'{name}_seed{seed}'
                        binding = dict(task=task, model=name, seed=seed, config=cfg,
                                       data_sha256=shas[task], git_commit=provenance['git_commit'], code=provenance['source_code'], runtime=provenance['runtime'],
                                       protocol_sha256=digest(workspace / 'protocols' / task / f'seed_{seed}.pt'))
                        bindings[name, seed] = binding
                        torch.cuda.reset_peak_memory_stats(args.device)
                        train_run(data, cfg, seed, folder, binding, args.device)
                        torch.cuda.empty_cache()
                # Only after all 24 fixed-epoch runs in this task are complete.
                for seed in range(8):
                    indices = protocol(data, seed, workspace / 'protocols' / task / f'seed_{seed}.pt')
                    for name in MODELS_NAMES:
                        measured = evaluate_run(data, all_configs[task][name], seed, workspace / task / f'{name}_seed{seed}', bindings[name,seed], indices, args.device)
                        row = dict(task=task, model=name, seed=seed, **measured)
                        rows.append(row)
                        write_summary(rows, workspace)
                        print(json.dumps(row, ensure_ascii=False), flush=True)
                        torch.cuda.empty_cache()
                del data
            if len(rows) != 72:
                raise RuntimeError('队列结果数量不完整。')
            atomic_json(dict(status='complete', training_runs=72, predictions=72), workspace / 'status.json')
        except BaseException as error:
            atomic_json(dict(status='failed', error=repr(error), traceback=traceback.format_exc()), workspace / 'status.json')
            raise


if __name__ == '__main__':
    main()
