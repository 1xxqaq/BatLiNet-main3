"""Read-only import of actual v2 cells and splits for the matched v3 queue.

This module never calls the old preparation or split-generation functions.  The
only writes are independent JSON receipts in the new workspace.  Legacy pair
indices are audited, then deliberately omitted from the imported protocols:
the new runner must train on every selected target/source pair.
"""
import json
import math
from collections import Counter
from pathlib import Path

import torch

from scripts import run_three_tasks as common
from scripts.run_cross_chem_two_models import MODELS as EXISTING_MODELS
from scripts.run_cross_chem_two_models import SHARED_POLICY

EXPECTED = {'LFP': 275, 'LCO': 37, 'NCA': 22, 'NMC': 69}
TEST_COUNTS = {'LCO': 21, 'NCA': 14, 'NMC': 53}
TRAIN_COUNTS = {'LCO': (1, 2, 4, 8, 16), 'NCA': (1, 2, 4, 8),
                'NMC': (1, 2, 4, 8, 16)}
SEEDS = tuple(range(8))
FEATURE_SHAPE = (6, 100, 1000)
EXPECTED_PROTOCOL_COUNT = 112
_ID_FIELDS = ('source_ids', 'target_train_ids', 'target_test_ids',
              'unused_target_ids')


def _read_json(path):
    with Path(path).open('r', encoding='utf-8') as handle:
        return json.load(handle)


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'旧工作区文件路径越界：{path}')
    return path


def _manifest_path(existing):
    return _inside(Path(existing) / 'dataset/manifest.json', existing)


def _cell_path(record, existing):
    folder = Path(existing) / 'dataset'
    return _inside(folder / record['feature_file'], folder)


def _check_cell(record, existing):
    path = _cell_path(record, existing)
    if not path.is_file() or common.digest(path) != record['feature_sha256']:
        raise ValueError(f'旧电池缓存缺失或SHA256变化：{path}')
    value = common.load(path)
    feature = value.get('feature')
    if (value.get('cell_id') != record['cell_id']
            or value.get('chemistry') != record['chemistry']
            or value.get('label_cycles') != record['label_cycles']
            or not torch.is_tensor(feature)
            or tuple(feature.shape) != FEATURE_SHAPE
            or feature.dtype != torch.float32
            or not bool(torch.isfinite(feature).all())
            or not math.isfinite(float(record['label_cycles']))
            or float(record['label_cycles']) <= 100):
        raise ValueError(f'旧电池缓存内容、形状或寿命与收据不符：{path}')
    capacity = record.get('nominal_capacity_in_Ah')
    if capacity is not None and (not math.isfinite(float(capacity))
            or float(capacity) <= 0
            or value.get('nominal_capacity_in_Ah') != capacity):
        raise ValueError(f'旧电池额定容量与收据不符：{path}')
    return value


def _ids(value, name, path):
    ids = value.get(name)
    if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
            or len(ids) != len(set(ids))):
        raise ValueError(f'旧协议名单类型错误或重复：{path}：{name}')
    return ids


def _check_pairs(value, count, path):
    parts = []
    for key in ('train_pairs', 'val_pairs'):
        part = value.get(key)
        if (not torch.is_tensor(part) or part.dtype != torch.int64
                or part.ndim != 1 or part.numel() == 0):
            raise ValueError(f'旧协议电池对索引缺失或无效：{path}：{key}')
        parts.append(part.cpu())
    merged = torch.cat(parts)
    if merged.numel() != count or not torch.equal(
            merged.sort().values, torch.arange(count, dtype=torch.int64)):
        raise ValueError(f'旧协议电池对有遗漏、越界或重复：{path}')
    return {'training_pairs': parts[0].numel(),
            'validation_pairs': parts[1].numel(),
            'new_training_pairs': count,
            'legacy_pair_split_reused_for_training': False}


def _check_test(path, p, manifest_sha, protocol_sha, queue, lookup):
    """Bind saved predictions to the actual cached IDs and raw lifetimes."""
    output = common.load(path)
    binding = output.get('binding', {})
    if (binding.get('queue') != queue
            or binding.get('manifest_sha256') != manifest_sha
            or binding.get('protocol_sha256') != protocol_sha
            or binding.get('target') != p['target_chemistry']
            or binding.get('train_count') != p['target_train_count']
            or binding.get('seed') != p['seed']
            or binding.get('model') not in EXISTING_MODELS
            or path.parent.name != f"{binding.get('model')}_seed{p['seed']}"
            or output.get('source_ids') != p['source_ids']
            or output.get('target_test_ids') != p['target_test_ids']):
        raise ValueError(f'旧测试输出与实际协议或队列绑定不同：{path}')
    truth = output.get('truth_cycles')
    expected = torch.tensor([lookup[i]['label_cycles']
                             for i in p['target_test_ids']], dtype=torch.float32)
    if (not torch.is_tensor(truth) or tuple(truth.shape) != tuple(expected.shape)
            or not torch.equal(truth.cpu(), expected)):
        raise ValueError(f'旧测试输出真值与缓存寿命不同：{path}')
    prediction = output.get('prediction_cycles')
    if (not torch.is_tensor(prediction)
            or tuple(prediction.shape) != tuple(expected.shape)
            or not bool(torch.isfinite(prediction).all())):
        raise ValueError(f'旧测试预测形状或数值无效：{path}')
    diagnostics = output.get('diagnostics', {})
    if 'support_index' in diagnostics:
        indices = torch.arange(len(p['source_ids'])).expand(len(expected), -1)
        if not torch.equal(diagnostics['support_index'].cpu(), indices):
            raise ValueError(f'旧测试实际参考顺序与保存名单不同：{path}')
    digest = common.digest(path)
    receipt_path = path.parent / 'run.json'
    if receipt_path.is_file():
        receipt = _read_json(receipt_path)
        if (receipt.get('binding') != binding
                or (receipt.get('test_sha256') is not None
                    and receipt['test_sha256'] != digest)):
            raise ValueError(f'旧测试文件与运行收据不符：{path}')
        if receipt.get('status') == 'complete':
            final = path.parent / 'final.pt'
            if (not final.is_file()
                    or common.digest(final) != receipt.get('final_sha256')
                    or output.get('checkpoint_sha256') != receipt['final_sha256']):
                raise ValueError(f'旧测试文件与完成权重指纹不符：{path}')
    return {'path': path.as_posix(), 'sha256': digest,
            'model': binding['model'], 'test_count': len(expected)}


def audit_existing(existing):
    """Return ``(manifest, actual_protocols, audit)`` without writing anything.

    The protocol mapping is keyed by ``(chemistry, train_count, seed)``.  Saved
    ID order is preserved.  Missing/changed evidence raises an error; there is
    no regeneration or fallback to the old data preparation code.
    """
    existing = Path(existing).resolve()
    manifest_path = _manifest_path(existing)
    queue_path = _inside(existing / 'queue.json', existing)
    if not manifest_path.is_file() or not queue_path.is_file():
        raise FileNotFoundError(f'需要已有v2队列及实际数据收据：{existing}')
    manifest_sha, queue_sha = common.digest(manifest_path), common.digest(queue_path)
    manifest, queue = _read_json(manifest_path), _read_json(queue_path)
    identity = manifest.get('identity', {})
    if (queue.get('schema') != 2 or queue.get('data_policy') != SHARED_POLICY
            or queue.get('models') != list(EXISTING_MODELS)
            or identity.get('policy') != SHARED_POLICY
            or not isinstance(identity.get('code_hashes'), dict)
            or identity['code_hashes'] != queue.get('code_hashes')):
        raise ValueError('旧队列必须是schema=2的SHARED_POLICY，且数据代码指纹绑定一致。')
    records = manifest.get('records', [])
    lookup = {r['cell_id']: r for r in records}
    counts = dict(Counter(r['chemistry'] for r in records))
    if (counts != EXPECTED or len(lookup) != len(records)
            or manifest.get('observed_counts') != EXPECTED
            or len({r['feature_file'] for r in records}) != len(records)):
        raise ValueError(f'旧数据收据数量、ID或缓存路径不符：实际{counts}，要求{EXPECTED}')
    for record in records:
        _check_cell(record, existing)
    keys = [(target, count, seed) for target, numbers in TRAIN_COUNTS.items()
            for count in numbers for seed in SEEDS]
    if len(keys) != EXPECTED_PROTOCOL_COUNT:
        raise ValueError('正式迁移协议必须含完整112个目标档位与种子组合。')
    expected_runs = {(*key, model) for key in keys for model in EXISTING_MODELS}
    actual_runs = [tuple(run) for run in queue.get('runs', [])]
    if len(actual_runs) != len(expected_runs) or set(actual_runs) != expected_runs:
        raise ValueError('旧v2队列不是完整两模型224组，不能静默补齐名单。')
    expected_files = {f'{target}/train_{count}/seed_{seed}.pt'
                      for target, count, seed in keys}
    actual_files = {path.relative_to(existing / 'protocols').as_posix()
                    for path in (existing / 'protocols').rglob('*.pt')}
    if actual_files != expected_files:
        raise ValueError('旧协议目录必须恰好含112份预定协议；有缺失或额外协议。')
    protocols, fingerprints, checked_tests = {}, {}, []
    source_set = {r['cell_id'] for r in records if r['chemistry'] == 'LFP'}
    source_order = None
    for key in keys:
        target, count, seed = key
        relative = f'{target}/train_{count}/seed_{seed}.pt'
        path = _inside(existing / 'protocols' / relative, existing)
        fingerprint = common.digest(path)
        value = common.load(path)
        if (value.get('policy') != SHARED_POLICY
                or value.get('target_chemistry') != target
                or value.get('target_train_count') != count
                or value.get('seed') != seed):
            raise ValueError(f'旧协议身份或策略不符：{path}')
        ids = {name: _ids(value, name, path) for name in _ID_FIELDS}
        source, training, testing, unused = (set(ids[name]) for name in _ID_FIELDS)
        target_set = {r['cell_id'] for r in records if r['chemistry'] == target}
        if (source != source_set or len(source) != EXPECTED['LFP']
                or len(training) != count or len(testing) != TEST_COUNTS[target]
                or training & testing or training & unused or testing & unused
                or source & (training | testing | unused)
                or training | testing | unused != target_set):
            raise ValueError(f'旧协议来源/训练/测试/未用名单存在遗漏、重叠或错误体系：{path}')
        if source_order is None:
            source_order = ids['source_ids']
        elif ids['source_ids'] != source_order:
            raise ValueError(f'旧协议之间的来源参考顺序不同：{path}；不能自动重排。')
        pairs = _check_pairs(value, len(source) * count, path)
        p = {name: value[name] for name in
             ('policy', 'target_chemistry', 'target_train_count', 'seed', *_ID_FIELDS)}
        p['existing_protocol_sha256'] = fingerprint
        protocols[key] = p
        fingerprints[relative] = {'sha256': fingerprint, **pairs}
        case = existing / target / f'train_{count}'
        for output in sorted(case.glob(f'*_seed{seed}/test.pt')):
            output = _inside(output, existing)
            checked_tests.append(_check_test(output, p, manifest_sha,
                                            fingerprint, queue, lookup))
    for target, numbers in TRAIN_COUNTS.items():
        for seed in SEEDS:
            biggest = protocols[(target, max(numbers), seed)]
            ordered_pool = biggest['target_train_ids'] + biggest['unused_target_ids']
            for count in numbers:
                p = protocols[(target, count, seed)]
                if (p['target_test_ids'] != biggest['target_test_ids']
                        or p['target_train_ids'] != ordered_pool[:count]
                        or p['unused_target_ids'] != ordered_pool[count:]):
                    raise ValueError(f'旧协议未满足同种子测试固定、训练名单嵌套：{target}/{count}/{seed}')
    # Detect concurrent replacement of evidence while the audit was in progress.
    if common.digest(manifest_path) != manifest_sha or common.digest(queue_path) != queue_sha:
        raise ValueError('审计过程中旧数据收据或队列发生变化，请在旧队列停止后重试。')
    for relative, metadata in fingerprints.items():
        if common.digest(existing / 'protocols' / relative) != metadata['sha256']:
            raise ValueError(f'审计过程中旧协议发生变化：{relative}')
    audit = {'schema': 1, 'existing_workspace': existing.as_posix(),
             'existing_queue_sha256': queue_sha,
             'existing_manifest_sha256': manifest_sha,
             'observed_counts': counts, 'feature_shape': list(FEATURE_SHAPE),
             'cells_verified': len(records), 'protocols_verified': len(protocols),
             'protocols': fingerprints, 'source_ids': source_order,
             'verified_existing_tests': checked_tests,
             'existing_workspace_read_only': True,
             'split_generation_called': False,
             'legacy_pair_split_reused_for_training': False,
             'new_relation_training': 'all_selected_target_times_all_saved_LFP_source',
             'target_test_fixed_within_seed': True,
             'target_training_nested_within_seed': True}
    return manifest, protocols, audit


def import_protocols(existing, new_workspace):
    """Audit the old queue, then write only lists and provenance in the new one."""
    existing, new_workspace = Path(existing).resolve(), Path(new_workspace).resolve()
    if (new_workspace == existing or new_workspace.is_relative_to(existing)
            or existing.is_relative_to(new_workspace)):
        raise ValueError('新旧工作区必须互相独立，禁止在旧目录内导入或覆盖旧目录。')
    manifest, protocols, audit = audit_existing(existing)
    receipt = _inside(new_workspace / 'protocol_import_receipt.json', new_workspace)
    if receipt.exists() and _read_json(receipt) != audit:
        raise ValueError('新工作区已有导入收据与当前旧文件指纹不同；禁止混合恢复。')
    for (target, count, seed), p in protocols.items():
        path = _inside(new_workspace / 'protocols' / target / f'train_{count}'
                       / f'seed_{seed}.json', new_workspace)
        value = {'schema': 1, 'existing_workspace': existing.as_posix(),
                 'existing_manifest_sha256': audit['existing_manifest_sha256'],
                 'legacy_pair_split_reused_for_training': False,
                 'new_relation_training': audit['new_relation_training'], **p}
        if path.exists():
            if _read_json(path) != value:
                raise ValueError(f'新工作区已有协议名单或指纹不同：{path}')
        else:
            common.atomic_json(value, path)
    if not receipt.exists():
        common.atomic_json(audit, receipt)
    return manifest, protocols, audit


def load_part(ids, manifest, existing, clip=True):
    """Load cells directly from the audited old cache, preserving requested order."""
    ids = list(ids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('载入名单必须非空且不重复。')
    lookup = {record['cell_id']: record for record in manifest['records']}
    features, labels = [], []
    for cell_id in ids:
        if cell_id not in lookup:
            raise ValueError(f'载入名单ID不在已审计数据收据中：{cell_id}')
        value = _check_cell(lookup[cell_id], existing)
        feature = value['feature'].clone()
        if clip:
            feature[feature.abs() > 10] = 0
        features.append(feature)
        labels.append(float(value['label_cycles']))
    return {'feature': torch.stack(features),
            'label': torch.tensor(labels, dtype=torch.float32), 'ids': ids}
