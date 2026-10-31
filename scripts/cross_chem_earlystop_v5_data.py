"""Read-only v3 IDs/rate and actual v2 pair partition for formal v5 training."""
import hashlib
import json
from pathlib import Path

import torch

from scripts import run_cross_chem_matched as previous
from scripts import cross_chem_matched_data as data
from scripts import run_three_tasks as common

ROOT = Path(__file__).resolve().parents[1]
FROZEN_MODEL_PATHS = (
    'scripts/cross_chem_matched_training.py', 'scripts/run_three_tasks.py',
    'src/models/rul_predictors/latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/cycle_mixer_latent_cross_attention_batlinet.py',
    'src/models/rul_predictors/soft_aligned_difference_batlinet.py',
    'src/models/rul_predictors/cycle_mixer_soft_aligned_difference_batlinet.py',
    'src/models/rul_predictors/batlinet.py', 'src/models/nn_model.py',
)
ID_FIELDS = ('source_ids', 'target_train_ids', 'target_test_ids', 'unused_target_ids')
PAIR_CONVENTION = 'source_major_target_interleaved'
LEGACY_CONVENTION = 'target_major_source_interleaved'
KNOWN_QUEUES = ('cross_chem_matched_v3', 'cross_chem_two_models_v2',
    'cross_chem_cycle_soft_difference_v1', 'mix_cycle_soft_difference_v1',
    'three_tasks_three_models_v1', 'mix20_matched_baselines_v1',
    'mix_branch_ablation_v1', 'mix_naive_ensemble_v1', 'mix_soft_difference_refinements_v1',
    'cross_chem_earlystop_v4', 'cross_chem_earlystop_v5')
KNOWN_WORKTREES = ('BatLiNet-cycle-mixer', 'BatLiNet-cross-chem-earlystop-v4',
                   'BatLiNet-cross-chem-earlystop-v5')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def separate_output(workspace, *protected):
    workspace = Path(workspace).resolve()
    for other in protected:
        other = Path(other).resolve()
        if workspace == other or workspace.is_relative_to(other) or other.is_relative_to(workspace):
            raise ValueError('v5输出必须独立于旧数据和其他实验目录，不能互相嵌套。')
    return workspace


def _safe(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('名单或输出路径超出指定工作区。')
    return path


def protected_folders(previous_workspace, old_data, workspace):
    previous_workspace, workspace = Path(previous_workspace).resolve(), Path(workspace).resolve()
    parents = {ROOT / 'workspaces', previous_workspace.parent}
    for parent in {ROOT.parent, previous_workspace.parent.parent.parent}:
        parents.update(parent / tree / 'workspaces' for tree in KNOWN_WORKTREES)
    folders = [previous_workspace, Path(old_data).resolve()]
    for parent in sorted(parents):
        for name in KNOWN_QUEUES:
            path = (parent / name).resolve()
            if name != 'cross_chem_earlystop_v5' or path != workspace:
                folders.append(path)
    return list(dict.fromkeys(path.resolve() for path in folders))


def _pair_tensor(indices):
    if torch.is_tensor(indices):
        if indices.dtype != torch.int64 or indices.ndim != 1:
            raise ValueError('电池对索引须为一维64位整数。')
        return indices.detach().cpu().clone()
    if not isinstance(indices, list) or any(not isinstance(i, int) or isinstance(i, bool) for i in indices):
        raise ValueError('电池对索引须为整数列表或64位整数张量。')
    return torch.tensor(indices, dtype=torch.int64)


def legacy_to_native(indices, n_source, n_target):
    """Preserve pairs and their saved order, converting the flattening convention."""
    if any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in (n_source, n_target)):
        raise ValueError('目标和来源电池数量须为正整数。')
    values = _pair_tensor(indices)
    if not len(values) or (values < 0).any() or (values >= n_source * n_target).any():
        raise ValueError('旧电池对索引为空或越界。')
    target = values // n_source
    source = values % n_source
    return source * n_target + target


def native_pairs(protocol):
    if (protocol.get('pair_index_convention') != PAIR_CONVENTION
            or protocol.get('legacy_pair_index_convention') != LEGACY_CONVENTION):
        raise ValueError('电池对编码约定不同，不能直接复用整数索引。')
    ns, nt = len(protocol['source_ids']), len(protocol['target_train_ids'])
    total = ns * nt
    train, validation = (_pair_tensor(protocol[key]) for key in ('train_pairs', 'val_pairs'))
    if (len(train) != int(total * .8) or len(validation) != total - len(train)
            or not torch.equal(torch.cat((train, validation)).sort().values, torch.arange(total))):
        raise ValueError('实际电池对须互斥、完整覆盖，并保持原80%/20%数量。')
    for name, value in (('train_pairs', train), ('val_pairs', validation)):
        if not torch.equal(value, legacy_to_native(protocol['legacy_' + name], ns, nt)):
            raise ValueError('原生电池对与旧实际目标/参考组合不同。')
    return train, validation


def inspect_previous(previous_workspace):
    previous_workspace = Path(previous_workspace).resolve()
    qpath = previous_workspace / 'queue.json'
    old_queue = read_json(qpath)
    if (old_queue.get('schema') != 3 or old_queue.get('jobs') != previous.formal_plan()
            or old_queue.get('config') != previous.read_config()):
        raise ValueError('必须使用完整v3匹配队列及其原配置。')
    for relative in FROZEN_MODEL_PATHS:
        if old_queue['code_hashes'].get(relative) != common.digest(ROOT / relative):
            raise ValueError(f'原模型或优化器实现已变化：{relative}')
    old = Path(old_queue['existing_workspace']).resolve()
    manifest, actual_protocols, audit = data.audit_existing(old)
    imported = old_queue['imported_data']
    for key in ('existing_manifest_sha256', 'existing_queue_sha256'):
        if imported[key] != audit[key]:
            raise ValueError('v3记录的实际数据与当前底层数据指纹不同。')
    for relative, value in audit['protocols'].items():
        if imported['protocols'][relative]['sha256'] != value['sha256']:
            raise ValueError('实际旧电池对划分与v3来源指纹不同。')
    selected_path, scores_path = previous_workspace / 'selection.json', previous_workspace / 'pilot/scores.json'
    selected = read_json(selected_path)
    if previous.select_budget(read_json(scores_path), old_queue['config']) != selected:
        raise ValueError('v3已选学习率不能从原来源开发记录复算。')
    files = {str(path.relative_to(previous_workspace).as_posix()): common.digest(path)
             for path in (qpath, selected_path, scores_path)}
    protocols, fingerprints = {}, {}
    for (target, count, seed), actual in actual_protocols.items():
        relative = f'protocols/{target}/train_{count}/seed_{seed}.json'
        path = _safe(previous_workspace, relative)
        saved = read_json(path)
        if (saved.get('existing_manifest_sha256') != audit['existing_manifest_sha256']
                or saved.get('existing_protocol_sha256') != actual['existing_protocol_sha256']
                or saved.get('target_chemistry') != target
                or saved.get('target_train_count') != count or saved.get('seed') != seed
                or any(saved[field] != actual[field] for field in ID_FIELDS)):
            raise ValueError('v3实际名单、任务身份或来源收据不同。')
        legacy_path = _safe(old, f'protocols/{target}/train_{count}/seed_{seed}.pt')
        legacy = common.load(legacy_path)
        if (common.digest(legacy_path) != actual['existing_protocol_sha256']
                or any(legacy[field] != actual[field] for field in ID_FIELDS)):
            raise ValueError('审计后的旧实际电池对文件发生变化。')
        value = dict(saved, schema=5, pair_index_convention=PAIR_CONVENTION,
            legacy_pair_index_convention=LEGACY_CONVENTION,
            legacy_pair_split_reused_for_training=True,
            new_relation_training='saved_80_percent_pairs_only',
            relation_validation='saved_20_percent_pairs_share_training_targets')
        for name in ('train_pairs', 'val_pairs'):
            value['legacy_' + name] = _pair_tensor(legacy[name]).tolist()
            value[name] = legacy_to_native(legacy[name], len(actual['source_ids']), count).tolist()
        native_pairs(value)
        protocols[(target, count, seed)] = value
        fingerprints[relative] = fingerprint(value)
        files[relative] = common.digest(path)
    for relative, sha in files.items():
        if common.digest(previous_workspace / relative) != sha:
            raise ValueError('审计过程中v3证据发生变化。')
    receipt = dict(schema=5, previous_workspace=str(previous_workspace), previous_files=files,
        previous_queue_sha256=files['queue.json'], previous_selection_sha256=files['selection.json'],
        learning_rate=selected['learning_rate'], data_workspace=str(old), imported_data=audit,
        protocol_fingerprints=fingerprints, previous_results_modified=False,
        split_generation_called=False, development_runs=0, lfp_validation_holdout=0,
        reuse_previous_weights=False, pair_index_convention=PAIR_CONVENTION)
    return manifest, protocols, receipt


def verify_receipt(receipt, workspace=None):
    previous_workspace = Path(receipt['previous_workspace'])
    for relative, sha in receipt['previous_files'].items():
        if common.digest(_safe(previous_workspace, relative)) != sha:
            raise ValueError('原v3名单或已选学习率指纹变化。')
    old, audit = Path(receipt['data_workspace']), receipt['imported_data']
    for path, sha in ((old / 'queue.json', audit['existing_queue_sha256']),
                      (old / 'dataset/manifest.json', audit['existing_manifest_sha256'])):
        if common.digest(path) != sha:
            raise ValueError('底层数据收据指纹变化。')
    for relative, spec in audit['protocols'].items():
        if common.digest(_safe(old / 'protocols', relative)) != spec['sha256']:
            raise ValueError('底层实际名单或电池对指纹变化。')
    if workspace is not None:
        if read_json(Path(workspace) / 'source_receipt.json') != receipt:
            raise ValueError('v5来源收据与队列身份不同。')
        for relative, sha in receipt['protocol_fingerprints'].items():
            value = read_json(_safe(workspace, relative))
            if fingerprint(value) != sha:
                raise ValueError('v5实际名单或转换后的电池对指纹变化。')
            native_pairs(value)


def prepare_inputs(previous_workspace, workspace):
    manifest, protocols, receipt = inspect_previous(previous_workspace)
    workspace = separate_output(workspace, previous_workspace, receipt['data_workspace'])
    path = _safe(workspace, 'source_receipt.json')
    if path.exists() and read_json(path) != receipt:
        raise ValueError('v5已有来源收据不同，禁止混合恢复。')
    for (target, count, seed), value in protocols.items():
        path = _safe(workspace, f'protocols/{target}/train_{count}/seed_{seed}.json')
        if path.exists() and read_json(path) != value:
            raise ValueError('v5已有实际名单或电池对不同。')
        if not path.exists():
            common.atomic_json(value, path)
    path = workspace / 'source_receipt.json'
    if not path.exists():
        common.atomic_json(receipt, path)
    return manifest, protocols, receipt
