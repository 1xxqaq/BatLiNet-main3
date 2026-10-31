"""Read-only v3 evidence, including actual LFP splits and source-selected rate."""
import json
from pathlib import Path

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


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def separate_output(workspace, *protected):
    workspace = Path(workspace).resolve()
    for other in protected:
        other = Path(other).resolve()
        if workspace == other or workspace.is_relative_to(other) or other.is_relative_to(workspace):
            raise ValueError('v4输出必须独立于v3和底层旧数据工作区，不能互相嵌套。')
    return workspace


def _safe(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('导入或输出路径超出指定工作区。')
    return path


def validate_split(split, source_ids):
    fields = ('source_ids', 'training_pool_ids', 'validation_ids')
    sizes = (219, 16, 40)
    combined = []
    for key, count in zip(fields, sizes):
        values = split.get(key)
        if not isinstance(values, list) or len(values) != count or any(not isinstance(v, str) for v in values):
            raise ValueError('既有LFP开发划分数量或编号类型不同。')
        combined.extend(values)
    if len(set(combined)) != 275 or set(combined) != set(source_ids):
        raise ValueError('既有开发划分重叠、遗漏或包含非LFP目标电池。')


def inspect_previous(previous_workspace):
    """Read all evidence without regenerating IDs or evaluating target scores."""
    previous_workspace = Path(previous_workspace).resolve()
    qpath = previous_workspace / 'queue.json'
    old_queue = read_json(qpath)
    if (old_queue.get('schema') != 3 or old_queue.get('jobs') != previous.formal_plan()
            or old_queue.get('config') != previous.read_config()):
        raise ValueError('必须使用完整v3匹配队列及其原配置。')
    for relative in FROZEN_MODEL_PATHS:
        if old_queue['code_hashes'].get(relative) != common.digest(ROOT / relative):
            raise ValueError(f'原模型或优化器实现已变化，无法只比较停止规则：{relative}')
    old = Path(old_queue['existing_workspace']).resolve()
    manifest, protocols, audit = data.audit_existing(old)
    imported = old_queue['imported_data']
    for key in ('existing_manifest_sha256', 'existing_queue_sha256'):
        if imported[key] != audit[key]:
            raise ValueError('v3记录的实际数据与当前底层数据指纹不同。')
    for relative, value in audit['protocols'].items():
        if imported['protocols'][relative]['sha256'] != value['sha256']:
            raise ValueError('底层实际名单与v3记录的名单不同。')
    files = {'queue.json': common.digest(qpath)}
    selected_path = previous_workspace / 'selection.json'
    scores_path = previous_workspace / 'pilot/scores.json'
    selected = read_json(selected_path)
    if previous.select_budget(read_json(scores_path), old_queue['config']) != selected:
        raise ValueError('v3已选学习率不能从原来源开发记录复算。')
    files['selection.json'] = common.digest(selected_path)
    files['pilot/scores.json'] = common.digest(scores_path)
    splits = {}
    for seed in (0, 1):
        relative = f'pilot/split_{seed}.json'
        split = read_json(previous_workspace / relative)
        validate_split(split, audit['source_ids'])
        splits[str(seed)] = split
        files[relative] = common.digest(previous_workspace / relative)
    for (target, count, seed), actual in protocols.items():
        relative = f'protocols/{target}/train_{count}/seed_{seed}.json'
        path = _safe(previous_workspace, relative)
        saved = read_json(path)
        if (saved.get('existing_manifest_sha256') != audit['existing_manifest_sha256']
                or saved.get('existing_protocol_sha256') != actual['existing_protocol_sha256']
                or saved.get('target_chemistry') != target
                or saved.get('target_train_count') != count or saved.get('seed') != seed):
            raise ValueError('v3名单来源收据或任务身份变化。')
        if any(saved[field] != actual[field] for field in ID_FIELDS):
            raise ValueError('v3实际电池名单与底层正式协议不同。')
        files[relative] = common.digest(path)
    for relative, sha in files.items():
        if common.digest(previous_workspace / relative) != sha:
            raise ValueError('审计过程中v3证据发生变化。')
    receipt = dict(schema=4, previous_workspace=str(previous_workspace),
        previous_files=files, previous_queue_sha256=files['queue.json'],
        previous_selection_sha256=files['selection.json'],
        learning_rate=selected['learning_rate'], development_splits=splits,
        data_workspace=str(old), imported_data=audit,
        previous_results_modified=False, split_generation_called=False,
        source_only_budget_selection=True)
    return manifest, protocols, receipt


def verify_receipt(receipt):
    """Cheap read-only version checks; saved source and target scores are not used."""
    previous_workspace = Path(receipt['previous_workspace'])
    for relative, sha in receipt['previous_files'].items():
        if common.digest(_safe(previous_workspace, relative)) != sha:
            raise ValueError('原v3名单、学习率或开发划分指纹变化。')
    old = Path(receipt['data_workspace'])
    audit = receipt['imported_data']
    for path, sha in ((old / 'queue.json', audit['existing_queue_sha256']),
                      (old / 'dataset/manifest.json', audit['existing_manifest_sha256'])):
        if common.digest(path) != sha:
            raise ValueError('底层旧数据收据指纹变化。')
    for relative, spec in audit['protocols'].items():
        if common.digest(_safe(old / 'protocols', relative)) != spec['sha256']:
            raise ValueError('底层旧实际名单指纹变化。')


def prepare_inputs(previous_workspace, workspace):
    manifest, protocols, receipt = inspect_previous(previous_workspace)
    workspace = separate_output(workspace, previous_workspace, receipt['data_workspace'])
    path = _safe(workspace, 'source_receipt.json')
    if path.exists() and read_json(path) != receipt:
        raise ValueError('v4已有来源收据不同，禁止混合恢复。')
    for (target, count, seed), value in protocols.items():
        path = _safe(workspace, f'protocols/{target}/train_{count}/seed_{seed}.json')
        if path.exists():
            if read_json(path) != value:
                raise ValueError('v4已有实际名单不同。')
        else:
            common.atomic_json(value, path)
    for seed, split in receipt['development_splits'].items():
        path = _safe(workspace, f'development/split_{seed}.json')
        if path.exists() and read_json(path) != split:
            raise ValueError('v4已有LFP开发划分不同。')
        if not path.exists():
            common.atomic_json(split, path)
    path = workspace / 'source_receipt.json'
    if not path.exists():
        common.atomic_json(receipt, path)
    return manifest, protocols, receipt
