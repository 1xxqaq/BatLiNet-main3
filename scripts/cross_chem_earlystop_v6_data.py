"""V6 input routes: recorded v3 selection, or actual v2 data with an explicit rate."""
import math
from pathlib import Path

from scripts import cross_chem_earlystop_v5_data as legacy

ROOT = legacy.ROOT
V3_MODE = 'v3'
V2_MODE = 'v2_fixed_learning_rate'
FIXED_RATE_SOURCE = 'explicit_fixed_value_without_v3_selection'
PAIR_CONVENTION = legacy.PAIR_CONVENTION
LEGACY_CONVENTION = legacy.LEGACY_CONVENTION
ID_FIELDS = legacy.ID_FIELDS
KNOWN_QUEUES = legacy.KNOWN_QUEUES
KNOWN_WORKTREES = legacy.KNOWN_WORKTREES
read_json = legacy.read_json
fingerprint = legacy.fingerprint
native_pairs = legacy.native_pairs
legacy_to_native = legacy.legacy_to_native
separate_output = legacy.separate_output
prepare_inputs = legacy.prepare_inputs
common, data = legacy.common, legacy.data


def fixed_rate(value):
    if (not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0):
        raise ValueError('直接使用v2时必须显式指定有限、正数的固定学习率。')
    return float(value)


def inspect_existing(existing, learning_rate):
    """No v3 files, model weights, raw-data discovery or new split generation."""
    rate = fixed_rate(learning_rate)
    existing = Path(existing).resolve()
    manifest, actual_protocols, audit = data.audit_existing(existing)
    protocols, fingerprints = {}, {}
    for (target, count, seed), actual in actual_protocols.items():
        relative = f'protocols/{target}/train_{count}/seed_{seed}.pt'
        path = legacy._safe(existing, relative)
        raw = common.load(path)
        if (common.digest(path) != actual['existing_protocol_sha256']
                or any(raw[field] != actual[field] for field in ID_FIELDS)):
            raise ValueError('审计后的v2实际电池名单或电池对文件发生变化。')
        value = dict(actual, schema=5, existing_workspace=str(existing),
            existing_manifest_sha256=audit['existing_manifest_sha256'],
            pair_index_convention=PAIR_CONVENTION, legacy_pair_index_convention=LEGACY_CONVENTION,
            legacy_pair_split_reused_for_training=True,
            new_relation_training='saved_80_percent_pairs_only',
            relation_validation='saved_20_percent_pairs_share_training_targets')
        for name in ('train_pairs', 'val_pairs'):
            value['legacy_' + name] = legacy._pair_tensor(raw[name]).tolist()
            value[name] = legacy_to_native(raw[name], len(actual['source_ids']), count).tolist()
        native_pairs(value)
        protocols[(target, count, seed)] = value
        json_relative = f'protocols/{target}/train_{count}/seed_{seed}.json'
        fingerprints[json_relative] = fingerprint(value)
    # audit_existing was originally written for v3's all-pair training. Its
    # factual SHA/ID checks are reused, while these workflow descriptions must
    # describe the actual v6 80/20 use rather than the older all-pair workflow.
    imported = dict(audit, legacy_pair_split_reused_for_training=True,
                    new_relation_training='saved_80_percent_pairs_only')
    receipt = dict(schema=5, input_mode=V2_MODE, learning_rate_source=FIXED_RATE_SOURCE,
        v3_selection_available=False, previous_workspace=str(existing),
        previous_files={'queue.json': audit['existing_queue_sha256']},
        previous_queue_sha256=audit['existing_queue_sha256'], previous_selection_sha256=None,
        learning_rate=rate, data_workspace=str(existing), imported_data=imported,
        protocol_fingerprints=fingerprints, previous_results_modified=False,
        split_generation_called=False, development_runs=0, lfp_validation_holdout=0,
        reuse_previous_weights=False, pair_index_convention=PAIR_CONVENTION,
        legacy_pair_split_reused_for_training=True)
    verify_receipt(receipt)
    return manifest, protocols, receipt


def verify_receipt(receipt, workspace=None):
    mode = receipt.get('input_mode', V3_MODE)
    if mode == V2_MODE:
        fixed_rate(receipt['learning_rate'])
        if (receipt.get('learning_rate_source') != FIXED_RATE_SOURCE
                or receipt.get('v3_selection_available') is not False
                or receipt.get('previous_selection_sha256') is not None
                or Path(receipt['previous_workspace']).resolve() != Path(receipt['data_workspace']).resolve()
                or receipt['previous_files'] != {
                    'queue.json': receipt['imported_data']['existing_queue_sha256']}):
            raise ValueError('v2固定学习率来源记录不一致，不能冒充v3选择结果。')
    elif mode != V3_MODE:
        raise ValueError('未知的数据或学习率来源模式。')
    legacy.verify_receipt(receipt, workspace)


def prepare_existing(existing, workspace, learning_rate):
    workspace = separate_output(workspace, existing)
    manifest, protocols, receipt = inspect_existing(existing, learning_rate)
    path = legacy._safe(workspace, 'source_receipt.json')
    if path.exists() and read_json(path) != receipt:
        raise ValueError('已有v6来源、实际划分或固定学习率不同，禁止混合恢复。')
    for (target, count, seed), value in protocols.items():
        path = legacy._safe(workspace, f'protocols/{target}/train_{count}/seed_{seed}.json')
        if path.exists() and read_json(path) != value:
            raise ValueError('v6已有实际电池名单或电池对不同。')
        if not path.exists():
            common.atomic_json(value, path)
    path = workspace / 'source_receipt.json'
    if not path.exists():
        common.atomic_json(receipt, path)
    return manifest, protocols, receipt
