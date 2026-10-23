"""Matched optimizer-step training; no dataset discovery or formal-test selection.

The single-cell network is extracted *after* constructing the full combination,
so its retained parameters match the combination at the same initialization seed.
Every method uses ln(cycles / 1000), fixed final weights, and AdamW updates.
"""
import copy
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import run_three_tasks as common
from src.models.rul_predictors.cycle_mixer_soft_aligned_difference_batlinet import (
    CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor,
)

KINDS = ('direct', 'finetune', 'pretrain', 'combination')
FIXED_CYCLE_UNIT = 1000.0
DEFAULT_MODEL_CONFIG = dict(
    in_channels=6, channels=32, input_height=100, input_width=1000,
    epochs=100, train_batch_size=128, test_batch_size=1,
    train_support_size=1, test_support_size=275,
    gradient_accumulation_steps=1, evaluate_freq=100,
    checkpoint_freq=1, filter_cycles=False, attention_channels=64,
    attention_heads=4, attention_layers=1, attention_dropout=.1,
    attention_mlp_ratio=2, head_hidden_channels=64, encoder_dropout=.1,
    alpha=.5, lr=.001, cycle_mixer_hidden=16, workspace=None,
)


def _kind(kind):
    if kind not in KINDS:
        raise ValueError(f'未知训练方法：{kind}；允许 {KINDS}')
    return kind


def _config(model_config):
    cfg = copy.deepcopy(DEFAULT_MODEL_CONFIG)
    if model_config is not None:
        override = dict(model_config)
        name = override.pop('name', 'CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor')
        if name != 'CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor':
            raise ValueError('必须从循环轴残差＋软匹配差分组合构造模型。')
        override.pop('seed', None)
        cfg.update(override)
    return cfg


class SingleCellModel(nn.Module):
    """Retain the exact encoder and own head; do not register relation modules."""

    def __init__(self, combination):
        super().__init__()
        self.cell_encoder = combination.cell_encoder
        self.ori_head = combination.ori_head

    def forward(self, feature):
        return self.ori_head(self.cell_encoder(feature)).reshape(-1)


def make_model(kind, seed, device, model_config=None):
    _kind(kind)
    cfg = _config(model_config)
    common.seed_all(seed)
    combination = CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor(**cfg, seed=seed)
    model = combination if kind == 'combination' else SingleCellModel(combination)
    model._matched_model_config = cfg
    return model.to(device)


def _check_part(part, name):
    if not isinstance(part, dict) or 'feature' not in part or 'label' not in part:
        raise ValueError(f'{name}必须包含feature与label。')
    x, y = part['feature'], part['label']
    if (not torch.is_tensor(x) or not torch.is_tensor(y) or x.ndim != 4
            or y.ndim != 1 or len(y) < 1 or len(x) != len(y)):
        raise ValueError(f'{name}特征与标签尺寸异常。')
    if not torch.isfinite(x).all() or not torch.isfinite(y).all() or (y <= 0).any():
        raise ValueError(f'{name}包含非有限特征或非正寿命。')
    ids = part.get('ids')
    if ids is not None and (len(ids) != len(y) or len(set(ids)) != len(y)):
        raise ValueError(f'{name}电池编号缺失或重复。')


def encode_labels(part):
    """Return a shallow copy with z; no empirical statistics are estimated."""
    _check_part(part, '数据')
    result = dict(part)
    result['z'] = (part['label'].double() / FIXED_CYCLE_UNIT).log().float()
    return result


def decode_labels(prediction):
    result = prediction.double().exp() * FIXED_CYCLE_UNIT
    if not torch.isfinite(result).all() or (result <= 0).any():
        raise FloatingPointError('寿命逆变换溢出或非正，不裁剪或删除样本。')
    return result


def next_indices(size, batch_size, cursor):
    """Sequential circular sampling; cursor always denotes the next sample."""
    if (not isinstance(size, int) or size < 1 or not isinstance(batch_size, int)
            or batch_size < 1 or not isinstance(cursor, int) or not 0 <= cursor < size):
        raise ValueError('顺序采样器的样本数、批量或游标异常。')
    return (torch.arange(batch_size, dtype=torch.long) + cursor) % size, (cursor + batch_size) % size


def pair_indices(indices, n_source, n_target):
    """Interleave targets within each source block for balanced step exposure."""
    if (not isinstance(n_source, int) or n_source < 1
            or not isinstance(n_target, int) or n_target < 1
            or not torch.is_tensor(indices) or indices.dtype != torch.int64
            or indices.ndim != 1 or not len(indices)
            or (indices < 0).any() or (indices >= n_source * n_target).any()):
        raise ValueError('配对索引、来源数或目标数异常。')
    return indices % n_target, indices // n_target


def train_step(model, optimizer, source, target, kind, indices, device, micro_batch=8):
    """One optimizer update; gradients use actual microbatch sample weights."""
    _kind(kind)
    if micro_batch < 1 or indices.ndim != 1 or not len(indices):
        raise ValueError('微批量或本步样本索引异常。')
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = own_total = relation_total = 0.0
    part = source if kind == 'pretrain' else target
    for offset in range(0, len(indices), micro_batch):
        chosen = indices[offset:offset + micro_batch]
        if kind == 'combination':
            ti, si = pair_indices(chosen, len(source['label']), len(target['label']))
            y = target['z'][ti].to(device)
            own, relation, _, _, _ = model.compute_prediction_components(
                target['feature'][ti].to(device),
                source['feature'][si].to(device).unsqueeze(1),
                source['z'][si].to(device).unsqueeze(1))
            own_loss = (own - y).square().mean()
            relation_loss = (relation[:, 0] - y).square().mean()
            loss = (1 - model.alpha) * own_loss + model.alpha * relation_loss
        else:
            y = part['z'][chosen].to(device)
            own_loss = (model(part['feature'][chosen].to(device)) - y).square().mean()
            relation_loss = own_loss.detach() * 0
            loss = own_loss
        if not torch.isfinite(loss):
            raise FloatingPointError('训练损失非有限。')
        weight = len(chosen) / len(indices)
        (loss * weight).backward()
        total += float(loss.detach()) * weight
        own_total += float(own_loss.detach()) * weight
        relation_total += float(relation_loss.detach()) * weight
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
        raise FloatingPointError('训练梯度非有限。')
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise FloatingPointError('参数更新产生非有限值。')
    return dict(train_loss=total, own_loss=own_total,
                relation_loss=relation_total, batch_size=len(indices))


def _tensor_digest(tensor):
    value = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(value.dtype).encode())
    h.update(str(tuple(value.shape)).encode())
    # Hash contiguous storage without making another full LFP feature copy.
    h.update(memoryview(value.numpy()).cast('B'))
    return h.hexdigest()


def _part_identity(part):
    if part is None:
        return None
    _check_part(part, '绑定数据')
    return dict(ids=list(part.get('ids', range(len(part['label'])))),
                feature_sha256=_tensor_digest(part['feature']),
                label_sha256=_tensor_digest(part['label']))


def _state_identity(state):
    if state is None:
        return None
    return {name: _tensor_digest(value) for name, value in sorted(state.items())}


def _assert_binding(actual, expected, path):
    # Caller bindings must be plain serializable metadata, not live tensors.
    def canonical(value):
        return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    if canonical(actual) != canonical(expected):
        raise ValueError(f'检查点与本次代码、数据、配置或训练预算绑定不同：{path}')


def _align_log(folder, step):
    path = folder / 'train.jsonl'
    lines = path.read_text(encoding='utf-8').splitlines() if path.is_file() else []
    keep = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row.get('step', step + 1) > step:
            break
        if (row.get('step') != len(keep) + 1
                or not math.isfinite(row.get('train_loss', float('nan')))):
            raise ValueError(f'训练日志步数或损失不连续：{path}')
        keep.append(line)
    if len(keep) != step:
        raise ValueError(f'训练日志没有覆盖检查点的{step}步：{path}')
    if len(lines) != len(keep):
        archive = folder / f'interrupted_log_{time.time_ns()}.jsonl'
        archive.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if len(lines) != len(keep) or not path.exists():
        temporary = path.with_name(path.name + '.tmp')
        temporary.write_text(('\n'.join(keep) + '\n') if keep else '', encoding='utf-8')
        os.replace(temporary, path)


def _checked(path, binding, training_binding=None):
    state = common.load(path)
    _assert_binding(state.get('binding'), binding, path)
    if training_binding is not None:
        _assert_binding(state.get('training_binding'), training_binding, path)
    required = {'state', 'optimizer', 'rng', 'cursor', 'step', 'samples_seen',
                'training_seconds', 'training_binding', 'binding'}
    if not required.issubset(state):
        raise ValueError(f'检查点缺少恢复所需字段：{path}')
    spec = state['training_binding']
    if (not isinstance(state['step'], int) or not 0 <= state['step'] <= spec['steps']
            or state['cursor'] != state['step'] * spec['step_batch'] % spec['sample_count']
            or state['samples_seen'] != state['step'] * spec['step_batch']
            or not math.isfinite(state['training_seconds']) or state['training_seconds'] < 0):
        raise ValueError(f'检查点步数、采样游标或耗时异常：{path}')
    return state


def load_completed(folder, binding, kind=None, steps=None):
    """Read/validate final weights and their completion receipt without writing."""
    folder = Path(folder)
    final, receipt = folder / 'final.pt', folder / 'run.json'
    if not final.is_file() or not receipt.is_file():
        raise FileNotFoundError(f'训练尚未完整完成：{folder}')
    record = json.loads(receipt.read_text(encoding='utf-8'))
    _assert_binding(record.get('binding'), binding, receipt)
    if record.get('status') != 'trained' or record.get('final_sha256') != common.digest(final):
        raise ValueError(f'最终权重指纹或完成状态异常：{folder}')
    state = _checked(final, binding)
    spec = state['training_binding']
    if (state['step'] != spec['steps'] or record.get('step') != state['step']
            or record.get('selection') != 'fixed_final_step'
            or (kind is not None and spec['kind'] != kind)
            or (steps is not None and state['step'] != steps)):
        raise ValueError(f'最终权重不是要求的固定训练步数：{folder}')
    _assert_binding(record.get('training_binding'), spec, receipt)
    return state


validate_completed = load_completed


def train_run(source, target, kind, seed, steps, folder, binding, device,
              initial_state=None, checkpoint_every=25, micro_batch=8,
              logical_batch=128, report_every=25, step_callback=None,
              lr=.001, weight_decay=.01, model_config=None):
    """Train a predeclared step budget; callback(step, model) is RNG-isolated.

    Pretraining updates source cells in batches up to logical_batch. Direct and
    fine-tuned models use the complete selected target set on every step.
    Combination updates traverse *all* target/source pairs with wraparound;
    there is no validation-pair exclusion, early stopping, or test selection.
    """
    _kind(kind)
    device = str(device)
    if (not isinstance(steps, int) or steps < 1 or checkpoint_every < 1
            or micro_batch < 1 or logical_batch < 1 or report_every < 1
            or not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError('固定训练预算、批量、学习率或权重衰减异常。')
    if kind == 'finetune' and initial_state is None:
        raise ValueError('微调必须独立加载来源预训练权重。')
    if kind != 'finetune' and initial_state is not None:
        raise ValueError('只有微调方法可以加载来源预训练权重。')
    if kind in ('combination', 'pretrain'):
        source = encode_labels(source)
    if kind != 'pretrain':
        target = encode_labels(target)
    sample_count = (len(source['label']) * len(target['label']) if kind == 'combination'
                    else len(source['label']) if kind == 'pretrain' else len(target['label']))
    step_batch = (min(sample_count, logical_batch) if kind in ('combination', 'pretrain') else sample_count)
    cfg = _config(model_config)
    spec = dict(schema=1, kind=kind, seed=seed, steps=steps, model_config=cfg,
                label_transform='ln_cycles_divided_by_fixed_1000',
                optimizer='AdamW', lr=lr, weight_decay=weight_decay,
                checkpoint_every=checkpoint_every, micro_batch=micro_batch,
                logical_batch=logical_batch, step_batch=step_batch,
                pair_order='source_major_target_interleaved' if kind == 'combination' else None,
                sample_count=sample_count, precision='float32', shuffle=False,
                selection='fixed_final_step', device=device,
                source=_part_identity(source) if kind in ('combination', 'pretrain') else None,
                target=_part_identity(target) if kind != 'pretrain' else None,
                initial_state=_state_identity(initial_state))
    # Reject nonserializable bindings before any output is created.
    _assert_binding(binding, binding, folder)
    folder = Path(folder)
    final, progress, receipt = folder / 'final.pt', folder / 'progress.pt', folder / 'run.json'
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding='utf-8'))
        _assert_binding(record.get('binding'), binding, receipt)
        _assert_binding(record.get('training_binding'), spec, receipt)
    if final.exists():
        state = _checked(final, binding, spec)
        if state['step'] != steps:
            raise ValueError('最终权重没有达到固定训练步数。')
        _align_log(folder, state['step'])
        # Atomic final write may precede its receipt when a process stops.
        if not receipt.exists() or record.get('status') != 'trained':
            common.atomic_json(dict(status='trained', binding=binding, training_binding=spec,
                step=steps, selection='fixed_final_step', final_sha256=common.digest(final)), receipt)
        return load_completed(folder, binding, kind, steps)
    model = make_model(kind, seed, device, cfg)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    step = cursor = samples_seen = 0
    training_seconds = 0.0
    if progress.exists():
        saved = _checked(progress, binding, spec)
        step, cursor, samples_seen = saved['step'], saved['cursor'], saved['samples_seen']
        training_seconds = saved['training_seconds']
        model.load_state_dict(saved['state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        common.restore_rng(saved['rng'], generator, device)
    folder.mkdir(parents=True, exist_ok=True)
    _align_log(folder, step)
    common.atomic_json(dict(status='training', binding=binding, training_binding=spec), receipt)

    def snapshot():
        return dict(schema=1, binding=copy.deepcopy(binding), training_binding=spec,
                    state=common.cpu(model.state_dict()), optimizer=common.cpu(optimizer.state_dict()),
                    rng=common.rng_state(generator, device), cursor=cursor, step=step,
                    samples_seen=samples_seen, training_seconds=training_seconds,
                    selection='fixed_final_step')

    if step == 0:
        common.save(snapshot(), progress)
    while step < steps:
        begin = time.monotonic()
        indices, next_cursor = next_indices(sample_count, step_batch, cursor)
        measured = train_step(model, optimizer, source, target, kind, indices, device, micro_batch)
        seconds = time.monotonic() - begin
        step += 1
        cursor = next_cursor
        samples_seen += len(indices)
        training_seconds += seconds
        row = dict(step=step, optimizer_steps=step, samples_seen=samples_seen,
                   target_sample_exposures=samples_seen if kind != 'pretrain' else 0,
                   source_sample_exposures=samples_seen if kind in ('combination', 'pretrain') else 0,
                   seconds=seconds, cumulative_training_seconds=training_seconds, **measured)
        if step_callback is not None:
            rng = common.rng_state(generator, device)
            was_training = model.training
            try:
                callback_value = step_callback(step, model)
            finally:
                common.restore_rng(rng, generator, device)
                model.train(was_training)
            if callback_value is not None:
                # Callback diagnostics must be JSON-safe, without tensors or NaN.
                json.dumps(callback_value, allow_nan=False)
                row['callback'] = callback_value
        with (folder / 'train.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        if step % checkpoint_every == 0 or step == steps:
            common.save(snapshot(), progress)
        if step % report_every == 0 or step == steps:
            remaining_minutes = (steps - step) * training_seconds / step / 60
            print(f'{kind} 种子{seed} 更新{step}/{steps} 损失{measured["train_loss"]:.6f} '
                  f'累计样本暴露{samples_seen} 训练耗时{training_seconds:.1f}秒 '
                  f'本次训练预计剩余{remaining_minutes:.1f}分钟', flush=True)
    common.save(snapshot(), final)
    common.atomic_json(dict(status='trained', binding=binding, training_binding=spec,
        step=steps, selection='fixed_final_step', final_sha256=common.digest(final)), receipt)
    return load_completed(folder, binding, kind, steps)


@torch.no_grad()
def predict(model, source, testing, kind, device, chunk=8):
    """One input per target; combination uses all provided source references."""
    _kind(kind)
    if chunk < 1:
        raise ValueError('推理批量必须为正数。')
    _check_part(testing, '待评估电池')
    was_training = model.training
    model.eval()
    try:
        if kind == 'combination':
            source = encode_labels(source)
            tokens = torch.cat([model.cell_encoder(source['feature'][i:i + chunk].to(device))
                                for i in range(0, len(source['label']), chunk)])
            own_rows, reference_rows = [], []
            for i in range(len(testing['label'])):
                query = model.cell_encoder(testing['feature'][i:i + 1].to(device))
                own_rows.append(model.ori_head(query).reshape(()).cpu())
                estimates = []
                for start in range(0, len(tokens), chunk):
                    ref = tokens[start:start + chunk]
                    relation = query.expand(len(ref), -1, -1)
                    for block in model.cross_attention:
                        relation = block(relation, ref)
                    estimates.append((model.support_head(relation).reshape(-1)
                        + source['z'][start:start + len(ref)].to(device)).cpu())
                reference_rows.append(torch.cat(estimates))
            own = torch.stack(own_rows)
            references = torch.stack(reference_rows)
            median = references.median(dim=1).values
            prediction_log = model.combine_predictions(own, median)
            diagnostics = dict(own_log=own, reference_log=references,
                               reference_median_log=median,
                               own_cycles=decode_labels(own),
                               reference_cycles=decode_labels(references),
                               reference_median_cycles=decode_labels(median),
                               y_ori=own, y_sup=references, y_sup_agg=median,
                               support_index=torch.arange(len(source['label'])).expand(len(testing['label']), -1).clone())
        else:
            prediction_log = torch.cat([model(testing['feature'][i:i + chunk].to(device)).cpu()
                                       for i in range(0, len(testing['label']), chunk)])
            diagnostics = dict(own_log=prediction_log, own_cycles=decode_labels(prediction_log),
                               y_ori=prediction_log)
        prediction_cycles = decode_labels(prediction_log)
        truth = testing['label'].double().cpu()
        return dict(prediction=prediction_log, prediction_log=prediction_log,
                    prediction_cycles=prediction_cycles, truth_cycles=truth,
                    scores=common.scores(prediction_cycles, truth), diagnostics=diagnostics,
                    source_ids=list(source.get('ids', range(len(source['label'])))) if kind == 'combination' else [],
                    target_test_ids=list(testing.get('ids', range(len(testing['label'])))),
                    label_transform='ln_cycles_divided_by_fixed_1000', fixed_cycle_unit=FIXED_CYCLE_UNIT)
    finally:
        model.train(was_training)
