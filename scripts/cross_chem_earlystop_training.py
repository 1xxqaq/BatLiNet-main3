"""Source-only development with real patience-based early stopping.

The network, labels, optimizer updates, circular sampler and prediction rule are
reused unchanged from matched v3.  This module selects a development checkpoint;
the independent formal refits use the selected update budget elsewhere.
"""
import copy
import json
import math
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import cross_chem_matched_training as matched

common = matched.common
make_model = matched.make_model
encode_labels = matched.encode_labels
predict = matched.predict
DEFAULT_SETTINGS = dict(max_steps=1000, min_stop=100, validate_every=50, patience=5)


def read_settings(settings=None):
    result = dict(DEFAULT_SETTINGS)
    if settings is not None:
        if not isinstance(settings, dict) or set(settings) - set(result):
            raise ValueError('早停设置包含未知字段。')
        result.update(settings)
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 1 for v in result.values())
            or result['min_stop'] > result['max_steps']
            or result['validate_every'] > result['max_steps']
            or result['max_steps'] % result['validate_every']):
        raise ValueError('早停预算须为正整数，且最大步数必须覆盖并整除验证间隔。')
    return result


def _clone_state(state):
    return {name: tensor.detach().cpu().clone() for name, tensor in state.items()}


def _disjoint_validation(source, target, validation, kind):
    valid_ids = set(validation.get('ids', []))
    parts = [source] if kind == 'pretrain' else [target]
    if kind == 'combination':
        parts.append(source)
        if set(source.get('ids', [])) & set(target.get('ids', [])):
            raise ValueError('来源与开发训练目标电池编号重叠。')
    if any(valid_ids & set(part.get('ids', [])) for part in parts):
        raise ValueError('开发验证电池与梯度训练电池重叠。')


@torch.no_grad()
def validation_loss(model, source, validation, kind, device, chunk=8):
    """MSE of the final fixed-log prediction, including full reference fusion."""
    output = predict(model, source, validation, kind, device, chunk=chunk)
    truth = (validation['label'].double().cpu() / matched.FIXED_CYCLE_UNIT).log()
    value = float((output['prediction_log'].double().cpu() - truth).square().mean())
    if not math.isfinite(value) or value < 0:
        raise FloatingPointError('来源开发验证对数均方误差非有限或为负。')
    return value


def _terminal_reason(step, bad_validations, settings):
    if step >= settings['max_steps']:
        return 'max_steps'
    if step >= settings['min_stop'] and bad_validations >= settings['patience']:
        return 'early_stopping'
    return None


def _check_progress(state, binding, spec, path):
    matched._assert_binding(state.get('binding'), binding, path)
    matched._assert_binding(state.get('training_binding'), spec, path)
    required = {'state', 'optimizer', 'rng', 'cursor', 'step', 'samples_seen',
                'training_seconds', 'validation_seconds', 'best_state', 'best_step',
                'best_loss', 'bad_validations', 'history', 'stop_reason'}
    if not required.issubset(state):
        raise ValueError(f'早停检查点缺少恢复字段：{path}')
    step, settings = state['step'], spec['settings']
    if (not isinstance(step, int) or not 0 <= step <= settings['max_steps']
            or state['cursor'] != step * spec['step_batch'] % spec['sample_count']
            or state['samples_seen'] != step * spec['step_batch']
            or any(not math.isfinite(state[key]) or state[key] < 0
                   for key in ('training_seconds', 'validation_seconds'))):
        raise ValueError(f'早停检查点步数、采样游标或耗时异常：{path}')
    history = state['history']
    if not isinstance(history, list) or len(history) != step // settings['validate_every']:
        raise ValueError(f'早停验证历史未覆盖检查点：{path}')
    best_loss, best_step, bad = None, 0, 0
    for index, row in enumerate(history, 1):
        loss = row.get('validation_log_mse')
        if (row.get('step') != index * settings['validate_every']
                or not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0):
            raise ValueError(f'早停验证历史步数或误差异常：{path}')
        improved = best_loss is None or loss < best_loss
        if improved:
            best_loss, best_step, bad = loss, row['step'], 0
        else:
            bad += 1
        if (row.get('improved') != improved or row.get('best_loss') != best_loss
                or row.get('best_step') != best_step or row.get('bad_validations') != bad):
            raise ValueError(f'早停历史的最佳权重或坏计数异常：{path}')
    if (state['best_step'] != best_step or state['best_loss'] != best_loss
            or state['bad_validations'] != bad
            or (state['best_state'] is None) != (best_step == 0)
            or state['stop_reason'] != _terminal_reason(step, bad, settings)):
        raise ValueError(f'早停检查点与严格下降规则不一致：{path}')
    return state


def _checked(path, binding, spec=None):
    value = common.load(path)
    if spec is None:
        spec = value['training_binding']
    return _check_progress(value, binding, spec, path)


def _check_final(value, binding, spec, path):
    matched._assert_binding(value.get('binding'), binding, path)
    matched._assert_binding(value.get('training_binding'), spec, path)
    terminal = _check_progress(value['terminal'], binding, spec, path)
    if (terminal['stop_reason'] is None or value.get('trained_steps') != terminal['step']
            or value.get('step') != terminal['step']
            or value.get('best_step') != terminal['best_step']
            or value.get('best_loss') != terminal['best_loss']
            or value.get('bad_validations') != terminal['bad_validations']
            or value.get('history') != terminal['history']
            or value.get('stop_reason') != terminal['stop_reason']
            or value.get('hit_max_steps') != (terminal['step'] == spec['settings']['max_steps'])
            or value.get('selection') != 'first_strictly_lowest_validation_log_mse'
            or not value.get('state') or terminal['best_state'] is None
            or set(value['state']) != set(terminal['best_state'])
            or any(not torch.equal(value['state'][key], terminal['best_state'][key]) for key in value['state'])):
        raise ValueError(f'早停最终权重不是已记录的最佳验证权重：{path}')
    return value


def load_completed(folder, binding, kind=None, settings=None):
    """Read selected weights and terminal recovery state without writing files."""
    folder = Path(folder)
    final, receipt = folder / 'final.pt', folder / 'run.json'
    if not final.is_file() or not receipt.is_file():
        raise FileNotFoundError(f'早停开发训练尚未完整完成：{folder}')
    record = json.loads(receipt.read_text(encoding='utf-8'))
    matched._assert_binding(record.get('binding'), binding, receipt)
    if record.get('status') != 'trained' or record.get('final_sha256') != common.digest(final):
        raise ValueError(f'早停最终权重指纹或完成状态异常：{folder}')
    value = common.load(final)
    spec = value['training_binding']
    matched._assert_binding(record.get('training_binding'), spec, receipt)
    _check_final(value, binding, spec, final)
    if ((kind is not None and kind != spec['kind'])
            or (settings is not None and read_settings(settings) != spec['settings'])
            or any(record.get(key) != value[key]
                   for key in ('best_step', 'trained_steps', 'best_loss', 'stop_reason', 'hit_max_steps'))):
        raise ValueError(f'早停完成收据与选定权重、训练设置不一致：{folder}')
    return value


validate_completed = load_completed


def train_run(source, target, validation, kind, seed, folder, binding, device,
              lr=.001, weight_decay=.01, initial_state=None, model_config=None,
              settings=None, checkpoint_every=25, micro_batch=8, logical_batch=128,
              report_every=25, step_callback=None, inference_chunk=8):
    """Train on source-development cells and select real validation improvements.

    Validation begins after validate_every updates, never at zero. A strict
    floating-point decrease resets patience; equality counts as no improvement.
    At the maximum update both ``stop_reason=max_steps`` and ``hit_max_steps``
    record reaching the cap. Callback(step, model) is isolated like validation.
    """
    matched._kind(kind)
    settings = read_settings(settings)
    device = str(device)
    if (any(not isinstance(v, int) or isinstance(v, bool) or v < 1
            for v in (checkpoint_every, micro_batch, logical_batch, report_every, inference_chunk))
            or not math.isfinite(lr) or lr <= 0 or not math.isfinite(weight_decay) or weight_decay < 0):
        raise ValueError('开发训练批量、检查点间隔或优化器设置异常。')
    if kind == 'finetune' and initial_state is None:
        raise ValueError('开发微调必须独立加载来源预训练最佳权重。')
    if kind != 'finetune' and initial_state is not None:
        raise ValueError('只有开发微调可以加载来源预训练权重。')
    matched._check_part(validation, '来源开发验证')
    if kind in ('pretrain', 'combination'):
        source = encode_labels(source)
    if kind != 'pretrain':
        target = encode_labels(target)
    _disjoint_validation(source, target, validation, kind)
    sample_count = (len(source['label']) * len(target['label']) if kind == 'combination'
                    else len(source['label']) if kind == 'pretrain' else len(target['label']))
    step_batch = min(sample_count, logical_batch) if kind in ('pretrain', 'combination') else sample_count
    cfg = matched._config(model_config)
    spec = dict(schema=1, kind=kind, seed=seed, model_config=cfg, settings=settings,
        label_transform='ln_cycles_divided_by_fixed_1000', optimizer='AdamW',
        lr=lr, weight_decay=weight_decay, checkpoint_every=checkpoint_every,
        micro_batch=micro_batch, logical_batch=logical_batch, step_batch=step_batch,
        inference_chunk=inference_chunk, sample_count=sample_count,
        pair_order='source_major_target_interleaved' if kind == 'combination' else None,
        precision='float32', shuffle=False, device=device,
        selection='first_strictly_lowest_validation_log_mse', evaluation_at_zero=False,
        source=matched._part_identity(source) if kind in ('combination', 'pretrain') else None,
        target=matched._part_identity(target) if kind != 'pretrain' else None,
        validation=matched._part_identity(validation), initial_state=matched._state_identity(initial_state))
    matched._assert_binding(binding, binding, folder)
    folder = Path(folder)
    final, progress, receipt = folder / 'final.pt', folder / 'progress.pt', folder / 'run.json'
    if receipt.exists():
        record = json.loads(receipt.read_text(encoding='utf-8'))
        matched._assert_binding(record.get('binding'), binding, receipt)
        matched._assert_binding(record.get('training_binding'), spec, receipt)

    def finish_receipt(value):
        common.atomic_json(dict(status='trained', binding=binding, training_binding=spec,
            best_step=value['best_step'], trained_steps=value['trained_steps'], best_loss=value['best_loss'],
            stop_reason=value['stop_reason'], hit_max_steps=value['hit_max_steps'],
            final_sha256=common.digest(final)), receipt)

    if final.exists():
        value = _check_final(common.load(final), binding, spec, final)
        matched._align_log(folder, value['trained_steps'])
        if not receipt.exists() or record.get('status') != 'trained':
            finish_receipt(value)
        return load_completed(folder, binding, kind, settings)
    model = make_model(kind, seed, device, cfg)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    step = cursor = samples_seen = best_step = bad_validations = 0
    training_seconds = validation_seconds = 0.0
    best_loss = best_state = stop_reason = None
    history = []
    if progress.exists():
        saved = _checked(progress, binding, spec)
        step, cursor, samples_seen = saved['step'], saved['cursor'], saved['samples_seen']
        training_seconds, validation_seconds = saved['training_seconds'], saved['validation_seconds']
        best_step, best_loss = saved['best_step'], saved['best_loss']
        best_state = saved['best_state']
        bad_validations, stop_reason, history = saved['bad_validations'], saved['stop_reason'], saved['history']
        model.load_state_dict(saved['state'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        common.restore_rng(saved['rng'], generator, device)
    folder.mkdir(parents=True, exist_ok=True)
    matched._align_log(folder, step)
    common.atomic_json(dict(status='training', binding=binding, training_binding=spec), receipt)

    def snapshot():
        return dict(schema=1, binding=copy.deepcopy(binding), training_binding=spec,
            state=_clone_state(model.state_dict()), optimizer=common.cpu(optimizer.state_dict()),
            rng=common.rng_state(generator, device), step=step, cursor=cursor, samples_seen=samples_seen,
            training_seconds=training_seconds, validation_seconds=validation_seconds,
            best_state=best_state, best_step=best_step, best_loss=best_loss,
            bad_validations=bad_validations, history=copy.deepcopy(history), stop_reason=stop_reason)

    def isolated(call):
        rng, was_training = common.rng_state(generator, device), model.training
        try:
            return call()
        finally:
            common.restore_rng(rng, generator, device)
            model.train(was_training)

    if step == 0:
        common.save(snapshot(), progress)
    while stop_reason is None:
        begin = time.monotonic()
        indices, cursor = matched.next_indices(sample_count, step_batch, cursor)
        measured = matched.train_step(model, optimizer, source, target, kind, indices, device, micro_batch)
        seconds = time.monotonic() - begin
        step += 1
        samples_seen += len(indices)
        training_seconds += seconds
        loss = None
        improved = False
        if step % settings['validate_every'] == 0:
            begin_validation = time.monotonic()
            loss = isolated(lambda: validation_loss(model, source, validation, kind, device, inference_chunk))
            validation_seconds += time.monotonic() - begin_validation
            if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
                raise FloatingPointError('开发验证误差非有限或为负。')
            improved = best_loss is None or loss < best_loss
            if improved:
                best_loss, best_step, bad_validations = loss, step, 0
                best_state = _clone_state(model.state_dict())
            else:
                bad_validations += 1
            history.append(dict(step=step, validation_log_mse=loss, improved=improved,
                best_step=best_step, best_loss=best_loss, bad_validations=bad_validations))
        stop_reason = _terminal_reason(step, bad_validations, settings)
        row = dict(step=step, optimizer_steps=step, samples_seen=samples_seen,
            target_sample_exposures=samples_seen if kind != 'pretrain' else 0,
            source_sample_exposures=samples_seen if kind in ('pretrain', 'combination') else 0,
            seconds=seconds, cumulative_training_seconds=training_seconds,
            cumulative_validation_seconds=validation_seconds, validation_log_mse=loss,
            improved=improved, best_step=best_step, best_loss=best_loss,
            bad_validations=bad_validations, stop_reason=stop_reason,
            hit_max_steps=step == settings['max_steps'], **measured)
        if step_callback is not None:
            callback_value = isolated(lambda: step_callback(step, model))
            if callback_value is not None:
                json.dumps(callback_value, allow_nan=False)
                row['callback'] = callback_value
        with (folder / 'train.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        if step % checkpoint_every == 0 or stop_reason is not None:
            common.save(snapshot(), progress)
        if step % report_every == 0 or loss is not None or stop_reason is not None:
            value = f'{loss:.8f}' if loss is not None else '未到验证步'
            print(f'来源开发 {kind} 种子{seed} 更新{step}/{settings["max_steps"]} '
                  f'训练损失{measured["train_loss"]:.6f} 验证误差{value} '
                  f'最佳步{best_step} 连续未改善{bad_validations}/{settings["patience"]} '
                  f'停止原因{stop_reason or "继续"}', flush=True)
    if best_state is None:
        raise ValueError('没有正步数验证最佳权重，不能完成来源开发。')
    terminal = snapshot()
    value = dict(schema=1, binding=copy.deepcopy(binding), training_binding=spec,
        state=_clone_state(best_state), step=step, best_step=best_step, best_loss=best_loss,
        trained_steps=step, bad_validations=bad_validations, history=copy.deepcopy(history),
        stop_reason=stop_reason, hit_max_steps=step == settings['max_steps'],
        selection='first_strictly_lowest_validation_log_mse', terminal=terminal)
    common.save(value, final)
    finish_receipt(value)
    return load_completed(folder, binding, kind, settings)
