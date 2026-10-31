"""Tiny synthetic CPU checks for source-only early stopping and v4 budgets.

Generated evidence lives only in TemporaryDirectory.  These checks neither
load real battery data nor start the formal experimental queue.
"""
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cross_chem_earlystop_training as early
from scripts import cross_chem_matched_training as matched
from scripts import run_three_tasks as common
from scripts import run_cross_chem_earlystop as suite
from scripts import cross_chem_earlystop_data as inputs
from scripts import run_cross_chem_matched as previous
from scripts import cross_chem_matched_data as data


class EarlyStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model_config(self, dropout=0.):
        return dict(input_height=20, input_width=128, attention_channels=8,
                    attention_heads=2, head_hidden_channels=8, cycle_mixer_hidden=4,
                    encoder_dropout=dropout, attention_dropout=dropout)

    def fixture(self):
        generator = torch.Generator().manual_seed(613)
        parts = []
        for name, count in [('LFP_source', 3), ('LFP_train', 2), ('LFP_validation', 2)]:
            parts.append(dict(feature=torch.randn(count, 6, 20, 128, generator=generator),
                              label=torch.linspace(225., 1000., count),
                              ids=[f'{name}_{index}' for index in range(count)]))
        return parts

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def kwargs(self, kind='direct', dropout=0., settings=None):
        source, target, validation = self.fixture()
        return dict(source=source, target=target, validation=validation,
                    kind=kind, seed=4, binding={'synthetic': 'early_stop', 'kind': kind},
                    device='cpu', checkpoint_every=2, micro_batch=2, logical_batch=4,
                    report_every=100, model_config=self.model_config(dropout),
                    settings=settings or dict(max_steps=14, min_stop=4, validate_every=2, patience=5))

    def test_first_best_retained_and_five_nonimprovements_stop_at_sixth_validation(self):
        kwargs = self.kwargs()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'validation_loss', side_effect=[1.] * 6):
            final = early.train_run(folder=Path(tmp), **kwargs)
            self.assertEqual(final['best_step'], 2)
            self.assertEqual(final['best_loss'], 1.)
            self.assertEqual(final['trained_steps'], 12)
            self.assertEqual(final['terminal']['step'], 12)
            progress = common.load(Path(tmp) / 'progress.pt')
            self.assertEqual(progress['bad_validations'], 5)
            self.assertEqual(progress['best_step'], 2)
            self.assert_nested_equal(final['state'], progress['best_state'])
            self.assertTrue(any(not torch.equal(value, progress['state'][key])
                                for key, value in progress['best_state'].items()))
            # The tiny cadence 2/4 represents formal validation 50/minimum 100:
            # the first validation may remain best even though stopping comes later.
            self.assertEqual(final['best_step'] * 25, 50)
            self.assertEqual(final['trained_steps'] * 25, 300)

    def test_only_strict_decrease_resets_patience_and_best_can_precede_minimum_stop(self):
        settings = dict(max_steps=12, min_stop=4, validate_every=2, patience=2)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'validation_loss', side_effect=[1., .5, .5, .6]):
            final = early.train_run(folder=Path(tmp), **self.kwargs(settings=settings))
            self.assertEqual(final['trained_steps'], 8)
            self.assertEqual(final['best_step'], 4)
            self.assertEqual(final['best_loss'], .5)
            self.assertEqual(common.load(Path(tmp) / 'progress.pt')['bad_validations'], 2)
        settings = dict(max_steps=8, min_stop=6, validate_every=2, patience=1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'validation_loss', side_effect=[1., 1., 1.]):
            final = early.train_run(folder=Path(tmp), **self.kwargs(settings=settings))
            self.assertEqual(final['best_step'], 2)
            self.assertEqual(final['trained_steps'], 6, 'Patience alone must not bypass minimum stopping step')
        settings = dict(max_steps=8, min_stop=4, validate_every=2, patience=2)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'validation_loss', side_effect=[1., 1. - 1e-10, 1., 1.]):
            final = early.train_run(folder=Path(tmp), **self.kwargs(settings=settings))
            self.assertEqual(final['best_step'], 4, 'Any strict reduction counts, without an implicit tolerance')
            self.assertEqual(final['trained_steps'], 8)
            self.assertEqual(final['stop_reason'], 'max_steps')
            self.assertTrue(final['hit_max_steps'])

    def test_finetunes_independently_reload_each_split_best_and_validation_ids_are_disjoint(self):
        settings = dict(max_steps=4, min_stop=4, validate_every=2, patience=1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'validation_loss', return_value=1.):
            root = Path(tmp)
            chosen = []
            for seed in (0, 1):
                kwargs = self.kwargs('pretrain', settings=settings)
                kwargs['seed'] = seed
                pre = early.train_run(folder=root / f'pre{seed}', **kwargs)
                initial = copy.deepcopy(pre['state'])
                chosen.append(initial)
                for count in (1, 2):
                    fine = self.kwargs('finetune', settings=settings)
                    fine['seed'] = seed
                    fine['target'] = {key: value[:count] for key, value in fine['target'].items()}
                    fine['initial_state'] = initial
                    result = early.train_run(folder=root / f'fine{seed}_{count}', **fine)
                    self.assertEqual(result['training_binding']['initial_state'], matched._state_identity(initial))
                    self.assert_nested_equal(initial, pre['state'])
                with self.assertRaises(ValueError):
                    early.train_run(folder=root / f'missing{seed}', **self.kwargs('finetune', settings=settings))
            self.assertTrue(any(not torch.equal(value, chosen[1][key])
                                for key, value in chosen[0].items()))
            for kind in ('direct', 'pretrain', 'combination'):
                kwargs = self.kwargs(kind, settings=settings)
                kwargs['validation']['ids'][0] = (kwargs['source'] if kind == 'pretrain'
                                                 else kwargs['target'])['ids'][0]
                with self.assertRaises(ValueError):
                    early.train_run(folder=root / f'overlap{kind}', **kwargs)

    def test_validation_metric_matches_fixed_log_prediction_for_each_method(self):
        source, _, validation = self.fixture()
        for kind in ('pretrain', 'direct', 'finetune', 'combination'):
            with self.subTest(kind=kind):
                model = matched.make_model(kind, 8, 'cpu', self.model_config()).train()
                result = matched.predict(model, source, validation, kind, 'cpu', chunk=1)
                expected = float(((result['prediction_cycles'] / 1000.).log()
                                  - (validation['label'].double() / 1000.).log()).square().mean())
                actual = early.validation_loss(model, source, validation, kind, 'cpu', chunk=1)
                self.assertAlmostEqual(actual, expected, places=6)

    def test_dropout_resume_restores_current_best_optimizer_rng_cursor_and_bad_count(self):
        for kind in ('direct', 'combination'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root = Path(tmp)
                kwargs = self.kwargs(kind, dropout=.1,
                                     settings=dict(max_steps=10, min_stop=4, validate_every=2, patience=3))

                def loss(model, source, validation, method, device, chunk=8):
                    # Validation diagnostics must never consume training RNG or
                    # leave the model in evaluation mode.
                    model.eval()
                    torch.randn(7)
                    np.random.random(5)
                    random.random()
                    return 1.

                with patch.object(early, 'validation_loss', return_value=1.):
                    clean = early.train_run(folder=root / 'clean_validation', **kwargs)
                with patch.object(early, 'validation_loss', side_effect=loss):
                    complete = early.train_run(folder=root / 'complete', **kwargs)
                    self.assert_nested_equal(clean['state'], complete['state'])
                    for key in ('state', 'optimizer', 'rng', 'cursor', 'step'):
                        self.assert_nested_equal(clean['terminal'][key], complete['terminal'][key])

                    def interrupt(step, model):
                        model.eval()
                        torch.randn(9)
                        np.random.random(3)
                        random.random()
                        if step == 5:
                            raise RuntimeError('synthetic interruption')

                    with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                        early.train_run(folder=root / 'restart', step_callback=interrupt, **kwargs)
                    saved = common.load(root / 'restart/progress.pt')
                    self.assertEqual(saved['step'], 4)
                    self.assertEqual(saved['best_step'], 2)
                    self.assertEqual(saved['bad_validations'], 1)
                    progress = root / 'restart/progress.pt'
                    original_progress = progress.read_bytes()
                    bad = copy.deepcopy(saved)
                    bad['bad_validations'] += 1
                    common.save(bad, progress)
                    with self.assertRaises(ValueError):
                        early.train_run(folder=root / 'restart', **kwargs)
                    progress.write_bytes(original_progress)
                    with (root / 'restart/train.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write('{"step": 5, "train_loss": 99.}\n{"step":')
                    restarted = early.train_run(folder=root / 'restart', **kwargs)
                for key in ('state', 'best_step', 'best_loss', 'trained_steps', 'stop_reason', 'history'):
                    self.assert_nested_equal(complete[key], restarted[key])
                for key in ('state', 'optimizer', 'rng', 'cursor', 'step', 'samples_seen'):
                    self.assert_nested_equal(complete['terminal'][key], restarted['terminal'][key])
                a = common.load(root / 'complete/progress.pt')
                b = common.load(root / 'restart/progress.pt')
                for key in ('state', 'best_state', 'best_step', 'best_loss', 'bad_validations',
                            'history', 'optimizer', 'rng', 'cursor', 'step'):
                    self.assert_nested_equal(a[key], b[key])
                self.assertFalse((root / 'restart/test.pt').exists())

    def test_nonfinite_validation_and_changed_binding_or_weights_are_rejected(self):
        for value in (float('nan'), float('inf')):
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                 patch.object(early, 'validation_loss', return_value=value):
                with self.assertRaises((ValueError, FloatingPointError)):
                    early.train_run(folder=Path(tmp), **self.kwargs())
                self.assertFalse((Path(tmp) / 'final.pt').exists())
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'validation_loss', return_value=1.):
            root = Path(tmp)
            kwargs = self.kwargs(settings=dict(max_steps=4, min_stop=4, validate_every=2, patience=1))
            early.train_run(folder=root, **kwargs)
            early.load_completed(root, kwargs['binding'], 'direct', kwargs['settings'])
            with self.assertRaises(ValueError):
                early.train_run(folder=root, **dict(kwargs, binding={'changed': True}))
            final_path = root / 'final.pt'
            value = common.load(final_path)
            key = next(iter(value['state']))
            value['state'][key] = value['state'][key] + .1
            common.save(value, final_path)
            with self.assertRaises(ValueError):
                early.load_completed(root, kwargs['binding'], 'direct', kwargs['settings'])

    def development_records(self, cfg):
        records = []
        for seed in cfg['development']['split_seeds']:
            for kind, count in [('pretrain', 0)] + [
                    (kind, count) for count in cfg['development']['target_counts'] for kind in suite.KINDS]:
                if kind == 'pretrain':
                    best = 50 + 50 * seed
                else:
                    position = cfg['development']['target_counts'].index(count)
                    best = 50 + 50 * position + {'direct': 0, 'finetune': 100, 'combination': 400}[kind] + 50 * seed
                records.append(dict(seed=seed, kind=kind, train_count=count,
                                    best_step=best, trained_steps=best + 250,
                                    best_validation_log_mse=.1 + seed * .01,
                                    stop_reason='early_stopping'))
        return records

    def test_budgets_use_two_split_medians_rounded_up_separately_per_method_and_count(self):
        cfg = suite.read_config()
        self.assertEqual(early.read_settings(), cfg['early_stopping'])
        records = self.development_records(cfg)
        self.assertEqual(len(records), 32)
        selection = suite.select_budgets(records, cfg, .0003)
        self.assertEqual(selection['learning_rate'], .0003)
        self.assertEqual(selection['pretrain_steps'], 100)
        self.assertEqual(selection['target_steps']['direct']['1'], 100)
        self.assertEqual(selection['target_steps']['finetune']['1'], 200)
        self.assertEqual(selection['target_steps']['combination']['1'], 500)
        self.assertEqual(selection['target_steps']['direct']['16'], 300)
        self.assertEqual(len(selection['groups']), 16)
        self.assertTrue(selection['source_only'])
        self.assertFalse(selection['formal_target_scores_used'])
        for job in suite.formal_plan():
            expected = (selection['pretrain_steps'] if job['kind'] == 'pretrain' else
                        selection['target_steps'][job['kind']][str(job['train_count'])])
            self.assertEqual(suite.steps_for(job, selection), expected)
        same_first = [dict(row, best_step=50, trained_steps=300) for row in records]
        earliest = suite.select_budgets(same_first, cfg, .0003)
        self.assertEqual(earliest['pretrain_steps'], 50)
        self.assertEqual(earliest['target_steps']['direct']['1'], 50)
        for bad in (records[:-1], records + [records[0]],
                    [dict(records[0], best_validation_log_mse=float('nan'))] + records[1:],
                    [dict(records[0], best_step=75)] + records[1:],
                    [dict(records[0], trained_steps=50)] + records[1:]):
            with self.assertRaises(ValueError):
                suite.select_budgets(bad, cfg, .0003)
        for rate in (0., float('inf'), float('nan')):
            with self.assertRaises(ValueError):
                suite.select_budgets(records, cfg, rate)

    def previous_fixture(self, root):
        # Reuse the existing synthetic evidence builder without changing any v3
        # code or running its tests as part of this suite.
        from scripts.test_cross_chem_matched import MatchedProtocolTests
        helper = MatchedProtocolTests()
        old, previous_workspace, workspace = root / 'v2', root / 'v3', root / 'v4'
        shape, manifest, _ = helper.existing_fixture(old)
        with patch.object(data, 'FEATURE_SHAPE', shape):
            _, protocols, audit = data.import_protocols(old, previous_workspace)
        cfg = previous.read_config()
        records = helper.pilot_records(cfg)
        selection = previous.select_budget(records, cfg)
        common.atomic_json(records, previous_workspace / 'pilot/scores.json')
        common.atomic_json(selection, previous_workspace / 'selection.json')
        splits = {}
        for seed in (0, 1):
            split = previous.pilot_split(audit['source_ids'], seed)
            common.atomic_json(split, previous_workspace / 'pilot' / f'split_{seed}.json')
            splits[str(seed)] = split
        identity = dict(schema=3, jobs=previous.formal_plan(), config=cfg,
                        existing_workspace=str(old), imported_data=audit,
                        code_hashes={relative: common.digest(ROOT / relative)
                                     for relative in inputs.FROZEN_MODEL_PATHS})
        common.atomic_json(identity, previous_workspace / 'queue.json')
        return old, previous_workspace, workspace, shape, manifest, protocols, selection, splits

    def fingerprints(self, *roots):
        return {str(path): common.digest(path) for root in roots
                for path in Path(root).rglob('*') if path.is_file()}

    def test_prepare_reuses_all_112_actual_lists_v3_rate_and_saved_splits_without_randomization(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, previous_workspace, workspace, shape, manifest, protocols, selected, splits = self.previous_fixture(Path(tmp))
            before = self.fingerprints(old, previous_workspace)
            with patch.object(data, 'FEATURE_SHAPE', shape), \
                 patch.object(previous, 'pilot_split', side_effect=AssertionError('Saved source splits must not be redrawn')), \
                 patch('scripts.transfer_cycle_soft_data.build_protocol',
                       side_effect=AssertionError('Actual target lists must not be redrawn')):
                result, imported, receipt = suite.prepare_inputs(previous_workspace, workspace)
                self.assertEqual(result, manifest)
                self.assertEqual(imported, protocols)
                self.assertEqual(len(imported), 112)
                self.assertEqual(receipt['learning_rate'], selected['learning_rate'])
                self.assertEqual(receipt['development_splits'], splits)
                self.assertFalse(receipt['split_generation_called'])
                self.assertEqual(len(list((workspace / 'protocols').rglob('*.json'))), 112)
                self.assertFalse((workspace / 'selection.json').exists())
                inputs.verify_receipt(receipt)
                self.assertEqual(suite.prepare_inputs(previous_workspace, workspace), (result, imported, receipt))
                for invalid in (old, previous_workspace, previous_workspace / 'child', Path(tmp)):
                    with self.assertRaises(ValueError):
                        suite.prepare_inputs(previous_workspace, invalid)
            self.assertEqual(before, self.fingerprints(old, previous_workspace))

    def test_previous_selection_split_protocol_and_source_fingerprint_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, previous_workspace, workspace, shape, manifest, protocols, selected, splits = self.previous_fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                _, _, receipt = suite.prepare_inputs(previous_workspace, workspace)
                for path in (previous_workspace / 'selection.json', previous_workspace / 'pilot/scores.json',
                             previous_workspace / 'pilot/split_0.json',
                             previous_workspace / 'protocols/LCO/train_1/seed_0.json'):
                    original = path.read_bytes()
                    value = json.loads(original.decode('utf-8'))
                    if path.name == 'selection.json':
                        value['learning_rate'] = .02
                    elif path.name == 'scores.json':
                        value[0]['validation_log_mse'] += 1.
                    elif path.name == 'split_0.json':
                        value['validation_ids'][0] = value['source_ids'][0]
                    else:
                        value['target_test_ids'].reverse()
                    common.atomic_json(value, path)
                    with self.assertRaises(ValueError):
                        inputs.verify_receipt(receipt)
                    with self.assertRaises(ValueError):
                        suite.prepare_inputs(previous_workspace, workspace)
                    path.write_bytes(original)
                cell = old / 'dataset' / manifest['records'][0]['feature_file']
                cell.write_bytes(cell.read_bytes() + b'synthetic corruption')
                with self.assertRaises(ValueError):
                    suite.prepare_inputs(previous_workspace, workspace)

    def test_source_development_runs_32_saved_split_jobs_with_independent_split_pretraining(self):
        cfg = suite.read_config()
        source = dict(feature=torch.zeros(275, 6, 1, 1), label=torch.linspace(201., 1001., 275),
                      ids=[f'LFP_{index:03}' for index in range(275)])
        splits = {}
        for seed in (0, 1):
            ids = source['ids'][seed * 7:] + source['ids'][:seed * 7]
            splits[str(seed)] = dict(source_ids=ids[:219], training_pool_ids=ids[219:235], validation_ids=ids[235:])
        receipt = dict(learning_rate=.0003, development_splits=splits)
        identity = dict(config=cfg, source_receipt=receipt)
        calls = []

        def fake_train(src, target, validation, kind, seed, folder, binding, device,
                       initial_state=None, lr=None, **kwargs):
            split = splits[str(seed)]
            self.assertEqual(src['ids'], split['source_ids'])
            self.assertEqual(validation['ids'], split['validation_ids'])
            self.assertTrue(all(i.startswith('LFP_') for i in src['ids'] + target['ids'] + validation['ids']))
            self.assertEqual(set(validation['ids']) & set(src['ids'] + target['ids']), set())
            self.assertEqual(lr, .0003)
            if kind == 'pretrain':
                self.assertIsNone(initial_state)
                state = {'split_marker': torch.tensor([seed + 1.])}
            else:
                self.assertEqual(target['ids'], split['training_pool_ids'][:binding['train_count']])
                if kind == 'finetune':
                    self.assert_nested_equal(initial_state, {'split_marker': torch.tensor([seed + 1.])})
                    pre = Path(folder).parents[1] / 'pretrain/final.pt'
                    self.assertEqual(binding['initial_source_checkpoint_sha256'], common.digest(pre))
                else:
                    self.assertIsNone(initial_state)
                state = {'split_marker': torch.tensor([-99.])}
            calls.append((seed, kind, binding['train_count']))
            result = dict(binding=binding, state=state, best_step=50, trained_steps=300, best_loss=.1,
                          stop_reason='early_stopping', hit_max_steps=False)
            common.save(result, Path(folder) / 'final.pt')
            return result

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'train_run', side_effect=fake_train), \
             patch.object(data, 'load_part', side_effect=AssertionError('Development may not load formal targets')), \
             patch.object(previous, 'pilot_split', side_effect=AssertionError('Use actual saved splits')):
            selected = suite.run_development(source, Path(tmp), identity, cfg, receipt, 'cpu')
            self.assertEqual(len(calls), 32)
            self.assertEqual(sum(kind == 'pretrain' for _, kind, _ in calls), 2)
            self.assertEqual(sum(kind == 'finetune' for _, kind, _ in calls), 10)
            self.assertEqual(selected['development_runs'], 32)
            self.assertEqual(selected['learning_rate'], .0003)
            self.assertTrue(selected['source_only'])
            def fake_load(folder, binding, kind, settings):
                value = common.load(Path(folder) / 'final.pt')
                self.assertEqual(value['binding'], binding)
                self.assertEqual(kind, binding['kind'])
                self.assertEqual(settings, cfg['early_stopping'])
                return value

            # This test isolates the 32-job orchestration. Real checkpoint and
            # receipt validation is covered by the tiny training tests above.
            with patch.object(early, 'load_completed', side_effect=fake_load) as checked:
                self.assertEqual(suite._verify_selection(Path(tmp), identity), selected)
                self.assertEqual(checked.call_count, 32)
            path = Path(tmp) / selected['development_records'][0]['checkpoint']
            path.write_bytes(path.read_bytes() + b'synthetic corruption')
            with self.assertRaises(ValueError):
                suite._verify_selection(Path(tmp), identity)

    def test_formal_gate_requires_exact_344_validated_completions(self):
        jobs = suite.formal_plan()
        self.assertEqual(len(jobs), 344)
        self.assertEqual(sum(job['kind'] == 'pretrain' for job in jobs), 8)
        self.assertEqual(sum(job['kind'] != 'pretrain' for job in jobs), 336)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(suite, 'validate_training_job', return_value={}) as validate:
                self.assertFalse(suite.all_training_complete(root, {}, jobs))
                self.assertEqual(validate.call_count, 0)
                for job in jobs:
                    path = suite.job_folder(root, job) / 'final.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'synthetic gate placeholder')
                self.assertTrue(suite.all_training_complete(root, {}, jobs))
                self.assertEqual(validate.call_count, 344)
                for invalid in (jobs[:-1], [jobs[0]] * 344):
                    with self.assertRaises(ValueError):
                        suite.all_training_complete(root, {}, invalid)
                validate.side_effect = ValueError('synthetic incomplete receipt')
                with self.assertRaises(ValueError):
                    suite.all_training_complete(root, {}, jobs)


if __name__ == '__main__':
    unittest.main(verbosity=2)
