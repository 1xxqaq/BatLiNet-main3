"""Synthetic CPU checks for the approved per-run stopping protocol (v5).

All generated evidence is temporary. No real battery data, target test scores,
formal GPU queues, or original experimental outputs are loaded or changed.
"""
import copy
import io
import json
import random
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cross_chem_earlystop_v5_training as early
from scripts import cross_chem_earlystop_v5_data as inputs
from scripts import cross_chem_matched_training as fixed
from scripts import cross_chem_matched_data as data
from scripts import run_cross_chem_matched as previous
from scripts import run_three_tasks as common


class PerRunStoppingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model_config(self, dropout=0.):
        return dict(input_height=20, input_width=128, attention_channels=8,
                    attention_heads=2, head_hidden_channels=8, cycle_mixer_hidden=4,
                    encoder_dropout=dropout, attention_dropout=dropout)

    def fixture(self):
        generator = torch.Generator().manual_seed(613)
        return tuple(dict(feature=torch.randn(count, 6, 20, 128, generator=generator),
                          label=torch.linspace(225., 1000., count),
                          ids=[f'{name}_{index}' for index in range(count)])
                     for name, count in [('LFP_source', 3), ('LCO_target', 2)])

    def pairs(self):
        # Native source-major encoding, deliberately neither sorted nor grouped.
        return torch.tensor([5, 0, 2, 1]), torch.tensor([4, 3])

    def kwargs(self, kind='direct', dropout=0., settings=None):
        source, target = self.fixture()
        args = dict(source=source, target=target, kind=kind, seed=4,
                    binding={'synthetic': 'per_run_stopping', 'kind': kind},
                    device='cpu', checkpoint_every=2, micro_batch=2,
                    logical_batch=3, inference_chunk=2, report_every=100,
                    model_config=self.model_config(dropout),
                    settings=settings or dict(max_steps=14, min_stop=4,
                                              validate_every=2, patience=5))
        if kind == 'combination':
            args['train_pairs'], args['val_pairs'] = self.pairs()
        return args

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

    def test_first_best_retained_and_five_nonimprovements_stop_at_sixth_check(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', side_effect=[1.] * 6):
            result = early.train_run(folder=Path(tmp), **self.kwargs())
            self.assertEqual(result['best_step'], 2)
            self.assertEqual(result['trained_steps'], 12)
            self.assertEqual(result['best_loss'], 1.)
            self.assertEqual(result['bad_validations'], 5)
            self.assertEqual(result['stop_reason'], 'early_stopping')
            self.assertEqual(result['monitor_policy'], 'training_cell_log_mse')
            self.assertFalse(result['independent_target_validation'])
            terminal = result['terminal']
            self.assert_nested_equal(result['state'], terminal['best_state'])
            self.assertTrue(any(not torch.equal(value, terminal['state'][key])
                                for key, value in result['state'].items()))
            # Tiny validation=2/minimum=4 represents formal 50/100 cadence.
            self.assertEqual(result['best_step'] * 25, 50)
            self.assertEqual(result['trained_steps'] * 25, 300)

    def test_strict_improvement_minimum_stop_and_maximum_reason(self):
        cases = [
            (dict(max_steps=12, min_stop=4, validate_every=2, patience=2),
             [1., .5, .5, .6], 8, 4, 'early_stopping'),
            (dict(max_steps=8, min_stop=6, validate_every=2, patience=1),
             [1., 1., 1.], 6, 2, 'early_stopping'),
            (dict(max_steps=8, min_stop=4, validate_every=2, patience=2),
             [1., 1. - 1e-10, 1., 1.], 8, 4, 'max_steps'),
        ]
        for settings, losses, steps, best, reason in cases:
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as tmp, \
                 redirect_stdout(io.StringIO()), patch.object(early, 'monitoring_loss', side_effect=losses):
                result = early.train_run(folder=Path(tmp), **self.kwargs(settings=settings))
                self.assertEqual(result['trained_steps'], steps)
                self.assertEqual(result['best_step'], best)
                self.assertEqual(result['stop_reason'], reason)
                self.assertEqual(result['hit_max_steps'], reason == 'max_steps')

    def test_single_cell_monitor_is_training_log_mse_with_dropout_disabled(self):
        source, target = self.fixture()
        for kind in ('pretrain', 'direct', 'finetune'):
            with self.subTest(kind=kind):
                model = fixed.make_model(kind, 8, 'cpu', self.model_config(.1))
                part = source if kind == 'pretrain' else target
                model.eval()
                with torch.no_grad():
                    prediction = model(part['feature']).double()
                    truth = fixed.encode_labels(part)['z'].double()
                    expected = float((prediction - truth).square().mean())
                model.train()
                actual = early.monitoring_loss(model, source, target, kind, 'cpu', chunk=1)
                self.assertAlmostEqual(actual, expected, places=7)
                self.assertTrue(model.training)
                self.assertEqual(early.monitor_policy(kind), 'training_cell_log_mse')
                # A second call is deterministic despite nonzero dropout.
                self.assertEqual(actual, early.monitoring_loss(model, source, target, kind, 'cpu', chunk=1))

    def test_combo_monitor_is_only_heldout_relation_mse_in_native_pair_order(self):
        source, target = self.fixture()
        _, val_pairs = self.pairs()
        model = fixed.make_model('combination', 8, 'cpu', self.model_config(.1))
        ti, si = fixed.pair_indices(val_pairs, len(source['label']), len(target['label']))
        source_z, target_z = fixed.encode_labels(source)['z'], fixed.encode_labels(target)['z']
        model.eval()
        with torch.no_grad():
            own, relation, *_ = model.compute_prediction_components(
                target['feature'][ti], source['feature'][si].unsqueeze(1), source_z[si].unsqueeze(1))
            expected = float((relation[:, 0].double() - target_z[ti].double()).square().mean())
            own_loss = float((own.double() - target_z[ti].double()).square().mean())
        model.train()
        actual = early.monitoring_loss(model, source, target, 'combination', 'cpu',
                                       chunk=1, val_pairs=val_pairs)
        self.assertAlmostEqual(actual, expected, places=7)
        self.assertNotAlmostEqual(actual, .5 * (expected + own_loss), places=5)
        self.assertTrue(model.training)
        self.assertEqual(early.monitor_policy('combination'), 'heldout_pair_relation_log_mse')

    def test_only_saved_training_pairs_receive_gradient_and_validation_pairs_only_monitor(self):
        kwargs = self.kwargs('combination', settings=dict(max_steps=4, min_stop=4,
                                                        validate_every=2, patience=1))
        training, validation = kwargs['train_pairs'], kwargs['val_pairs']
        train_calls, monitor_calls = [], []
        original_step = fixed.train_step

        def train_step(model, optimizer, source, target, kind, indices, device, micro_batch=8):
            train_calls.append(indices.clone())
            self.assertEqual(set(indices.tolist()) & set(validation.tolist()), set())
            return original_step(model, optimizer, source, target, kind, indices, device, micro_batch)

        def monitor(model, source, target, kind, device, chunk=8, val_pairs=None):
            self.assertTrue(torch.equal(val_pairs, validation))
            monitor_calls.append(val_pairs.clone())
            return 1.

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(fixed, 'train_step', side_effect=train_step), \
             patch.object(early, 'monitoring_loss', side_effect=monitor):
            result = early.train_run(folder=Path(tmp), **kwargs)
        expected = training[torch.arange(12) % len(training)]
        self.assertTrue(torch.equal(torch.cat(train_calls), expected))
        self.assertEqual(len(monitor_calls), 2)
        self.assertEqual(result['terminal']['samples_seen'], 12)
        self.assertEqual(result['terminal']['cursor'], 0)

    def test_invalid_pair_partitions_and_baseline_pair_arguments_are_rejected(self):
        args = self.kwargs('combination')
        invalid = [
            dict(train_pairs=None, val_pairs=None),
            dict(train_pairs=torch.tensor([5, 0, 2, 1]), val_pairs=torch.tensor([4, 1])),
            dict(train_pairs=torch.tensor([5, 0, 2]), val_pairs=torch.tensor([4, 3])),
            dict(train_pairs=torch.tensor([5, 0, 2]), val_pairs=torch.tensor([4, 3, 1])),
            dict(train_pairs=torch.tensor([5., 0., 2., 1.]), val_pairs=torch.tensor([4, 3])),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for index, changes in enumerate(invalid):
                with self.subTest(changes=changes), self.assertRaises(ValueError):
                    early.train_run(folder=Path(tmp) / str(index), **dict(args, **changes))
            for kind in ('direct', 'pretrain'):
                with self.subTest(kind=kind), self.assertRaises(ValueError):
                    early.train_run(folder=Path(tmp) / kind,
                                    **dict(self.kwargs(kind), train_pairs=args['train_pairs'], val_pairs=args['val_pairs']))

    def test_one_target_cell_trains_and_monitors_different_pairs_without_extra_cells(self):
        args = self.kwargs('combination', settings=dict(max_steps=4, min_stop=4,
                                                      validate_every=2, patience=1))
        args['target'] = {key: value[:1] for key, value in args['target'].items()}
        args['train_pairs'], args['val_pairs'] = torch.tensor([2, 0]), torch.tensor([1])
        calls = []
        original = early.monitoring_loss

        def monitor(model, source, target, kind, device, chunk=8, val_pairs=None):
            self.assertEqual(target['ids'], ['LCO_target_0'])
            self.assertEqual(len(source['ids']), 3)
            self.assertTrue(torch.equal(val_pairs, torch.tensor([1])))
            calls.append(True)
            return original(model, source, target, kind, device, chunk, val_pairs)

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', side_effect=monitor):
            result = early.train_run(folder=Path(tmp), **args)
            self.assertEqual(len(calls), 2)
            self.assertEqual(result['monitor_policy'], 'heldout_pair_relation_log_mse')
            self.assertFalse(result['independent_target_validation'])
            self.assertEqual(len(result['training_binding']['target']['ids']), 1)

    def test_finetunes_independently_reload_the_seed_source_best(self):
        settings = dict(max_steps=4, min_stop=4, validate_every=2, patience=1)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', return_value=1.):
            chosen = []
            for seed in (0, 1):
                args = dict(self.kwargs('pretrain', settings=settings), seed=seed)
                pre = early.train_run(folder=Path(tmp) / f'pre{seed}', **args)
                initial = copy.deepcopy(pre['state'])
                chosen.append(initial)
                for n in (1, 2):
                    fine = dict(self.kwargs('finetune', settings=settings), seed=seed,
                                initial_state=initial)
                    fine['target'] = {key: value[:n] for key, value in fine['target'].items()}
                    result = early.train_run(folder=Path(tmp) / f'fine{seed}_{n}', **fine)
                    self.assertEqual(result['training_binding']['initial_state'], fixed._state_identity(initial))
                    self.assert_nested_equal(initial, pre['state'])
                with self.assertRaises(ValueError):
                    early.train_run(folder=Path(tmp) / f'missing{seed}',
                                    **self.kwargs('finetune', settings=settings))
            self.assertTrue(any(not torch.equal(value, chosen[1][key])
                                for key, value in chosen[0].items()))

    def test_dropout_resume_preserves_current_best_optimizer_rng_cursor_and_patience(self):
        for kind in ('direct', 'combination'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
                root = Path(tmp)
                args = self.kwargs(kind, dropout=.1,
                                   settings=dict(max_steps=10, min_stop=4, validate_every=2, patience=3))

                def diagnostic(model, source, target, method, device, chunk=8, val_pairs=None):
                    model.eval()
                    torch.randn(7)
                    np.random.random(5)
                    random.random()
                    return 1.

                with patch.object(early, 'monitoring_loss', return_value=1.):
                    clean = early.train_run(folder=root / 'clean', **args)
                with patch.object(early, 'monitoring_loss', side_effect=diagnostic):
                    complete = early.train_run(folder=root / 'complete', **args)
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
                        early.train_run(folder=root / 'restart', step_callback=interrupt, **args)
                    saved = common.load(root / 'restart/progress.pt')
                    self.assertEqual(saved['step'], 4)
                    self.assertEqual(saved['best_step'], 2)
                    self.assertEqual(saved['bad_validations'], 1)
                    progress = root / 'restart/progress.pt'
                    original = progress.read_bytes()
                    altered = copy.deepcopy(saved)
                    altered['bad_validations'] += 1
                    common.save(altered, progress)
                    with self.assertRaises(ValueError):
                        early.train_run(folder=root / 'restart', **args)
                    progress.write_bytes(original)
                    with (root / 'restart/train.jsonl').open('a', encoding='utf-8') as handle:
                        handle.write('{"step":5,"train_loss":99.}\n{"step":')
                    resumed = early.train_run(folder=root / 'restart', **args)
                for key in ('state', 'best_step', 'best_loss', 'trained_steps', 'stop_reason', 'history'):
                    self.assert_nested_equal(complete[key], resumed[key])
                for key in ('state', 'best_state', 'optimizer', 'rng', 'cursor', 'step',
                            'samples_seen', 'bad_validations', 'history'):
                    self.assert_nested_equal(complete['terminal'][key], resumed['terminal'][key])
                self.assertFalse((root / 'restart/test.pt').exists())

    def test_nonfinite_monitor_changed_source_pair_order_and_weights_are_rejected(self):
        for value in (float('nan'), float('inf'), -1.):
            with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                 patch.object(early, 'monitoring_loss', return_value=value):
                with self.assertRaises((ValueError, FloatingPointError)):
                    early.train_run(folder=Path(tmp), **self.kwargs())
                self.assertFalse((Path(tmp) / 'final.pt').exists())
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', return_value=1.):
            root = Path(tmp)
            args = self.kwargs('combination', settings=dict(max_steps=4, min_stop=4,
                                                          validate_every=2, patience=1))
            early.train_run(folder=root, **args)
            early.load_completed(root, args['binding'], 'combination', args['settings'])
            changed_source = copy.deepcopy(args['source'])
            changed_source['feature'][0, 0, 0, 0] += 1.
            changes = [dict(source=changed_source), dict(train_pairs=args['train_pairs'].flip(0)),
                       dict(binding={'changed': True}), dict(model_config=self.model_config(.1))]
            for change in changes:
                with self.subTest(change=list(change)), self.assertRaises(ValueError):
                    early.train_run(folder=root, **dict(args, **change))
            path = root / 'final.pt'
            result = common.load(path)
            key = next(iter(result['state']))
            result['state'][key] += .1
            common.save(result, path)
            with self.assertRaises(ValueError):
                early.load_completed(root, args['binding'], 'combination', args['settings'])

    def test_legacy_pair_conversion_preserves_physical_cells_and_order(self):
        # Legacy target-major [target0/source0..2, target1/source0..2].
        old = torch.tensor([5, 0, 2, 1, 4, 3])
        native = inputs.legacy_to_native(old, 3, 2)
        self.assertTrue(torch.equal(native, torch.tensor([5, 0, 4, 2, 3, 1])))
        self.assertTrue(torch.equal(inputs.legacy_to_native(old.tolist(), 3, 2), native))
        ti, si = fixed.pair_indices(native, 3, 2)
        self.assertTrue(torch.equal(ti, old // 3))
        self.assertTrue(torch.equal(si, old % 3))
        one = torch.tensor([2, 0, 1])
        self.assertTrue(torch.equal(inputs.legacy_to_native(one, 3, 1), one))
        for invalid in (torch.tensor([-1]), torch.tensor([6]), torch.tensor([1.]), [True]):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                inputs.legacy_to_native(invalid, 3, 2)

    def test_formal_prediction_loads_selected_best_not_terminal_weights(self):
        from scripts import run_cross_chem_earlystop_v5 as suite
        args = self.kwargs(settings=dict(max_steps=4, min_stop=4, validate_every=2, patience=1))
        job = dict(kind='direct', target='LCO', train_count=2, seed=args['seed'])
        original_make, original_predict = fixed.make_model, fixed.predict
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', return_value=1.):
            root = Path(tmp)
            selected = early.train_run(folder=suite.job_folder(root, job), **args)
            self.assertEqual(selected['best_step'], 2)
            self.assertEqual(selected['trained_steps'], 4)
            testing = dict(feature=args['target']['feature'].flip(0).clone(),
                           label=args['target']['label'].flip(0).clone(),
                           ids=['LCO_test_0', 'LCO_test_1'])
            protocol = dict(target_test_ids=testing['ids'], source_ids=args['source']['ids'])
            manifest = dict(records=[dict(cell_id=cell_id, label_cycles=float(label))
                                     for cell_id, label in zip(testing['ids'], testing['label'])])

            def predict(model, *positional, **keywords):
                self.assert_nested_equal(model.state_dict(), selected['state'])
                self.assertTrue(any(not torch.equal(value, selected['terminal']['state'][key])
                                    for key, value in model.state_dict().items()))
                return original_predict(model, *positional, **keywords)

            with patch.object(suite, 'validate_training_job', return_value=selected), \
                 patch.object(fixed, 'make_model', side_effect=lambda kind, seed, device:
                              original_make(kind, seed, device, self.model_config())), \
                 patch.object(fixed, 'predict', side_effect=predict) as prediction:
                row = suite.evaluate_job(root, {}, job, args['source'], testing, protocol,
                                         manifest, suite.read_config(), 'cpu')
                self.assertEqual(prediction.call_count, 1)
                self.assertEqual(row['best_step'], 2)
                self.assertEqual(row['trained_steps'], 4)
                output = common.load(suite.job_folder(root, job) / 'test.pt')
                self.assertEqual(output['training_metadata']['best_step'], 2)
                self.assertEqual(output['training_metadata']['trained_steps'], 4)
                # A resumed evaluation verifies the saved output without a fresh model.
                with patch.object(fixed, 'make_model', side_effect=AssertionError('Reuse verified saved prediction')):
                    self.assertEqual(suite.evaluate_job(root, {}, job, args['source'], testing, protocol,
                                                       manifest, suite.read_config(), 'cpu'), row)

    def test_main_gate_rejects_before_any_target_test_loading_or_prediction(self):
        from scripts import run_cross_chem_earlystop_v5 as suite
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old, previous_workspace, workspace = root / 'old', root / 'v3', root / 'v5'
            common.atomic_json(dict(existing_workspace=str(old)), previous_workspace / 'queue.json')
            source_ids = [f'LFP_{index:03}' for index in range(275)]
            source = dict(feature=torch.zeros(275, 6, 1, 1),
                          label=torch.full((275,), 1000.), ids=source_ids)
            protocols, fingerprints, test_ids = {}, {}, set()
            for chemistry, counts in data.TRAIN_COUNTS.items():
                for n in counts:
                    for seed in range(8):
                        training = [f'{chemistry}_{seed}_train_{index}' for index in range(n)]
                        testing = [f'{chemistry}_{seed}_test_0']
                        test_ids.update(testing)
                        legacy = list(range(275 * n))
                        cut = int(.8 * len(legacy))
                        p = dict(source_ids=source_ids, target_train_ids=training, target_test_ids=testing,
                                 legacy_train_pairs=legacy[:cut], legacy_val_pairs=legacy[cut:],
                                 legacy_pair_index_convention=inputs.LEGACY_CONVENTION,
                                 pair_index_convention=inputs.PAIR_CONVENTION,
                                 train_pairs=inputs.legacy_to_native(legacy[:cut], 275, n).tolist(),
                                 val_pairs=inputs.legacy_to_native(legacy[cut:], 275, n).tolist())
                        protocols[(chemistry, n, seed)] = p
                        fingerprints[f'protocols/{chemistry}/train_{n}/seed_{seed}.json'] = 'synthetic'
            receipt = dict(learning_rate=.0003, imported_data=dict(source_ids=source_ids),
                           protocol_fingerprints=fingerprints)
            calls, states, loaded_ids = [], {}, []

            def load_part(ids, manifest, existing):
                self.assertFalse(set(ids) & test_ids, 'Target test cells were read before the full training gate')
                loaded_ids.extend(ids)
                if ids == source_ids:
                    return source
                return dict(feature=torch.zeros(len(ids), 6, 1, 1),
                            label=torch.full((len(ids),), 1000.), ids=list(ids))

            def train(src, target, kind, seed, folder, binding, device, initial_state=None,
                      train_pairs=None, val_pairs=None, **kwargs):
                self.assertEqual(src['ids'], source_ids)
                self.assertEqual(kwargs['lr'], .0003)
                self.assertEqual(kwargs['settings'], early.read_settings())
                if kind == 'finetune':
                    self.assert_nested_equal(initial_state, {'marker': torch.tensor([seed + 1.])})
                else:
                    self.assertIsNone(initial_state)
                if kind == 'combination':
                    self.assertEqual(set(train_pairs.tolist()) & set(val_pairs.tolist()), set())
                    self.assertEqual(len(train_pairs), int(.8 * len(target['ids']) * 275))
                else:
                    self.assertIsNone(train_pairs)
                    self.assertIsNone(val_pairs)
                calls.append((kind, seed))
                value = dict(state={'marker': torch.tensor([seed + 1.])}, binding=binding,
                             step=300, trained_steps=300, best_step=50, best_loss=.1,
                             monitor_policy=early.monitor_policy(kind), stop_reason='early_stopping',
                             hit_max_steps=False, samples_seen=300, training_seconds=0., monitor_seconds=0.)
                common.save(value, Path(folder) / 'final.pt')
                states[str(Path(folder))] = value
                return value

            argv = ['run_cross_chem_earlystop_v5.py', '--previous-workspace', str(previous_workspace),
                    '--workspace', str(workspace), '--device', 'cuda:0']
            with ExitStack() as stack:
                contexts = [
                    patch.object(sys, 'argv', argv),
                    patch.object(suite, 'os', SimpleNamespace(name='posix')),
                    patch.object(torch.cuda, 'is_available', return_value=True),
                    patch.object(torch.cuda, 'empty_cache'), patch.object(torch, 'set_num_threads'),
                    patch.object(common, 'seed_all'), patch.object(common, 'runtime', return_value={}),
                    patch.object(common, 'queue_lock', side_effect=lambda folder: nullcontext()),
                    patch.object(previous, 'legacy_read_lock', side_effect=lambda folder: nullcontext()),
                    patch.object(inputs, 'protected_folders', return_value=[previous_workspace, old]),
                    patch.object(suite, 'prepare_inputs', return_value=({}, protocols, receipt)),
                    patch.object(suite, 'code_hashes', return_value={}),
                    patch.object(suite.subprocess, 'check_output', return_value='synthetic_commit'),
                    patch.object(data, 'load_part', side_effect=load_part),
                    patch.object(previous, 'preflight', return_value={'synthetic': 'passed'}),
                    patch.object(early, 'train_run', side_effect=train),
                    patch.object(suite, 'validate_training_job', side_effect=lambda work, identity, job:
                                 states[str(suite.job_folder(work, job))]),
                ]
                for context in contexts:
                    stack.enter_context(context)
                gate = stack.enter_context(patch.object(suite, 'all_training_complete', return_value=False))
                completed = stack.enter_context(patch.object(suite, 'completed_rows'))
                evaluate = stack.enter_context(patch.object(suite, 'evaluate_job'))
                predict = stack.enter_context(patch.object(fixed, 'predict'))
                with self.assertRaisesRegex(ValueError, '344'):
                    suite.main()
                self.assertEqual(len(calls), 344)
                self.assertEqual(sum(kind == 'pretrain' for kind, _ in calls), 8)
                self.assertEqual(sum(kind == 'finetune' for kind, _ in calls), 112)
                self.assertEqual(gate.call_count, 1)
                self.assertEqual(set(loaded_ids) & test_ids, set())
                completed.assert_not_called()
                evaluate.assert_not_called()
                predict.assert_not_called()

    def previous_fixture(self, root):
        from scripts.test_cross_chem_matched import MatchedProtocolTests
        helper = MatchedProtocolTests()
        old, previous_workspace, workspace = root / 'v2', root / 'v3', root / 'v5'
        shape, manifest, _ = helper.existing_fixture(old)
        with patch.object(data, 'FEATURE_SHAPE', shape):
            _, protocols, audit = data.import_protocols(old, previous_workspace)
        cfg = previous.read_config()
        records = helper.pilot_records(cfg)
        selection = previous.select_budget(records, cfg)
        common.atomic_json(records, previous_workspace / 'pilot/scores.json')
        common.atomic_json(selection, previous_workspace / 'selection.json')
        identity = dict(schema=3, jobs=previous.formal_plan(), config=cfg,
                        existing_workspace=str(old), imported_data=audit,
                        code_hashes={relative: common.digest(ROOT / relative)
                                     for relative in inputs.FROZEN_MODEL_PATHS})
        common.atomic_json(identity, previous_workspace / 'queue.json')
        return old, previous_workspace, workspace, shape, manifest, protocols, selection

    def fingerprints(self, *roots):
        return {str(path): common.digest(path) for root in roots
                for path in Path(root).rglob('*') if path.is_file()}

    def test_prepare_reuses_112_actual_lists_pairs_and_rate_without_development(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, previous_workspace, workspace, shape, manifest, protocols, selected = self.previous_fixture(Path(tmp))
            before = self.fingerprints(old, previous_workspace)
            with patch.object(data, 'FEATURE_SHAPE', shape), \
                 patch.object(previous, 'pilot_split', side_effect=AssertionError('No new development split')), \
                 patch('scripts.transfer_cycle_soft_data.build_protocol',
                       side_effect=AssertionError('Do not regenerate actual target lists')):
                result, imported, receipt = inputs.prepare_inputs(previous_workspace, workspace)
                self.assertEqual(result, manifest)
                self.assertEqual(len(imported), 112)
                self.assertEqual(receipt['learning_rate'], selected['learning_rate'])
                self.assertFalse(receipt['split_generation_called'])
                for key, p in imported.items():
                    for field in ('source_ids', 'target_train_ids', 'target_test_ids', 'unused_target_ids'):
                        self.assertEqual(p[field], protocols[key][field])
                    raw = common.load(old / 'protocols' / key[0] / f'train_{key[1]}' / f'seed_{key[2]}.pt')
                    self.assertEqual(p['legacy_train_pairs'], raw['train_pairs'].tolist())
                    self.assertEqual(p['legacy_val_pairs'], raw['val_pairs'].tolist())
                    for field in ('train_pairs', 'val_pairs'):
                        expected = inputs.legacy_to_native(raw[field], 275, key[1]).tolist()
                        self.assertEqual(p[field], expected)
                    self.assertEqual(p['pair_index_convention'], 'source_major_target_interleaved')
                    self.assertEqual(p['legacy_pair_index_convention'], 'target_major_source_interleaved')
                self.assertEqual(len(list((workspace / 'protocols').rglob('*.json'))), 112)
                self.assertFalse((workspace / 'selection.json').exists())
                self.assertFalse((workspace / 'development').exists())
                inputs.verify_receipt(receipt)
                self.assertEqual(inputs.prepare_inputs(previous_workspace, workspace), (result, imported, receipt))
                for invalid in (old, previous_workspace, previous_workspace / 'child', Path(tmp)):
                    with self.assertRaises(ValueError):
                        inputs.prepare_inputs(previous_workspace, invalid)
            self.assertEqual(before, self.fingerprints(old, previous_workspace))

    def test_previous_rate_pair_protocol_and_source_fingerprints_cannot_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            old, previous_workspace, workspace, shape, manifest, _, _ = self.previous_fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                _, _, receipt = inputs.prepare_inputs(previous_workspace, workspace)
                for path in (previous_workspace / 'selection.json', previous_workspace / 'pilot/scores.json',
                             previous_workspace / 'protocols/LCO/train_1/seed_0.json'):
                    original = path.read_bytes()
                    value = json.loads(original.decode('utf-8'))
                    if path.name == 'selection.json':
                        value['learning_rate'] = .02
                    elif path.name == 'scores.json':
                        value[0]['validation_log_mse'] += 1.
                    else:
                        value['target_test_ids'].reverse()
                    common.atomic_json(value, path)
                    with self.assertRaises(ValueError):
                        inputs.verify_receipt(receipt)
                    with self.assertRaises(ValueError):
                        inputs.prepare_inputs(previous_workspace, workspace)
                    path.write_bytes(original)
                pair_file = old / 'protocols/LCO/train_1/seed_0.pt'
                original = pair_file.read_bytes()
                raw = common.load(pair_file)
                raw['train_pairs'] = raw['train_pairs'].flip(0)
                common.save(raw, pair_file)
                with self.assertRaises(ValueError):
                    inputs.verify_receipt(receipt)
                with self.assertRaises(ValueError):
                    inputs.prepare_inputs(previous_workspace, workspace)
                pair_file.write_bytes(original)
                cell = old / 'dataset' / manifest['records'][0]['feature_file']
                cell.write_bytes(cell.read_bytes() + b'synthetic corruption')
                with self.assertRaises(ValueError):
                    inputs.prepare_inputs(previous_workspace, workspace)

    def test_formal_plan_has_no_development_and_gate_requires_all_344_receipts(self):
        from scripts import run_cross_chem_earlystop_v5 as suite
        cfg = suite.read_config()
        self.assertEqual(cfg['early_stopping'], early.read_settings())
        self.assertNotIn('development', cfg)
        self.assertNotIn('budget', cfg)
        self.assertFalse(hasattr(suite, 'run_development'))
        jobs = suite.formal_plan()
        self.assertEqual(len(jobs), 344)
        self.assertEqual(sum(job['kind'] == 'pretrain' for job in jobs), 8)
        self.assertEqual(sum(job['kind'] != 'pretrain' for job in jobs), 336)
        self.assertEqual({job['seed'] for job in jobs}, set(range(8)))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(suite, 'validate_training_job', return_value={}) as validate:
                self.assertFalse(suite.all_training_complete(root, {}, jobs))
                self.assertEqual(validate.call_count, 0)
                for job in jobs:
                    path = suite.job_folder(root, job) / 'final.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'synthetic completed placeholder')
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
