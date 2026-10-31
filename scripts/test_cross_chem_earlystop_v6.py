"""Focused CPU checks for changing the monitoring cadence from 50 to 10.

The v5 training engine is reused unchanged. All generated runs are tiny,
temporary fixtures; no real battery data or formal queue is loaded.
"""
import copy
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cross_chem_earlystop_v5_training as early
from scripts import cross_chem_earlystop_v5_data as inputs
from scripts import cross_chem_matched_training as fixed
from scripts import run_cross_chem_earlystop_v5 as old_suite
from scripts import run_cross_chem_earlystop_v6 as suite
from scripts import run_three_tasks as common
from scripts import test_cross_chem_earlystop_v5 as fixtures


class TenStepMonitoringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def kwargs(self, kind='direct', old_cadence=False):
        settings = copy.deepcopy(suite.read_config()['early_stopping'])
        if old_cadence:
            settings['validate_every'] = 50
        value = fixtures.PerRunStoppingTests().kwargs(kind=kind, settings=settings)
        value.update(checkpoint_every=25, report_every=1000)
        return value

    @staticmethod
    def no_update(model, optimizer, source, target, kind, indices, device, micro_batch):
        # Only the 1000-step boundary check skips expensive gradients. The
        # engine still performs sampling, callbacks, monitoring and persistence.
        return dict(train_loss=0., own_loss=0., relation_loss=0., batch_size=len(indices))

    def test_only_profile_identity_and_monitor_interval_change(self):
        expected = copy.deepcopy(old_suite.read_config())
        expected.update(schema=6, experiment='cross_chem_earlystop_v6')
        expected['early_stopping']['validate_every'] = 10
        config = suite.read_config()
        self.assertEqual(config, expected)
        self.assertIs(suite.early, early)
        self.assertIs(suite.inputs, inputs)
        kwargs = suite._kwargs(config, .0003)
        self.assertEqual(kwargs['settings'], dict(max_steps=1000, min_stop=100,
                                                validate_every=10, patience=5))
        self.assertEqual(kwargs['lr'], .0003)
        self.assertNotIn('development', config)
        self.assertNotIn('budget', config)
        jobs = suite.formal_plan()
        self.assertEqual(jobs, old_suite.formal_plan())
        self.assertEqual(len(jobs), 344)
        self.assertEqual(sum(job['kind'] == 'pretrain' for job in jobs), 8)
        self.assertEqual(sum(job['kind'] != 'pretrain' for job in jobs), 336)

    def test_flat_monitor_keeps_step_10_best_but_waits_until_minimum_100(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', return_value=1.) as monitor:
            value = early.train_run(folder=Path(tmp), **self.kwargs())
            self.assertEqual(monitor.call_count, 10)
            self.assertEqual([row['step'] for row in value['history']], list(range(10, 101, 10)))
            self.assertEqual(value['best_step'], 10)
            self.assertEqual(value['trained_steps'], 100)
            self.assertEqual(value['bad_validations'], 9)
            self.assertEqual(value['stop_reason'], 'early_stopping')
            self.assertFalse(value['hit_max_steps'])
            self.assertEqual(value['monitor_policy'], 'training_cell_log_mse')
            self.assertFalse(value['independent_target_validation'])
            fixtures.PerRunStoppingTests().assert_nested_equal(value['state'], value['terminal']['best_state'])
            self.assertTrue(any(not torch.equal(tensor, value['terminal']['state'][key])
                                for key, tensor in value['state'].items()))

    def test_strict_tiny_improvement_at_60_resets_patience_and_stops_at_110(self):
        losses = [1.] * 5 + [1. - 1e-10] + [1.] * 5
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(early, 'monitoring_loss', side_effect=losses) as monitor:
            value = early.train_run(folder=Path(tmp), **self.kwargs())
            self.assertEqual(monitor.call_count, 11)
            self.assertEqual(value['history'][4]['bad_validations'], 4)
            self.assertTrue(value['history'][5]['improved'])
            self.assertEqual(value['history'][5]['bad_validations'], 0)
            self.assertEqual(value['best_step'], 60)
            self.assertEqual(value['trained_steps'], 110)
            self.assertEqual(value['bad_validations'], 5)
            self.assertEqual(value['stop_reason'], 'early_stopping')

    def test_full_cap_evaluates_100_times_and_maximum_wins_over_patience(self):
        # Best at 950; the five final ties hit patience at exactly 1000.
        losses = [1. / (index + 1) for index in range(95)] + [1. / 95] * 5
        observed_steps, callbacks = [], []

        def monitor(*args, **kwargs):
            observed_steps.append(len(callbacks) + 1)
            return losses[len(observed_steps) - 1]

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(fixed, 'train_step', side_effect=self.no_update), \
             patch.object(early, 'monitoring_loss', side_effect=monitor):
            value = early.train_run(folder=Path(tmp), step_callback=lambda step, model: callbacks.append(step),
                                    **self.kwargs())
            self.assertEqual(callbacks, list(range(1, 1001)))
            self.assertEqual(observed_steps, list(range(10, 1001, 10)))
            self.assertEqual([row['step'] for row in value['history']], observed_steps)
            self.assertEqual(value['best_step'], 950)
            self.assertEqual(value['trained_steps'], 1000)
            self.assertEqual(value['bad_validations'], 5)
            self.assertEqual(value['stop_reason'], 'max_steps')
            self.assertTrue(value['hit_max_steps'])

    def test_50_step_checkpoint_cannot_resume_under_10_step_settings(self):
        class Interruption(Exception):
            pass

        def interrupt(step, model):
            if step == 75:
                raise Interruption()

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(fixed, 'train_step', side_effect=self.no_update), \
             patch.object(early, 'monitoring_loss', return_value=1.):
            root = Path(tmp)
            with self.assertRaises(Interruption):
                early.train_run(folder=root, step_callback=interrupt, **self.kwargs(old_cadence=True))
            progress = common.load(root / 'progress.pt')
            self.assertEqual(progress['step'], 50)
            self.assertEqual([row['step'] for row in progress['history']], [50])
            before = {path.name: common.digest(path) for path in root.iterdir() if path.is_file()}
            with self.assertRaises(ValueError):
                early.train_run(folder=root, **self.kwargs())
            self.assertEqual(before, {path.name: common.digest(path) for path in root.iterdir() if path.is_file()})

    def test_completed_10_step_run_rejects_loading_as_old_50_step_policy(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(fixed, 'train_step', side_effect=self.no_update), \
             patch.object(early, 'monitoring_loss', return_value=1.):
            kwargs = self.kwargs()
            early.train_run(folder=Path(tmp), **kwargs)
            old_settings = dict(kwargs['settings'], validate_every=50)
            with self.assertRaises(ValueError):
                early.load_completed(Path(tmp), kwargs['binding'], 'direct', old_settings)

    def test_new_workspace_allowed_and_old_v5_sibling_remains_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / 'BatLiNet-cross-chem-earlystop-v6'
            old_root = parent / 'BatLiNet-cycle-mixer'
            previous = old_root / 'workspaces/cross_chem_matched_v3'
            old_data = old_root / 'workspaces/cross_chem_two_models_v2'
            workspace = root / 'workspaces/cross_chem_earlystop_v6'
            v5 = parent / 'BatLiNet-cross-chem-earlystop-v5/workspaces/cross_chem_earlystop_v5'
            v5.mkdir(parents=True)
            (v5 / '.queue.lock').write_text('synthetic existing lock', encoding='utf-8')
            with patch.object(suite, 'ROOT', root):
                protected = suite.protected_folders(previous, old_data, workspace)
                self.assertEqual(inputs.separate_output(workspace, *protected), workspace.resolve())
                self.assertNotIn(workspace.resolve(), protected)
                self.assertIn(v5.resolve(), protected)
                self.assertTrue((v5 / '.queue.lock').is_file())
                self.assertIn(root / 'workspaces/cross_chem_earlystop_v5', protected)
                wrong = suite.protected_folders(previous, old_data, v5)
                with self.assertRaises(ValueError):
                    inputs.separate_output(v5, *wrong)
                for invalid in (previous, old_data, v5 / 'child'):
                    with self.assertRaises(ValueError):
                        inputs.separate_output(invalid, *protected)

    def test_fingerprints_include_new_profile_and_reused_training_engine(self):
        hashes = suite.code_hashes()
        for relative in ('scripts/run_cross_chem_earlystop_v6.py',
                         'configs/cross_chem_earlystop_v6/protocol.yaml',
                         'scripts/cross_chem_earlystop_v5_training.py',
                         'scripts/cross_chem_earlystop_v5_data.py'):
            self.assertIn(relative, hashes)
            self.assertEqual(hashes[relative], common.digest(suite.ROOT / relative))

    def test_report_rejects_old_identity_before_reading_any_source_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            old_identity = dict(schema=5, config=old_suite.read_config())
            stale_config = copy.deepcopy(suite.read_config())
            stale_config['early_stopping']['validate_every'] = 50
            for identity in (old_identity, dict(schema=6, config=stale_config)):
                common.atomic_json(identity, workspace / 'queue.json')
                with patch.object(suite, 'load_protocols', side_effect=AssertionError('Wrong profile may not read sources')):
                    with self.assertRaises(ValueError):
                        suite.report(workspace)


if __name__ == '__main__':
    unittest.main(verbosity=2)
