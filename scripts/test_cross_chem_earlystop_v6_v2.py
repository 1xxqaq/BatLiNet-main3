"""Synthetic checks for running v6 directly from saved v2 materials.

These fixtures contain the full 403-cell / 112-protocol topology with tiny
curves. No real battery data, formal training or original results are changed.
"""
import argparse
import copy
import io
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cross_chem_earlystop_v6_data as inputs
from scripts import cross_chem_matched_data as data
from scripts import run_cross_chem_matched as previous
from scripts import run_cross_chem_earlystop_v6 as suite
from scripts import run_three_tasks as common
from scripts import test_cross_chem_matched as fixtures


class V2InputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self, root):
        existing = root / 'cross_chem_two_models_v2'
        workspace = root / 'cross_chem_earlystop_v6_v2'
        shape, manifest, protocols = fixtures.MatchedProtocolTests().existing_fixture(existing)
        return existing, workspace, shape, manifest, protocols

    @staticmethod
    def fingerprints(root):
        return {str(path.relative_to(root)): common.digest(path)
                for path in root.rglob('*') if path.is_file()}

    def test_missing_v3_does_not_prevent_readonly_import_of_actual_v2_ids_and_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing, workspace, shape, manifest, actual = self.fixture(root)
            before = self.fingerprints(existing)
            with (patch.object(data, 'FEATURE_SHAPE', shape),
                  patch.object(previous, 'select_budget', side_effect=AssertionError('No v3 learning-rate selection')),
                  patch('scripts.transfer_cycle_soft_data.build_protocol',
                        side_effect=AssertionError('Do not regenerate saved target lists'))):
                checked, protocols, receipt = inputs.prepare_existing(existing, workspace, .001)
                self.assertEqual(checked, manifest)
                self.assertEqual(len(protocols), 112)
                self.assertEqual(receipt['input_mode'], 'v2_fixed_learning_rate')
                self.assertEqual(receipt['learning_rate'], .001)
                self.assertEqual(receipt['learning_rate_source'], 'explicit_fixed_value_without_v3_selection')
                self.assertFalse(receipt['v3_selection_available'])
                self.assertTrue(receipt['legacy_pair_split_reused_for_training'])
                self.assertEqual(Path(receipt['data_workspace']), existing.resolve())
                self.assertEqual(receipt['development_runs'], 0)
                self.assertEqual(receipt['lfp_validation_holdout'], 0)
                self.assertFalse(receipt['reuse_previous_weights'])
                self.assertFalse(receipt['split_generation_called'])
                for key, imported in protocols.items():
                    saved = actual[key]
                    for field in inputs.ID_FIELDS:
                        self.assertEqual(imported[field], saved[field])
                    for field in ('train_pairs', 'val_pairs'):
                        self.assertEqual(imported['legacy_' + field], saved[field].tolist())
                        native = torch.tensor(imported[field], dtype=torch.int64)
                        ti, si = native % key[1], native // key[1]
                        self.assertEqual(ti.tolist(), (saved[field] // 275).tolist())
                        self.assertEqual(si.tolist(), (saved[field] % 275).tolist())
                    train, valid = inputs.native_pairs(imported)
                    self.assertFalse(set(train.tolist()) & set(valid.tolist()))
                    self.assertEqual(len(train), int(key[1] * 275 * .8))
                inputs.verify_receipt(receipt, workspace)
                self.assertEqual(inputs.prepare_existing(existing, workspace, .001),
                                 (checked, protocols, receipt))
            self.assertEqual(self.fingerprints(existing), before)
            self.assertEqual(len(list((workspace / 'protocols').rglob('*.json'))), 112)
            self.assertFalse((root / 'cross_chem_matched_v3').exists())
            for name in ('selection.json', 'pilot', 'development'):
                self.assertFalse((existing / name).exists())
                self.assertFalse((workspace / name).exists())

    def test_fixed_learning_rate_must_be_explicit_finite_and_positive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing, workspace, shape, _, _ = self.fixture(root)
            with patch.object(data, 'FEATURE_SHAPE', shape):
                for value in (None, 0., -.001, float('nan'), float('inf')):
                    with self.subTest(learning_rate=value), self.assertRaises((ValueError, TypeError)):
                        inputs.prepare_existing(existing, workspace, value)
            self.assertFalse((workspace / 'source_receipt.json').exists())

    def test_a_different_fixed_rate_cannot_resume_the_same_workspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, workspace, shape, _, _ = self.fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                inputs.prepare_existing(existing, workspace, .001)
                before = self.fingerprints(workspace)
                with self.assertRaises(ValueError):
                    inputs.prepare_existing(existing, workspace, .0003)
                self.assertEqual(self.fingerprints(workspace), before)

    def test_changed_original_queue_manifest_and_pair_fingerprints_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, workspace, shape, _, _ = self.fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                _, _, receipt = inputs.prepare_existing(existing, workspace, .001)
                for path in (existing / 'queue.json', existing / 'dataset/manifest.json',
                             existing / 'protocols/LCO/train_1/seed_0.pt'):
                    before = path.read_bytes()
                    path.write_bytes(before + b'\n')
                    with self.subTest(path=path), self.assertRaises(ValueError):
                        inputs.verify_receipt(receipt, workspace)
                    path.write_bytes(before)
                path = existing / 'protocols/LCO/train_1/seed_0.pt'
                before = path.read_bytes()
                value = common.load(path)
                value['train_pairs'] = value['train_pairs'].flip(0)
                common.save(value, path)
                with self.assertRaises(ValueError):
                    inputs.prepare_existing(existing, workspace, .001)
                path.write_bytes(before)

    def test_changed_cell_cache_is_rejected_during_resume_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, workspace, shape, manifest, _ = self.fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                inputs.prepare_existing(existing, workspace, .001)
                path = existing / 'dataset' / manifest['records'][0]['feature_file']
                path.write_bytes(path.read_bytes() + b'synthetic corruption')
                with self.assertRaises(ValueError):
                    inputs.prepare_existing(existing, workspace, .001)

    def test_new_pair_order_or_receipt_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, workspace, shape, _, _ = self.fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                _, _, receipt = inputs.prepare_existing(existing, workspace, .001)
                path = workspace / 'protocols/LCO/train_1/seed_0.json'
                original = path.read_bytes()
                value = inputs.read_json(path)
                value['train_pairs'].reverse()
                common.atomic_json(value, path)
                with self.assertRaises(ValueError):
                    inputs.verify_receipt(receipt, workspace)
                with self.assertRaises(ValueError):
                    inputs.prepare_existing(existing, workspace, .001)
                path.write_bytes(original)
                value = copy.deepcopy(receipt)
                value['learning_rate'] = .0003
                common.atomic_json(value, workspace / 'source_receipt.json')
                with self.assertRaises(ValueError):
                    inputs.verify_receipt(receipt, workspace)

    def test_output_must_not_overlap_original_v2_materials(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, workspace, shape, _, _ = self.fixture(Path(tmp))
            before = self.fingerprints(existing)
            with patch.object(data, 'FEATURE_SHAPE', shape):
                for invalid in (existing, existing / 'child', Path(tmp)):
                    with self.assertRaises(ValueError):
                        inputs.prepare_existing(existing, invalid, .001)
            self.assertEqual(self.fingerprints(existing), before)

    def test_cli_v2_source_uses_separate_default_workspace_and_explicit_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            existing = root / 'old/workspaces/cross_chem_two_models_v2'
            parser = argparse.ArgumentParser()
            args = SimpleNamespace(previous_workspace=None, existing_workspace=existing,
                                   workspace=None, learning_rate=.001)
            with patch.object(suite, 'ROOT', root / 'code'):
                source = suite.resolve_source(args, parser)
            self.assertEqual(source['input_mode'], inputs.V2_MODE)
            self.assertEqual(source['source_workspace'], existing.resolve())
            self.assertEqual(source['old_data'], existing.resolve())
            self.assertEqual(source['workspace'], (root / 'code/workspaces/cross_chem_earlystop_v6_v2').resolve())
            self.assertEqual(source['learning_rate'], .001)
            for invalid in (None, 0., -.1, float('nan'), float('inf')):
                args.learning_rate = invalid
                with redirect_stdout(io.StringIO()), patch('sys.stderr', new=io.StringIO()):
                    with self.assertRaises(SystemExit):
                        suite.resolve_source(args, parser)

    def test_v2_profile_preserves_training_rules_and_report_needs_no_v3_files(self):
        cfg = suite.read_config(inputs.V2_MODE)
        expected = copy.deepcopy(suite.read_config())
        expected['learning_rate'] = 'explicit_fixed_value_without_v3_selection'
        self.assertEqual(cfg, expected)
        self.assertEqual(cfg['early_stopping'], dict(max_steps=1000, min_stop=100,
                                                   validate_every=10, patience=5))
        with tempfile.TemporaryDirectory() as tmp:
            existing, workspace, shape, _, _ = self.fixture(Path(tmp))
            with patch.object(data, 'FEATURE_SHAPE', shape):
                _, _, receipt = inputs.prepare_existing(existing, workspace, .001)
                identity = dict(schema=6, config=cfg, source_receipt=receipt)
                common.atomic_json(identity, workspace / 'queue.json')
                stream = io.StringIO()
                with redirect_stdout(stream), patch.object(previous, 'select_budget',
                        side_effect=AssertionError('Report cannot search a missing v3')):
                    suite.report(workspace)
                self.assertIn('0/344', stream.getvalue())
                self.assertIn('0.001', stream.getvalue())
            self.assertFalse((Path(tmp) / 'cross_chem_matched_v3').exists())

    def test_main_prepare_only_reads_v2_source_and_never_requires_v3_or_trains(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            existing, workspace, shape, _, _ = self.fixture(root)
            real_load = data.load_part
            loaded = []

            def load(ids, manifest, folder, **kwargs):
                self.assertTrue(all(cell_id.startswith('LFP_') for cell_id in ids))
                loaded.extend(ids)
                return real_load(ids, manifest, folder, **kwargs)

            argv = ['run_cross_chem_earlystop_v6.py', '--existing-workspace', str(existing),
                    '--learning-rate', '0.001', '--workspace', str(workspace),
                    '--device', 'cuda:0', '--prepare-only']
            with ExitStack() as stack:
                for context in (
                    patch.object(sys, 'argv', argv),
                    patch.object(suite, 'os', SimpleNamespace(name='posix')),
                    patch.object(data, 'FEATURE_SHAPE', shape),
                    patch.object(data, 'load_part', side_effect=load),
                    patch.object(torch.cuda, 'is_available', return_value=True),
                    patch.object(torch, 'set_num_threads'),
                    patch.object(common, 'seed_all'),
                    patch.object(common, 'runtime', return_value={}),
                    patch.object(common, 'queue_lock', side_effect=lambda folder: nullcontext()),
                    patch.object(previous, 'legacy_read_lock', side_effect=lambda folder: nullcontext()),
                    patch.object(suite, 'protected_folders', return_value=[existing]),
                    patch.object(suite, 'prepare_inputs', side_effect=AssertionError('No v3 route')),
                    patch.object(suite, 'code_hashes', return_value={}),
                    patch.object(suite.subprocess, 'check_output', return_value='synthetic_commit'),
                    patch.object(previous, 'preflight', return_value={'synthetic': 'passed'}),
                ):
                    stack.enter_context(context)
                prepared = stack.enter_context(patch.object(inputs, 'prepare_existing', wraps=inputs.prepare_existing))
                train = stack.enter_context(patch.object(suite.early, 'train_run'))
                evaluate = stack.enter_context(patch.object(suite, 'evaluate_job'))
                suite.main()
                prepared.assert_called_once_with(existing.resolve(), workspace.resolve(), .001)
                train.assert_not_called()
                evaluate.assert_not_called()
            self.assertEqual(len(loaded), 275)
            self.assertEqual(len(set(loaded)), 275)
            identity = inputs.read_json(workspace / 'queue.json')
            self.assertEqual(identity['schema'], 6)
            self.assertEqual(identity['source_receipt']['input_mode'], inputs.V2_MODE)
            self.assertEqual(identity['source_receipt']['learning_rate'], .001)
            self.assertEqual(identity['config'], suite.read_config(inputs.V2_MODE))
            self.assertEqual(inputs.read_json(workspace / 'status.json')['status'], 'prepared')
            self.assertFalse((root / 'cross_chem_matched_v3').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
