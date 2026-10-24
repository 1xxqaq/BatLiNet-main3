"""Independent synthetic CPU checks; never read real training workspaces."""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, nullcontext, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import preview_cross_chem_matched as preview
from scripts import run_cross_chem_matched as suite
from scripts import cross_chem_matched_training as training
from scripts import cross_chem_matched_data as data
from scripts import run_three_tasks as common


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small_config(self):
        return dict(input_height=20, input_width=128, attention_channels=8,
                    attention_heads=2, head_hidden_channels=8, cycle_mixer_hidden=4,
                    encoder_dropout=0., attention_dropout=0.)

    def fingerprints(self, *roots):
        return {str(path): common.digest(path) for root in roots
                for path in Path(root).rglob('*') if path.is_file()}

    def fixture(self, root):
        """Real tiny training templates, saved as complete synthetic job evidence."""
        old, formal, output = root / 'old', root / 'formal', root / 'preview/LCO'
        generator = torch.Generator().manual_seed(37)
        parts = []
        for name, count in [('LFP', 3), ('LCO_train', 1), ('LCO_test', 2)]:
            parts.append(dict(feature=torch.randn(count, 6, 20, 128, generator=generator),
                              label=torch.linspace(250., 1000., count),
                              ids=[f'{name}_{i}' for i in range(count)]))
        source, target, testing = parts
        records = []
        for part in parts:
            for index, cell_id in enumerate(part['ids']):
                value = dict(cell_id=cell_id, chemistry='LFP' if cell_id.startswith('LFP') else 'LCO',
                             label_cycles=float(part['label'][index]), feature=part['feature'][index])
                relative = f'cells/{cell_id}.pt'
                common.save(value, old / 'dataset' / relative)
                records.append({key: value[key] for key in ('cell_id', 'chemistry', 'label_cycles')} |
                               dict(feature_file=relative, feature_sha256=common.digest(old / 'dataset' / relative)))
        common.atomic_json(dict(records=records), old / 'dataset/manifest.json')
        common.atomic_json({'synthetic': 'old_queue'}, old / 'queue.json')
        jobs = [dict(kind='pretrain', seed=seed) for seed in range(8)] + [
            dict(kind=kind, seed=seed, target='LCO', train_count=1)
            for seed in range(8) for kind in suite.KINDS]
        imported = dict(existing_manifest_sha256=common.digest(old / 'dataset/manifest.json'),
                        existing_queue_sha256=common.digest(old / 'queue.json'), protocols={})
        for seed in range(8):
            protocol = dict(source_ids=source['ids'], target_train_ids=target['ids'],
                            target_test_ids=testing['ids'], unused_target_ids=[])
            relative = f'LCO/train_1/seed_{seed}.pt'
            common.save(protocol, old / 'protocols' / relative)
            imported['protocols'][relative] = {'sha256': common.digest(old / 'protocols' / relative)}
            common.atomic_json(protocol, formal / 'protocols/LCO/train_1' / f'seed_{seed}.json')
        protected_code = 'scripts/cross_chem_matched_training.py'
        identity = dict(schema=3, jobs=jobs, existing_workspace=str(old),
                        imported_data=imported, config=suite.read_config(),
                        code_hashes={protected_code: common.digest(ROOT / protected_code)})
        selection = dict(pretrain_steps=1, target_steps=1, learning_rate=.001,
                         source_only=True, synthetic=True)
        common.atomic_json(identity, formal / 'queue.json')
        common.atomic_json(selection, formal / 'selection.json')
        common.atomic_json({'status': 'formal_training', 'completed_training': 154}, formal / 'status.json')
        templates = {}
        with redirect_stdout(io.StringIO()):
            for kind in ('pretrain', *suite.KINDS):
                templates[kind] = training.train_run(
                    source, source if kind == 'pretrain' else target, kind, 0, 1,
                    root / 'templates' / kind, {'synthetic_template': kind}, 'cpu',
                    initial_state=templates['pretrain']['state'] if kind == 'finetune' else None,
                    checkpoint_every=1, logical_batch=4, micro_batch=2,
                    model_config=self.small_config())
        for job in jobs:
            state = copy.deepcopy(templates[job['kind']])
            state['binding'] = suite.job_binding(identity, job, selection, formal)
            state['training_binding']['seed'] = job['seed']
            # An intentionally poor final seed tests that preview statistics
            # retain large but finite errors instead of filtering a run.
            if job['seed'] == 7 and job['kind'] != 'pretrain':
                state['state']['ori_head.net.4.bias'].fill_(8.)
            folder = suite.job_folder(formal, job)
            common.save(state, folder / 'final.pt')
            common.atomic_json(dict(status='trained', binding=state['binding'],
                                    training_binding=state['training_binding'], step=1,
                                    selection='fixed_final_step',
                                    final_sha256=common.digest(folder / 'final.pt')), folder / 'run.json')
        return old, formal, output, identity, jobs

    def patched(self, jobs):
        stack = ExitStack()
        stack.enter_context(patch.object(suite, 'formal_plan', return_value=jobs))
        stack.enter_context(patch.object(data, 'TRAIN_COUNTS', {'LCO': (1,)}))
        stack.enter_context(patch.object(data, 'FEATURE_SHAPE', (6, 20, 128)))
        stack.enter_context(patch.object(common, 'queue_lock', side_effect=lambda path: nullcontext()))
        original_make = training.make_model
        stack.enter_context(patch.object(training, 'make_model',
                                        side_effect=lambda kind, seed, device:
                                        original_make(kind, seed, device, self.small_config())))
        stack.enter_context(patch.object(training, 'train_run', side_effect=AssertionError('Preview must never train')))
        stack.enter_context(patch.object(suite, 'all_training_complete',
                                        side_effect=AssertionError('Preview must not require the 344-job formal gate')))
        return stack

    def test_full_lco_plan_requires_120_targets_and_eight_source_completions(self):
        identity = dict(schema=3, jobs=suite.formal_plan())
        jobs = preview.preview_plan(identity, 'LCO')
        self.assertEqual(len(jobs), 120)
        self.assertEqual(len(preview.preview_plan(identity, 'LCO', [16])), 24)
        self.assertEqual(len(preview.preview_plan(identity, 'NCA')), 96)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dependencies = [dict(kind='pretrain', seed=s) for s in range(8)]
            for job in dependencies + jobs:
                folder = suite.job_folder(root, job)
                folder.mkdir(parents=True, exist_ok=True)
                (folder / 'final.pt').write_bytes(b'synthetic final')
                (folder / 'run.json').write_text('{}', encoding='utf-8')
            with patch.object(suite, 'validate_training_job', return_value={}) as validate:
                fingerprints = preview.check_ready(root, identity, jobs)
                self.assertEqual(validate.call_count, 128)
                self.assertEqual(len(fingerprints), 256)
        for counts in ([1, 1], [3], []):
            with self.assertRaises(ValueError):
                preview.preview_plan(identity, 'LCO', counts)

    def test_preview_is_read_only_reuses_predictions_and_keeps_bad_seeds(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            old, formal, output, identity, jobs = self.fixture(Path(tmp))
            before = self.fingerprints(old, formal)
            with self.patched(jobs), patch.object(training, 'predict', wraps=training.predict) as predict:
                summary = preview.run_preview(formal, output, chunk=1)
                self.assertEqual(summary['expected_predictions'], 24)
                self.assertEqual(summary['completed_predictions'], 24)
                self.assertEqual(len(summary['groups']), 3)
                self.assertEqual(predict.call_count, 24)
                rows = json.loads((output / 'per_seed.json').read_text(encoding='utf-8'))
                for group in summary['groups']:
                    selected = [row for row in rows if row['kind'] == group['kind']]
                    self.assertEqual(group['seeds'], 8)
                    self.assertEqual({row['seed'] for row in selected}, set(range(8)))
                    self.assertAlmostEqual(group['metrics']['RMSE']['mean'],
                                           sum(row['RMSE'] for row in selected) / 8)
                    self.assertGreater(next(row['RMSE'] for row in selected if row['seed'] == 7), 10000.)
                predict.side_effect = AssertionError('Valid preview cache must not rerun inference')
                self.assertEqual(preview.run_preview(formal, output, chunk=1), summary)
            self.assertEqual(before, self.fingerprints(old, formal))
            self.assertFalse(list(formal.rglob('test.pt')))
            self.assertFalse(list(formal.rglob('evaluation.json')))

    def test_missing_or_corrupt_completion_stops_before_test_loading(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            old, formal, output, identity, jobs = self.fixture(Path(tmp))
            last = suite.job_folder(formal, jobs[-1])
            with self.patched(jobs), patch.object(data, 'load_part',
                                                  side_effect=AssertionError('No test data before all selected jobs are ready')):
                for name in ('final.pt', 'run.json'):
                    path = last / name
                    original = path.read_bytes()
                    path.unlink()
                    with self.assertRaises(ValueError):
                        preview.run_preview(formal, output)
                    self.assertFalse(output.exists())
                    path.write_bytes(original)
                path = last / 'final.pt'
                path.write_bytes(path.read_bytes() + b'synthetic corruption')
                with self.assertRaises(ValueError):
                    preview.run_preview(formal, output)
                self.assertFalse(output.exists())

    def test_preview_cache_or_weight_hash_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            old, formal, output, identity, jobs = self.fixture(Path(tmp))
            with self.patched(jobs):
                preview.run_preview(formal, output)
                test_path = output / 'train_1/direct_seed0/test.pt'
                # A process can stop after the durable prediction write but
                # before its SHA receipt.  Recovery still validates the saved
                # prediction against immutable weights and actual truth.
                evaluation = test_path.with_name('evaluation.json')
                evaluation.unlink()
                with patch.object(training, 'predict',
                                  side_effect=AssertionError('Durable predictions should be recovered')):
                    preview.run_preview(formal, output)
                self.assertTrue(evaluation.is_file())
                original = test_path.read_bytes()
                value = common.load(test_path)
                value['prediction_cycles'][0] += 1
                common.save(value, test_path)
                with self.assertRaises(ValueError):
                    preview.run_preview(formal, output)
                test_path.write_bytes(original)
                final = suite.job_folder(formal, jobs[-1]) / 'final.pt'
                value = common.load(final)
                key = next(iter(value['state']))
                value['state'][key] = value['state'][key] + .1
                common.save(value, final)
                with self.assertRaises(ValueError):
                    preview.run_preview(formal, output)

    def test_changed_selected_weights_during_inference_stop_before_prediction_save(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            old, formal, output, identity, jobs = self.fixture(Path(tmp))
            original_predict = training.predict
            altered = False

            def changed(*args, **kwargs):
                nonlocal altered
                result = original_predict(*args, **kwargs)
                if not altered:
                    current_job = next(job for job in jobs if job['kind'] != 'pretrain')
                    final = suite.job_folder(formal, current_job) / 'final.pt'
                    final.write_bytes(final.read_bytes() + b'synthetic concurrent replacement')
                    altered = True
                return result

            with self.patched(jobs), patch.object(training, 'predict', side_effect=changed):
                with self.assertRaisesRegex(ValueError, '预览过程中'):
                    preview.run_preview(formal, output)
            self.assertTrue(altered)
            self.assertFalse(list(output.rglob('test.pt')))

    def test_manifest_protocol_and_protected_code_fingerprints_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            old, formal, output, identity, jobs = self.fixture(Path(tmp))
            with self.patched(jobs):
                manifest, protocols = preview.read_protocols(formal, identity, 'LCO')
                self.assertEqual(len(protocols), 8)
                for path in (old / 'dataset/manifest.json', old / 'protocols/LCO/train_1/seed_0.pt',
                             formal / 'protocols/LCO/train_1/seed_0.json'):
                    original = path.read_bytes()
                    if path.suffix == '.pt':
                        value = common.load(path)
                        value['target_test_ids'] = ['wrong']
                        common.save(value, path)
                    else:
                        value = json.loads(original.decode('utf-8'))
                        value['target_test_ids'] = ['wrong']
                        common.atomic_json(value, path)
                    with self.assertRaises(ValueError):
                        preview.read_protocols(formal, identity, 'LCO')
                    path.write_bytes(original)
                bad_identity = copy.deepcopy(identity)
                key = next(iter(bad_identity['code_hashes']))
                bad_identity['code_hashes'][key] = 'invalid code fingerprint'
                with self.assertRaises(ValueError):
                    preview.read_protocols(formal, bad_identity, 'LCO')
                self.assertFalse(output.exists())

    def test_output_rejects_formal_old_nested_and_ancestor_paths(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            old, formal, output, identity, jobs = self.fixture(Path(tmp))
            before = self.fingerprints(old, formal)
            for invalid in (old, formal, old / 'child', formal / 'child', Path(tmp)):
                with self.assertRaises(ValueError):
                    preview.validate_output_path(formal, old, invalid)
                with self.assertRaises(ValueError):
                    preview.run_preview(formal, invalid)
            self.assertEqual(preview.validate_output_path(formal, old, output), output.resolve())
            self.assertEqual(before, self.fingerprints(old, formal))


if __name__ == '__main__':
    unittest.main(verbosity=2)
