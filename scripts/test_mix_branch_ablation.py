"""Synthetic CPU checks for branch-ablation training, recovery and readout.

These checks never read the experimental battery cache or start a real run.
"""
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix_branch_ablation as suite
from src.data.databundle import Dataset


class ScalarPredictor(torch.nn.Module):
    """A mean loss whose exact logical-batch SGD update is easy to check."""
    def __init__(self, mode='full', random_draws=0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.branch_mode = mode
        self.requires_reference = mode != 'self_only'
        self.train_batch_size = 4
        self.train_support_size = 2
        self.random_draws = random_draws

    def get_support_set(self, feature, pool, labels, fixed_indices=None,
                        support_is_prepared=True):
        return pool[fixed_indices], labels[fixed_indices]

    def forward(self, feature, label=None, support_feature=None,
                support_label=None, return_loss=False, prepared=True):
        if self.random_draws:
            torch.rand(self.random_draws)
        prediction = self.weight * feature.reshape(len(feature), -1).mean(1)
        if return_loss:
            return 0.5 * (prediction - label).square().mean()
        return prediction


class BranchRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def fixture(self):
        generator = torch.Generator().manual_seed(23)
        mean, scale = torch.tensor(5.3), torch.tensor(.5)
        result = dict(mean=mean, scale=scale)
        for part, n in (('train', 5), ('test', 3)):
            z = torch.linspace(-.2, .8, n)
            result[part] = dict(
                feature=torch.randn(n, 6, 20, 128, generator=generator),
                label_z=z, label=(z * scale + mean).exp(),
                ids=[f'{part}_{i}' for i in range(n)])
        return result

    def small(self, mode='full'):
        cfg = copy.deepcopy(suite.model_config('mix_20', mode))
        cfg['model'].update(input_width=128, train_batch_size=4, epochs=2,
                            attention_channels=8, attention_heads=2,
                            head_hidden_channels=8, evaluate_freq=1)
        cfg['runner'].update(micro_batch_size=2, checkpoint_every=1, pair_chunk=8)
        return cfg

    def binding(self, cfg, mode, seed=3):
        return dict(config=cfg, task='synthetic', model=mode, seed=seed,
                    data_sha256='synthetic-data', protocol_sha256='synthetic-protocol',
                    queue=dict(synthetic=True))

    def indices(self):
        return torch.arange(96).reshape(3, 32) % 5

    def test_reference_rng_isolated_from_dropout_and_branch_mode(self):
        ds = Dataset(torch.arange(1, 6).float().reshape(5, 1),
                     torch.tensor([.1, .4, .2, .5, .9]))
        fingerprints = []
        states = []
        for mode, draws in (('full', 0), ('full', 97), ('reference_only', 211),
                            ('self_only', 401)):
            model = ScalarPredictor(mode, draws)
            generator = suite.reference_generator(9)
            optimizer = torch.optim.SGD(model.parameters(), lr=.03)
            row = suite.train_epoch(model, optimizer, ds, 2, generator)
            fingerprints.append(row['reference_fingerprint'])
            states.append(generator.get_state())
        self.assertEqual(len(set(fingerprints)), 1)
        for state in states[1:]:
            self.assertTrue(torch.equal(state, states[0]))
        expected = torch.Generator(device='cpu').manual_seed(310009)
        torch.randint(len(ds), (4, 2), generator=expected)
        torch.randint(len(ds), (1, 2), generator=expected)
        self.assertTrue(torch.equal(states[0], expected.get_state()))

    def test_micro_batches_and_tail_use_actual_logical_batch_size(self):
        ds = Dataset(torch.arange(1, 6).float().reshape(5, 1),
                     torch.tensor([.1, .4, .2, .5, .9]))
        expected = ScalarPredictor()
        optimizer = torch.optim.SGD(expected.parameters(), lr=.03)
        losses = []
        for start in (0, 4):
            x, y = ds.feature[start:start + 4], ds.label[start:start + 4]
            optimizer.zero_grad()
            loss = expected(x, y, return_loss=True)
            losses.append(float(loss.detach()) * len(x))
            loss.backward()
            optimizer.step()
        for micro in (1, 2, 3, 128):
            actual = ScalarPredictor()
            row = suite.train_epoch(actual, torch.optim.SGD(actual.parameters(), lr=.03),
                                    ds, micro, suite.reference_generator(1))
            torch.testing.assert_close(actual.weight, expected.weight, atol=1e-7, rtol=0)
            self.assertAlmostEqual(row['train_loss'], sum(losses) / 5, places=6)
            self.assertEqual(row['optimizer_steps'], 2)
            self.assertEqual(row['micro_batches'], (4 + micro - 1) // micro + 1)

    def test_self_only_training_and_prediction_never_fetch_references(self):
        data, cfg = self.fixture(), self.small('self_only')
        model = suite.build_branch_model(cfg['model'], 7, 'self_only')
        ds = suite.prepared(model, data['train'], 'cpu')
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        with patch.object(model, 'get_support_set', side_effect=AssertionError('reference access')):
            suite.train_epoch(model, optimizer, ds, 2, suite.reference_generator(7))
            first = suite.predict(model, data, self.indices(), 'cpu', 8)
            changed = copy.deepcopy(data)
            changed['train']['feature'].fill_(float('nan'))
            changed['train']['label_z'].fill_(float('nan'))
            second = suite.predict(model, changed, self.indices(), 'cpu', 8)
        torch.testing.assert_close(first['prediction'], second['prediction'], atol=0, rtol=0)
        self.assertIsNone(first['diagnostics']['y_sup'])
        self.assertIsNone(first['diagnostics']['y_sup_agg'])
        self.assertTrue(torch.equal(first['diagnostics']['support_index'], self.indices()))

    def test_chunked_references_aggregate_all_32_with_lower_median(self):
        data = self.fixture()
        data['train'] = dict(
            feature=data['train']['feature'][torch.arange(32) % 5].clone(),
            label_z=torch.arange(32).float(),
            label=(torch.arange(32).float() * data['scale'] + data['mean']).exp(),
            ids=[f'train_{i}' for i in range(32)])
        indices = torch.arange(32).repeat(3, 1)
        for mode in ('full', 'reference_only'):
            cfg = self.small(mode)
            model = suite.build_branch_model(cfg['model'], 8, mode)
            with torch.no_grad():
                model.support_head.net[-1].weight.zero_()
                model.support_head.net[-1].bias.zero_()
            reference = suite.predict(model, data, indices, 'cpu', 32)
            for chunk in (1, 7, 8, 13):
                actual = suite.predict(model, data, indices, 'cpu', chunk)
                torch.testing.assert_close(actual['prediction'], reference['prediction'],
                                           atol=1e-6, rtol=1e-5)
                sup = actual['diagnostics']['y_sup']
                self.assertEqual(tuple(sup.shape), (3, 32))
                expected = sup.sort(1).values[:, 15]
                self.assertTrue(torch.equal(expected, torch.full((3,), 15.)))
                torch.testing.assert_close(actual['diagnostics']['y_sup_agg'], expected,
                                           atol=0, rtol=0)
                self.assertTrue(torch.equal(actual['diagnostics']['support_index'], indices))
            if mode == 'reference_only':
                self.assertIsNone(reference['diagnostics']['y_ori'])

    def test_interrupted_run_resumes_weights_and_reference_rng_exactly(self):
        data, cfg = self.fixture(), self.small()
        binding = self.binding(cfg, 'full')
        with tempfile.TemporaryDirectory() as tmp:
            complete, interrupted = Path(tmp) / 'complete', Path(tmp) / 'interrupted'
            suite.train_run(data, cfg, 3, complete, binding, 'cpu')
            original_save = suite.common.save
            crashed = False

            def crash(value, path):
                nonlocal crashed
                original_save(value, path)
                if Path(path).name == 'progress.pt' and value.get('epoch') == 1 and not crashed:
                    crashed = True
                    raise RuntimeError('synthetic crash')

            with patch.object(suite.common, 'save', side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, 'synthetic crash'):
                    suite.train_run(data, cfg, 3, interrupted, binding, 'cpu')
            with (interrupted / 'train.jsonl').open('a') as log:
                log.write('{"epoch":2,"train_loss":0.1}\n{"epoch":')
            suite.train_run(data, cfg, 3, interrupted, binding, 'cpu')
            one = suite.common.load(complete / 'final.pt')
            two = suite.common.load(interrupted / 'final.pt')
            for key, value in one['state'].items():
                self.assertTrue(torch.equal(value, two['state'][key]), key)
            one = suite.common.load(complete / 'progress.pt')['rng']
            two = suite.common.load(interrupted / 'progress.pt')['rng']
            for key in ('torch', 'references'):
                self.assertTrue(torch.equal(one[key], two[key]), key)
            self.assertTrue(list(interrupted.glob('interrupted_log_*.jsonl')))
            sha = suite.common.digest(interrupted / 'final.pt')
            suite.train_run(data, cfg, 3, interrupted, binding, 'cpu')
            self.assertEqual(sha, suite.common.digest(interrupted / 'final.pt'))
            with self.assertRaises(ValueError):
                suite.train_run(data, cfg, 3, interrupted, dict(binding, seed=4), 'cpu')

    def trained_result(self, data, root, mode, task='synthetic'):
        cfg = self.small(mode)
        binding = self.binding(cfg, mode)
        binding['task'] = task
        folder = root / mode
        suite.train_run(data, cfg, 3, folder, binding, 'cpu')
        suite.evaluate_run(data, cfg, 3, folder, binding, self.indices(), 'cpu')
        result = suite.common.load(folder / 'test.pt')
        result['test_sha256'] = suite.common.digest(folder / 'test.pt')
        return cfg, binding, folder, result

    def test_independent_ensemble_is_log_space_mean_and_binds_both_sources(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, _, _, own = self.trained_result(data, root, 'self_only')
            _, _, _, ref = self.trained_result(data, root, 'reference_only')
            result = suite.ensemble_result(own, ref)
            expected_z = .5 * own['prediction'] + .5 * ref['prediction']
            expected_cycles = (own['prediction_cycles'] * ref['prediction_cycles']).sqrt()
            torch.testing.assert_close(result['prediction'], expected_z, atol=0, rtol=0)
            torch.testing.assert_close(result['prediction_cycles'], expected_cycles, atol=1e-4, rtol=1e-6)
            self.assertEqual(result['source_checkpoint_sha256'],
                             dict(self_only=own['checkpoint_sha256'],
                                  reference_only=ref['checkpoint_sha256']))
            self.assertEqual(result['source_test_sha256'],
                             dict(self_only=own['test_sha256'], reference_only=ref['test_sha256']))
            changed = copy.deepcopy(ref)
            changed['binding']['data_sha256'] = 'different-data'
            with self.assertRaises(ValueError):
                suite.ensemble_result(own, changed)

    def test_saved_prediction_reference_and_checkpoint_tampering_are_rejected(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            cfg, binding, folder, result = self.trained_result(data, Path(tmp), 'reference_only')
            clean = suite.common.load(folder / 'test.pt')
            for field in ('reference', 'prediction', 'checkpoint', 'coherent_prediction'):
                changed = copy.deepcopy(clean)
                if field == 'reference':
                    changed['diagnostics']['support_index'][0, 0] = 4
                elif field == 'prediction':
                    changed['prediction'][0] += .25
                elif field == 'checkpoint':
                    changed['checkpoint_sha256'] = 'different-weight'
                else:
                    changed['diagnostics']['y_sup'] = changed['diagnostics']['y_sup'] + .25
                    changed['diagnostics']['y_sup_agg'] = changed['diagnostics']['y_sup'].median(1).values
                    changed['prediction'] = changed['diagnostics']['y_sup_agg'].clone()
                    changed['prediction_cycles'] = (
                        changed['prediction'] * changed['label_sample_std']
                        + changed['label_mean']).exp()
                    changed['scores'] = suite.common.scores(
                        changed['prediction_cycles'], changed['truth_cycles'])
                suite.common.save(changed, folder / 'test.pt')
                with self.assertRaises((ValueError, AssertionError)):
                    suite.evaluate_run(data, cfg, 3, folder, binding, self.indices(), 'cpu')
            suite.common.save(clean, folder / 'test.pt')
            self.assertEqual(suite.evaluate_run(data, cfg, 3, folder, binding, self.indices(), 'cpu')['model'],
                             'reference_only')

    def queue_fixture(self, workspace):
        configs = {task:{mode:self.small(mode) for mode in suite.TRAIN_MODES}
                   for task in suite.TASKS}
        for task in suite.TASKS:
            suite.common.save(self.fixture(), workspace/'datasets'/task/'data.pt')
            for seed in suite.SEEDS:
                suite.common.save(dict(indices=self.indices()),
                                  workspace/'protocols'/task/f'seed_{seed}.pt')
        identity = dict(synthetic=True)
        shas = {task:suite.common.digest(workspace/'datasets'/task/'data.pt')
                for task in suite.TASKS}
        bindings = {(task, mode, seed):suite.make_binding(
            task, mode, seed, configs[task][mode], identity, shas[task], workspace)
            for task in suite.TASKS for mode in suite.TRAIN_MODES for seed in suite.SEEDS}
        return configs, identity, shas, bindings

    def test_all_48_trainings_finish_before_any_test_or_ensemble(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            configs, identity, shas, bindings = self.queue_fixture(workspace)
            trained, evaluated = [], []

            def train(data, cfg, seed, folder, binding, device):
                trained.append((binding['task'], binding['model'], seed))

            def row(task, mode, seed):
                self.assertEqual(len(trained), 48)
                evaluated.append((task, mode, seed))
                return dict(task=task, model=mode, seed=seed,
                            RMSE=1., MAE=1., MAPE=.01, ACC15=1.)

            def evaluate(data, cfg, seed, folder, binding, indices, device):
                return row(binding['task'], binding['model'], seed)

            def ensemble(data, seed, folder, task, indices):
                return row(task, 'independent_ensemble', seed)

            def gate(*args):
                self.assertEqual(len(trained), 48)
                return bindings

            with redirect_stdout(io.StringIO()), \
                    patch.object(suite, 'train_run', side_effect=train), \
                    patch.object(suite, 'evaluate_run', side_effect=evaluate), \
                    patch.object(suite, 'evaluate_ensemble', side_effect=ensemble), \
                    patch.object(suite, 'check_all_trainings', side_effect=gate):
                suite.execute_queue(workspace, configs, identity, shas, 'cpu')
            self.assertEqual(len(set(trained)), 48)
            self.assertEqual(len(set(evaluated)), 64)
            status = json.loads((workspace/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'complete')
            self.assertFalse(status['independent_ensemble_trained'])

    def test_last_training_failure_prevents_all_test_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            configs, identity, shas, _ = self.queue_fixture(workspace)
            calls = 0

            def train(*args):
                nonlocal calls
                calls += 1
                if calls == 48:
                    raise RuntimeError('last training failed')

            with redirect_stdout(io.StringIO()), \
                    patch.object(suite, 'train_run', side_effect=train), \
                    patch.object(suite, 'evaluate_run') as evaluate, \
                    patch.object(suite, 'evaluate_ensemble') as ensemble, \
                    patch.object(suite, 'check_all_trainings') as gate:
                with self.assertRaisesRegex(RuntimeError, 'last training failed'):
                    suite.execute_queue(workspace, configs, identity, shas, 'cpu')
            evaluate.assert_not_called()
            ensemble.assert_not_called()
            gate.assert_not_called()

    def test_training_gate_rejects_different_reference_fingerprints(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            task, seed = 'mix_20', 3
            suite.common.save(dict(indices=self.indices()),
                              workspace/'protocols'/task/f'seed_{seed}.pt')
            configs = {task:{mode:self.small(mode) for mode in suite.TRAIN_MODES}}
            identity, shas = dict(synthetic=True), {task:'synthetic-data'}
            for mode in suite.TRAIN_MODES:
                binding = suite.make_binding(task, mode, seed, configs[task][mode],
                                             identity, shas[task], workspace)
                suite.train_run(data, configs[task][mode], seed,
                                workspace/task/f'{mode}_seed{seed}', binding, 'cpu')
            with patch.object(suite, 'TASKS', (task,)), patch.object(suite, 'SEEDS', (seed,)):
                bindings = suite.check_all_trainings(workspace, configs, identity, shas)
                self.assertEqual(len(bindings), 3)
                path = workspace/task/f'reference_only_seed{seed}'/'train.jsonl'
                records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
                records[-1]['reference_fingerprint'] = 'f' * 64
                path.write_text(''.join(json.dumps(row) + '\n' for row in records), encoding='utf-8')
                with self.assertRaisesRegex(ValueError, '参考'):
                    suite.check_all_trainings(workspace, configs, identity, shas)

    def test_report_is_read_only_and_never_starts_training_or_inference(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            cfg, binding, folder, result = self.trained_result(
                data, workspace/'mix_20', 'self_only', task='mix_20')
            target = workspace/'mix_20'/'self_only_seed3'
            folder.rename(target)
            suite.common.save(dict(indices=self.indices()), workspace/'protocols'/'mix_20'/'seed_3.pt')
            row = dict(task='mix_20', model='self_only', seed=3, **result['scores'])
            suite.common.atomic_json([row], workspace/'per_seed.json')

            def snapshot():
                return {p.relative_to(workspace).as_posix():suite.common.digest(p)
                        for p in workspace.rglob('*') if p.is_file()}

            before = snapshot()
            with redirect_stdout(io.StringIO()), \
                    patch.object(suite, 'train_run', side_effect=AssertionError('training')), \
                    patch.object(suite, 'predict', side_effect=AssertionError('inference')), \
                    patch.object(suite.common, 'save', side_effect=AssertionError('file write')), \
                    patch.object(suite.common, 'atomic_json', side_effect=AssertionError('file write')):
                self.assertEqual(suite.report_workspace(workspace), [])
            self.assertEqual(before, snapshot())


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(BranchRunnerTests))
    if '--report' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--report') + 1])
        suite.common.atomic_json(dict(
            status='passed' if result.wasSuccessful() else 'failed', tests=result.testsRun,
            failures=len(result.failures), errors=len(result.errors), synthetic_only=True,
            formal_training_started=False, cuda_checked=False,
            runtime=suite.base.runtime('cpu'), source_code=suite.sources()), path)
    sys.exit(0 if result.wasSuccessful() else 1)
