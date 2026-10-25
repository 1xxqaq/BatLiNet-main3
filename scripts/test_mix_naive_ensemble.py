"""Synthetic checks for simultaneous, untied two-branch ensemble training.

No historical battery cache is read and no formal experiment is started.
"""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix_branch_ablation as old
from scripts import run_mix_naive_ensemble as suite


class NaiveEnsembleTests(unittest.TestCase):
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

    def small(self):
        cfg = copy.deepcopy(suite.model_config('mix_20'))
        cfg['model'].update(input_width=128, train_batch_size=4, epochs=2,
                            attention_channels=8, attention_heads=2,
                            head_hidden_channels=8, evaluate_freq=1)
        cfg['runner'].update(micro_batch_size=2, checkpoint_every=1, pair_chunk=8)
        return cfg

    def binding(self, cfg, task='synthetic', seed=3):
        return dict(config=cfg, task=task, model='naive_ensemble', seed=seed,
                    data_sha256='synthetic-data', protocol_sha256='synthetic-protocol',
                    queue=dict(synthetic=True))

    def indices(self):
        return torch.arange(96).reshape(3, 32) % 5

    def assert_nested_equal(self, one, two, path='state'):
        if isinstance(one, torch.Tensor):
            self.assertIsInstance(two, torch.Tensor, path)
            self.assertEqual(one.dtype, two.dtype, path)
            self.assertEqual(one.shape, two.shape, path)
            self.assertTrue(torch.equal(one.cpu(), two.cpu()), path)
        elif isinstance(one, dict):
            self.assertEqual(set(one), set(two), path)
            for key in one:
                self.assert_nested_equal(one[key], two[key], f'{path}/{key}')
        elif isinstance(one, (tuple, list)):
            self.assertEqual(type(one), type(two), path)
            self.assertEqual(len(one), len(two), path)
            for index, (a, b) in enumerate(zip(one, two)):
                self.assert_nested_equal(a, b, f'{path}/{index}')
        else:
            self.assertEqual(one, two, path)

    def parameter_snapshot(self, model, optimizer):
        return {
            name: dict(value=p.detach().cpu().clone(),
                       optimizer=old.common.cpu(copy.deepcopy(optimizer.state[p])))
            for name, p in model.named_parameters()}

    def independent_baseline(self, data, cfg, seed, mode, micro, device):
        old.common.seed_all(seed)
        model_cfg = dict(cfg['model'], branch_mode=mode)
        model = old.build_branch_model(model_cfg, seed, mode).to(device)
        ds = old.prepared(model, data['train'], device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'],
                                     weight_decay=cfg['runner']['weight_decay'])
        generator = old.reference_generator(seed)
        snapshots, records = [], []
        step = optimizer.step

        def capture(*args, **kwargs):
            result = step(*args, **kwargs)
            snapshots.append(self.parameter_snapshot(model, optimizer))
            return result

        with patch.object(optimizer, 'step', side_effect=capture):
            for _ in range(2):
                records.append(old.train_epoch(model, optimizer, ds, micro, generator))
        rng = dict(cpu=torch.get_rng_state().clone(),
                   cuda=torch.cuda.get_rng_state(device).cpu().clone()
                   if device.startswith('cuda') else None)
        return snapshots, records, generator.get_state().clone(), rng

    def compare_single_optimizer_to_independent(self, device):
        data, cfg, seed = self.fixture(), self.small(), 3
        for micro in (2, 3):
            with self.subTest(device=device, micro=micro):
                expected = {mode:self.independent_baseline(data, cfg, seed, mode, micro, device)
                            for mode in ('self_only', 'reference_only')}
                model = suite.build_naive_model(cfg['model'], seed).to(device)
                ds = old.prepared(model, data['train'], device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['model']['lr'],
                                             weight_decay=cfg['runner']['weight_decay'])
                generator = suite.reference_generator(seed)
                streams = suite.BranchRandomStreams(seed, device)
                actual, records = [], []
                step = optimizer.step

                def capture(*args, **kwargs):
                    result = step(*args, **kwargs)
                    actual.append(self.parameter_snapshot(model, optimizer))
                    return result

                old.common.seed_all(917)
                ambient_cpu = torch.get_rng_state().clone()
                ambient_cuda = (torch.cuda.get_rng_state(device).cpu().clone()
                                if device.startswith('cuda') else None)
                with patch.object(optimizer, 'step', side_effect=capture):
                    for _ in range(2):
                        records.append(suite.train_epoch(model, optimizer, ds, micro,
                                                         generator, streams))
                self.assertEqual(len(actual), 4)
                self.assertTrue(torch.equal(ambient_cpu, torch.get_rng_state()))
                if ambient_cuda is not None:
                    self.assertTrue(torch.equal(ambient_cuda, torch.cuda.get_rng_state(device).cpu()))
                for mode, prefix in (('self_only', 'self_branch.'),
                                     ('reference_only', 'reference_branch.')):
                    baseline, old_records, reference_state, rng = expected[mode]
                    for index, snapshot in enumerate(actual):
                        branch = {name[len(prefix):]:state for name, state in snapshot.items()
                                  if name.startswith(prefix)}
                        self.assert_nested_equal(branch, baseline[index], f'{mode}/step{index+1}')
                    self.assertEqual([r['reference_fingerprint'] for r in records],
                                     [r['reference_fingerprint'] for r in old_records])
                    self.assertTrue(torch.equal(generator.get_state(), reference_state))
                    self.assert_nested_equal(streams.state_dict()['states'][mode], rng, mode)
                for epoch, record in enumerate(records):
                    expected_loss = sum(expected[mode][1][epoch]['train_loss']
                                        for mode in ('self_only', 'reference_only'))
                    self.assertAlmostEqual(record['train_loss'], expected_loss, places=6)

    def test_single_adamw_matches_two_independent_adamw_after_every_cpu_step(self):
        self.compare_single_optimizer_to_independent('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; synthetic CPU verification only')
    def test_single_adamw_matches_two_independent_adamw_after_every_cuda_step(self):
        self.compare_single_optimizer_to_independent('cuda:0')

    def resume_matches_uninterrupted(self, device):
        data, cfg = self.fixture(), self.small()
        binding = self.binding(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            complete, interrupted = Path(tmp)/'complete', Path(tmp)/'interrupted'
            suite.train_run(data, cfg, 3, complete, binding, device)
            original_save = suite.common.save
            crashed = False

            def crash(value, path):
                nonlocal crashed
                original_save(value, path)
                if Path(path).name == 'progress.pt' and value.get('epoch') == 1 and not crashed:
                    crashed = True
                    raise RuntimeError('synthetic interruption')

            with patch.object(suite.common, 'save', side_effect=crash):
                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    suite.train_run(data, cfg, 3, interrupted, binding, device)
            with (interrupted/'train.jsonl').open('a') as stream:
                stream.write('{"epoch":2,"train_loss":0.1}\n{"epoch":')
            suite.train_run(data, cfg, 3, interrupted, binding, device)
            expected = suite.common.load(complete/'final.pt')
            actual = suite.common.load(interrupted/'final.pt')
            self.assert_nested_equal(expected['state'], actual['state'], 'model')
            expected = suite.common.load(complete/'progress.pt')
            actual = suite.common.load(interrupted/'progress.pt')
            for key in ('optimizer', 'rng', 'branch_rng'):
                self.assert_nested_equal(expected[key], actual[key], key)
            self.assertTrue(list(interrupted.glob('interrupted_log_*.jsonl')))
            sha = suite.common.digest(interrupted/'final.pt')
            suite.train_run(data, cfg, 3, interrupted, binding, device)
            self.assertEqual(sha, suite.common.digest(interrupted/'final.pt'))
            with self.assertRaises(ValueError):
                suite.train_run(data, cfg, 3, interrupted, dict(binding, seed=4), device)

    def test_resume_restores_model_optimizer_cpu_reference_and_branch_rng_exactly(self):
        self.resume_matches_uninterrupted('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable; synthetic CPU verification only')
    def test_resume_also_restores_cuda_rng_exactly(self):
        self.resume_matches_uninterrupted('cuda:0')

    def test_prediction_joins_all_32_references_before_lower_median_and_fusion(self):
        data, cfg = self.fixture(), self.small()
        data['train'] = dict(
            feature=data['train']['feature'][torch.arange(32) % 5].clone(),
            label_z=torch.arange(32).float(),
            label=(torch.arange(32).float()*data['scale']+data['mean']).exp(),
            ids=[f'train_{i}' for i in range(32)])
        indices = torch.arange(32).repeat(3, 1)
        model = suite.build_naive_model(cfg['model'], 8)
        with torch.no_grad():
            model.reference_branch.support_head.net[-1].weight.zero_()
            model.reference_branch.support_head.net[-1].bias.zero_()
        reference = suite.predict(model, data, indices, 'cpu', 32)
        for chunk in (1, 7, 8, 13):
            actual = suite.predict(model, data, indices, 'cpu', chunk)
            self.assertEqual(tuple(actual['diagnostics']['y_sup'].shape), (3, 32))
            torch.testing.assert_close(actual['diagnostics']['y_sup_agg'],
                                       torch.full((3,), 15.), rtol=0, atol=0)
            expected = .5*(actual['diagnostics']['y_ori']+15.)
            torch.testing.assert_close(actual['prediction'], expected, rtol=0, atol=0)
            torch.testing.assert_close(actual['prediction'], reference['prediction'],
                                       rtol=1e-5, atol=1e-6)
            self.assertTrue(torch.equal(actual['diagnostics']['support_index'], indices))
            geometric = (actual['diagnostics']['y_ori']*data['scale']+data['mean']).exp()
            geometric *= (actual['diagnostics']['y_sup_agg']*data['scale']+data['mean']).exp()
            torch.testing.assert_close(actual['prediction_cycles'], geometric.sqrt(),
                                       rtol=1e-6, atol=1e-3)

    def trained_result(self, data, workspace):
        cfg = self.small()
        task, seed = 'mix_20', 3
        folder = workspace/task/f'naive_ensemble_seed{seed}'
        protocol = workspace/'protocols'/task/f'seed_{seed}.pt'
        suite.common.save(dict(indices=self.indices()), protocol)
        suite.common.save(data, workspace/'datasets'/task/'data.pt')
        identity = dict(synthetic=True, runtime=dict(synthetic=True))
        suite.common.atomic_json(identity, workspace/'queue.json')
        data_sha = suite.common.digest(workspace/'datasets'/task/'data.pt')
        suite.common.atomic_json(dict(sha256=data_sha), workspace/'datasets'/task/'receipt.json')
        binding = self.binding(cfg, task=task, seed=seed)
        binding.update(queue=identity, data_sha256=data_sha,
                       protocol_sha256=suite.common.digest(protocol))
        suite.train_run(data, cfg, seed, folder, binding, 'cpu')
        row = suite.evaluate_run(data, cfg, seed, folder, binding, self.indices(), 'cpu')
        return cfg, binding, folder, row

    def test_saved_self_consistent_prediction_tampering_is_rejected(self):
        data = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            cfg, binding, folder, _ = self.trained_result(data, Path(tmp))
            clean = suite.common.load(folder/'test.pt')
            changed = copy.deepcopy(clean)
            changed['diagnostics']['y_sup'] = changed['diagnostics']['y_sup']+.25
            changed['diagnostics']['y_sup_agg'] = changed['diagnostics']['y_sup'].median(1).values
            changed['prediction'] = .5*(changed['diagnostics']['y_ori']
                                         + changed['diagnostics']['y_sup_agg'])
            changed['prediction_cycles'] = (changed['prediction']*changed['label_sample_std']
                                             + changed['label_mean']).exp()
            changed['scores'] = suite.common.scores(changed['prediction_cycles'], changed['truth_cycles'])
            suite.common.save(changed, folder/'test.pt')
            with self.assertRaises((ValueError, AssertionError)):
                suite.evaluate_run(data, cfg, 3, folder, binding, self.indices(), 'cpu')

    def test_report_is_read_only_and_cannot_start_training_or_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            _, _, _, row = self.trained_result(self.fixture(), workspace)
            suite.common.atomic_json([row], workspace/'per_seed.json')

            def snapshot():
                return {p.relative_to(workspace).as_posix():suite.common.digest(p)
                        for p in workspace.rglob('*') if p.is_file()}

            before = snapshot()
            with redirect_stdout(io.StringIO()), \
                    patch.object(suite, 'train_run', side_effect=AssertionError('training')), \
                    patch.object(suite, 'predict', side_effect=AssertionError('inference')), \
                    patch.object(suite.common, 'save', side_effect=AssertionError('write')), \
                    patch.object(suite.common, 'atomic_json', side_effect=AssertionError('write')):
                self.assertEqual(suite.report_workspace(workspace), [])
            self.assertEqual(before, snapshot())

    def queue_fixture(self, workspace):
        configs = {task:self.small() for task in suite.TASKS}
        for task in suite.TASKS:
            suite.common.save(self.fixture(), workspace/'datasets'/task/'data.pt')
            for seed in suite.SEEDS:
                suite.common.save(dict(indices=self.indices()),
                                  workspace/'protocols'/task/f'seed_{seed}.pt')
        identity = dict(synthetic=True)
        shas = {task:suite.common.digest(workspace/'datasets'/task/'data.pt')
                for task in suite.TASKS}
        bindings = {(task, seed):suite.make_binding(
            task, seed, configs[task], identity, shas[task], workspace)
            for task in suite.TASKS for seed in suite.SEEDS}
        return configs, identity, shas, bindings

    def test_all_16_trainings_complete_before_first_test_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            configs, identity, shas, bindings = self.queue_fixture(workspace)
            trained, evaluated = [], []

            def train(data, cfg, seed, folder, binding, device):
                trained.append((binding['task'], seed))

            def evaluate(data, cfg, seed, folder, binding, indices, device):
                self.assertEqual(len(trained), 16)
                evaluated.append((binding['task'], seed))
                return dict(task=binding['task'], model='naive_ensemble', seed=seed,
                            RMSE=1., MAE=1., MAPE=.01, ACC15=1.)

            def gate(*args):
                self.assertEqual(len(trained), 16)
                return bindings

            with redirect_stdout(io.StringIO()), \
                    patch.object(suite, 'train_run', side_effect=train), \
                    patch.object(suite, 'evaluate_run', side_effect=evaluate), \
                    patch.object(suite, 'check_all_trainings', side_effect=gate), \
                    patch.object(suite, 'compare_previous', return_value=dict(status='unavailable')):
                suite.execute_queue(workspace, configs, identity, shas, 'cpu')
            self.assertEqual(len(set(trained)), 16)
            self.assertEqual(len(set(evaluated)), 16)
            status = json.loads((workspace/'status.json').read_text(encoding='utf-8'))
            self.assertEqual(status['status'], 'complete')
            self.assertEqual(status['predictions'], 16)

    def test_crash_after_final_weight_before_receipt_recovers_only_matching_progress(self):
        data, cfg = self.fixture(), self.small()
        binding = self.binding(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'interrupted_final'
            original = suite.common.atomic_json
            def fail_receipt(value, path):
                if Path(path).name == 'run.json' and value.get('status') == 'trained':
                    raise RuntimeError('final receipt crash')
                return original(value, path)
            with patch.object(suite.common, 'atomic_json', side_effect=fail_receipt):
                with self.assertRaisesRegex(RuntimeError, 'final receipt crash'):
                    suite.train_run(data, cfg, 3, folder, binding, 'cpu')
            self.assertTrue((folder/'final.pt').exists())
            final = suite.common.load(folder/'final.pt')
            changed = copy.deepcopy(final)
            changed['state'][next(iter(changed['state']))].add_(1.)
            suite.common.save(changed, folder/'final.pt')
            with self.assertRaises(ValueError):
                suite.train_run(data, cfg, 3, folder, binding, 'cpu')
            suite.common.save(final, folder/'final.pt')
            before = suite.common.digest(folder/'final.pt')
            suite.train_run(data, cfg, 3, folder, binding, 'cpu')
            self.assertEqual(before, suite.common.digest(folder/'final.pt'))
            receipt = json.loads((folder/'run.json').read_text(encoding='utf-8'))
            self.assertTrue(receipt['receipt_recovered_from_verified_final_progress'])

    def test_previous_comparison_summary_requires_all_eight_verified_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            baseline = dict(RMSE=3., MAE=2., MAPE=.2, ACC15=.5)
            current = dict(RMSE=2., MAE=1., MAPE=.1, ACC15=.6)
            rows = [dict(task='mix_20', seed=seed, status='verified_comparison',
                         current_scores=current, previous_scores=dict(full=baseline))
                    for seed in range(8)]
            suite.write_previous_comparisons(rows[:7]+[dict(status='unavailable')], workspace)
            self.assertEqual(json.loads((workspace/'paired_previous_comparisons.json').read_text()), [])
            suite.write_previous_comparisons(rows, workspace)
            result = json.loads((workspace/'paired_previous_comparisons.json').read_text())
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['baseline'], 'full')
            self.assertEqual(result[0]['metrics']['RMSE']['mean_delta'], -1.)
            for key in suite.METRICS:
                self.assertEqual(result[0]['metrics'][key]['wins'], 8)

    def test_last_training_failure_prevents_test_and_prior_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            configs, identity, shas, _ = self.queue_fixture(workspace)
            calls = 0

            def train(*args):
                nonlocal calls
                calls += 1
                if calls == 16:
                    raise RuntimeError('last training failed')

            with redirect_stdout(io.StringIO()), \
                    patch.object(suite, 'train_run', side_effect=train), \
                    patch.object(suite, 'evaluate_run') as evaluate, \
                    patch.object(suite, 'compare_previous') as compare, \
                    patch.object(suite, 'check_all_trainings') as gate:
                with self.assertRaisesRegex(RuntimeError, 'last training failed'):
                    suite.execute_queue(workspace, configs, identity, shas, 'cpu')
            evaluate.assert_not_called()
            compare.assert_not_called()
            gate.assert_not_called()

    def test_previous_single_branch_comparison_is_read_only_and_verified(self):
        data = self.fixture()
        data['historical_binding'] = dict(dataset=dict(synthetic=True))
        with tempfile.TemporaryDirectory() as tmp:
            current, previous = Path(tmp)/'current', Path(tmp)/'previous'
            cfg, binding, folder, _ = self.trained_result(data, current)
            task, seed = 'mix_20', 3
            suite.common.atomic_json(dict(historical_binding=data['historical_binding']),
                                     previous/'datasets'/task/'receipt.json')
            for mode in ('self_only', 'reference_only'):
                old_cfg = copy.deepcopy(old.model_config(task, mode))
                old_cfg['model'].update({key:value for key,value in cfg['model'].items()
                                         if key != 'branch_mode'})
                old_cfg['runner'].update(micro_batch_size=2, checkpoint_every=1, pair_chunk=8)
                source = previous/task/f'{mode}_seed{seed}'
                old_binding = dict(binding, model=mode, config=old_cfg)
                old.train_run(data, old_cfg, seed, source, old_binding, 'cpu')
                old.evaluate_run(data, old_cfg, seed, source, old_binding, self.indices(), 'cpu')

            def snapshot():
                return {p.relative_to(Path(tmp)).as_posix():suite.common.digest(p)
                        for p in Path(tmp).rglob('*') if p.is_file()}

            before = snapshot()
            with patch.object(suite.common, 'save', side_effect=AssertionError('write')), \
                    patch.object(suite.common, 'atomic_json', side_effect=AssertionError('write')):
                comparison = suite.compare_previous(data, task, seed, folder, previous)
            self.assertEqual(before, snapshot())
            self.assertEqual(comparison['status'], 'verified_comparison', comparison)
            self.assertEqual(comparison['prediction_max_absolute_difference_z'], 0.)
            for mode in ('self_only', 'reference_only'):
                self.assertTrue(comparison['branches'][mode]['parameters_bitwise_equal'])
                self.assertEqual(comparison['branches'][mode]['maximum_parameter_absolute_difference'], 0.)
            receipt = previous/'datasets'/task/'receipt.json'
            altered = json.loads(receipt.read_text(encoding='utf-8'))
            altered['historical_binding']['dataset'] = dict(different=True)
            suite.common.atomic_json(altered, receipt)
            rejected = suite.compare_previous(data, task, seed, folder, previous)
            self.assertEqual(rejected['status'], 'unavailable_or_rejected')


if __name__ == '__main__':
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NaiveEnsembleTests))
    if '--report' in sys.argv:
        path = Path(sys.argv[sys.argv.index('--report')+1])
        suite.common.atomic_json(dict(
            status='passed' if result.wasSuccessful() else 'failed',
            tests=result.testsRun, skipped=len(result.skipped),
            failures=len(result.failures), errors=len(result.errors),
            synthetic_only=True, formal_training_started=False,
            cuda_checked=torch.cuda.is_available(),
            runtime=suite.branch.base.runtime('cpu'), source_code=suite.sources()), path)
    sys.exit(0 if result.wasSuccessful() else 1)
