"""Check weighted partial batches, frozen models, references and all-16 ordering."""
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
from scripts import run_mix100_cycle_mixer as runner
from src.data.databundle import DataBundle, Dataset
from src.data.transformation.sequential import SequentialDataTransformation
from src.models.rul_predictors.latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor


class SmallRegressor(torch.nn.Module):
    """No dropout: microbatch and whole-batch updates must agree numerically."""
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.4))
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.lr, self.train_batch_size, self.grad_accum_steps = .001, 5, 1
        self.train_support_size, self.test_support_size = 2, 32
        self.test_batch_size, self.evaluate_freq = 1, 2

    def build_cell_dataset(self, training):
        return training

    def get_support_set(self, x, feature, labels, fixed_indices=None, **kwargs):
        if fixed_indices is None:
            fixed_indices = torch.randint(len(feature), (len(x) * self.train_support_size,)).view(len(x), -1)
        return feature[fixed_indices], labels[fixed_indices]

    def forward(self, x, y, support_x, support_y, return_loss=True):
        prediction = self.weight * (x[:, 0] + support_x.mean((1, 2))) + self.bias
        return ((prediction - y) ** 2).mean()


class Mix100Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def fixture(self):
        transform = SequentialDataTransformation([
            dict(name='LogScaleDataTransformation'), dict(name='ZScoreDataTransformation')])
        data = DataBundle(torch.randn(5, 6, 100, 64), torch.tensor([100., 200., 300., 400., 500.]),
                          torch.randn(2, 6, 100, 64), torch.tensor([150., 350.]),
                          label_transformation=transform)
        configs = runner.model_configs()
        for config in configs.values():
            config.update(input_width=64, train_batch_size=4, attention_channels=8,
                          attention_heads=2, head_hidden_channels=8)
        protocols = [torch.arange(64).reshape(2, 32) % 5 for _ in range(8)]
        rows = [dict(model='latent_cross_attention_history', seed=s,
                     RMSE=100., MAE=80., MAPE=.2, ACC15=.5) for s in range(8)]
        history = dict(dataset=dict(train=dict(ids=[str(s) for s in range(5)])))
        return data, configs, protocols, rows, history

    def test_frozen_mix100_architectures_and_same_initial_weights(self):
        configs = runner.model_configs()
        runner.shared.set_seed(12)
        baseline = runner.shared.MODELS.build(configs[runner.BASELINE], seed=12)
        baseline_rng = torch.get_rng_state().clone()
        runner.shared.set_seed(12)
        enhanced = runner.shared.MODELS.build(configs[runner.ENHANCED], seed=12)
        torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in baseline.parameters()), 209890)
        self.assertEqual(sum(p.numel() for p in enhanced.parameters()), 211556)
        self.assertEqual(baseline.cell_encoder.num_tokens, 775)
        self.assertEqual(enhanced.cell_encoder.cycle_mixer.net[0].in_features, 50)
        self.assertEqual(enhanced.cell_encoder.cycle_mixer.net[0].out_features, 16)
        state = enhanced.state_dict()
        for key, value in baseline.state_dict().items():
            target = key.replace('cell_encoder.', 'cell_encoder.base.', 1) if key.startswith('cell_encoder.') else key
            torch.testing.assert_close(state[target], value, rtol=0, atol=0)

    def test_partial_microbatches_match_whole_batch_optimizer_updates_and_rng(self):
        training = Dataset(torch.arange(7).float().view(-1, 1) / 7,
                           torch.linspace(.1, .9, 7))
        original, split = SmallRegressor(), SmallRegressor()
        torch.manual_seed(92)
        runner.shared.legacy_train(original, copy.deepcopy(training), 3, 2, 2)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(92)
        records = []
        runner.train(split, copy.deepcopy(training), 3, 2, 2, records.append)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(split.state_dict()[key], value, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        self.assertEqual([(r['optimizer_steps'], r['micro_batches']) for r in records], [(2, 4)] * 3)

    def test_protocol_rejects_mix20_and_out_of_range_references(self):
        valid = dict(seed=0, num_train_samples=205, num_test_samples=137,
                     test_support_size=32, indices=torch.zeros(137, 32, dtype=torch.long))
        runner.validate_protocol(valid, 0)
        for key, value in (('num_train_samples', 207), ('num_test_samples', 147),
                           ('seed', 1), ('indices', torch.full((137, 32), 205, dtype=torch.long))):
            with self.assertRaises(ValueError):
                runner.validate_protocol(dict(valid, **{key: value}), 0)

    def test_all_16_train_before_testing_and_completed_runs_are_reused(self):
        data, configs, protocols, rows, history = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / 'runs'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace), '--micro-batch-size', '2']
            completed = []
            actual_train = runner.train
            actual_predict = LatentCrossAttentionBatLiNetRULPredictor.predict
            def train(model, training, epochs, micro, num_test, progress):
                self.assertEqual((epochs, micro, num_test), (1000, 2, 2))
                records = []
                # One real epoch per model/seed verifies updates; only the
                # orchestration fixture synthesizes the remaining log records.
                actual_train(model, training, 1, micro, num_test, records.append)
                for epoch in range(1, 1001):
                    progress(dict(records[0], epoch=epoch))
                completed.append(True)
            def predict(model, dataset, **kwargs):
                self.assertEqual(len(completed), 16)
                return actual_predict(model, dataset, **kwargs)
            with patch.object(sys, 'argv', argv), \
                    patch.object(runner, 'audit_history', side_effect=lambda *a: (data, protocols, rows, copy.deepcopy(history))), \
                    patch.object(runner, 'model_configs', return_value=configs), \
                    patch.object(runner, 'train', side_effect=train) as training, \
                    patch.object(LatentCrossAttentionBatLiNetRULPredictor, 'predict', predict), \
                    redirect_stdout(io.StringIO()):
                runner.main()
                self.assertEqual(len(json.loads((workspace / 'summary.json').read_text())), 24)
                files = [workspace / f'{name}_seed{s}/test.pt' for s in range(8) for name in runner.ARCHITECTURES]
                hashes = [runner.shared.digest(path) for path in files]
                for name in runner.ARCHITECTURES:
                    result = runner.shared.load(workspace / f'{name}_seed0/test.pt')
                    torch.testing.assert_close(result['diagnostics']['support_index'], protocols[0])
                    self.assertEqual(result['scores'], runner.shared.scores(data, result['prediction']))
                runner.main()
                self.assertEqual(training.call_count, 16)
                self.assertEqual(hashes, [runner.shared.digest(path) for path in files])
                # A changed physical batch must never silently reuse old runs.
                argv[-1] = '1'
                with self.assertRaises(ValueError):
                    runner.main()
                self.assertEqual(training.call_count, 16)

    def test_audit_only_is_read_only_and_missing_history_fails_before_training(self):
        data, configs, protocols, rows, history = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / 'output'
            argv = ['runner', '--audit-only', '--workspace', str(workspace)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(runner, 'audit_history', return_value=(data, protocols, rows, history)), \
                    patch.object(runner, 'train') as training, redirect_stdout(io.StringIO()):
                runner.main()
            training.assert_not_called()
            self.assertFalse(workspace.exists())
            argv = ['runner', '--history-root', directory, '--workspace', str(workspace), '--device', 'cpu']
            with patch.object(sys, 'argv', argv), self.assertRaises(ValueError):
                runner.main()
            self.assertFalse(workspace.exists())


if __name__ == '__main__':
    unittest.main()
