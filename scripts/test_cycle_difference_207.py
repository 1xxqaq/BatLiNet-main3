"""Input semantics, reference consistency and the 16-run train/test barrier."""
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import run_mix20_cycle_difference_207 as suite
from src.data.databundle import DataBundle
from src.data.transformation.sequential import SequentialDataTransformation
from src.models.rul_predictors.latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor

shared = suite.shared


def small_config(name):
    config = suite.model_config(name)
    config.update(input_width=64, train_batch_size=2, attention_channels=8,
                  attention_heads=2, head_hidden_channels=8)
    return config


class InputTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def models(self):
        models = []
        for name in suite.NEW_MODELS:
            shared.set_seed(9)
            models.append(suite.build_model(small_config(name), 9))
        return models

    def test_raw_mode_preserves_old_weights_rng_preprocessing_and_predictions(self):
        config = shared.model_config()
        shared.set_seed(9)
        old = shared.MODELS.build(config, seed=9)
        expected_rng = torch.get_rng_state().clone()
        models = []
        for name in suite.NEW_MODELS:
            shared.set_seed(9)
            model = suite.build_model(suite.model_config(name), 9)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            self.assertEqual(sum(p.numel() for p in model.parameters()), 170556)
            self.assertEqual(old.state_dict().keys(), model.state_dict().keys())
            for key, value in model.state_dict().items():
                torch.testing.assert_close(value, old.state_dict()[key], rtol=0, atol=0)
            models.append(model)
        x = torch.randn(2, 6, 20, 1000)
        x[:, :, 10] = 0
        snapshot = x.clone()
        torch.testing.assert_close(models[0]._prepare_feature(x), old._prepare_feature(x), rtol=0, atol=0)
        prepared = old._prepare_feature(x)
        references = prepared.flip(0).unsqueeze(1).expand(-1, 2, -1, -1, -1).contiguous()
        labels = torch.randn(2, 2)
        with torch.no_grad():
            old.eval()
            models[0].eval()
            expected = old.compute_prediction_components(prepared, references, labels)
            actual = models[0].compute_prediction_components(prepared, references, labels)
            for a, b in zip(actual[:3], expected[:3]):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(old.combine_predictions(expected[0], expected[2]),
                                       models[0].combine_predictions(actual[0], actual[2]),
                                       rtol=0, atol=0)
        self.assertTrue(torch.equal(x, snapshot))

    def test_first_cycle_zero_invalid_cycles_preserved_and_no_cross_cell_subtraction(self):
        raw, difference = self.models()
        # Different cells have different constant offsets and different slopes.
        x = (torch.arange(20)[None, None, :, None].float() * torch.tensor([1., 3.])[:, None, None, None]
             + torch.tensor([10., 100.])[:, None, None, None]).expand(2, 6, 20, 64).clone()
        x[:, :, 10] = 0
        x[1, :, 7] = 0
        snapshot = x.clone()
        cleaned = raw._prepare_feature(x)
        expected = cleaned - cleaned[:, :, :1]
        expected[:, :, 10] = 0
        expected[1, :, 7] = 0
        actual = difference._prepare_feature(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(float(actual[:, :, 0].abs().sum()), 0.)
        self.assertEqual(float(actual[:, :, 10].abs().sum()), 0.)
        self.assertGreater(float(actual[:, :, 1].abs().sum()), 0.)
        self.assertTrue(torch.equal(x, snapshot))
        self.assertEqual(tuple(actual.shape), tuple(x.shape))

    def test_invalid_baseline_and_bad_options_are_rejected(self):
        _, model = self.models()
        x = torch.randn(2, 6, 20, 64)
        x[1, :, 0] = 0
        with self.assertRaisesRegex(ValueError, '基准循环无效'):
            model._prepare_feature(x)
        for changes in ({'input_mode': 'other'}, {'difference_base': -1}, {'difference_base': 20}):
            with self.assertRaises(ValueError):
                suite.build_model(dict(small_config(suite.NEW_MODELS[1]), **changes), 0)

    def test_reference_training_and_inference_apply_identical_single_preparation(self):
        _, model = self.models()
        x = torch.randn(3, 6, 20, 64)
        x[:, :, 10] = 0
        labels = torch.tensor([-.5, .2, .8])
        indices = torch.tensor([[2, 0], [1, 1]])
        expected = model._prepare_feature(x)[indices]
        for training in (True, False):
            model.train(training)
            prepared = model.build_cell_dataset(DataBundle(x, labels, x, labels).train_data).feature
            a = model.get_support_set(prepared[:2], x, labels, fixed_indices=indices)[0]
            b = model.get_support_set(prepared[:2], prepared, labels, fixed_indices=indices,
                                      support_is_prepared=True)[0]
            torch.testing.assert_close(a, expected, rtol=0, atol=0)
            torch.testing.assert_close(b, expected, rtol=0, atol=0)


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def fixture(self):
        transform = SequentialDataTransformation([
            dict(name='LogScaleDataTransformation'), dict(name='ZScoreDataTransformation')])
        x, test = torch.randn(3, 6, 20, 64), torch.randn(2, 6, 20, 64)
        x[:, :, 10] = 0
        test[:, :, 10] = 0
        data = DataBundle(x, torch.tensor([100., 200., 300.]), test, torch.tensor([150., 250.]),
                          label_transformation=transform)
        history = dict(policy=shared.POLICY, protocols_paths=['unused'] * 8)
        protocols = [torch.arange(64).reshape(2, 32) % 3 for _ in range(8)]
        rows = [dict(model=name, seed=s, RMSE=100., MAE=80., MAPE=.2, ACC15=.5)
                for name in ('batlinet', 'latent_cross_attention') for s in range(8)]
        return data, protocols, rows, history

    def test_16_runs_finish_before_test_reuse_and_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            fixture = self.fixture()
            data, protocols, _, _ = fixture
            snapshot = data.train_data.feature.clone()
            workspace = Path(directory) / 'new'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace)]
            configs = {name: small_config(name) for name in suite.NEW_MODELS}
            calls = []
            real_train = shared.legacy_train
            real_predict = LatentCrossAttentionBatLiNetRULPredictor.predict
            def train(model, training, epochs, every, num_test, progress):
                self.assertEqual((epochs, every, num_test), (1000, 100, 147))
                real_train(model, training, 1, 100, num_test)
                for epoch in range(1, 1001):
                    progress(dict(epoch=epoch, train_loss=.1, elapsed_seconds=0.))
                calls.append(model.input_mode)
            def predict(model, dataset, **kwargs):
                self.assertEqual(len(calls), 16)
                return real_predict(model, dataset, **kwargs)
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=fixture), \
                    patch.object(suite, 'model_config', side_effect=lambda name: configs[name]), \
                    patch.object(shared, 'legacy_train', side_effect=train) as training, \
                    patch.object(LatentCrossAttentionBatLiNetRULPredictor, 'predict', predict):
                suite.main()
                self.assertEqual(calls, ['raw'] * 8 + ['self_difference'] * 8)
                rows = json.loads((workspace / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(len(rows), 32)
                self.assertTrue(torch.equal(data.train_data.feature, snapshot))
                for name in suite.NEW_MODELS:
                    result = shared.load(workspace / f'{name}_seed0/test.pt')
                    self.assertTrue(torch.equal(result['diagnostics']['support_index'], protocols[0]))
                    self.assertEqual(result['scores'], shared.scores(data, result['prediction']))
                    self.assertIn('y_ori_RMSE', next(r for r in rows if r['model'] == name))
                hashes = {str(p): shared.digest(p) for p in workspace.rglob('*') if p.is_file()}
                suite.main()
                self.assertEqual(training.call_count, 16)
                self.assertEqual(hashes, {str(p): shared.digest(p) for p in workspace.rglob('*') if p.is_file()})
                path = workspace / f'{suite.NEW_MODELS[1]}_seed0/test.pt'
                bad = shared.load(path)
                bad['diagnostics']['support_index'][0, 0] = 2
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    suite.main()
                self.assertEqual(training.call_count, 16)

    def test_audit_only_invalid_base_and_incomplete_runs_never_start_training(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            fixture = self.fixture()
            workspace = Path(directory) / 'new'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace), '--audit-only']
            configs = {name: small_config(name) for name in suite.NEW_MODELS}
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=fixture), \
                    patch.object(suite, 'model_config', side_effect=lambda name: configs[name]), \
                    patch.object(shared, 'legacy_train') as training:
                suite.main()
                self.assertFalse(workspace.exists())
                fixture[0].train_data.feature[0, :, 0] = 0
                with self.assertRaisesRegex(ValueError, '基准循环无效'):
                    suite.main()
                self.assertFalse(workspace.exists())
                fixture[0].train_data.feature[0, :, 0] = 1
                (workspace / f'{suite.NEW_MODELS[0]}_seed0').mkdir(parents=True)
                with self.assertRaisesRegex(ValueError, '训练目录不完整'):
                    suite.main()
                training.assert_not_called()

    def test_missing_historical_protocol_fails_without_creating_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / 'new'
            argv = ['runner', '--device', 'cpu', '--history-root', directory,
                    '--original-root', directory, '--workspace', str(workspace)]
            with patch.object(sys, 'argv', argv), self.assertRaises(FileNotFoundError):
                suite.main()
            self.assertFalse(workspace.exists())


if __name__ == '__main__':
    unittest.main()
