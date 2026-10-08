"""Mathematical and small-CPU integration checks; not a formal battery experiment."""
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
import run_mix20_soft_aligned_difference_207 as suite
from src.data.databundle import DataBundle
from src.data.transformation.sequential import SequentialDataTransformation
from src.models.rul_predictors.soft_aligned_difference_batlinet import SoftAlignedDifferenceBlock
from src.models.rul_predictors.latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor


class DifferenceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_formula_against_explicit_attention(self):
        block = SoftAlignedDifferenceBlock(8, 2, dropout=0).eval()
        t, r = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
        wq, wk, wv = block.attn.in_proj_weight.chunk(3)
        bq, bk, bv = block.attn.in_proj_bias.chunk(3)
        project = torch.nn.functional.linear
        split = lambda x: x.reshape(2, -1, 2, 4).transpose(1, 2)
        q = split(project(block.q_norm(t), wq, bq))
        k = split(project(block.kv_norm(r), wk, bk))
        v = split(project(block.kv_norm(r), wv, bv))
        vt = split(project(block.kv_norm(t), wv, bv))
        attention = (q @ k.transpose(-1, -2) / 2).softmax(-1)
        expected = (vt - attention @ v).transpose(1, 2).reshape(2, 3, 8)
        torch.testing.assert_close(block.projected_difference(t, r), expected)
        x = block.attn.out_proj(expected)
        torch.testing.assert_close(block(t, r), x + block.ffn(block.ffn_norm(x)))

    def test_identical_single_token_and_reference_permutation(self):
        block = SoftAlignedDifferenceBlock(8, 2, dropout=.1).eval()
        t, r = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
        torch.testing.assert_close(block.projected_difference(t[:, :1], t[:, :1]),
                                   torch.zeros(2, 1, 8), atol=1e-6, rtol=0)
        torch.testing.assert_close(block(t, r), block(t, r[:, [3, 0, 4, 1, 2]]))
        self.assertFalse(torch.allclose(block(t, r), block(t, torch.randn_like(r))))

    def test_initialization_parameters_rng_and_shapes(self):
        for height, count in ((20, 170210), (100, 209890)):
            models = []
            for name in suite.NEW_MODELS:
                config = suite.model_config(name)
                config['input_height'] = height
                torch.manual_seed(7)
                models.append(suite.shared.MODELS.build(config, seed=7).eval())
                state = torch.get_rng_state()
                if len(models) == 1:
                    expected_rng = state.clone()
                else:
                    self.assertTrue(torch.equal(state, expected_rng))
            a, b = models
            self.assertEqual(sum(p.numel() for p in b.parameters()), count)
            self.assertEqual(a.state_dict().keys(), b.state_dict().keys())
            for key, value in a.state_dict().items():
                torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0)
            with torch.no_grad():
                x = torch.randn(1, 6, height, 1000)
                y = b.compute_prediction_components(x, x[:, None], torch.zeros(1, 1))
                self.assertEqual(y[0].shape, (1,))
                self.assertEqual(y[1].shape, (1, 1))
                self.assertTrue(torch.isfinite(y[1]).all())
        config['attention_layers'] = 2
        with self.assertRaises(ValueError):
            suite.shared.MODELS.build(config, seed=7)

    def test_gradients_reach_both_inputs_and_projections(self):
        block = SoftAlignedDifferenceBlock(8, 2, dropout=0)
        t = torch.randn(2, 3, 8, requires_grad=True)
        r = torch.randn(2, 5, 8, requires_grad=True)
        (block(t, r) * torch.randn_like(t)).sum().backward()
        gradients = [t.grad, r.grad, block.attn.out_proj.weight.grad,
                     *block.attn.in_proj_weight.grad.chunk(3)]
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_training_mean_evaluation_median_and_original_loss(self):
        config = suite.model_config(suite.NEW_MODELS[1])
        config.update(input_width=64, attention_channels=8, attention_heads=2,
                      encoder_dropout=0, attention_dropout=0, head_hidden_channels=8)
        model = suite.shared.MODELS.build(config, seed=1)
        x, r = torch.randn(2, 6, 20, 64), torch.randn(2, 4, 6, 20, 64)
        labels, support = torch.randn(2), torch.randn(2, 4)
        ori, individual, agg, _, _ = model.compute_prediction_components(x, r, support)
        torch.testing.assert_close(agg, individual.mean(1))
        expected = .5 * ((ori-labels)**2).mean() + .5 * ((agg-labels)**2).mean()
        torch.testing.assert_close(model(x, labels, r, support, return_loss=True), expected)
        model.eval()
        ori, individual, agg, _, _ = model.compute_prediction_components(x, r, support)
        torch.testing.assert_close(agg, individual.median(1).values)
        torch.testing.assert_close(model(x, labels, r, support), .5*(ori+agg))

    def test_full_workflow_completion_reuse_and_corruption_rejection(self):
        shared = suite.shared
        transform = SequentialDataTransformation([
            dict(name='LogScaleDataTransformation'), dict(name='ZScoreDataTransformation')])
        data = DataBundle(torch.randn(3, 6, 20, 64), torch.tensor([100., 200., 300.]),
                          torch.randn(2, 6, 20, 64), torch.tensor([150., 250.]),
                          label_transformation=transform)
        protocols = [torch.arange(64).reshape(2, 32) % 3 for _ in range(8)]
        history = dict(policy=shared.POLICY, protocols_paths=['unused']*8)
        configs = {name: suite.model_config(name) for name in suite.NEW_MODELS}
        for config in configs.values():
            config.update(input_width=64, train_batch_size=2, attention_channels=8,
                          attention_heads=2, head_hidden_channels=8)
        calls = []
        real_train, real_predict = shared.legacy_train, LatentCrossAttentionBatLiNetRULPredictor.predict
        def train(model, training, epochs, every, num_test, progress):
            self.assertEqual((epochs, every, num_test), (1000, 100, 147))
            real_train(model, training, 1, 100, num_test)
            # Simulated completion records test orchestration; only ONE real epoch.
            for epoch in range(1, 1001):
                progress(dict(epoch=epoch, train_loss=.1, elapsed_seconds=0.))
            calls.append(True)
        def predict(model, dataset, **kwargs):
            self.assertEqual(len(calls), 16)
            return real_predict(model, dataset, **kwargs)
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            workspace = Path(directory)/'runs'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace)]
            with patch.object(sys, 'argv', argv), \
                 patch.object(shared, 'audit_history', return_value=(data, protocols, [], history)), \
                 patch.object(suite, 'model_config', side_effect=lambda name: configs[name]), \
                 patch.object(shared, 'legacy_train', side_effect=train) as trainer, \
                 patch.object(LatentCrossAttentionBatLiNetRULPredictor, 'predict', predict):
                suite.main()
                self.assertEqual(len(json.loads((workspace/'summary.json').read_text())), 16)
                hashes = {str(p): shared.digest(p) for p in workspace.rglob('*') if p.is_file()}
                suite.main()
                self.assertEqual(trainer.call_count, 16)
                self.assertEqual(hashes, {str(p): shared.digest(p) for p in workspace.rglob('*') if p.is_file()})
                path = workspace/f'{suite.NEW_MODELS[1]}_seed0/test.pt'
                result = shared.load(path)
                torch.testing.assert_close(result['diagnostics']['support_index'], protocols[0])
                torch.testing.assert_close(result['truth_cycles'], torch.tensor([150., 250.]))
                result['diagnostics']['support_index'][0, 0] = 2
                torch.save(result, path)
                with self.assertRaises(ValueError):
                    suite.main()
                self.assertEqual(trainer.call_count, 16)
                path.unlink()
                (path.parent/'epoch1000.pt').unlink()
                with self.assertRaises(ValueError):
                    suite.main()
                self.assertEqual(trainer.call_count, 16)


if __name__ == '__main__':
    unittest.main()
