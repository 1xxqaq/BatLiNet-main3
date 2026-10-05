"""CPU checks for variant A, without downloading battery data."""

import copy
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.builders import MODELS
from src.data.databundle import DataBundle
from src.models.rul_predictors.latent_cross_attention_batlinet import (
    LatentCrossAttentionBatLiNetRULPredictor,
)
from src.models.rul_predictors.latent_self_cross_attention_batlinet import (
    LatentSelfCrossAttentionBatLiNetRULPredictor,
)


def model_kwargs(height=20):
    return dict(
        in_channels=6, channels=32, input_height=height, input_width=1000,
        train_support_size=2, test_support_size=32, attention_channels=64,
        attention_heads=4, attention_layers=1, attention_dropout=0.0,
        attention_mlp_ratio=2, head_hidden_channels=64, encoder_dropout=0.0,
        filter_cycles=False, epochs=1, train_batch_size=2, test_batch_size=1,
    )


class LatentSelfCrossAttentionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        torch.set_num_threads(2)

    def test_configs_preserve_base_experiment_and_register_model(self):
        for task in ('mix_20', 'mix_100'):
            with self.subTest(task=task):
                root = REPO_ROOT / 'configs/ablation/diff_branch'
                base = yaml.safe_load(
                    (root / f'batlinet_latent_cross_attention/{task}.yaml')
                    .read_text(encoding='utf-8'))
                variant = yaml.safe_load(
                    (root / f'batlinet_latent_self_cross_attention/{task}.yaml')
                    .read_text(encoding='utf-8'))
                for field in ('train_test_split', 'feature', 'label',
                              'label_transformation'):
                    self.assertEqual(base[field], variant[field])
                expected = dict(base['model'])
                actual = dict(variant['model'])
                expected.pop('name')
                actual.pop('name')
                expected.pop('checkpoint_freq')
                self.assertEqual(actual.pop('checkpoint_freq'), 1000)
                for key in list(actual):
                    if key.startswith('self_attention_'):
                        actual.pop(key)
                self.assertEqual(expected, actual)
                model = MODELS.build(copy.deepcopy(variant['model']))
                self.assertIsInstance(
                    model, LatentSelfCrossAttentionBatLiNetRULPredictor)
                self.assertEqual(model.checkpoint_freq, model.train_epochs)

    def test_both_windows_backward_and_aggregation(self):
        for height, token_count in ((20, 155), (100, 775)):
            with self.subTest(height=height):
                model = LatentSelfCrossAttentionBatLiNetRULPredictor(
                    **model_kwargs(height), self_attention_dropout=0.0)
                feature = torch.randn(1, 6, height, 1000)
                supports = torch.randn(1, 2, 6, height, 1000)
                labels = torch.tensor([[0.2, 0.9]])
                model.train()
                components = model.compute_prediction_components(
                    feature, supports, labels, return_features=True)
                self.assertEqual(components[5].shape, (1, token_count, 64))
                self.assertEqual(components[6].shape, (1, 2, token_count, 64))
                torch.testing.assert_close(components[2], components[1].mean(1))
                loss = model(feature, torch.tensor([0.4]), supports, labels,
                             return_loss=True)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                for module in (model.cell_encoder.local_encoder,
                               model.cell_encoder.blocks,
                               model.cross_attention, model.ori_head,
                               model.support_head):
                    self.assertGreater(
                        sum(p.grad.abs().sum().item() for p in module.parameters()),
                        0.0)
                model.eval()
                with torch.no_grad():
                    components = model.compute_prediction_components(
                        feature, supports, labels)
                    torch.testing.assert_close(
                        components[2], components[1].min(1).values)
                    prediction = model(feature, None, supports, labels)
                    torch.testing.assert_close(
                        prediction, 0.5 * components[0] + 0.5 * components[2])

    def test_zero_layers_matches_base_predictions_and_checkpoint_keys(self):
        base = LatentCrossAttentionBatLiNetRULPredictor(**model_kwargs())
        variant = LatentSelfCrossAttentionBatLiNetRULPredictor(
            **model_kwargs(), self_attention_layers=0)
        variant.load_state_dict(base.state_dict(), strict=True)
        self.assertEqual(list(base.state_dict()), list(variant.state_dict()))
        feature = torch.randn(1, 6, 20, 1000)
        supports = torch.randn(1, 2, 6, 20, 1000)
        labels = torch.randn(1, 2)
        for training in (True, False):
            base.train(training)
            variant.train(training)
            with torch.no_grad():
                torch.testing.assert_close(
                    base(feature, None, supports, labels),
                    variant(feature, None, supports, labels), rtol=0, atol=0)

    def test_cell_encoding_does_not_mix_batteries(self):
        model = LatentSelfCrossAttentionBatLiNetRULPredictor(
            **model_kwargs(), self_attention_dropout=0.0).eval()
        features = torch.randn(2, 6, 20, 1000)
        with torch.no_grad():
            together = model.cell_encoder(features)
            first = model.cell_encoder(features[:1])
            features[1] = features[1] * 20 + 10
            altered = model.cell_encoder(features)
        torch.testing.assert_close(together[:1], first, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(together[:1], altered[:1], atol=1e-6, rtol=1e-5)

    def test_fixed_32_references_roundtrip_and_reference_permutation(self):
        with tempfile.TemporaryDirectory(prefix='latent_self_attention_') as temp:
            temp = Path(temp)
            indices = torch.tensor([[0, 1, 2, 3] * 8, [3, 1, 0, 2] * 8])
            protocol = temp / 'protocol.pt'
            torch.save({'indices': indices}, protocol)
            kwargs = dict(model_kwargs(), self_attention_dropout=0.0,
                          fixed_test_support_index_path=str(protocol))
            model = LatentSelfCrossAttentionBatLiNetRULPredictor(**kwargs)
            data = DataBundle(
                torch.randn(4, 6, 20, 1000), torch.arange(4).float(),
                torch.randn(2, 6, 20, 1000), torch.tensor([0.4, 1.2]))
            data.train_data.feature[:, :, 10] = 0
            data.test_data.feature[:, :, 10] = 0
            prediction, diagnostics = model.predict(data, return_diagnostics=True)
            self.assertTrue(torch.isfinite(prediction).all())
            torch.testing.assert_close(diagnostics['support_index'], indices)
            self.assertEqual(diagnostics['y_sup'].shape, (2, 32))
            torch.testing.assert_close(
                diagnostics['y_sup_agg'], diagnostics['y_sup'].median(1).values)
            torch.testing.assert_close(
                prediction, 0.5 * diagnostics['y_ori']
                + 0.5 * diagnostics['y_sup_agg'])
            checkpoint = temp / 'model.ckpt'
            model.dump_checkpoint(checkpoint)
            restored = LatentSelfCrossAttentionBatLiNetRULPredictor(**kwargs)
            restored.load_checkpoint(checkpoint, device='cpu')
            torch.testing.assert_close(restored.predict(data), prediction)
            permutation = torch.arange(31, -1, -1)
            permuted_protocol = temp / 'permuted.pt'
            torch.save({'indices': indices[:, permutation]}, permuted_protocol)
            restored.fixed_test_support_index_path = str(permuted_protocol)
            restored._fixed_test_support_index = None
            torch.testing.assert_close(restored.predict(data), prediction,
                                       atol=1e-6, rtol=1e-5)

    def test_inherited_training_writes_final_epoch_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix='latent_self_training_') as temp:
            temp = Path(temp)
            model = LatentSelfCrossAttentionBatLiNetRULPredictor(
                **model_kwargs(), self_attention_dropout=0.0,
                workspace=temp, checkpoint_freq=1, evaluate_freq=100)
            data = DataBundle(
                torch.randn(2, 6, 20, 1000), torch.tensor([0.1, 0.8]),
                torch.randn(1, 6, 20, 1000), torch.tensor([0.4]))
            initial = model.cell_encoder.blocks[0].attention.in_proj_weight.detach().clone()
            model.fit(data, timestamp='smoke')
            checkpoint = temp / 'smoke_seed_0_epoch_1.ckpt'
            self.assertTrue(checkpoint.is_file())
            self.assertTrue((temp / 'latest.ckpt').is_file())
            self.assertFalse(torch.equal(
                initial, model.cell_encoder.blocks[0].attention.in_proj_weight))
            torch.testing.assert_close(
                torch.load(checkpoint, map_location='cpu', weights_only=True)
                ['cell_encoder.blocks.0.attention.in_proj_weight'],
                model.cell_encoder.blocks[0].attention.in_proj_weight)

    def test_invalid_self_attention_settings(self):
        for options in (
                dict(self_attention_layers=-1), dict(self_attention_layers=0.5),
                dict(self_attention_heads=3), dict(self_attention_heads=0),
                dict(self_attention_mlp_ratio=0), dict(self_attention_dropout=1.1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                LatentSelfCrossAttentionBatLiNetRULPredictor(
                    **model_kwargs(), **options)


if __name__ == '__main__':
    unittest.main(verbosity=2)
