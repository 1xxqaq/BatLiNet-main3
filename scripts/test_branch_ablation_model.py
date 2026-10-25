"""CPU checks for structural branch removal, aligned initialization and losses."""
import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.rul_predictors.branch_ablation_batlinet import (
    BRANCH_MODES, build_branch_model,
)
from src.models.rul_predictors.cycle_mixer_soft_aligned_difference_batlinet import (
    CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor,
)


class BranchModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self):
        return dict(in_channels=6, channels=32, input_height=20, input_width=128,
                    attention_channels=8, attention_heads=2, attention_layers=1,
                    head_hidden_channels=8, cycle_mixer_hidden=16, alpha=0.5,
                    filter_cycles=False, train_support_size=2,
                    test_support_size=4, encoder_dropout=0.1,
                    attention_dropout=0.1)

    def inputs(self):
        generator = torch.Generator().manual_seed(43)
        return (torch.randn(2, 6, 20, 128, generator=generator),
                torch.randn(2, 4, 6, 20, 128, generator=generator),
                torch.randn(2, 4, generator=generator),
                torch.randn(2, generator=generator))

    def test_initialization_matches_existing_complete_model_and_preserves_rng(self):
        config = self.config()
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(17)
            existing = CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor(
                **config, seed=17)
        before = torch.get_rng_state().clone()
        models = {mode: build_branch_model(dict(model=config), 17, mode)
                  for mode in BRANCH_MODES}
        self.assertTrue(torch.equal(before, torch.get_rng_state()))
        expected = existing.state_dict()
        self.assertEqual(set(models['full'].state_dict()), set(expected))
        for mode, model in models.items():
            for key, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]), f'{mode}: {key}')
            self.assertEqual(model.cell_encoder.num_tokens, existing.cell_encoder.num_tokens)
        self.assertFalse(hasattr(models['self_only'], 'cross_attention'))
        self.assertFalse(hasattr(models['self_only'], 'support_head'))
        self.assertFalse(hasattr(models['reference_only'], 'ori_head'))
        for mode in ('self_only', 'reference_only'):
            self.assertLess(sum(p.numel() for p in models[mode].parameters()),
                            sum(p.numel() for p in existing.parameters()))
        for mode, missing in (('self_only', ('cross_attention.', 'support_head.')),
                              ('reference_only', ('ori_head.',))):
            self.assertFalse(any(key.startswith(missing)
                                 for key in models[mode].state_dict()))
        # Each model must own its parameters, rather than share live tensors.
        self.assertNotEqual(models['full'].cell_encoder.base.position.data_ptr(),
                            models['self_only'].cell_encoder.base.position.data_ptr())

    def test_full_forward_loss_gradients_and_dropout_match_existing_model(self):
        model = build_branch_model(self.config(), 17, 'full')
        existing = CycleMixerSoftAlignedDifferenceBatLiNetRULPredictor(**self.config())
        existing.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
        feature, reference, ref_label, label = self.inputs()
        for training in (False, True):
            model.train(training)
            existing.train(training)
            state = torch.get_rng_state().clone()
            expected = existing.compute_prediction_components(feature, reference, ref_label)
            after = torch.get_rng_state().clone()
            torch.set_rng_state(state)
            actual = model.predict_components(feature, reference, ref_label)
            self.assertTrue(torch.equal(after, torch.get_rng_state()))
            for key, value in zip(('y_ori', 'y_sup', 'y_sup_agg'), expected[:3]):
                torch.testing.assert_close(actual[key], value, rtol=0, atol=0)
            torch.testing.assert_close(actual['prediction'],
                                       existing.combine_predictions(expected[0], expected[2]),
                                       rtol=0, atol=0)
            state = torch.get_rng_state().clone()
            expected_loss = existing(feature, label, reference, ref_label, return_loss=True)
            torch.set_rng_state(state)
            actual_loss = model(feature, label, reference, ref_label, return_loss=True)
            torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
            if training:
                existing.zero_grad()
                model.zero_grad()
                expected_loss.backward()
                actual_loss.backward()
                for (name, param), (_, original) in zip(model.named_parameters(),
                                                       existing.named_parameters()):
                    torch.testing.assert_close(param.grad, original.grad,
                                               rtol=0, atol=0, msg=name)

    def test_self_only_never_reads_or_encodes_references(self):
        feature, _, _, label = self.inputs()
        model = build_branch_model(self.config(), 9, 'self_only').eval()
        with patch.object(model.cell_encoder, 'forward',
                          wraps=model.cell_encoder.forward) as encoder:
            output = model.predict_components(feature, object(), object())
        self.assertEqual(encoder.call_count, 1)
        self.assertIsNone(output['y_sup'])
        self.assertIsNone(output['y_sup_agg'])
        torch.testing.assert_close(output['prediction'], output['y_ori'], rtol=0, atol=0)
        expected = 0.5 * ((output['y_ori'] - label) ** 2).mean()
        torch.testing.assert_close(model.loss_from_components(output, label),
                                   expected, rtol=0, atol=0)
        loss = model(feature, label=label, return_loss=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_reference_only_shared_encoder_gradient_and_mean_median(self):
        feature, reference, ref_label, label = self.inputs()
        feature.requires_grad_()
        reference.requires_grad_()
        model = build_branch_model(self.config(), 9, 'reference_only')
        for training in (False, True):
            model.train(training)
            with patch.object(model.cell_encoder, 'forward',
                              wraps=model.cell_encoder.forward) as encoder:
                output = model.predict_components(feature, reference, ref_label)
            self.assertEqual(encoder.call_count, 2)
            self.assertIsNone(output['y_ori'])
            expected = (output['y_sup'].mean(1) if training
                        else output['y_sup'].median(1)[0])
            torch.testing.assert_close(output['prediction'], expected, rtol=0, atol=0)
            loss = model.loss_from_components(output, label)
            torch.testing.assert_close(loss, 0.5 * ((expected - label) ** 2).mean(),
                                       rtol=0, atol=0)
            if training:
                loss.backward()
                for tensor in (feature, reference):
                    self.assertIsNotNone(tensor.grad)
                    self.assertGreater(float(tensor.grad.abs().sum()), 0)
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_raw_preparation_matches_once_prepared_and_does_not_mutate_inputs(self):
        feature, reference, ref_label, _ = self.inputs()
        feature_before = feature.clone()
        reference_before = reference.clone()
        config = dict(self.config(), cycles_to_drop=[10])
        for mode in BRANCH_MODES:
            model = build_branch_model(config, 2, mode).eval()
            expected_feature = model._prepare_feature(feature)
            expected_reference = model._prepare_feature(reference.flatten(0, 1)).reshape_as(reference)
            expected = model.predict_components(expected_feature, expected_reference,
                                                ref_label, prepared=True)
            actual = model.predict_components(feature, reference, ref_label, prepared=False)
            torch.testing.assert_close(actual['prediction'], expected['prediction'],
                                       rtol=0, atol=0)
            self.assertTrue(torch.equal(feature, feature_before))
            self.assertTrue(torch.equal(reference, reference_before))

    def test_reject_invalid_mode_weight_and_missing_reference(self):
        with self.assertRaises(ValueError):
            build_branch_model(self.config(), 2, 'independent_ensemble')
        with self.assertRaises(ValueError):
            build_branch_model(dict(self.config(), alpha=0), 2, 'self_only')
        feature, reference, ref_label, _ = self.inputs()
        model = build_branch_model(self.config(), 2, 'reference_only')
        with self.assertRaises(ValueError):
            model(feature)
        with self.assertRaises(ValueError):
            model.predict_components(feature, reference, ref_label.flatten())
        with self.assertRaises(ValueError):
            model(feature, support_feature=reference, support_label=ref_label,
                  return_loss=True)

    def test_factory_accepts_branch_mode_config_and_rejects_conflicts(self):
        for mode in BRANCH_MODES:
            for config in (dict(self.config(), branch_mode=mode),
                           dict(model=dict(self.config(), branch_mode=mode))):
                model = build_branch_model(config, 2, mode)
                self.assertEqual(model.branch_mode, mode)
                self.assertEqual(build_branch_model(config, 2).branch_mode, mode)
                other = 'reference_only' if mode != 'reference_only' else 'self_only'
                with self.assertRaises(ValueError):
                    build_branch_model(config, 2, other)


if __name__ == '__main__':
    unittest.main()
