"""Model structure, additive gradients and isolated branch-randomness checks."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.rul_predictors.branch_ablation_batlinet import build_branch_model
from src.models.rul_predictors.naive_branch_ensemble_batlinet import (
    BranchRandomStreams, build_naive_model,
)


class NaiveEnsembleModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def config(self, **updates):
        config = dict(in_channels=6, channels=32, input_height=20, input_width=128,
                      attention_channels=8, attention_heads=2, attention_layers=1,
                      head_hidden_channels=8, cycle_mixer_hidden=16, alpha=0.5,
                      filter_cycles=False, train_batch_size=4, train_support_size=2,
                      test_support_size=4, encoder_dropout=0.1,
                      attention_dropout=0.1)
        config.update(updates)
        return config

    def inputs(self):
        generator = torch.Generator().manual_seed(7)
        return (torch.randn(2, 6, 20, 128, generator=generator),
                torch.randn(2, 4, 6, 20, 128, generator=generator),
                torch.randn(2, 4, generator=generator),
                torch.randn(2, generator=generator))

    def test_parameter_counts_unshared_encoders_and_full_model_initial_values(self):
        before = torch.get_rng_state().clone()
        for height, expected in ((20, 290614), (100, 372614)):
            config = self.config(input_height=height, input_width=1000,
                                 attention_channels=64, attention_heads=4,
                                 head_hidden_channels=64)
            model = build_naive_model(dict(model=dict(config, branch_mode='naive_ensemble')), 13)
            full = build_branch_model(config, 13, 'full')
            self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
            self.assertEqual(model.branch_mode, 'naive_ensemble')
            self.assertTrue(model.requires_reference)
            self.assertEqual(model.train_batch_size, 4)
            self.assertIs(model.own, model.self_branch)
            self.assertIs(model.relation, model.reference_branch)
            pointers = {p.data_ptr() for p in model.self_branch.parameters()}
            self.assertFalse(pointers & {p.data_ptr() for p in model.reference_branch.parameters()})
            for branch in (model.self_branch, model.reference_branch):
                for name, value in branch.state_dict().items():
                    torch.testing.assert_close(value, full.state_dict()[name], rtol=0, atol=0)
            self.assertTrue(all(key.startswith(('self_branch.', 'reference_branch.'))
                                for key in model.state_dict()))
            self.assertFalse(hasattr(model.self_branch, 'support_head'))
            self.assertFalse(hasattr(model.reference_branch, 'ori_head'))
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_branch_losses_have_disjoint_gradients_and_no_fusion_loss(self):
        feature, reference, ref_label, label = self.inputs()
        model = build_naive_model(self.config(), 13).train()
        result = model.predict_components(feature, reference, ref_label,
                                          streams=BranchRandomStreams(13))
        own_loss = .5 * ((result['y_ori'] - label) ** 2).mean()
        ref_loss = .5 * ((result['y_sup_agg'] - label) ** 2).mean()
        own_loss.backward(retain_graph=True)
        own_grads = {name: parameter.grad.clone()
                     for name, parameter in model.self_branch.named_parameters()}
        self.assertTrue(all(p.grad is None for p in model.reference_branch.parameters()))
        model.zero_grad(set_to_none=True)
        ref_loss.backward(retain_graph=True)
        ref_grads = {name: parameter.grad.clone()
                     for name, parameter in model.reference_branch.named_parameters()}
        self.assertTrue(all(p.grad is None for p in model.self_branch.parameters()))
        model.zero_grad(set_to_none=True)
        loss = model.loss_from_components(result, label)
        torch.testing.assert_close(loss, own_loss + ref_loss, rtol=0, atol=0)
        loss.backward()
        for branch, expected in ((model.self_branch, own_grads),
                                 (model.reference_branch, ref_grads)):
            for name, parameter in branch.named_parameters():
                torch.testing.assert_close(parameter.grad, expected[name], rtol=0, atol=0)

    def test_mean_median_fusion_and_reference_encoder_used_for_both_inputs(self):
        feature, reference, ref_label, label = self.inputs()
        model = build_naive_model(self.config(), 13)
        for training in (True, False):
            model.train(training)
            with patch.object(model.self_branch.cell_encoder, 'forward',
                              wraps=model.self_branch.cell_encoder.forward) as own_encoder, \
                    patch.object(model.reference_branch.cell_encoder, 'forward',
                                 wraps=model.reference_branch.cell_encoder.forward) as ref_encoder:
                result = model.predict_components(feature, reference, ref_label)
            self.assertEqual(own_encoder.call_count, 1)
            self.assertEqual(ref_encoder.call_count, 2)
            aggregated = (result['y_sup'].mean(1) if training
                          else result['y_sup'].median(1).values)
            torch.testing.assert_close(result['y_sup_agg'], aggregated, rtol=0, atol=0)
            torch.testing.assert_close(result['prediction'],
                                       .5 * result['y_ori'] + .5 * aggregated, rtol=0, atol=0)
        expected = model.predict_components(feature, reference, ref_label)
        torch.testing.assert_close(model(feature, label, reference, ref_label, return_loss=True),
                                   model.loss_from_components(expected, label), rtol=0, atol=0)

    def test_prepare_and_support_sampling_delegate_without_mutating_inputs(self):
        feature, reference, ref_label, _ = self.inputs()
        feature_before = feature.clone()
        reference_before = reference.clone()
        model = build_naive_model(self.config(cycles_to_drop=[10]), 13).eval()
        query = model._prepare_feature(feature)
        refs = model._prepare_feature(reference.flatten(0, 1)).reshape_as(reference)
        expected = model.predict_components(query, refs, ref_label)
        actual = model.predict_components(feature, reference, ref_label, prepared=False)
        torch.testing.assert_close(actual['prediction'], expected['prediction'], rtol=0, atol=0)
        self.assertTrue(torch.equal(feature, feature_before))
        self.assertTrue(torch.equal(reference, reference_before))
        indices = torch.tensor([[0, 1], [1, 0]])
        sx, sy = model.get_support_set(query, query, torch.arange(2.),
                                       fixed_indices=indices, support_is_prepared=True)
        self.assertEqual(tuple(sx.shape), (2, 2, 6, 20, 128))
        self.assertTrue(torch.equal(sy, indices.float()))

    def test_random_streams_isolate_advance_restore_and_do_not_alias_saved_state(self):
        outside = torch.get_rng_state().clone()
        streams = BranchRandomStreams(13)
        self.assertTrue(torch.equal(outside, torch.get_rng_state()))
        expected = torch.Generator().manual_seed(13)
        first = torch.rand(4, generator=expected)
        second = torch.rand(4, generator=expected)
        torch.testing.assert_close(streams.call('self_only', torch.rand, 4), first, rtol=0, atol=0)
        saved = streams.state_dict()
        torch.testing.assert_close(streams.call('self_only', torch.rand, 4), second, rtol=0, atol=0)
        torch.testing.assert_close(streams.call('reference_only', torch.rand, 4), first, rtol=0, atol=0)
        self.assertTrue(torch.equal(outside, torch.get_rng_state()))
        restored = BranchRandomStreams(13)
        restored.load_state_dict(saved)
        saved['states']['self_only']['cpu'].zero_()
        torch.testing.assert_close(restored.call('self_only', torch.rand, 4), second, rtol=0, atol=0)
        self.assertTrue(torch.equal(outside, torch.get_rng_state()))
        def fail():
            torch.rand(3)
            raise RuntimeError('deliberate failure')
        with self.assertRaisesRegex(RuntimeError, 'deliberate'):
            restored.call('self_only', fail)
        self.assertTrue(torch.equal(outside, torch.get_rng_state()))

    def test_model_forward_with_streams_leaves_global_rng_unchanged(self):
        feature, reference, ref_label, label = self.inputs()
        model = build_naive_model(self.config(), 13).train()
        streams = BranchRandomStreams(13)
        before = torch.get_rng_state().clone()
        saved = streams.state_dict()
        first = model(feature, label, reference, ref_label, return_loss=True, streams=streams)
        streams.load_state_dict(saved)
        again = model(feature, label, reference, ref_label, return_loss=True, streams=streams)
        torch.testing.assert_close(first, again, rtol=0, atol=0)
        self.assertTrue(torch.equal(before, torch.get_rng_state()))

    def test_reject_invalid_mode_and_stream_states(self):
        with self.assertRaises(ValueError):
            build_naive_model(self.config(branch_mode='full'), 13)
        streams = BranchRandomStreams(13)
        with self.assertRaises(ValueError):
            streams.call('full', torch.rand, 4)
        for change in (dict(seed=14), dict(device='cuda:0'), dict(schema=2), dict(states={})):
            with self.assertRaises(ValueError):
                streams.load_state_dict(dict(streams.state_dict(), **change))
        invalid = streams.state_dict()
        invalid['states']['self_only']['cpu'] = torch.ones(3)
        with self.assertRaises(ValueError):
            streams.load_state_dict(invalid)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable on local test host')
    def test_cuda_stream_matches_seeded_local_generator_without_changing_caller(self):
        device = torch.device('cuda', torch.cuda.current_device())
        before_cpu = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state(device).clone()
        streams = BranchRandomStreams(13, device)
        expected = torch.rand(8, device=device,
                              generator=torch.Generator(device=device).manual_seed(13))
        actual = streams.call('self_only', torch.rand, 8, device=device)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertTrue(torch.equal(before_cpu, torch.get_rng_state()))
        self.assertTrue(torch.equal(before_cuda, torch.cuda.get_rng_state(device)))
        restored = BranchRandomStreams(13, device)
        restored.load_state_dict(streams.state_dict())
        torch.testing.assert_close(streams.call('self_only', torch.rand, 8, device=device),
                                   restored.call('self_only', torch.rand, 8, device=device),
                                   rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
