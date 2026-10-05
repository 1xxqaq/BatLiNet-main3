"""CPU checks for axial exchange, live gradients and the fixed 207-cell workflow."""
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
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import run_mix20_axis_interaction_207 as suite
import test_latent_current_207 as baseline_tests
from src.models.rul_predictors.axis_interaction_latent_cross_attention_batlinet import AxisGridFusion
from src.models.rul_predictors.cycle_mixer_latent_cross_attention_batlinet import CycleAxisResidual
from src.models.rul_predictors.dual_axis_latent_cross_attention_batlinet import CapacityAxisResidual
from src.models.rul_predictors.latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor

shared, current = suite.shared, suite.current


class AxisInteractionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_same_weights_rng_parameter_matched_control_and_local_grid(self):
        for cycles, tokens, counts in ((20, 155, (172618, 221610, 221610)),
                                      (100, 775, (213618, 265170, 265170))):
            config = shared.model_config()
            config['input_height'] = cycles
            torch.manual_seed(7)
            backbone = shared.MODELS.build(config, seed=7)
            expected_rng = torch.get_rng_state().clone()
            # A trained cycle skip must also survive a zero fusion projection.
            torch.nn.init.normal_(backbone.cell_encoder.cycle_mixer.net[-1].weight, std=.03)
            x = torch.randn(1, 6, cycles, 1000)
            states = []
            for index, architecture in enumerate(suite.NEW_MODELS):
                config = suite.model_config(architecture)
                config['input_height'] = cycles
                torch.manual_seed(7)
                model = shared.MODELS.build(config, seed=7)
                self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
                self.assertEqual(sum(p.numel() for p in model.parameters()), counts[index])
                for key, value in model.state_dict().items():
                    if 'capacity_mixer' not in key and 'axis_fusion' not in key:
                        # The cycle output layer was intentionally changed above.
                        if 'cycle_mixer.net.2.weight' not in key:
                            torch.testing.assert_close(value, backbone.state_dict()[key], rtol=0, atol=0)
                missing, unexpected = model.load_state_dict(backbone.state_dict(), strict=False)
                self.assertFalse(unexpected)
                self.assertTrue(all('capacity_mixer' in key or 'axis_fusion' in key for key in missing))
                model.eval()
                with torch.no_grad():
                    actual = model.cell_encoder(x)
                    expected = backbone.eval().cell_encoder(x)
                self.assertEqual(actual.shape, (1, tokens, 64))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                states.append(copy.deepcopy(model.state_dict()))
            self.assertEqual(states[1].keys(), states[2].keys())
            for key in states[1]:
                torch.testing.assert_close(states[1][key], states[2][key], rtol=0, atol=0)

    def test_cross_axis_context_depends_on_other_view_but_within_does_not(self):
        for mode in ('cross', 'within'):
            block = AxisGridFusion(8, 3, 5, heads=2, hidden=4, interaction_mode=mode)
            cycle = torch.randn(2, 8, 3, 5, requires_grad=True)
            capacity = torch.randn_like(cycle, requires_grad=True)
            a, b = block.axis_contexts(cycle, capacity)
            self.assertEqual(a.shape, (2, 3, 8))
            self.assertEqual(b.shape, (2, 5, 8))
            gradients = (
                torch.autograd.grad((a * torch.randn_like(a)).sum(), capacity,
                                    retain_graph=True, allow_unused=True)[0],
                torch.autograd.grad((b * torch.randn_like(b)).sum(), cycle,
                                    allow_unused=True)[0])
            for gradient in gradients:
                if mode == 'cross':
                    self.assertIsNotNone(gradient)
                    self.assertGreater(float(gradient.abs().sum()), 0.)
                else:
                    self.assertIsNone(gradient)

    def test_zero_projection_is_identity_and_every_new_path_can_learn(self):
        for mode in ('cross', 'within'):
            cycle = CycleAxisResidual(3, hidden=4)
            capacity = CapacityAxisResidual(5, hidden=4)
            fusion = AxisGridFusion(8, 3, 5, heads=2, hidden=4, interaction_mode=mode)
            grid, target = torch.randn(2, 8, 3, 5), torch.randn(2, 8, 3, 5)
            self.assertTrue(torch.equal(fusion(cycle(grid), capacity(grid)), torch.zeros_like(grid)))
            optimizer = torch.optim.SGD(list(cycle.parameters()) + list(capacity.parameters())
                                        + list(fusion.parameters()), lr=.1)
            for step in range(3):
                optimizer.zero_grad()
                c, p = cycle(grid), capacity(grid)
                torch.nn.functional.mse_loss(c + fusion(c, p), target).backward()
                if step == 0:
                    self.assertGreater(float(fusion.local_fusion[-1].weight.grad.abs().sum()), 0.)
                optimizer.step()
            for parameter in (cycle.net[0].weight, capacity.net[0].weight,
                              fusion.cycle_attention.in_proj_weight,
                              fusion.capacity_attention.in_proj_weight):
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0.)

    def test_auxiliary_forward_keeps_dropout_rng_and_backbone_output(self):
        config = shared.model_config()
        config.update(input_width=64, attention_channels=8, attention_heads=2)
        torch.manual_seed(9)
        backbone = shared.MODELS.build(config, seed=9).train()
        config = suite.model_config(suite.NEW_MODELS[2])
        config.update(input_width=64, attention_channels=8, attention_heads=2, axis_attention_heads=2)
        torch.manual_seed(9)
        enhanced = shared.MODELS.build(config, seed=9).train()
        x = torch.randn(2, 6, 20, 64)
        torch.manual_seed(90)
        a = backbone.cell_encoder(x)
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(90)
        b = enhanced.cell_encoder(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))


class InteractionWorkflowTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def fixture(self, directory):
        data, config, peer_config, history, protocols, historical, peer_root = baseline_tests.CurrentBaselineTests().fixture(directory)
        current_root = Path(directory) / 'current'
        with patch.object(shared, 'model_config', return_value=peer_config):
            _, inputs = current.audit_enhanced(peer_root, data, protocols, history)
        shared.set_seed(0)
        provenance = suite.current_provenance(history, inputs, current.runtime_info('cpu'))
        for seed in range(8):
            folder = current_root / f'{current.MODEL}_seed{seed}'
            folder.mkdir(parents=True)
            identity = dict(provenance=provenance, config=config, seed=seed, model=current.MODEL)
            model = shared.MODELS.build(config, seed=seed)
            (folder / 'run.json').write_text(json.dumps(dict(identity, status='complete')), encoding='utf-8')
            (folder / 'train.jsonl').write_text(''.join(
                json.dumps(dict(epoch=e, train_loss=.1)) + '\n' for e in range(1, 1001)), encoding='utf-8')
            shared.save_new(dict(identity, epoch=1000, state=model.state_dict()), folder / 'epoch1000.pt')
            model._fixed_test_support_index = protocols[seed]
            model.fixed_test_support_index_path = 'unused'
            prediction, diagnostics = model.predict(data, return_diagnostics=True)
            shared.save_new(dict(identity, epoch=1000, prediction=prediction, diagnostics=diagnostics,
                                 scores=shared.scores(data, prediction),
                                 checkpoint_sha256=shared.digest(folder / 'epoch1000.pt')), folder / 'test.pt')
        return data, config, peer_config, history, protocols, historical, peer_root, current_root

    def test_24_trainings_finish_before_testing_and_rerun_preserves_results(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            data, config, peer_config, history, protocols, historical, peer_root, current_root = self.fixture(directory)
            peer_hashes = {str(p): shared.digest(p) for root in (peer_root, current_root)
                           for p in root.rglob('*') if p.is_file()}
            workspace = Path(directory) / 'new'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace),
                    '--cycle-runs', str(peer_root), '--current-runs', str(current_root)]
            new_configs = {name: suite.model_config(name) for name in suite.NEW_MODELS}
            for item in new_configs.values():
                item.update(input_width=64, train_batch_size=2, attention_channels=8,
                            attention_heads=2, head_hidden_channels=8)
                if 'axis_attention_heads' in item:
                    item.update(axis_attention_heads=2, axis_fusion_hidden=4)
            calls = []
            real_train, real_predict = shared.legacy_train, LatentCrossAttentionBatLiNetRULPredictor.predict
            def train(model, training, epochs, every, num_test, progress):
                self.assertEqual((epochs, every, num_test), (1000, 100, 147))
                # One real CPU training epoch; completion records exercise the
                # 1000-epoch workflow without pretending to run a formal study.
                real_train(model, training, 1, 100, num_test)
                for epoch in range(1, 1001):
                    progress(dict(epoch=epoch, train_loss=.1, elapsed_seconds=0.))
                calls.append(model.cell_encoder.__class__.__name__)
            def predict(model, dataset, **kwargs):
                self.assertEqual(len(calls), 24)
                return real_predict(model, dataset, **kwargs)
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=(data, protocols, historical, history)), \
                    patch.object(shared, 'model_config', return_value=peer_config), \
                    patch.object(current, 'model_config', return_value=config), \
                    patch.object(suite, 'model_config', side_effect=lambda name: new_configs[name]), \
                    patch.object(shared, 'legacy_train', side_effect=train) as training, \
                    patch.object(LatentCrossAttentionBatLiNetRULPredictor, 'predict', predict):
                suite.main()
                rows = json.loads((workspace / 'summary.json').read_text(encoding='utf-8'))
                self.assertEqual(len(rows), 56)
                hashes = {str(p): shared.digest(p) for p in workspace.rglob('*') if p.is_file()}
                for name in suite.NEW_MODELS:
                    path = workspace / f'{name}_seed0/test.pt'
                    result = shared.load(path)
                    torch.testing.assert_close(result['diagnostics']['support_index'], protocols[0])
                    self.assertEqual(result['scores'], shared.scores(data, result['prediction']))
                    torch.testing.assert_close(result['truth_cycles'], torch.tensor([150., 250.]))
                suite.main()
                self.assertEqual(training.call_count, 24)
                self.assertEqual(hashes, {str(p): shared.digest(p) for p in workspace.rglob('*') if p.is_file()})
                # An existing bad test result must stop before any new training.
                path = workspace / f'{suite.NEW_MODELS[2]}_seed0/test.pt'
                bad = shared.load(path)
                bad['diagnostics']['support_index'][0, 0] = 2
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    suite.main()
                self.assertEqual(training.call_count, 24)
            self.assertEqual(peer_hashes, {str(p): shared.digest(p) for root in (peer_root, current_root)
                                           for p in root.rglob('*') if p.is_file()})

    def test_missing_history_fails_before_creating_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / 'new'
            argv = ['runner', '--history-root', directory, '--original-root', directory,
                    '--workspace', str(workspace), '--device', 'cpu']
            with patch.object(sys, 'argv', argv), self.assertRaises(FileNotFoundError):
                suite.main()
            self.assertFalse(workspace.exists())

    def test_audit_only_is_read_only_and_incomplete_runs_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            data, config, peer_config, history, protocols, historical, peer_root, current_root = self.fixture(directory)
            workspace = Path(directory) / 'new'
            argv = ['runner', '--audit-only', '--device', 'cpu', '--workspace', str(workspace),
                    '--cycle-runs', str(peer_root), '--current-runs', str(current_root)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=(data, protocols, historical, history)), \
                    patch.object(shared, 'model_config', return_value=peer_config), \
                    patch.object(current, 'model_config', return_value=config), \
                    patch.object(shared, 'legacy_train') as training:
                suite.main()
                training.assert_not_called()
                self.assertFalse(workspace.exists())
                (workspace / f'{suite.NEW_MODELS[0]}_seed0').mkdir(parents=True)
                with self.assertRaises(ValueError):
                    suite.main()
                training.assert_not_called()


if __name__ == '__main__':
    unittest.main()
