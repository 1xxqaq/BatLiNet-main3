"""CPU checks of encoder equivalence, cycle interactions and training isolation."""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import context_grid_lifetime as common
import matched_baselines as runner
import evaluate_cycle_mixer_suite as suite
from src.models.rul_predictors.cycle_mixer_latent_cross_attention_batlinet import CycleAxisResidual
from src.models.rul_predictors.matched_baselines import MatchedBaseline


class CycleMixerTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(21)
        torch.set_num_threads(2)

    def test_same_seed_original_weights_rng_and_prediction(self):
        torch.manual_seed(21)
        baseline = MatchedBaseline('latent_cross_attention')
        original_rng = torch.get_rng_state().clone()
        models = [baseline]
        for architecture in ('latent_cycle_mixer', 'latent_cycle_conv'):
            torch.manual_seed(21)
            model = MatchedBaseline(architecture)
            torch.testing.assert_close(torch.get_rng_state(), original_rng, rtol=0, atol=0)
            for key, value in baseline.base.cell_encoder.state_dict().items():
                torch.testing.assert_close(model.base.cell_encoder.base.state_dict()[key], value,
                                           rtol=0, atol=0)
            models.append(model)
        x, reference = torch.randn(2, 6, 20, 1000), torch.randn(2, 2, 6, 20, 1000)
        x[:, :, 10] = 0
        reference[:, :, :, 10] = 0
        before_x, before_r = x.clone(), reference.clone()
        labels = torch.tensor([[-.5, 1.], [1., .2]])
        for training in (False, True):
            outputs = []
            for model in models:
                model.train(training)
                torch.manual_seed(123)
                outputs.append(model({'raw': x}, {'raw': reference}, labels))
            for actual in outputs[1:]:
                for key in outputs[0]:
                    torch.testing.assert_close(actual[key], outputs[0][key], rtol=0, atol=0)
        torch.testing.assert_close(x, before_x, rtol=0, atol=0)
        torch.testing.assert_close(reference, before_r, rtol=0, atol=0)

    def test_full_cycle_interaction_keeps_capacity_channel_and_position(self):
        mixer = CycleAxisResidual(10, hidden=1)
        with torch.no_grad():
            mixer.net[0].weight.zero_()
            mixer.net[0].weight[0, 0] = 1.
            mixer.net[0].bias.zero_()
            mixer.net[-1].weight.zero_()
            mixer.net[-1].weight[9, 0] = 1.
        grid = torch.zeros(1, 2, 10, 3)
        grid[0, 0, 0, 1] = 2.
        difference = mixer(grid) - grid
        self.assertGreater(difference[0, 0, 9, 1].item(), 1.)
        self.assertEqual(difference.count_nonzero().item(), 1)
        # Moving the input to a different cycle is not equivalent to shuffling
        # unordered tokens: fixed input/output cycle positions have distinct roles.
        shifted = grid.flip(2)
        torch.testing.assert_close(mixer(shifted), shifted, rtol=0, atol=0)
        replicated = torch.zeros_like(grid)
        replicated[0, 1, 0, 2] = 2.
        other = mixer(replicated) - replicated
        torch.testing.assert_close(other[0, 1, 9, 2], difference[0, 0, 9, 1])

    def test_control_parameter_count_and_locality(self):
        for cycles in (10, 50):
            mlp = CycleAxisResidual(cycles)
            conv = CycleAxisResidual(cycles, kind='conv')
            counts = [sum(p.numel() for p in model.parameters()) for model in (mlp, conv)]
            self.assertLess(abs(counts[0] - counts[1]) / counts[0], .03)
        with torch.no_grad():
            for layer in (conv.net[0], conv.net[-1]):
                layer.weight.fill_(.01)
                layer.bias.zero_()
        grid = torch.zeros(1, 1, 50, 1)
        grid[:, :, 0] = 1.
        difference = conv(grid) - grid
        self.assertGreater(difference[:, :, :3].abs().sum().item(), 0.)
        self.assertEqual(difference[:, :, 3:].count_nonzero().item(), 0)

    def test_zero_output_initialization_can_learn_both_layers(self):
        for kind in ('mlp', 'conv'):
            mixer = CycleAxisResidual(10, kind=kind)
            grid = torch.randn(2, 3, 10, 4)
            torch.testing.assert_close(mixer(grid), grid, rtol=0, atol=0)
            optimizer = torch.optim.SGD(mixer.parameters(), lr=.01)
            for step in range(2):
                optimizer.zero_grad()
                loss = mixer(grid).square().mean()
                loss.backward()
                self.assertGreater(mixer.net[-1].weight.grad.abs().sum().item(), 0.)
                if step == 1:
                    self.assertGreater(mixer.net[0].weight.grad.abs().sum().item(), 0.)
                optimizer.step()
            self.assertFalse(torch.equal(mixer(grid), grid))

    def test_mix100_shape_and_checkpoint_round_trip(self):
        for architecture in ('latent_cycle_mixer', 'latent_cycle_conv'):
            model = MatchedBaseline(architecture, cycles=100).eval()
            feature = torch.randn(1, 6, 100, 1000)
            with torch.no_grad():
                encoded = model.base.cell_encoder(feature)
            self.assertEqual(encoded.shape, (1, 775, 64))
            self.assertEqual(model.base.cell_encoder.cycle_mixer.cycles, 50)
            restored = MatchedBaseline(architecture, cycles=100).eval()
            restored.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored.base.cell_encoder(feature), encoded, rtol=0, atol=0)

    def test_nonzero_mixer_chunking_and_single_median_over_32_references(self):
        training = dict(ids=['a', 'b', 'c'], labels=torch.tensor([100., 200., 400.]),
                        features={'raw': torch.randn(3, 6, 20, 1000)})
        query = dict(ids=['d'], labels=torch.tensor([150.]),
                     features={'raw': torch.randn(1, 6, 20, 1000)})
        indices = common.make_indices(1, 3, 55)
        for architecture in ('latent_cycle_mixer', 'latent_cycle_conv'):
            model = MatchedBaseline(architecture)
            with torch.no_grad():
                model.base.cell_encoder.cycle_mixer.net[-1].weight.normal_(0., .01)
            a = runner.evaluate(model, training, query, indices, 5., .5, 'cpu', pair_chunk=7)
            b = runner.evaluate(model, training, query, indices, 5., .5, 'cpu', pair_chunk=32)
            torch.testing.assert_close(a['prediction'], b['prediction'])
            torch.testing.assert_close(a['diagnostics']['y_sup'], b['diagnostics']['y_sup'])
            torch.testing.assert_close(a['diagnostics']['y_sup_agg'], a['diagnostics']['y_sup'].median(1).values)

    def test_matched_training_checkpoint_and_no_test_feature_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = {part: dict(ids=[part+str(i) for i in range(n)],
                                 labels=torch.arange(n).float()*30+120)
                       for part, n in [('train', 3), ('val', 1), ('test', 1)]}
            cp, bp = root / 'context.pt', root / 'baseline.pt'
            torch.save(context, cp)
            data = copy.deepcopy(context)
            data['metadata'] = {'context_sha256': common.digest(cp)}
            for part in ('train', 'val'):
                data[part]['features'] = {'raw': torch.randn(len(data[part]['ids']), 6, 20, 1000)}
            # Training would fail if it attempted to evaluate the test partition.
            torch.save(data, bp)
            for architecture in ('latent_cycle_mixer', 'latent_cycle_conv'):
                args = SimpleNamespace(context_data=str(cp), data=str(bp),
                    protocol_dir=str(root / 'protocols'), workspace=str(root / architecture),
                    model=architecture, seed=0, epochs=2, batch_size=2, accumulation=2,
                    evaluate_every=1, lr=.001, pair_chunk=8, device='cpu', amp=False,
                    skip_complete=True, command='train')
                with redirect_stdout(io.StringIO()):
                    runner.train(args)
                    runner.train(args)
                checkpoint = common.load(root / architecture / 'best.pt')
                self.assertEqual(checkpoint['config'], suite.model_config(architecture))
                restored = MatchedBaseline(**checkpoint['config'])
                restored.load_state_dict(checkpoint['state'], strict=True)
                self.assertGreater(restored.base.cell_encoder.cycle_mixer.net[-1].weight.abs().sum().item(), 0.)

    def test_suite_preflight_rejects_protocol_and_selection_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            protocol = root / 'val.pt'
            torch.save({'indices': torch.zeros(1, 32, dtype=torch.long)}, protocol)
            data = {'train': {'ids': ['a', 'b'], 'labels': torch.tensor([100., 200.])}}
            architecture = 'latent_cycle_mixer'
            config = suite.model_config(architecture)
            arguments = dict(model=architecture, seed=0, epochs=1000, batch_size=8,
                             accumulation=16, evaluate_every=25, lr=.001, amp=True, pair_chunk=8)
            info = dict(arguments=arguments, model=config, data_sha256='data',
                        context_sha256='context', train_ids=['a', 'b'],
                        validation_protocol_sha256=common.digest(protocol))
            (root / 'run.json').write_text(json.dumps(info), encoding='utf-8')
            records = [dict(epoch=epoch, **({'validation': {'RMSE': float(epoch)}}
                       if epoch % 25 == 0 else {})) for epoch in range(1, 1001)]
            (root / 'train.jsonl').write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
            label = data['train']['labels'].log()
            checkpoint = dict(config=config, epoch=25, seed=0, refit=False, selection_metric='RMSE',
                data_sha256='data', context_sha256='context', train_ids=['a', 'b'],
                label_mean=label.mean(), label_scale=label.std(unbiased=False))
            torch.save(checkpoint, root / 'best.pt')
            suite.preflight(root, data, 'data', 'context', 0, architecture, protocol)
            checkpoint['epoch'] = 50
            torch.save(checkpoint, root / 'best.pt')
            with self.assertRaisesRegex(ValueError, 'Best checkpoint'):
                suite.preflight(root, data, 'data', 'context', 0, architecture, protocol)
            checkpoint['epoch'] = 25
            torch.save(checkpoint, root / 'best.pt')
            torch.save({'indices': torch.ones(1, 32, dtype=torch.long)}, protocol)
            with self.assertRaisesRegex(ValueError, 'references changed'):
                suite.preflight(root, data, 'data', 'context', 0, architecture, protocol)

    def test_suite_entry_audits_three_models_without_test_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = {'metadata': {'dataset': 'mix20'}}
            for part, n in (('train', 166), ('val', 41), ('test', 147)):
                context[part] = dict(ids=[f'{part}{i}' for i in range(n)],
                                     labels=torch.arange(n).float() + 120)
            cp, bp = root / 'context.pt', root / 'data.pt'
            torch.save(context, cp)
            data = copy.deepcopy(context)
            data['metadata'] = {'context_sha256': common.digest(cp)}
            # An audit must not run any predictor, so features are omitted.
            torch.save(data, bp)
            protocol = root / 'protocols' / 'val_seed0.pt'
            runner.fixed_protocol(context, 0, 'val', protocol)
            records = [dict(epoch=epoch, **({'validation': dict(
                RMSE=float(epoch), MAE=10., MAPE=.1, ACC15=50.)}
                if epoch % 25 == 0 else {})) for epoch in range(1, 1001)]
            label = data['train']['labels'].log()
            for architecture in suite.ARCHITECTURES:
                folder = root / 'runs' / f'{architecture}_seed0'
                folder.mkdir(parents=True)
                config = suite.model_config(architecture)
                arguments = dict(model=architecture, seed=0, epochs=1000, batch_size=8,
                                 accumulation=16, evaluate_every=25, lr=.001, amp=True, pair_chunk=8)
                identity = dict(data_sha256=common.digest(bp), context_sha256=common.digest(cp),
                                train_ids=data['train']['ids'])
                info = dict(arguments=arguments, model=config, **identity,
                            validation_protocol_sha256=common.digest(protocol))
                (folder / 'run.json').write_text(json.dumps(info), encoding='utf-8')
                (folder / 'train.jsonl').write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
                torch.save(dict(config=config, epoch=25, seed=0, refit=False,
                    selection_metric='RMSE', label_mean=label.mean(),
                    label_scale=label.std(unbiased=False), **identity), folder / 'best.pt')
            argv = ['audit', '--context-data', str(cp), '--data', str(bp),
                    '--baseline-runs', str(root / 'runs'), '--runs', str(root / 'runs'),
                    '--protocol-dir', str(root / 'protocols'), '--output-dir', str(root / 'output'),
                    '--seeds', '0']
            with patch.object(sys, 'argv', argv), redirect_stdout(io.StringIO()):
                suite.main()
            summary = json.loads((root / 'output' / 'validation_summary.json').read_text())
            self.assertEqual([row['model'] for row in summary], list(suite.ARCHITECTURES))
            self.assertEqual(list((root / 'output').rglob('*.pt')), [])
            with patch.object(sys, 'argv', argv + ['--run-test']), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    suite.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
