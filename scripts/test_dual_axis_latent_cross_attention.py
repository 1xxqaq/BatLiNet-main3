"""CPU checks for parallel axes, matched training and complete-suite isolation."""
import copy
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]
import context_grid_lifetime as common
import matched_baselines as matched
import run_mix20_dual_axis as suite
from src.models.rul_predictors.dual_axis_latent_cross_attention_batlinet import CapacityAxisResidual
from src.models.rul_predictors.dual_axis_matched_baselines import DualAxisMatchedBaseline
from src.models.rul_predictors.matched_baselines import MatchedBaseline


def fixture(root):
    context = dict(metadata={'dataset': 'mix20'})
    for part, n in (('train', 3), ('val', 1), ('test', 1)):
        context[part] = dict(ids=[f'{part}{i}' for i in range(n)],
                             labels=torch.arange(n).float() * 30 + 120)
    cp, dp = root / 'context.pt', root / 'data.pt'
    torch.save(context, cp)
    data = copy.deepcopy(context)
    data['metadata'] = dict(context_sha256=common.digest(cp))
    for part in ('train', 'val'):
        data[part]['features'] = {'raw': torch.randn(len(data[part]['ids']), 6, 20, 1000)}
    # No test features: accidental test prediction during training must fail.
    torch.save(data, dp)
    protocols = root / 'protocols'
    for seed in range(8):
        for part in ('val', 'test'):
            matched.fixed_protocol(context, seed, part, protocols / f'{part}_seed{seed}.pt')
    args = SimpleNamespace(context_data=str(cp), data=str(dp), protocol_dir=str(protocols),
        cycle_runs=str(root / 'cycle'), workspace_root=str(root / 'new'),
        output_dir=str(root / 'output'), device='cpu', audit_only=False, run_test=False)
    return args, context, data


def peer_runs(args, data):
    name = suite.MODELS[0]
    config = suite.model_config(name)
    label = data['train']['labels'].log()
    identity = dict(data_sha256=common.digest(args.data), context_sha256=common.digest(args.context_data),
                    train_ids=data['train']['ids'])
    for seed in range(8):
        folder = Path(args.cycle_runs) / f'{name}_seed{seed}'
        folder.mkdir(parents=True)
        info = dict(arguments=dict(model=name, seed=seed, **suite.SETTINGS), model=config,
            validation_protocol_sha256=common.digest(Path(args.protocol_dir) / f'val_seed{seed}.pt'),
            **identity)
        (folder / 'run.json').write_text(json.dumps(info), encoding='utf-8')
        records = [dict(epoch=e, train_loss=1., **({'validation': dict(
            RMSE=float(e), MAE=10., MAPE=.1, ACC15=.5)} if e % 25 == 0 else {}))
                   for e in range(1, 1001)]
        (folder / 'train.jsonl').write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
        model = suite.build_model(config)
        torch.save(dict(state=model.state_dict(), config=config, epoch=25, seed=seed,
            label_mean=label.mean(), label_scale=label.std(unbiased=False), refit=False,
            selection_metric='RMSE', **identity), folder / 'best.pt')


class DualAxisTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_same_seed_weights_rng_counts_and_token_grid(self):
        for cycles, expected in ((20, (170556, 172272, 172618, 155)),
                                  (100, (211556, 211952, 213618, 775))):
            torch.manual_seed(7)
            cycle = MatchedBaseline('latent_cycle_mixer', cycles=cycles)
            rng = torch.get_rng_state().clone()
            for index, name in enumerate(suite.NEW_MODELS):
                torch.manual_seed(7)
                model = DualAxisMatchedBaseline(name, cycles=cycles)
                self.assertTrue(torch.equal(torch.get_rng_state(), rng))
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected[index + 1])
                for key, value in model.state_dict().items():
                    if 'capacity_mixer' not in key:
                        torch.testing.assert_close(value, cycle.state_dict()[key], rtol=0, atol=0)
                x = torch.randn(1, 6, cycles, 1000)
                with torch.no_grad():
                    a = cycle.eval().base.cell_encoder(x)
                    b = model.eval().base.cell_encoder(x)
                self.assertEqual(b.shape, (1, expected[-1], 64))
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertEqual(sum(p.numel() for p in cycle.parameters()), expected[0])

    def test_zero_capacity_and_disabling_it_preserve_trained_cycle_predictor(self):
        cycle = MatchedBaseline('latent_cycle_mixer').eval()
        nn = cycle.base.cell_encoder.cycle_mixer.net
        torch.nn.init.normal_(nn[-1].weight, std=.03)
        dual = DualAxisMatchedBaseline('latent_dual_axis_mixer').eval()
        missing, unexpected = dual.load_state_dict(cycle.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all('capacity_mixer' in key for key in missing))
        target = {'raw': torch.randn(1, 6, 20, 1000)}
        references = {'raw': torch.randn(1, 2, 6, 20, 1000)}
        labels = torch.tensor([[-.2, .8]])
        with torch.no_grad():
            expected = cycle(target, references, labels)
            for key, value in dual(target, references, labels).items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
            torch.nn.init.normal_(dual.base.cell_encoder.capacity_mixer.net[-1].weight)
            dual.base.cell_encoder.enable_capacity = False
            for key, value in dual(target, references, labels).items():
                torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

    def test_capacity_branch_preserves_channel_cycle_positions(self):
        capacity = CapacityAxisResidual(62)
        torch.nn.init.normal_(capacity.net[-1].weight)
        grid = torch.randn(2, 3, 10, 62)
        changed = grid.clone()
        changed[1, 2, 4, 8] += 10
        delta = capacity.correction(changed) - capacity.correction(grid)
        keep = delta[1, 2, 4].clone()
        delta[1, 2, 4] = 0
        self.assertEqual(delta.count_nonzero(), 0)
        self.assertGreater(keep.count_nonzero(), 1)

    def test_parallel_residuals_and_both_layers_learn(self):
        encoder = DualAxisMatchedBaseline('latent_dual_axis_mixer').base.cell_encoder
        grid = torch.randn(2, 64, 10, 62)
        optimizer = torch.optim.AdamW(encoder.parameters(), lr=.001)
        for iteration in range(2):
            optimizer.zero_grad()
            encoder.mix_grid(grid).square().mean().backward()
            for module in (encoder.cycle_mixer, encoder.capacity_mixer):
                self.assertGreater(module.net[-1].weight.grad.abs().sum().item(), 0)
                if iteration:
                    self.assertGreater(module.net[0].weight.grad.abs().sum().item(), 0)
            optimizer.step()
        expected = encoder.cycle_mixer(grid) + encoder.capacity_mixer.correction(grid)
        serial = encoder.capacity_mixer(encoder.cycle_mixer(grid))
        torch.testing.assert_close(encoder.mix_grid(grid), expected, rtol=0, atol=0)
        self.assertGreater((expected - serial).abs().max().item(), 1e-8)

    def test_training_loop_matches_existing_cycle_loop_including_partial_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, context, data = fixture(root)
            settings = dict(suite.SETTINGS, epochs=2, batch_size=2, accumulation=2,
                            evaluate_every=1, amp=False)
            gold = SimpleNamespace(context_data=args.context_data, data=args.data,
                protocol_dir=args.protocol_dir, workspace=str(root / 'gold'),
                model='latent_cycle_mixer', seed=0, **settings,
                device='cpu', skip_complete=False, command='train')
            identity = dict(data_sha256=common.digest(args.data), context_sha256=common.digest(args.context_data))
            protocol = Path(args.protocol_dir) / 'val_seed0.pt'
            with redirect_stdout(io.StringIO()):
                matched.train(gold)
                expected_rng = torch.get_rng_state().clone()
                suite.train_one(gold.model, 0, root / 'new_loop', data,
                    common.load(protocol)['indices'], identity, protocol,
                    suite.source_hashes(), suite.runtime_info('cpu'), 'cpu', settings, 0)
            self.assertTrue(torch.equal(expected_rng, torch.get_rng_state()))
            a, b = common.load(root / 'gold' / 'best.pt'), common.load(root / 'new_loop' / 'best.pt')
            self.assertEqual(a['epoch'], b['epoch'])
            for key in a['state']:
                torch.testing.assert_close(a['state'][key], b['state'][key], rtol=0, atol=0)
            for a, b in zip(suite.read_records(root / 'gold'), suite.read_records(root / 'new_loop')):
                self.assertEqual(a['train_loss'], b['train_loss'])
                self.assertEqual(a['validation'], b['validation'])

    def test_actual_sixteen_runs_validation_only_reuse_and_audit_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, context, data = fixture(root)
            peer_runs(args, data)
            settings = dict(suite.SETTINGS, epochs=2, batch_size=2, accumulation=2,
                            evaluate_every=1, amp=False)
            baseline_before = {str(p): common.digest(p) for p in Path(args.cycle_runs).rglob('*') if p.is_file()}
            protocols_before = {str(p): common.digest(p) for p in Path(args.protocol_dir).glob('*.pt')}
            with redirect_stdout(io.StringIO()):
                args.audit_only = True
                suite.execute(args, context, data, settings)
                self.assertFalse(Path(args.workspace_root).exists())
                self.assertFalse(Path(args.output_dir).exists())
                args.audit_only, args.run_test = False, True
                with self.assertRaisesRegex(ValueError, '全部16次'):
                    suite.execute(args, context, data, settings)
                self.assertFalse(Path(args.workspace_root).exists())
                args.run_test = False
                suite.execute(args, context, data, settings)
                files_before = {str(p): common.digest(p) for p in Path(args.workspace_root).rglob('*') if p.is_file()}
                suite.execute(args, context, data, settings)
                self.assertEqual(files_before, {str(p): common.digest(p)
                    for p in Path(args.workspace_root).rglob('*') if p.is_file()})
                args.audit_only = True
                with patch.object(suite, 'runtime_info', side_effect=AssertionError('audit touched GPU')):
                    suite.execute(args, context, data, settings)
            rows = json.loads((Path(args.output_dir) / 'validation_summary.json').read_text())
            self.assertEqual(len(rows), 24)
            self.assertEqual(list(Path(args.output_dir).rglob('*.pt')), [])
            self.assertEqual(baseline_before, {str(p): common.digest(p)
                for p in Path(args.cycle_runs).rglob('*') if p.is_file()})
            self.assertEqual(protocols_before, {str(p): common.digest(p)
                for p in Path(args.protocol_dir).glob('*.pt')})
            # Test features become available only for the explicit test stage.
            data['test']['features'] = {'raw': torch.randn(1, 6, 20, 1000)}
            args.audit_only, args.run_test = False, True
            with redirect_stdout(io.StringIO()):
                suite.execute(args, context, data, settings)
                results_before = {str(p): common.digest(p) for p in Path(args.output_dir).rglob('*.pt')}
                suite.execute(args, context, data, settings)
            self.assertEqual(len(results_before), 24)
            self.assertEqual(results_before, {str(p): common.digest(p)
                for p in Path(args.output_dir).rglob('*.pt')})
            info_path = Path(args.workspace_root) / f'{suite.NEW_MODELS[0]}_seed0' / 'run.json'
            info = json.loads(info_path.read_text())
            info['source_sha256']['scripts/run_mix20_dual_axis.py'] = 'changed'
            info_path.write_text(json.dumps(info), encoding='utf-8')
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, '源代码指纹'):
                suite.execute(args, context, data, settings)

    def test_missing_late_seed_reference_stops_before_any_new_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args, context, data = fixture(root)
            peer_runs(args, data)
            protocol = Path(args.protocol_dir) / 'test_seed7.pt'
            protocol.rename(protocol.with_suffix('.held'))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(FileNotFoundError, '固定参考名单'):
                suite.execute(args, context, data)
            self.assertFalse(Path(args.workspace_root).exists())
            self.assertFalse(Path(args.output_dir).exists())

    def test_full_32_reference_median_and_chunking_with_trained_residuals(self):
        train = dict(ids=['a', 'b', 'c'], labels=torch.tensor([100., 200., 400.]),
                     features={'raw': torch.randn(3, 6, 20, 1000)})
        query = dict(ids=['d'], labels=torch.tensor([150.]),
                     features={'raw': torch.randn(1, 6, 20, 1000)})
        model = DualAxisMatchedBaseline('latent_dual_axis_mixer')
        for module in (model.base.cell_encoder.cycle_mixer, model.base.cell_encoder.capacity_mixer):
            torch.nn.init.normal_(module.net[-1].weight, std=.03)
        indices = common.make_indices(1, 3, 55)
        a = matched.evaluate(model, train, query, indices, 5., .5, 'cpu', 7)
        b = matched.evaluate(model, train, query, indices, 5., .5, 'cpu', 32)
        torch.testing.assert_close(a['prediction'], b['prediction'])
        torch.testing.assert_close(a['diagnostics']['y_sup'], b['diagnostics']['y_sup'])
        torch.testing.assert_close(a['diagnostics']['y_sup_agg'], a['diagnostics']['y_sup'].median(1).values)


if __name__ == '__main__':
    unittest.main()
