"""Formula, independent controls, training recovery and queue contract checks."""
import copy
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mix_soft_difference_refinements as suite
from scripts.test_mix_cycle_soft_difference import CombinationTests
from src.models.rul_predictors.soft_difference_refinements import ContentDifferenceBlock
import torch
from torch.nn import functional as F


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def small(self, variant):
        cfg = copy.deepcopy(suite.model_config('mix_20', variant))
        cfg['model'].update(input_width=128, train_batch_size=4, epochs=2,
                            attention_channels=8, attention_heads=2, head_hidden_channels=8)
        cfg['runner'].update(micro_batch_size=2, checkpoint_every=1, periodic_rng_every=1)
        return cfg

    def test_content_formula_and_positions_only_in_matching(self):
        torch.manual_seed(3)
        block = ContentDifferenceBlock(8, 2, dropout=.1).eval()
        t, r = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
        tc, rc = torch.randn_like(t), torch.randn_like(r)
        wq, wk, wv = block.attn.in_proj_weight.chunk(3)
        bq, bk, bv = block.attn.in_proj_bias.chunk(3)
        split = lambda x: x.reshape(2, -1, 2, 4).transpose(1, 2)
        q = split(F.linear(block.q_norm(t), wq, bq))
        k = split(F.linear(block.kv_norm(r), wk, bk))
        vt = split(F.linear(block.kv_norm(tc), wv, bv))
        vr = split(F.linear(block.kv_norm(rc), wv, bv))
        expected = (vt - (q @ k.transpose(-1, -2) / 2).softmax(-1) @ vr)
        expected = expected.transpose(1, 2).reshape(2, 3, 8)
        torch.testing.assert_close(block.projected_difference(t, r, tc, rc), expected)
        # With one key the match is fixed; changing position cannot change values.
        a = block.projected_difference(t[:, :1], r[:, :1], tc[:, :1], rc[:, :1])
        b = block.projected_difference(t[:, :1]*3, r[:, :1]*5, tc[:, :1], rc[:, :1])
        torch.testing.assert_close(a, b)
        with self.assertRaises(ValueError):
            block.projected_difference(t, r)

    def test_parameter_counts_initialization_rng_and_real_shapes(self):
        for task, expected in (('mix_20', 170210), ('mix_100', 209890)):
            models = []
            for variant in suite.VARIANTS:
                cfg = suite.model_config(task, variant)['model']
                torch.manual_seed(19)
                model = suite.common.MODELS.build(cfg, seed=19).eval()
                rng = torch.get_rng_state().clone()
                self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
                if models:
                    self.assertTrue(torch.equal(rng, reference_rng))
                    for key, value in models[0].state_dict().items():
                        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
                else:
                    reference_rng = rng
                x = torch.randn(1, 6, cfg['input_height'], 1000)
                with torch.no_grad():
                    outputs = model.compute_prediction_components(x, x[:, None], torch.zeros(1, 1))
                self.assertEqual(outputs[0].shape, (1,))
                self.assertTrue(all(torch.isfinite(v).all() for v in outputs[:3]))
                models.append(model)
            x = torch.randn(1, 6, cfg['input_height'], 1000)
            with torch.no_grad():
                baseline = models[0](x, torch.zeros(1), x[:, None], torch.zeros(1, 1))
                for model in models[2:]:
                    torch.testing.assert_close(baseline, model(x, torch.zeros(1), x[:, None], torch.zeros(1, 1)), atol=0, rtol=0)

    def test_only_matching_probability_dropout_disabled(self):
        model = suite.common.MODELS.build(self.small('no_matching_dropout')['model'], seed=0).train()
        block = model.cross_attention[0]
        self.assertEqual(block.attn.dropout, 0.)
        self.assertEqual(block.attn_dropout.p, .1)
        self.assertEqual(block.ffn[2].p, .1)
        self.assertEqual(model.cell_encoder.net[-1].p, .1)
        self.assertEqual(model.ori_head.net[-2].p, .1)
        t = torch.randn(2, 3, 8)
        with patch('torch.nn.functional.scaled_dot_product_attention', wraps=F.scaled_dot_product_attention) as attention:
            block.projected_difference(t, t)
            self.assertEqual(attention.call_args.kwargs['dropout_p'], 0.)

    def test_pair_loss_exact_and_inference_unchanged(self):
        model = suite.common.MODELS.build(self.small('pair_supervision')['model'], seed=0)
        own, refs, labels = torch.tensor([1., 3.]), torch.tensor([[0., 2.], [2., 6.]]), torch.tensor([2., 4.])
        agg = refs.mean(1)
        expected = .5*(own-labels).square().mean()+.5*(agg-labels).square().mean()+.1*(refs-labels[:, None]).square().mean()
        with patch.object(model, 'compute_prediction_components', return_value=(own, refs, agg, None, None)):
            torch.testing.assert_close(model(None, labels, None, None, return_loss=True), expected)
            torch.testing.assert_close(model(None, labels, None, None), .5*(own+agg))
        for bad in (-1., float('nan'), float('inf')):
            cfg = self.small('pair_supervision')['model']; cfg['pair_loss_weight'] = bad
            with self.assertRaises(ValueError):
                suite.common.MODELS.build(cfg, seed=0)

    def test_gradients_reach_encoder_positions_and_qkv(self):
        for variant in suite.VARIANTS:
            model = suite.common.MODELS.build(self.small(variant)['model'], seed=0).train()
            x = torch.randn(2, 6, 20, 128, requires_grad=True)
            ref = torch.randn(2, 2, 6, 20, 128, requires_grad=True)
            loss = model(x, torch.randn(2), ref, torch.randn(2, 2), return_loss=True)
            loss.backward()
            for grad in (x.grad, ref.grad, model.cell_encoder.position.grad,
                         model.cell_encoder.net[0].weight.grad,
                         *model.cross_attention[0].attn.in_proj_weight.grad.chunk(3)):
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(grad.abs().sum().item(), 0)

    def test_reference_schedule_ignores_dropout_rng_and_epoch_varies(self):
        torch.manual_seed(1)
        rng = torch.get_rng_state().clone()
        a = suite.reference_indices(207, 2, 7, 23)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        torch.rand(2000)
        self.assertTrue(torch.equal(a, suite.reference_indices(207, 2, 7, 23)))
        self.assertFalse(torch.equal(a, suite.reference_indices(207, 2, 7, 24)))
        seen = []
        data = CombinationTests().fixture()
        for variant in suite.VARIANTS:
            model = suite.common.MODELS.build(self.small(variant)['model'], seed=7)
            ds = suite.base.prepared(model, data['train'], 'cpu')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
            with patch.object(model, 'get_support_set', wraps=model.get_support_set) as get_refs:
                suite.train_epoch(model, optimizer, ds, 2, 100, 23, 3)
                seen.append(torch.cat([c.kwargs['fixed_indices'] for c in get_refs.call_args_list]))
        for indices in seen[1:]:
            self.assertTrue(torch.equal(seen[0], indices))

    def test_all_variants_resume_exactly_and_validate_results(self):
        data = CombinationTests().fixture()
        indices = torch.arange(96).reshape(3, 32) % 5
        for variant in suite.VARIANTS:
            cfg = self.small(variant)
            binding = dict(task='mix_20', model=variant, seed=3, config=cfg)
            with tempfile.TemporaryDirectory() as tmp:
                a, b = Path(tmp)/'a', Path(tmp)/'b'
                run = lambda path: suite.base.train_run(data, cfg, 3, path, binding, 'cpu', epoch_runner=suite.train_epoch)
                run(a)
                save = suite.common.save
                def crash(value, path):
                    save(value, path)
                    if Path(path).name == 'progress.pt' and value.get('epoch') == 1:
                        raise RuntimeError('test interruption')
                with patch.object(suite.common, 'save', side_effect=crash):
                    with self.assertRaisesRegex(RuntimeError, 'test interruption'):
                        run(b)
                run(b)
                aa, bb = suite.common.load(a/'final.pt'), suite.common.load(b/'final.pt')
                for key, value in aa['state'].items():
                    torch.testing.assert_close(value, bb['state'][key], atol=0, rtol=0)
                row = suite.evaluate_run(data, cfg, 3, b, binding, indices, 'cpu')
                self.assertEqual(row['model'], variant)
                self.assertTrue(set(suite.METRICS).issubset(row))
                self.assertEqual(row, suite.evaluate_run(data, cfg, 3, b, binding, indices, 'cpu'))
                bad = suite.common.load(b/'test.pt'); bad['diagnostics']['support_index'][0, 0] = 4
                suite.common.save(bad, b/'test.pt')
                with self.assertRaises(ValueError):
                    suite.evaluate_run(data, cfg, 3, b, binding, indices, 'cpu')

    def test_content_chunked_prediction_and_reference_aggregation(self):
        data = CombinationTests().fixture()
        model = suite.common.MODELS.build(self.small('content_values')['model'], seed=4)
        indices = torch.arange(96).reshape(3, 32) % 5
        chunked = suite.base.predict(model, data, indices, 'cpu', 8)
        full = suite.base.predict(model, data, indices, 'cpu', 32)
        torch.testing.assert_close(chunked['prediction'], full['prediction'], rtol=1e-5, atol=1e-6)
        diag = chunked['diagnostics']
        torch.testing.assert_close(diag['y_sup_agg'], diag['y_sup'].median(1).values)
        torch.testing.assert_close(chunked['prediction'], .5*(diag['y_ori']+diag['y_sup_agg']))
        model.train()
        own, refs, agg, *_ = model.compute_prediction_components(
            data['train']['feature'][:2], data['train']['feature'][:4].reshape(2, 2, 6, 20, 128),
            data['train']['label_z'][:4].reshape(2, 2))
        torch.testing.assert_close(agg, refs.mean(1))

    def test_full_queue_has_64_runs_and_test_only_after_task_training(self):
        data = CombinationTests().fixture()
        events = []
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            configs = {t: {v: self.small(v) for v in suite.VARIANTS} for t in suite.TASKS}
            for task in suite.TASKS:
                suite.common.save(data, workspace/'datasets'/task/'data.pt')
                for seed in range(8):
                    suite.common.save(dict(indices=torch.zeros(3, 32, dtype=torch.long)), workspace/'protocols'/task/f'seed_{seed}.pt')
            def train(data, cfg, seed, folder, binding, device, **kwargs):
                events.append(('train', binding['task'], binding['model'], seed))
            def evaluate(data, cfg, seed, folder, binding, indices, device):
                events.append(('test', binding['task'], binding['model'], seed))
                return dict(task=binding['task'], model=binding['model'], seed=seed,
                            **{k: float(seed+1) for k in suite.METRICS})
            with patch.object(suite.base, 'train_run', side_effect=train), patch.object(suite, 'evaluate_run', side_effect=evaluate), redirect_stdout(io.StringIO()):
                suite.execute_queue(workspace, configs, {}, {t: 'synthetic' for t in suite.TASKS}, 'cpu')
            for task in suite.TASKS:
                filtered = [event[0] for event in events if event[1] == task]
                self.assertEqual(filtered, ['train']*32+['test']*32)
            self.assertEqual(len(json.loads((workspace/'summary.json').read_text())), 8)
            self.assertEqual(len(json.loads((workspace/'paired_comparisons.json').read_text())), 6)


if __name__ == '__main__':
    tests = unittest.defaultTestLoader.loadTestsFromTestCase(RefinementTests)
    result = unittest.TextTestRunner(verbosity=2).run(tests)
    sys.exit(0 if result.wasSuccessful() else 1)
