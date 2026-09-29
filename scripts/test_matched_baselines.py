"""CPU checks of original-path equivalence, protocols and training isolation."""
import copy
import json
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import context_grid_lifetime as common
import matched_baselines as runner
import evaluate_matched_suite as suite
from test_context_grid_lifetime import cell
from src.data.databundle import Dataset
from src.models.rul_predictors.matched_baselines import MatchedBaseline


class MatchedTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(21)

    def test_original_paths_and_gradients(self):
        # Actual configured dimensions, fixed target/reference pairing.
        x = torch.randn(2, 6, 20, 1000)
        pool = torch.randn(3, 6, 20, 1000)
        ix = torch.tensor([[0, 2], [1, 0]])
        y = torch.tensor([-.5, .2, 1.])
        before_x, before_pool = x.clone(), pool.clone()
        for name in ('batlinet', 'latent_cross_attention'):
            model = MatchedBaseline(name).eval()
            original = copy.deepcopy(model.base).eval()
            actual = model({'raw': x}, {'raw': pool[ix]}, y[ix])
            if name == 'batlinet':
                dataset = original.build_cycle_diff_dataset(Dataset(x.clone(), torch.zeros(2)))
                refs, labels = original.get_support_set(dataset.raw_feature, pool.clone(), y, fixed_indices=ix)
                target = dataset.feature
            else:
                target = original.build_cell_dataset(Dataset(x.clone(), torch.zeros(2))).feature
                refs, labels = original.get_support_set(target, pool.clone(), y, fixed_indices=ix)
            expected = original.compute_prediction_components(target, refs, labels)
            for key, value in zip(('y_ori', 'y_sup', 'y_sup_agg'), expected[:3]):
                torch.testing.assert_close(actual[key], value)
            torch.testing.assert_close(x, before_x, rtol=0, atol=0)
            torch.testing.assert_close(pool, before_pool, rtol=0, atol=0)
            model.train()
            out = model({'raw': x}, {'raw': pool[ix]}, y[ix])
            torch.testing.assert_close(out['y_sup_agg'], out['y_sup'].mean(1))
            (out['y_ori'].square().mean() + out['y_sup_agg'].square().mean()).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_chunking_and_median(self):
        train = dict(ids=['a', 'b', 'c'], labels=torch.tensor([100., 200., 400.]),
                     features={'raw': torch.randn(3, 6, 20, 1000)})
        query = dict(ids=['d'], labels=torch.tensor([150.]), features={'raw': torch.randn(1, 6, 20, 1000)})
        indices = common.make_indices(1, 3, 55)
        for name in ('batlinet', 'latent_cross_attention'):
            model = MatchedBaseline(name)
            a = runner.evaluate(model, train, query, indices, 5., .5, 'cpu', pair_chunk=7)
            b = runner.evaluate(model, train, query, indices, 5., .5, 'cpu', pair_chunk=32)
            torch.testing.assert_close(a['prediction'], b['prediction'])
            torch.testing.assert_close(a['diagnostics']['y_sup'], b['diagnostics']['y_sup'])
            torch.testing.assert_close(a['diagnostics']['y_sup_agg'], a['diagnostics']['y_sup'].median(1).values)

    def test_protocol_reproduces_context_and_rejects_mismatch(self):
        context = {'train': {'ids': ['a', 'b']}, 'val': {'ids': ['c']}, 'test': {'ids': ['d']}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'val.pt'
            protocol = runner.fixed_protocol(context, 3, 'val', path)
            torch.testing.assert_close(protocol['indices'], common.make_indices(1, 2, 100003))
            runner.fixed_protocol(context, 3, 'val', path)
            protocol['train_ids'].reverse()
            torch.save(protocol, path)
            with self.assertRaises(ValueError):
                runner.fixed_protocol(context, 3, 'val', path)

    def test_train_checkpoint_and_no_test_access(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = {p: dict(ids=[p+str(i) for i in range(n)], labels=torch.arange(n).float()*30+120)
                       for p, n in [('train', 3), ('val', 1), ('test', 1)]}
            cp, bp = root/'context.pt', root/'baseline.pt'
            torch.save(context, cp)
            baseline = copy.deepcopy(context)
            baseline['metadata'] = {'context_sha256': common.digest(cp)}
            for part in ('train', 'val'):
                baseline[part]['features'] = {'raw': torch.randn(len(baseline[part]['ids']), 6, 20, 1000)}
            # No test features at all: accessing test during training would fail.
            torch.save(baseline, bp)
            for name in ('batlinet', 'latent_cross_attention'):
                args = SimpleNamespace(context_data=str(cp), data=str(bp), protocol_dir=str(root/'protocols'),
                    workspace=str(root/name), model=name, seed=0, epochs=2, batch_size=2,
                    accumulation=2, evaluate_every=1, lr=.001, pair_chunk=8, device='cpu',
                    amp=False, skip_complete=True, command='train')
                runner.train(args)
                runner.train(args)  # Validated skip, no overwrite.
                checkpoint = common.load(root/name/'best.pt')
                torch.testing.assert_close(checkpoint['label_mean'], context['train']['labels'].log().mean())
                model = MatchedBaseline(**checkpoint['config'])
                model.load_state_dict(checkpoint['state'])
                self.assertEqual(checkpoint['train_ids'], context['train']['ids'])
                args.lr = .002
                with self.assertRaises(ValueError):
                    runner.train(args)

    def test_prepare_alignment_and_source_tamper(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = {'metadata': {'dataset': 'mix20', 'source_files': []}}
            paths = []
            for part in ('train', 'val', 'test'):
                source = cell()
                source.update(cell_id=part, charge_protocol=None, discharge_protocol=None)
                path = root / f'{part}.pkl'
                with path.open('wb') as stream:
                    pickle.dump(source, stream)
                paths.append(path)
                context[part] = dict(ids=[part], labels=torch.tensor([120.]))
                context['metadata']['source_files'].append(dict(cell_id=part, split=part,
                    path=path.name, sha256=common.digest(path)))
            cp = root/'context.pt'
            torch.save(context, cp)
            args = SimpleNamespace(context_data=str(cp), data_root=str(root), output=str(root/'legacy.pt'))
            with patch('src.train_test_split.MIX20_split.MIX20TrainTestSplitter') as splitter, \
                    patch.object(runner.RULLabelAnnotator, 'process_cell', return_value=120.):
                splitter.return_value.split.return_value = (paths[:2], paths[2:])
                runner.prepare(args)
                result = common.load(args.output)
                runner.check_alignment(context, result, cp)
                self.assertEqual(result['train']['features']['raw'].shape, (1, 6, 20, 1000))
                self.assertEqual(result['train']['features']['raw'][:, :, 10].count_nonzero(), 0)
                runner.prepare(args)  # Existing cache validated, not overwritten.
                args.output = str(root/'another.pt')
                with paths[0].open('ab') as stream:
                    stream.write(b'changed')
                with self.assertRaisesRegex(ValueError, 'Source changed'):
                    runner.prepare(args)
                bad = copy.deepcopy(result)
                bad['val']['labels'] += 1
                with self.assertRaises(ValueError):
                    runner.check_alignment(context, bad, cp)

    def test_final_evaluation_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            data = {'train': {'ids': ['a']}}
            args = dict(seed=0, epochs=1000, batch_size=8, accumulation=16,
                        evaluate_every=25, lr=.001, amp=True)
            info = dict(arguments=args, data_sha256='digest', train_ids=['a'])
            (folder/'run.json').write_text(json.dumps(info))
            records = []
            for epoch in range(1, 1001):
                record = dict(epoch=epoch)
                if epoch % 25 == 0:
                    record['validation'] = {'RMSE': float(epoch)}
                records.append(record)
            (folder/'train.jsonl').write_text('\n'.join(map(json.dumps, records)))
            checkpoint = dict(epoch=25, seed=0, refit=False, selection_metric='RMSE',
                data_sha256='digest', train_ids=['a'],
                config=dict(architecture='batlinet', cycles=20, width=1000))
            torch.save(checkpoint, folder/'best.pt')
            suite.preflight(folder, data, 'digest', 0, 'batlinet')
            checkpoint['epoch'] = 50
            torch.save(checkpoint, folder/'best.pt')
            with self.assertRaises(ValueError):
                suite.preflight(folder, data, 'digest', 0, 'batlinet')


if __name__ == '__main__':
    unittest.main()
