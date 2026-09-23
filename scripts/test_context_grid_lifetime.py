"""CPU integration and invariant tests; temporary outputs are automatically removed."""
import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
import pickle
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from src.feature.lifetime_curves import LifetimeCurveExtractor
from src.models.rul_predictors.context_grid_lifetime import ContextGridLifetime, early_difference
from context_grid_lifetime import stack_features, subset, evaluate, make_indices, read_official_splits, validate_disjoint, prepare, protocol


def cell(cycles=20, offset=0):
    data = []
    for i in range(cycles):
        q = np.linspace(0, 1 - i * .0005, 40)
        data.append(dict(cycle_number=i + 1,
            voltage_in_V=np.r_[3 + q + offset + i * .001, 4 - q + offset],
            current_in_A=np.r_[np.ones(40), -np.ones(40)],
            charge_capacity_in_Ah=np.r_[q, np.zeros(40)],
            discharge_capacity_in_Ah=np.r_[np.zeros(40), q],
            time_in_s=np.arange(80) * 10.))
    return dict(cell_id=f'cell{offset}', nominal_capacity_in_Ah=1., cycle_data=data)


def batch(cycles=20, n=3):
    extract = LifetimeCurveExtractor(cycles=cycles, phase_points=32, drop_cycles=[10])
    return stack_features([extract(cell(cycles, i * .1)) for i in range(n)])


class Invariants(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        torch.set_num_threads(2)

    def model(self, data, use_context=True):
        model = ContextGridLifetime(cycles=data['curves'].shape[2], phase_points=32,
                                    channels=16, bins=8, dropout=0., use_context=use_context)
        model.encoder.statistics.fit(data)
        return model

    def test_early_only_phase_boundaries_and_masks(self):
        original = cell(30)
        modified = copy.deepcopy(original)
        modified['cycle_data'][25]['voltage_in_V'][:] = 9999
        extractor = LifetimeCurveExtractor(cycles=20, phase_points=32, drop_cycles=[10])
        a, b = extractor(original), extractor(modified)
        for key in a:
            self.assertTrue(torch.equal(a[key], b[key]))
        self.assertFalse(a['mask'][10].any())
        self.assertFalse(a['mask'][:, -1].any())
        self.assertGreater(a['curves'][1, 0, 5], 0)
        self.assertLess(a['curves'][1, 0, 37], 0)
        self.assertTrue(a['descriptor_mask'][0].all())
        self.assertAlmostEqual(a['descriptors'][0, 4].item(), 1.)

    def test_missing_placeholders_and_checkpoint(self):
        data = batch()
        model = self.model(data).eval()
        with torch.no_grad():
            a = model.encoder(data)
            dirty = {k: v.clone() for k, v in data.items()}
            dirty['curves'].masked_fill_(~data['mask'].unsqueeze(1), 98765)
            dirty['descriptors'].masked_fill_(~data['descriptor_mask'], -12345)
            b = model.encoder(dirty)
        self.assertTrue(torch.allclose(a[0], b[0], atol=1e-6))
        stream = io.BytesIO()
        torch.save(model.state_dict(), stream)
        stream.seek(0)
        fresh = ContextGridLifetime(cycles=20, phase_points=32, channels=16, bins=8, dropout=0.)
        fresh.load_state_dict(torch.load(stream, weights_only=True))
        with torch.no_grad():
            restored = fresh.eval().encoder(data)
        self.assertTrue(torch.equal(a[0], restored[0]))
        changed = {k: v.clone() for k, v in data.items()}
        changed['curves'][:, 0] += .2 * changed['mask']
        with torch.no_grad():
            new = model.encoder(changed)
        self.assertGreater((a[0] - new[0]).abs().max().item(), 1e-3)

    def test_pair_order_gradients_and_masked_window(self):
        for cycles in [20, 100]:
            data = batch(cycles)
            model = self.model(data)
            indices = torch.tensor([[1, 2], [2, 0]])
            target = subset(data, torch.tensor([0, 1]))
            reference = subset(data, indices)
            labels = torch.tensor([[.5, 1.], [1., -.5]])
            out = model(target, reference, labels)
            loss = out['y_ori'].square().mean() + out['y_sup_agg'].square().mean()
            loss.backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            model.eval()
            with torch.no_grad():
                out = model(target, reference, labels)
                swapped = model(target, {k: v.flip(1) for k, v in reference.items()}, labels.flip(1))
                self.assertTrue(torch.allclose(out['prediction'], swapped['prediction'], atol=1e-6))
                for i in range(2):
                    for j in range(2):
                        single = model(subset(target, slice(i, i + 1)),
                            {k: v[i:i + 1, j:j + 1] for k, v in reference.items()}, labels[i:i + 1, j:j + 1])
                        self.assertTrue(torch.allclose(single['y_sup'][0, 0], out['y_sup'][i, j], atol=2e-6))
            control = self.model(data, False).eval()
            with torch.no_grad():
                self.assertEqual(control.encoder(data)[0].shape, (3, cycles // 4 * 8, 16))

    def test_earliest_observed_baseline(self):
        x = torch.arange(5.).view(1, 1, 5, 1).expand(1, 3, 5, 2).clone()
        mask = torch.ones(1, 5, 2, dtype=torch.bool)
        mask[:, 0] = False
        delta, valid = early_difference(x, mask)
        self.assertAlmostEqual(delta[0, 0, 4, 0].item(), 2.)
        self.assertFalse(valid[:, 0].any())

    def test_official_split_parser_and_overlap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'splits.py'
            path.write_text("class split_recorder:\n    a = ['a.pkl']\n    MIX_large_train_files = a + ['b.pkl']\n    MIX_large_val_files = ['c.pkl']\n    MIX_large_test_files = ['d.pkl']\n")
            parts = read_official_splits(path)
            validate_disjoint(parts)
            self.assertEqual(parts['train'], ['a.pkl', 'b.pkl'])
            parts['test'].append('a.pkl')
            with self.assertRaises(ValueError):
                validate_disjoint(parts)

    def test_evaluation_chunk_equivalence(self):
        data = batch()
        model = self.model(data)
        train = dict(features=data, labels=torch.tensor([100., 200., 300.]), ids=['a', 'b', 'c'])
        ix = make_indices(3, 3, 99)
        a = evaluate(model, train, train, ix, torch.tensor(5.), torch.tensor(.5), 'cpu', 2, 3)
        b = evaluate(model, train, train, ix, torch.tensor(5.), torch.tensor(.5), 'cpu', 3, 32)
        self.assertTrue(torch.allclose(a['prediction'], b['prediction'], rtol=1e-5))

    def test_training_cli_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = dict(metadata=dict(extraction=dict(cycles=20, phase_points=32)))
            for part, n in [('train', 4), ('val', 2), ('test', 2)]:
                data[part] = dict(features=batch(n=n), labels=torch.arange(n).float() * 100 + 200,
                                  ids=[f'{part}_{i}' for i in range(n)])
            path = tmp / 'data.pt'
            torch.save(data, path)
            base = [sys.executable, '-B', str(ROOT / 'scripts/context_grid_lifetime.py')]
            command = base + ['train', '--data', str(path), '--workspace', str(tmp / 'run'),
                '--device', 'cpu', '--epochs', '2', '--channels', '16', '--batch-size', '2',
                '--accumulation', '2', '--evaluate-every', '1']
            subprocess.run(command, check=True, capture_output=True, text=True)
            subprocess.run(base + ['evaluate', '--data', str(path), '--checkpoint', str(tmp / 'run/best.pt'),
                '--device', 'cpu', '--output', str(tmp / 'result.pt')], check=True, capture_output=True, text=True)
            result = torch.load(tmp / 'result.pt', weights_only=False)
            self.assertTrue(np.isfinite(result['scores']['RMSE']))
            self.assertEqual(result['diagnostics']['support_index'].shape, (2, 32))
            self.assertEqual(json.loads((tmp / 'run/run.json').read_text())['train_ids'], data['train']['ids'])

    def test_prepare_official_schema_and_refit_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / 'official/data_provider').mkdir(parents=True)
            (tmp / 'MICH').mkdir()
            (tmp / 'Life labels').mkdir()
            split = tmp / 'official/data_provider/data_split_recorder.py'
            split.write_text("class split_recorder:\n    MIX_large_train_files=['MICH_a.pkl', 'MICH_b.pkl']\n    MIX_large_val_files=['MICH_c.pkl']\n    MIX_large_test_files=['MICH_d.pkl']\n")
            labels = {}
            for i, name in enumerate(['a', 'b', 'c', 'd']):
                with (tmp / 'MICH' / f'MICH_{name}.pkl').open('wb') as f:
                    pickle.dump(cell(20, i * .1), f)
                labels[f'MICH_{name}.pkl'] = 200 + 100 * i
            (tmp / 'Life labels/MICH_labels.json').write_text(json.dumps(labels))
            args = SimpleNamespace(output=str(tmp / 'data.pt'), data_root=str(tmp),
                dataset='batterylife', cycles=20, phase_points=32, official_repo=str(tmp / 'official'),
                data_version='fixture', split_seed=0, val_fraction=.2)
            with patch('context_grid_lifetime.revision', return_value='fixture'):
                prepare(args)
            data = torch.load(args.output, weights_only=False)
            self.assertEqual(data['test']['ids'], ['MICH_d'])
            self.assertEqual(data['train']['labels'].tolist(), [200., 300.])
            protocol(SimpleNamespace(data=args.output, refit=True, partition='test', seed=0,
                legacy_predictions=None, output=str(tmp / 'protocol.pt')))
            pr = torch.load(tmp / 'protocol.pt', weights_only=False)
            self.assertEqual(pr['train_ids'], ['MICH_a', 'MICH_b', 'MICH_c'])
            self.assertEqual(pr['indices'].shape, (1, 32))

    def test_legacy_protocol_remaps_both_axes(self):
        from src.data.databundle import DataBundle
        from src.data.transformation.sequential import SequentialDataTransformation
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            training = dict(features=batch(n=2), labels=torch.tensor([200., 300.]), ids=['a', 'b'])
            val = dict(features=batch(n=1), labels=torch.tensor([400.]), ids=['c'])
            test = dict(features=batch(n=2), labels=torch.tensor([500., 600.]), ids=['d', 'e'])
            prepared = tmp / 'data.pt'
            torch.save(dict(train=training, val=val, test=test), prepared)
            transform = SequentialDataTransformation([
                dict(name='LogScaleDataTransformation'), dict(name='ZScoreDataTransformation')])
            bundle = DataBundle(torch.zeros(3, 1), torch.tensor([400., 200., 300.]),
                                torch.zeros(2, 1), torch.tensor([600., 500.]),
                                train_metadata=[dict(cell_id=k) for k in ['c', 'a', 'b']],
                                test_metadata=[dict(cell_id=k) for k in ['e', 'd']],
                                label_transformation=transform)
            old_indices = torch.stack((torch.zeros(32, dtype=torch.long), torch.ones(32, dtype=torch.long)))
            legacy = tmp / 'old.pkl'
            with legacy.open('wb') as f:
                pickle.dump(dict(seed=3, data=bundle, diagnostics=dict(support_index=old_indices)), f)
            protocol(SimpleNamespace(data=str(prepared), refit=True, partition='test', seed=3,
                legacy_predictions=str(legacy), output=str(tmp / 'mapped.pt')))
            result = torch.load(tmp / 'mapped.pt', weights_only=False)
            self.assertEqual(result['query_ids'], ['d', 'e'])
            self.assertTrue((result['indices'][0] == 0).all())
            self.assertTrue((result['indices'][1] == 2).all())


if __name__ == '__main__':
    unittest.main()
