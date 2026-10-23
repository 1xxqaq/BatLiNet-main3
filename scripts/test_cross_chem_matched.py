"""Small synthetic CPU checks for the matched cross-chemistry protocol.

All generated data and checkpoints live in TemporaryDirectory.  This suite
does not read the real cells, start formal training, or alter old results.
"""
import copy
import io
import json
import math
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import cross_chem_matched_training as training
from scripts import cross_chem_matched_data as data
from scripts import run_cross_chem_matched as suite
from scripts import run_three_tasks as common


class MatchedProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model_config(self, dropout=0.):
        return dict(input_height=20, input_width=128,
                    attention_channels=8, attention_heads=2,
                    head_hidden_channels=8, encoder_dropout=dropout,
                    attention_dropout=dropout, cycle_mixer_hidden=4)

    def fixture(self):
        generator = torch.Generator().manual_seed(714)
        parts = []
        for name, count in [('LFP', 3), ('train', 5), ('test', 2)]:
            parts.append(dict(
                feature=torch.randn(count, 6, 20, 128, generator=generator),
                label=torch.linspace(225., 1025., count),
                ids=[f'{name}_{i}' for i in range(count)]))
        return parts

    def assert_nested_equal(self, first, second):
        if torch.is_tensor(first):
            self.assertTrue(torch.equal(first, second))
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first:
                self.assert_nested_equal(first[key], second[key])
        elif isinstance(first, (tuple, list)):
            self.assertEqual(len(first), len(second))
            for a, b in zip(first, second):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(first, second)

    def test_same_seed_encoder_and_own_head_initialization(self):
        models = {kind: training.make_model(kind, 5, 'cpu', self.model_config())
                  for kind in training.KINDS}
        baseline = models['direct']
        for kind, model in models.items():
            self.assert_nested_equal(baseline.cell_encoder.state_dict(),
                                     model.cell_encoder.state_dict())
            self.assert_nested_equal(baseline.ori_head.state_dict(),
                                     model.ori_head.state_dict())
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()), kind)
        self.assertEqual(set(baseline._modules), {'cell_encoder', 'ori_head'})
        other = training.make_model('direct', 6, 'cpu', self.model_config())
        self.assertTrue(any(not torch.equal(p, q)
                            for p, q in zip(baseline.parameters(), other.parameters())))

    def test_fixed_log_labels_accept_single_cell_and_round_trip(self):
        for labels in (torch.tensor([275.]), torch.tensor([125., 500., 1000., 2500.])):
            original = dict(label=labels.clone(), feature=torch.zeros(len(labels), 6, 20, 128),
                            ids=[f'cell{i}' for i in range(len(labels))])
            encoded = training.encode_labels(original)
            expected = (labels.double() / 1000.).log()
            torch.testing.assert_close(encoded['z'].double(), expected, rtol=1e-6, atol=1e-7)
            self.assertNotIn('z', original)
            restored = training.decode_labels(encoded['z'])
            self.assertEqual(restored.dtype, torch.float64)
            torch.testing.assert_close(restored, labels.double(), rtol=1e-6, atol=1e-6)
            self.assertTrue(torch.isfinite(encoded['z']).all())
        for bad in (0., -1., float('nan'), float('inf')):
            with self.assertRaises((ValueError, FloatingPointError)):
                training.encode_labels(dict(label=torch.tensor([bad]), feature=torch.zeros(1, 6, 20, 128)))

    def test_sequential_cursor_wraps_without_dropping_tail(self):
        first, cursor = training.next_indices(7, 5, 0)
        second, cursor = training.next_indices(7, 5, cursor)
        self.assertEqual(first.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(second.tolist(), [5, 6, 0, 1, 2])
        self.assertEqual(cursor, 3)
        for args in ((0, 1, 0), (7, 0, 0), (7, 1, 7), (7, 1, -1)):
            with self.assertRaises(ValueError):
                training.next_indices(*args)

    def test_combination_pair_order_balances_every_target_and_uses_actual_mapping(self):
        for target_count in (1, 2, 4, 8, 16):
            size = 275 * target_count
            full, _ = training.next_indices(size, size, 0)
            ti, si = training.pair_indices(full, 275, target_count)
            # Every target/source pair appears exactly once in a complete
            # cycle, even though targets are interleaved within each source.
            self.assertEqual((ti * 275 + si).sort().values.tolist(), list(range(size)))
            for budget in (100, 300, 1000):
                cursor = 0
                exposures = torch.zeros(target_count, dtype=torch.int64)
                for _ in range(budget):
                    chosen, cursor = training.next_indices(size, 128, cursor)
                    targets, _ = training.pair_indices(chosen, 275, target_count)
                    count = torch.bincount(targets, minlength=target_count)
                    self.assertEqual(count.tolist(), [128 // target_count] * target_count)
                    exposures += count
                self.assertEqual(exposures.tolist(), [budget * 128 // target_count] * target_count)
        for indices, source_count, target_count in (
                (torch.tensor([0.]), 3, 2), (torch.tensor([-1]), 3, 2),
                (torch.tensor([6]), 3, 2), (torch.tensor([0]), 0, 2),
                (torch.tensor([0]), 3, 0)):
            with self.assertRaises(ValueError):
                training.pair_indices(indices, source_count, target_count)

        source, target, _ = self.fixture()
        for index in range(len(source['label'])):
            source['feature'][index, 0, 0, 0] = 10. + index
        for index in range(len(target['label'])):
            target['feature'][index, 0, 0, 0] = 100. + index
        source, target = training.encode_labels(source), training.encode_labels(target)
        model = training.make_model('combination', 5, 'cpu', self.model_config())
        original = model.compute_prediction_components
        actual_targets, actual_sources = [], []

        def observe(x, reference, labels):
            actual_targets.extend((x[:, 0, 0, 0] - 100.).long().tolist())
            actual_sources.extend((reference[:, 0, 0, 0, 0] - 10.).long().tolist())
            return original(x, reference, labels)

        chosen = torch.tensor([0, 1, 4, 5, 6, 14])
        optimizer = torch.optim.SGD(model.parameters(), lr=.001)
        with patch.object(model, 'compute_prediction_components', side_effect=observe):
            training.train_step(model, optimizer, source, target, 'combination',
                                chosen, 'cpu', micro_batch=2)
        self.assertEqual(actual_targets, [0, 1, 4, 0, 1, 4])
        self.assertEqual(actual_sources, [0, 0, 0, 1, 1, 2])

    def test_single_and_combination_microbatch_tail_weighting(self):
        source, target, _ = [training.encode_labels(p) for p in self.fixture()]
        for kind in ('direct', 'combination'):
            with self.subTest(kind=kind):
                first = training.make_model(kind, 11, 'cpu', self.model_config())
                second = training.make_model(kind, 11, 'cpu', self.model_config())
                # Five examples split 2 + 2 + 1; the final sample must receive
                # exactly the same weight as every other example.
                indices = torch.tensor([0, 1, 2, 3, 4])
                oa = torch.optim.SGD(first.parameters(), lr=.002)
                ob = torch.optim.SGD(second.parameters(), lr=.002)
                a = training.train_step(first, oa, source, target, kind, indices,
                                        'cpu', micro_batch=2)
                b = training.train_step(second, ob, source, target, kind, indices,
                                        'cpu', micro_batch=5)
                self.assertEqual(a['batch_size'], 5)
                self.assertAlmostEqual(a['train_loss'], b['train_loss'], places=6)
                for key, value in first.state_dict().items():
                    torch.testing.assert_close(value, second.state_dict()[key],
                                               rtol=1e-5, atol=3e-7, msg=key)

    def test_dropout_resume_matches_continuous_parameters_optimizer_rng_and_cursor(self):
        source, target, _ = self.fixture()
        for kind in ('direct', 'combination'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp, \
                 redirect_stdout(io.StringIO()):
                root = Path(tmp)
                binding = dict(test='synthetic_resume', kind=kind)
                kwargs = dict(source=source, target=target, kind=kind, seed=12, steps=5,
                              binding=binding, device='cpu', checkpoint_every=2,
                              micro_batch=2, logical_batch=4, report_every=10,
                              model_config=self.model_config(.1))
                complete = training.train_run(folder=root / 'complete', **kwargs)

                def interrupt(step, model):
                    # Both the RNG and module mode must be restored even when
                    # a source-only monitoring callback raises an exception.
                    model.eval()
                    torch.randn(9)
                    np.random.random(3)
                    random.random()
                    if step == 3:
                        raise RuntimeError('synthetic interruption')

                with self.assertRaisesRegex(RuntimeError, 'synthetic interruption'):
                    training.train_run(folder=root / 'restart', step_callback=interrupt, **kwargs)
                progress = common.load(root / 'restart/progress.pt')
                self.assertEqual(progress['step'], 2)
                with (root / 'restart/train.jsonl').open('a', encoding='utf-8') as handle:
                    handle.write('{"step": 3, "train_loss": 999.}\n{"step":')
                resumed = training.train_run(folder=root / 'restart', **kwargs)
                for key in ('state', 'optimizer', 'rng', 'cursor', 'step', 'samples_seen'):
                    self.assert_nested_equal(complete[key], resumed[key])
                logs = [json.loads(line) for line in
                        (root / 'restart/train.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual([r['step'] for r in logs], [1, 2, 3, 4, 5])
                self.assertTrue(list((root / 'restart').glob('interrupted_log_*.jsonl')))
                self.assertFalse((root / 'restart/test.pt').exists())
                before = common.digest(root / 'restart/final.pt')
                training.train_run(folder=root / 'restart', **kwargs)
                self.assertEqual(before, common.digest(root / 'restart/final.pt'))
                for changed in (dict(binding=dict(binding, changed=True)), dict(lr=.003),
                                dict(micro_batch=1)):
                    with self.assertRaises(ValueError):
                        training.train_run(folder=root / 'restart', **dict(kwargs, **changed))
                altered = copy.deepcopy(target)
                altered['feature'][0, 0, 0, 0] += .25
                with self.assertRaises(ValueError):
                    training.train_run(folder=root / 'restart', **dict(kwargs, target=altered))
                corrupt = common.load(root / 'restart/final.pt')
                first_key = next(iter(corrupt['state']))
                corrupt['state'][first_key] = corrupt['state'][first_key] + .1
                common.save(corrupt, root / 'restart/final.pt')
                with self.assertRaises(ValueError):
                    training.load_completed(root / 'restart', binding, kind, 5)

    def test_finetuning_requires_fresh_source_checkpoint_and_matching_architecture(self):
        source, target, _ = self.fixture()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            kwargs = dict(source=source, target=target, seed=2, steps=2,
                          device='cpu', model_config=self.model_config(),
                          logical_batch=3, micro_batch=2)
            pretrained = training.train_run(kind='pretrain', folder=root / 'pretrain',
                                             binding={'test': 'pretrain'}, **kwargs)
            initial = copy.deepcopy(pretrained['state'])
            for count in (1, 2):
                selected = {key: value[:count] if torch.is_tensor(value) else value[:count]
                            for key, value in target.items()}
                training.train_run(kind='finetune', folder=root / f'fine{count}',
                                   binding={'test': 'finetune', 'count': count},
                                   initial_state=initial, **dict(kwargs, target=selected))
                self.assert_nested_equal(initial, pretrained['state'])
            with self.assertRaises(ValueError):
                training.train_run(kind='finetune', folder=root / 'bad_fine',
                                   binding={}, **kwargs)
            with self.assertRaises(ValueError):
                training.train_run(kind='direct', folder=root / 'bad_direct',
                                   binding={}, initial_state=initial, **kwargs)

    def test_cached_reference_prediction_matches_direct_forward_and_metrics(self):
        source, _, testing = self.fixture()
        model = training.make_model('combination', 13, 'cpu', self.model_config()).train()
        seen = []
        forward = model.cell_encoder.forward

        def count_encoding(x):
            seen.append(len(x))
            return forward(x)

        with patch.object(model.cell_encoder, 'forward', side_effect=count_encoding):
            first = training.predict(model, source, testing, 'combination', 'cpu', chunk=1)
        self.assertTrue(model.training, 'Prediction must restore the original model mode')
        self.assertEqual(sum(seen), len(source['ids']) + len(testing['ids']))
        second = training.predict(model, source, testing, 'combination', 'cpu', chunk=3)
        model.eval()
        count = len(testing['label'])
        encoded_source = training.encode_labels(source)
        with torch.no_grad():
            own, references, median, _, _ = model.compute_prediction_components(
                testing['feature'],
                source['feature'][None].expand(count, -1, -1, -1, -1).contiguous(),
                encoded_source['z'][None].expand(count, -1))
        expected = training.decode_labels(model.combine_predictions(own, median))
        for actual in (first, second):
            self.assertEqual(actual['source_ids'], source['ids'])
            self.assertEqual(actual['target_test_ids'], testing['ids'])
            torch.testing.assert_close(actual['diagnostics']['own_log'], own, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual['diagnostics']['reference_log'], references,
                                       rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(actual['prediction_cycles'], expected, rtol=1e-5, atol=1e-4)
            self.assertEqual(actual['scores'], common.scores(actual['prediction_cycles'],
                                                            testing['label'].double()))
        single = training.make_model('direct', 13, 'cpu', self.model_config()).eval()
        a = training.predict(single, None, testing, 'direct', 'cpu', chunk=1)
        b = training.predict(single, None, testing, 'direct', 'cpu', chunk=3)
        torch.testing.assert_close(a['prediction_cycles'], b['prediction_cycles'],
                                   rtol=1e-5, atol=1e-4)

    def test_formal_plan_has_all_344_training_jobs(self):
        jobs = suite.formal_plan()
        self.assertEqual(len(jobs), 344)
        self.assertEqual(jobs[:8], [dict(kind='pretrain', seed=s) for s in range(8)])
        targets = jobs[8:]
        self.assertEqual(len(targets), 336)
        self.assertEqual(len({tuple(sorted(j.items())) for j in jobs}), 344)
        for kind in ('direct', 'finetune', 'combination'):
            self.assertEqual(sum(j['kind'] == kind for j in targets), 112)
        for key in {(j['target'], j['train_count'], j['seed']) for j in targets}:
            self.assertEqual({j['kind'] for j in targets
                              if (j['target'], j['train_count'], j['seed']) == key},
                             {'direct', 'finetune', 'combination'})

    def test_pilot_split_only_uses_lfp_and_keeps_input_order_intact(self):
        ids = [f'LFP_{i}' for i in range(275)]
        original = list(ids)
        for seed in (0, 1):
            split = suite.pilot_split(ids, seed)
            self.assertEqual([len(split[k]) for k in
                              ('source_ids', 'training_pool_ids', 'validation_ids')],
                             [219, 16, 40])
            flattened = sum((split[k] for k in
                             ('source_ids', 'training_pool_ids', 'validation_ids')), [])
            self.assertEqual(set(flattened), set(ids))
            self.assertEqual(len(set(flattened)), 275)
            self.assertEqual(split, suite.pilot_split(ids, seed))
        self.assertEqual(ids, original)

    def pilot_records(self, cfg):
        records = []
        pilot = cfg['pilot']
        for lr in pilot['learning_rates']:
            for seed in pilot['split_seeds']:
                for step in pilot['candidate_steps']:
                    records.append(dict(lr=lr, split_seed=seed, kind='pretrain',
                                        train_count=0, step=step,
                                        validation_log_mse=1. / step))
                for count in pilot['target_counts']:
                    for kind in ('direct', 'finetune', 'combination'):
                        for step in pilot['candidate_steps']:
                            # The lower learning rate wins at 300 updates;
                            # the 1,000-step endpoint is intentionally worse.
                            value = {100: .3, 300: .1, 1000: .2}[step]
                            value += .05 if lr == pilot['learning_rates'][0] else 0.
                            records.append(dict(lr=lr, split_seed=seed, kind=kind,
                                                train_count=count, step=step,
                                                validation_log_mse=value))
        return records

    def test_source_only_budget_selection_rejects_partial_duplicates_and_nonfinite(self):
        cfg = suite.read_config()
        records = self.pilot_records(cfg)
        self.assertEqual(len(records), 84)
        result = suite.select_budget(records, cfg)
        self.assertEqual(result['learning_rate'], cfg['pilot']['learning_rates'][1])
        self.assertEqual(result['pretrain_steps'], 1000)
        self.assertEqual(result['target_steps'], 300)
        for bad in (records[:-1], records + [records[0]],
                    [dict(records[0], validation_log_mse=float('nan'))] + records[1:]):
            with self.assertRaises(ValueError):
                suite.select_budget(bad, cfg)

    def test_pilot_orchestration_never_loads_formal_target_or_test_cells(self):
        cfg = suite.read_config()
        source = dict(feature=torch.zeros(275, 6, 1, 1),
                      label=torch.linspace(201., 1001., 275),
                      ids=[f'LFP_{i:03}' for i in range(275)])
        calls = []

        def fake_train(src, target, kind, seed, steps, folder, binding, device,
                       initial_state=None, step_callback=None, **kwargs):
            self.assertTrue(all(i.startswith('LFP_') for i in src['ids'] + target['ids']))
            self.assertEqual(len(src['ids']), 219)
            self.assertEqual(set(src['ids']) & set(binding['ids']['validation_ids']), set())
            if kind != 'pretrain':
                self.assertEqual(len(target['ids']), binding['train_count'])
                self.assertEqual(set(target['ids']) & set(binding['ids']['validation_ids']), set())
            self.assertEqual(initial_state is not None, kind == 'finetune')
            calls.append((kind, seed, len(target['ids'])))
            model = torch.nn.Linear(1, 1)
            for step in cfg['pilot']['candidate_steps']:
                step_callback(step, model)
            return {'state': model.state_dict()}

        def fake_predict(model, src, validation, kind, device, chunk):
            self.assertEqual(len(validation['ids']), 40)
            self.assertTrue(all(i.startswith('LFP_') for i in validation['ids']))
            self.assertEqual(set(src['ids']) & set(validation['ids']), set())
            return {'prediction_cycles': validation['label'].double()}

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
             patch.object(data, 'load_part', side_effect=AssertionError('Formal cells may not be loaded')), \
             patch.object(training, 'train_run', side_effect=fake_train), \
             patch.object(training, 'predict', side_effect=fake_predict):
            selected = suite.run_pilot(source, Path(tmp), {'test': 'source_only'}, cfg, 'cpu')
            self.assertTrue(selected['source_only'])
            self.assertFalse(selected['formal_target_scores_used'])
            self.assertEqual(selected['learning_rate'], cfg['pilot']['learning_rates'][0])
            self.assertEqual(selected['pretrain_steps'], 100)
            self.assertEqual(selected['target_steps'], 100)
            self.assertEqual(len(calls), 28)
            scores = json.loads((Path(tmp) / 'pilot/scores.json').read_text(encoding='utf-8'))
            self.assertEqual(len(scores), 84)

    def existing_fixture(self, existing):
        """The real 403-cell/112-protocol topology with tiny synthetic curves."""
        shape = (6, 4, 32)
        records = []
        for chem, count in data.EXPECTED.items():
            for index in range(count):
                cell_id = f'{chem}_{index:03}'
                relative = f'cells/{cell_id}.pt'
                feature = torch.zeros(shape, dtype=torch.float32)
                feature[0, 0, 0] = 11. if index == 0 else .5
                value = dict(cell_id=cell_id, chemistry=chem,
                             label_cycles=201. + index, feature=feature)
                path = existing / 'dataset' / relative
                common.save(value, path)
                records.append(dict(cell_id=cell_id, chemistry=chem,
                                    label_cycles=value['label_cycles'],
                                    feature_file=relative, feature_sha256=common.digest(path)))
        hashes = {'synthetic_fixture': 'not_real_data'}
        manifest = dict(identity=dict(policy=data.SHARED_POLICY, code_hashes=hashes),
                        records=records, observed_counts=data.EXPECTED)
        common.atomic_json(manifest, existing / 'dataset/manifest.json')
        runs = []
        source = list(reversed([r['cell_id'] for r in records if r['chemistry'] == 'LFP']))
        protocols = {}
        for chem, numbers in data.TRAIN_COUNTS.items():
            ids = list(reversed([r['cell_id'] for r in records if r['chemistry'] == chem]))
            for seed in range(8):
                # Deliberately use a saved ordering different from the old
                # pseudo-random generator.  Import must preserve actual IDs.
                ordered = ids[seed:] + ids[:seed]
                test_count = data.TEST_COUNTS[chem]
                tests, pool = ordered[:test_count], ordered[test_count:]
                for count in numbers:
                    pairs = torch.arange(count * len(source), dtype=torch.int64)
                    cut = int(len(pairs) * .8)
                    protocol = dict(policy=data.SHARED_POLICY,
                                    target_chemistry=chem, target_train_count=count, seed=seed,
                                    source_ids=source, target_train_ids=pool[:count],
                                    target_test_ids=tests, unused_target_ids=pool[count:],
                                    train_pairs=pairs[:cut], val_pairs=pairs[cut:])
                    path = existing / 'protocols' / chem / f'train_{count}' / f'seed_{seed}.pt'
                    common.save(protocol, path)
                    protocols[(chem, count, seed)] = protocol
                    runs.extend([chem, count, seed, model] for model in data.EXISTING_MODELS)
        common.atomic_json(dict(schema=2, data_policy=data.SHARED_POLICY,
                                models=list(data.EXISTING_MODELS), code_hashes=hashes, runs=runs),
                           existing / 'queue.json')
        return shape, manifest, protocols

    def test_actual_112_protocols_import_without_randomization_or_old_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, new = Path(tmp) / 'old', Path(tmp) / 'new'
            shape, manifest, original = self.existing_fixture(existing)
            fingerprints = {p.relative_to(existing).as_posix(): common.digest(p)
                            for p in existing.rglob('*') if p.is_file()}
            with patch.object(data, 'FEATURE_SHAPE', shape), \
                 patch('scripts.transfer_cycle_soft_data.build_protocol',
                       side_effect=AssertionError('Import must never redraw actual lists')):
                checked, protocols, audit = data.import_protocols(existing, new)
                self.assertEqual(checked, manifest)
                self.assertEqual(len(protocols), 112)
                self.assertEqual(audit['cells_verified'], 403)
                self.assertFalse(audit['split_generation_called'])
                self.assertFalse(audit['legacy_pair_split_reused_for_training'])
                self.assertEqual(len(list((new / 'protocols').rglob('*.json'))), 112)
                for key, imported in protocols.items():
                    for field in ('source_ids', 'target_train_ids', 'target_test_ids', 'unused_target_ids'):
                        self.assertEqual(imported[field], original[key][field])
                    self.assertNotIn('train_pairs', imported)
                    self.assertNotIn('val_pairs', imported)
                self.assertEqual(data.import_protocols(existing, new), (checked, protocols, audit))
                part = data.load_part(protocols[('LCO', 1, 0)]['source_ids'][-1:], checked, existing)
                self.assertEqual(part['ids'], ['LFP_000'])
                self.assertEqual(float(part['feature'][0, 0, 0, 0]), 0.)
                raw = data.load_part(part['ids'], checked, existing, clip=False)
                self.assertEqual(float(raw['feature'][0, 0, 0, 0]), 11.)
                for invalid in (existing, existing / 'child', Path(tmp)):
                    with self.assertRaises(ValueError):
                        data.import_protocols(existing, invalid)
            after = {p.relative_to(existing).as_posix(): common.digest(p)
                     for p in existing.rglob('*') if p.is_file()}
            self.assertEqual(after, fingerprints)

    def test_bad_nested_lists_corrupt_cache_and_changed_import_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing, new = Path(tmp) / 'old', Path(tmp) / 'new'
            shape, manifest, _ = self.existing_fixture(existing)
            with patch.object(data, 'FEATURE_SHAPE', shape):
                data.import_protocols(existing, new)
                path = existing / 'protocols/LCO/train_1/seed_0.pt'
                original = path.read_bytes()
                corrupted = common.load(path)
                corrupted['target_train_ids'][0], corrupted['unused_target_ids'][0] = (
                    corrupted['unused_target_ids'][0], corrupted['target_train_ids'][0])
                common.save(corrupted, path)
                with self.assertRaisesRegex(ValueError, '嵌套'):
                    data.audit_existing(existing)
                path.write_bytes(original)
                cell_path = existing / 'dataset' / manifest['records'][0]['feature_file']
                cell_bytes = cell_path.read_bytes()
                cell_path.write_bytes(b'synthetic corruption')
                with self.assertRaises(ValueError):
                    data.audit_existing(existing)
                cell_path.write_bytes(cell_bytes)
                imported = new / 'protocols/LCO/train_1/seed_0.json'
                changed = json.loads(imported.read_text(encoding='utf-8'))
                changed['target_test_ids'].reverse()
                common.atomic_json(changed, imported)
                with self.assertRaises(ValueError):
                    data.import_protocols(existing, new)

    def test_all_344_training_jobs_must_complete_before_formal_test_gate(self):
        jobs = suite.formal_plan()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            identity = {'test': 'synthetic_gate'}
            with patch.object(suite, 'validate_training_job', return_value={}) as validate:
                self.assertFalse(suite.all_training_complete(root, identity, jobs))
                self.assertEqual(validate.call_count, 0)
                for job in jobs:
                    path = suite.job_folder(root, job) / 'final.pt'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'synthetic gate placeholder, not model weights')
                self.assertTrue(suite.all_training_complete(root, identity, jobs))
                self.assertEqual(validate.call_count, 344)
                with self.assertRaises(ValueError):
                    suite.all_training_complete(root, identity, jobs[:-1])
                with self.assertRaises(ValueError):
                    suite.all_training_complete(root, identity, [jobs[0]] * 344)
                validate.side_effect = ValueError('synthetic invalid checkpoint')
                with self.assertRaisesRegex(ValueError, 'synthetic invalid checkpoint'):
                    suite.all_training_complete(root, identity, jobs)

    def test_real_tiny_training_round_trip_selection_binding_and_completion_receipts(self):
        source, target, _ = self.fixture()
        target = {key: value[:1] for key, value in target.items()}
        cfg = copy.deepcopy(suite.read_config())
        cfg['pilot']['candidate_steps'] = [1, 2]
        records = [dict(lr=lr, split_seed=seed, kind=kind, train_count=count,
                        step=step, validation_log_mse=1. / step)
                   for lr in cfg['pilot']['learning_rates']
                   for seed in cfg['pilot']['split_seeds']
                   for kind, count in [('pretrain', 0)] + [
                       (kind, count) for count in cfg['pilot']['target_counts']
                       for kind in suite.KINDS]
                   for step in cfg['pilot']['candidate_steps']]
        selection = suite.select_budget(records, cfg)
        self.assertEqual(selection['pretrain_steps'], 2)
        self.assertEqual(selection['target_steps'], 2)
        self.assertEqual(set(selection['candidates'][0]['pretrain_scores']), {'1', '2'})
        jobs = [dict(kind='pretrain', seed=3)] + [
            dict(kind=kind, seed=3, target='LCO', train_count=1) for kind in suite.KINDS]
        identity = {'test': 'real_tiny_training_receipts'}
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            common.atomic_json(selection, root / 'selection.json')
            self.assertEqual(selection, json.loads((root / 'selection.json').read_text(encoding='utf-8')))
            pretrained = None
            for job in jobs:
                binding = suite.job_binding(identity, job, selection, root)
                trained = training.train_run(
                    source, source if job['kind'] == 'pretrain' else target,
                    job['kind'], 3, 2, suite.job_folder(root, job), binding, 'cpu',
                    initial_state=pretrained if job['kind'] == 'finetune' else None,
                    checkpoint_every=1, logical_batch=4, micro_batch=2,
                    report_every=10, model_config=self.model_config())
                checked = suite.validate_training_job(root, identity, job)
                self.assert_nested_equal(trained['state'], checked['state'])
                self.assertEqual(checked['binding']['selection'], selection)
                if job['kind'] == 'pretrain':
                    pretrained = copy.deepcopy(trained['state'])
            with patch.object(suite, 'formal_plan', return_value=jobs):
                self.assertTrue(suite.all_training_complete(root, identity, jobs))
                job = jobs[1]
                folder = suite.job_folder(root, job)
                final = folder / 'final.pt'
                original_final = final.read_bytes()
                damaged = common.load(final)
                key = next(iter(damaged['state']))
                damaged['state'][key] = damaged['state'][key] + .1
                common.save(damaged, final)
                with self.assertRaises(ValueError):
                    suite.validate_training_job(root, identity, job)
                with self.assertRaises(ValueError):
                    suite.all_training_complete(root, identity, jobs)
                final.write_bytes(original_final)
                receipt = folder / 'run.json'
                original_receipt = receipt.read_bytes()
                changed = json.loads(original_receipt.decode('utf-8'))
                changed['status'] = 'training'
                common.atomic_json(changed, receipt)
                with self.assertRaises(ValueError):
                    suite.validate_training_job(root, identity, job)
                receipt.write_bytes(original_receipt)
                self.assertTrue(suite.all_training_complete(root, identity, jobs))

    def test_report_before_start_or_budget_is_read_only_and_checks_actual_fingerprints(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()) as output:
            root = Path(tmp)
            old, new = root / 'old', root / 'new'
            suite.report(new)
            self.assertFalse(new.exists(), 'Reporting an unstarted queue must not create it')
            self.assertIn('尚未开始', output.getvalue())
            manifest = old / 'dataset/manifest.json'
            old_queue = old / 'queue.json'
            common.atomic_json({'records': []}, manifest)
            common.atomic_json({'test': 'existing_queue'}, old_queue)
            imported = dict(existing_manifest_sha256=common.digest(manifest),
                            existing_queue_sha256=common.digest(old_queue), protocols={})
            for seed in range(8):
                relative = f'LCO/train_1/seed_{seed}.pt'
                protocol = dict(source_ids=['LFP_000'], target_train_ids=['LCO_000'],
                                target_test_ids=['LCO_001'], unused_target_ids=['LCO_002'])
                common.save(protocol, old / 'protocols' / relative)
                imported['protocols'][relative] = {'sha256': common.digest(old / 'protocols' / relative)}
                common.atomic_json(protocol, new / 'protocols/LCO/train_1' / f'seed_{seed}.json')
            common.atomic_json(dict(existing_workspace=str(old), imported_data=imported), new / 'queue.json')
            common.atomic_json({'status': 'source_only_development'}, new / 'status.json')
            snapshot = {p.relative_to(root).as_posix(): common.digest(p)
                        for p in root.rglob('*') if p.is_file()}
            with patch.object(data, 'TRAIN_COUNTS', {'LCO': (1,)}), \
                 patch.object(data, 'import_protocols', side_effect=AssertionError('Read-only report cannot import')), \
                 patch.object(suite, 'completed_rows', side_effect=AssertionError('No formal scores before selection')):
                suite.report(new)
                self.assertIn('尚未锁定正式预算', output.getvalue())
                after = {p.relative_to(root).as_posix(): common.digest(p)
                         for p in root.rglob('*') if p.is_file()}
                self.assertEqual(snapshot, after)
                for path in (manifest, old_queue, old / 'protocols/LCO/train_1/seed_0.pt',
                             new / 'protocols/LCO/train_1/seed_0.json'):
                    with self.subTest(changed=path.relative_to(root).as_posix()):
                        original = path.read_bytes()
                        if path.suffix == '.pt':
                            value = common.load(path)
                            value['target_train_ids'] = ['wrong_id']
                            common.save(value, path)
                        else:
                            value = json.loads(original.decode('utf-8'))
                            value['target_train_ids'] = ['wrong_id']
                            common.atomic_json(value, path)
                        with self.assertRaises(ValueError):
                            suite.report(new)
                        path.write_bytes(original)
                final = {p.relative_to(root).as_posix(): common.digest(p)
                         for p in root.rglob('*') if p.is_file()}
                self.assertEqual(snapshot, final)

    def test_reused_predictions_verify_hashes_truth_reference_order_and_recomputed_metrics(self):
        source, _, testing = self.fixture()
        cfg = suite.read_config()
        manifest = dict(records=[dict(cell_id=i, label_cycles=float(y))
                                 for i, y in zip(testing['ids'], testing['label'])])
        protocol = dict(source_ids=source['ids'], target_test_ids=testing['ids'])
        job = dict(kind='combination', seed=13, target='LCO', train_count=1)
        binding = {'test': 'synthetic_prediction'}
        identity = {'test': 'synthetic_queue'}
        model = training.make_model('combination', 13, 'cpu', self.model_config())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = suite.job_folder(root, job)
            state = dict(state=model.state_dict(), binding=binding, step=3)
            common.save(state, folder / 'final.pt')
            original_make = training.make_model

            def tiny(kind, seed, device):
                return original_make(kind, seed, device, self.model_config())

            with patch.object(suite, 'validate_training_job', return_value=state), \
                 patch.object(suite, 'all_training_complete', return_value=True), \
                 patch.object(training, 'make_model', side_effect=tiny), \
                 patch.object(training, 'predict', wraps=training.predict) as predict:
                first = suite.evaluate_job(root, identity, job, source, testing,
                                           protocol, manifest, cfg, 'cpu')
                second = suite.evaluate_job(root, identity, job, source, testing,
                                            protocol, manifest, cfg, 'cpu')
                self.assertEqual(first, second)
                self.assertEqual(predict.call_count, 1, 'Completed predictions should be validated and reused')
                output = common.load(folder / 'test.pt')
                expected = common.scores(output['prediction_cycles'], output['truth_cycles'])
                self.assertEqual(output['scores'], expected)
                for modification in ('metric', 'truth', 'reference_order', 'prediction'):
                    with self.subTest(modification=modification):
                        bad = copy.deepcopy(output)
                        if modification == 'metric':
                            bad['scores']['RMSE'] += 1
                        elif modification == 'truth':
                            bad['truth_cycles'][0] += 1
                        elif modification == 'reference_order':
                            bad['source_ids'].reverse()
                        else:
                            bad['prediction_cycles'][0] += 1
                        with self.assertRaises((ValueError, AssertionError)):
                            suite.verify_test(bad, state, job, binding,
                                              common.digest(folder / 'final.pt'), protocol, manifest)
                output['prediction_cycles'][0] += 1
                common.save(output, folder / 'test.pt')
                with self.assertRaises(ValueError):
                    suite.evaluate_job(root, identity, job, source, testing,
                                       protocol, manifest, cfg, 'cpu')

    def test_summary_preserves_large_error_seeds_and_requires_unique_results(self):
        rows = []
        for job in suite.formal_plan():
            if job['kind'] == 'pretrain':
                continue
            error = 10001. if job['seed'] == 7 else 1.
            rows.append(dict(**job, optimizer_steps=300,
                             RMSE=error, MAE=error, MAPE=error, ACC15=0.))
        summary = suite.summarize(rows)
        self.assertEqual(summary['completed_predictions'], 336)
        self.assertEqual(len(summary['groups']), 42)
        self.assertEqual(len(summary['paired_differences']), 28)
        for group in summary['groups']:
            self.assertEqual(group['seeds'], 8)
            self.assertEqual(group['metrics']['RMSE']['mean'], 1251.)
            self.assertGreater(group['metrics']['RMSE']['sample_std'], 3000.)
        with self.assertRaises(ValueError):
            suite.summarize(rows + [rows[0]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
