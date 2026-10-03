"""Verify historical training equivalence and reject mismatched references."""
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
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
import run_mix20_cycle_mixer_207 as runner
from src.data.databundle import DataBundle
from src.data.transformation.sequential import SequentialDataTransformation
from src.models.rul_predictors.latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor


class HistoricalProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def test_training_weights_and_rng_match_original_fit(self):
        data = DataBundle(torch.randn(3, 6, 20, 64), torch.randn(3),
                          torch.randn(2, 6, 20, 64), torch.randn(2))
        model = LatentCrossAttentionBatLiNetRULPredictor(
            in_channels=6, channels=32, input_height=20, input_width=64,
            epochs=3, train_batch_size=2, test_batch_size=1,
            train_support_size=2, test_support_size=3, evaluate_freq=1,
            attention_channels=8, attention_heads=2, filter_cycles=False)
        original, replacement = copy.deepcopy(model), copy.deepcopy(model)
        torch.manual_seed(90)
        with redirect_stdout(io.StringIO()):
            original.fit(copy.deepcopy(data), timestamp='test')
        expected_rng = torch.get_rng_state().clone()
        torch.manual_seed(90)
        # Only training data and the number of queries enter the new loop.
        runner.legacy_train(replacement, copy.deepcopy(data.train_data), 3, 1, 2)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, rtol=0, atol=0)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(replacement.state_dict()[key], value, rtol=0, atol=0)

    def test_rng_advance_uses_no_features_and_matches_prediction_draws(self):
        for batch_size in (1, 2, 4):
            torch.manual_seed(91)
            for batch in torch.utils.data.DataLoader(torch.ones(7), batch_size, shuffle=False):
                torch.randint(207, (len(batch) * 32,), device='cpu')
            expected = torch.get_rng_state().clone()
            torch.manual_seed(91)
            runner.advance_old_monitor_rng(207, 7, 32, batch_size, 'cpu')
            torch.testing.assert_close(torch.get_rng_state(), expected, rtol=0, atol=0)

    def test_protocol_rejects_new_split_and_out_of_range_indices(self):
        valid = dict(seed=0, num_train_samples=207, num_test_samples=147,
                     test_support_size=32, indices=torch.zeros(147, 32, dtype=torch.long))
        runner.validate_protocol(valid, 0)
        for key, value in (('num_train_samples', 166), ('seed', 1),
                           ('indices', torch.full((147, 32), 207, dtype=torch.long))):
            wrong = dict(valid, **{key: value})
            with self.assertRaises(ValueError):
                runner.validate_protocol(wrong, 0)

    def test_complete_checkpoint_and_fixed_epoch_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            provenance = dict(policy=runner.POLICY, protocols_sha256=['historical'])
            config = runner.model_config()
            identity = dict(provenance=provenance, config=config, seed=0, model=runner.MODEL)
            (folder / 'run.json').write_text(json.dumps(dict(identity, status='complete')))
            (folder / 'train.jsonl').write_text(''.join(
                json.dumps(dict(epoch=epoch, train_loss=.1)) + '\n' for epoch in range(1, 1001)))
            checkpoint = dict(identity, epoch=1000, state={})
            runner.save_new(checkpoint, folder / 'epoch1000.pt')
            runner.validate_complete(folder, provenance, config, 0)
            with self.assertRaises(ValueError):
                runner.validate_complete(folder, dict(provenance, policy='new_protocol'), config, 0)
            checkpoint['epoch'] = 800
            torch.save(checkpoint, folder / 'epoch1000.pt')
            with self.assertRaises(ValueError):
                runner.validate_complete(folder, provenance, config, 0)

    def test_missing_history_fails_before_creating_training_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory) / 'output'
            argv = ['runner', '--history-root', directory, '--original-root', directory,
                    '--workspace', str(workspace), '--device', 'cpu']
            with patch.object(sys, 'argv', argv), self.assertRaises(FileNotFoundError):
                runner.main()
            self.assertFalse(workspace.exists())

    def test_all_seeds_finish_before_fixed_test_and_completed_runs_are_reused(self):
        transform = SequentialDataTransformation([
            dict(name='LogScaleDataTransformation'), dict(name='ZScoreDataTransformation')])
        data = DataBundle(torch.randn(3, 6, 20, 64), torch.tensor([100., 200., 300.]),
                          torch.randn(2, 6, 20, 64), torch.tensor([150., 250.]),
                          label_transformation=transform)
        config = runner.model_config()
        config.update(input_width=64, train_batch_size=2, attention_channels=8,
                      attention_heads=2, head_hidden_channels=8)
        protocols = [torch.arange(64).reshape(2, 32) % 3 for _ in range(8)]
        rows = [dict(model=model, seed=seed, RMSE=100., MAE=80., MAPE=.2, ACC15=.5)
                for model in ('batlinet', 'latent_cross_attention') for seed in range(8)]
        provenance = dict(policy=runner.POLICY, protocols_paths=['unused'] * 8)
        completed = []
        def train(model, training, epochs, evaluate_every, num_test, progress):
            self.assertEqual(len(training), 3)
            for epoch in range(1, epochs + 1):
                progress(dict(epoch=epoch, train_loss=.1, elapsed_seconds=0.))
            completed.append(True)
        original_predict = LatentCrossAttentionBatLiNetRULPredictor.predict
        def predict(model, dataset, **kwargs):
            self.assertEqual(len(completed), 8)
            return original_predict(model, dataset, **kwargs)
        with tempfile.TemporaryDirectory() as directory:
            argv = ['runner', '--workspace', directory, '--device', 'cpu']
            with patch.object(sys, 'argv', argv), patch.object(runner, 'model_config', return_value=config), \
                    patch.object(runner, 'audit_history', return_value=(data, protocols, rows, provenance)), \
                    patch.object(runner, 'legacy_train', side_effect=train) as training, \
                    patch.object(LatentCrossAttentionBatLiNetRULPredictor, 'predict', predict), \
                    redirect_stdout(io.StringIO()):
                runner.main()
                result_path = Path(directory) / f'{runner.MODEL}_seed0/test.pt'
                result = runner.load(result_path)
                torch.testing.assert_close(result['diagnostics']['support_index'], protocols[0])
                expected = runner.scores(data, result['prediction'])
                self.assertEqual(result['scores'], expected)
                torch.testing.assert_close(result['truth_cycles'], torch.tensor([150., 250.]))
                original_hash = runner.digest(result_path)
                runner.main()
                self.assertEqual(training.call_count, 8)
                self.assertEqual(runner.digest(result_path), original_hash)


if __name__ == '__main__':
    unittest.main()
