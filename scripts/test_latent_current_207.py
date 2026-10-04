"""Check the original encoder, saved peer identity and the eight-seed workflow."""
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
import run_mix20_latent_current_207 as current
from src.data.databundle import DataBundle
from src.data.transformation.sequential import SequentialDataTransformation
from src.models.rul_predictors.latent_cross_attention_batlinet import LatentCrossAttentionBatLiNetRULPredictor

shared = current.shared


class CurrentBaselineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(31)

    def fixture(self, directory):
        transform = SequentialDataTransformation([
            dict(name='LogScaleDataTransformation'), dict(name='ZScoreDataTransformation')])
        data = DataBundle(torch.randn(3, 6, 20, 64), torch.tensor([100., 200., 300.]),
                          torch.randn(2, 6, 20, 64), torch.tensor([150., 250.]),
                          label_transformation=transform)
        config = current.model_config()
        config.update(input_width=64, train_batch_size=2, attention_channels=8,
                      attention_heads=2, head_hidden_channels=8)
        enhanced_config = dict(config, name='CycleMixerLatentCrossAttentionBatLiNetRULPredictor',
                               cycle_mixer_hidden=16, cycle_mixer_type='mlp', cycle_conv_kernel=3)
        history = dict(policy=shared.POLICY, protocols_paths=['unused'] * 8)
        provenance = current.enhanced_provenance(history)
        protocols = [torch.arange(64).reshape(2, 32) % 3 for _ in range(8)]
        rows = [dict(model=model, seed=seed, RMSE=100., MAE=80., MAPE=.2, ACC15=.5)
                for model in ('batlinet', 'latent_cross_attention') for seed in range(8)]
        peer_root = Path(directory) / 'enhanced'
        peer_root.mkdir()
        for seed in range(8):
            model = shared.MODELS.build(enhanced_config, seed=seed)
            folder = peer_root / f'{shared.MODEL}_seed{seed}'
            folder.mkdir()
            identity = dict(provenance=provenance, config=enhanced_config, seed=seed, model=shared.MODEL)
            (folder / 'run.json').write_text(json.dumps(dict(identity, status='complete')))
            (folder / 'train.jsonl').write_text(''.join(
                json.dumps(dict(epoch=epoch, train_loss=.1)) + '\n' for epoch in range(1, 1001)))
            shared.save_new(dict(identity, state=model.state_dict(), epoch=1000), folder / 'epoch1000.pt')
            model._fixed_test_support_index = protocols[seed]
            model.fixed_test_support_index_path = 'unused'
            prediction, diagnostics = model.predict(data, return_diagnostics=True)
            shared.save_new(dict(model=shared.MODEL, seed=seed, epoch=1000,
                provenance=provenance, checkpoint_sha256=shared.digest(folder / 'epoch1000.pt'),
                prediction=prediction, diagnostics=diagnostics, scores=shared.scores(data, prediction)), folder / 'test.pt')
        return data, config, enhanced_config, history, protocols, rows, peer_root

    def test_original_architecture_and_initial_weights(self):
        config = current.model_config()
        self.assertEqual(config['name'], 'LatentCrossAttentionBatLiNetRULPredictor')
        self.assertFalse(any(key.startswith('cycle_mixer') for key in config))
        torch.manual_seed(11)
        original = shared.MODELS.build(config, seed=11)
        self.assertIs(type(original), LatentCrossAttentionBatLiNetRULPredictor)
        self.assertEqual(sum(p.numel() for p in original.parameters()), 170210)
        torch.manual_seed(11)
        enhanced = shared.MODELS.build(shared.model_config(), seed=11)
        for key, value in original.cell_encoder.state_dict().items():
            torch.testing.assert_close(enhanced.cell_encoder.base.state_dict()[key], value,
                                       rtol=0, atol=0)

    def test_train_original_eight_seeds_then_test_and_reuse_without_touching_peer(self):
        with tempfile.TemporaryDirectory() as directory:
            data, config, peer_config, history, protocols, rows, peer_root = self.fixture(directory)
            hashes = {str(path): shared.digest(path) for path in peer_root.rglob('*') if path.is_file()}
            workspace = Path(directory) / 'current'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace),
                    '--enhanced-runs', str(peer_root)]
            calls = []
            original_train = shared.legacy_train
            original_predict = LatentCrossAttentionBatLiNetRULPredictor.predict
            def train(model, training, epochs, every, num_test, progress):
                self.assertIs(type(model), LatentCrossAttentionBatLiNetRULPredictor)
                self.assertEqual((epochs, every, num_test), (1000, 100, 147))
                original_train(model, training, 1, 100, num_test)
                for epoch in range(1, 1001):
                    progress(dict(epoch=epoch, train_loss=.1, elapsed_seconds=0.))
                calls.append(True)
            def predict(model, dataset, **kwargs):
                self.assertEqual(len(calls), 8)
                return original_predict(model, dataset, **kwargs)
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=(data, protocols, rows, history)), \
                    patch.object(shared, 'model_config', return_value=peer_config), \
                    patch.object(current, 'model_config', return_value=config), \
                    patch.object(shared, 'legacy_train', side_effect=train) as training, \
                    patch.object(LatentCrossAttentionBatLiNetRULPredictor, 'predict', predict), \
                    redirect_stdout(io.StringIO()):
                current.main()
                result_path = workspace / f'{current.MODEL}_seed0/test.pt'
                result = shared.load(result_path)
                self.assertEqual(result['model'], current.MODEL)
                self.assertEqual(result['scores'], shared.scores(data, result['prediction']))
                torch.testing.assert_close(result['diagnostics']['support_index'], protocols[0])
                self.assertEqual(result['provenance']['runtime']['device'], 'cpu')
                self.assertEqual(len(json.loads((workspace / 'summary.json').read_text())), 32)
                result_hash = shared.digest(result_path)
                current.main()
                self.assertEqual(training.call_count, 8)
                self.assertEqual(shared.digest(result_path), result_hash)
            self.assertEqual(hashes, {str(path): shared.digest(path)
                                     for path in peer_root.rglob('*') if path.is_file()})

    def test_wrong_peer_references_fail_before_new_training_or_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            data, config, peer_config, history, protocols, rows, peer_root = self.fixture(directory)
            path = peer_root / f'{shared.MODEL}_seed3/test.pt'
            result = shared.load(path)
            result['diagnostics']['support_index'][0, 0] = 2
            torch.save(result, path)
            workspace = Path(directory) / 'current'
            argv = ['runner', '--device', 'cpu', '--workspace', str(workspace),
                    '--enhanced-runs', str(peer_root)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=(data, protocols, rows, history)), \
                    patch.object(shared, 'model_config', return_value=peer_config), \
                    patch.object(current, 'model_config', return_value=config), \
                    patch.object(shared, 'legacy_train') as training, \
                    redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                current.main()
            training.assert_not_called()
            self.assertFalse(workspace.exists())

    def test_audit_only_is_read_only(self):
        with tempfile.TemporaryDirectory() as directory:
            data, config, peer_config, history, protocols, rows, peer_root = self.fixture(directory)
            workspace = Path(directory) / 'current'
            argv = ['runner', '--audit-only', '--workspace', str(workspace),
                    '--enhanced-runs', str(peer_root)]
            with patch.object(sys, 'argv', argv), \
                    patch.object(shared, 'audit_history', return_value=(data, protocols, rows, history)), \
                    patch.object(shared, 'model_config', return_value=peer_config), \
                    patch.object(shared, 'legacy_train') as training, \
                    redirect_stdout(io.StringIO()):
                current.main()
            training.assert_not_called()
            self.assertFalse(workspace.exists())


if __name__ == '__main__':
    unittest.main()
