"""Check redirected progress, unchanged optimization and result summaries."""

import contextlib
import io
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'scripts'))

from test_latent_self_cross_attention import model_kwargs
from summarize_latent_self_cross_attention import collect_results, display_results
from src.data.databundle import DataBundle
from src.models.rul_predictors.latent_cross_attention_batlinet import (
    LatentCrossAttentionBatLiNetRULPredictor,
)
from src.models.rul_predictors.latent_self_cross_attention_batlinet import (
    LatentSelfCrossAttentionBatLiNetRULPredictor,
)
from src.utils.training_progress import TrainingProgress


def write_prediction(workspace, seed, errors=2.0, bad_score=False):
    data = DataBundle(torch.zeros(2, 1), torch.tensor([10., 20.]),
                      torch.zeros(2, 1), torch.tensor([10., 20.]))
    prediction = torch.tensor([10. + errors, 20. - errors])
    scores = {metric: data.evaluate(prediction, metric)
              for metric in ('RMSE', 'MAE', 'MAPE')}
    if bad_score:
        scores['RMSE'] = 999.
    payload = dict(seed=seed, data=data, prediction=prediction, scores=scores,
                   evaluation_protocol=dict(fixed_test_support_index_path='fixed.pt'))
    with (workspace / f'predictions_seed_{seed}_test.pkl').open('wb') as stream:
        pickle.dump(payload, stream)


class ProgressAndResultsTests(unittest.TestCase):
    def test_progress_eta_and_remaining_training_count(self):
        now = [0.0]
        stream = io.StringIO()
        progress = TrainingProgress(
            '测试模型', 2, task='mix_20', training_number=3, total_trainings=8,
            clock=lambda: now[0], stream=stream)
        progress.start(4, 2)
        self.assertIn('当前训练预计剩余=估算中', stream.getvalue())
        self.assertIn('后续训练=5次', stream.getvalue())
        progress.epoch_started(1)
        now[0] = 10.0
        progress.batch_finished(1, 1)
        self.assertIn('当前训练预计剩余=00:01:10', stream.getvalue())
        self.assertIn('轮次=1/4', stream.getvalue())
        now[0] = 20.0
        progress.batch_finished(1, 2)
        progress.epoch_finished(1)
        self.assertIn('当前训练预计剩余=00:01:00', stream.getvalue())
        now[0] = 80.0
        progress.epoch_finished(4)
        progress.finish()
        self.assertIn('当前训练预计剩余=00:00:00', stream.getvalue())
        self.assertNotIn('\r', stream.getvalue())

    def test_invalid_progress_interval(self):
        for interval in (0, -1, math.inf, math.nan):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                TrainingProgress('test', 0, log_interval=interval)

    def test_logging_preserves_seeded_training_with_periodic_evaluation(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)
        kwargs = dict(model_kwargs(), epochs=2, evaluate_freq=1,
                      checkpoint_freq=None)
        base = LatentCrossAttentionBatLiNetRULPredictor(**kwargs)
        logged = LatentSelfCrossAttentionBatLiNetRULPredictor(
            **kwargs, self_attention_layers=0)
        logged.load_state_dict(base.state_dict())
        data = DataBundle(
            torch.randn(2, 6, 20, 1000), torch.tensor([0.1, 0.8]),
            torch.randn(1, 6, 20, 1000), torch.tensor([0.4]))
        with contextlib.redirect_stdout(io.StringIO()):
            torch.manual_seed(23)
            base.fit(data, timestamp='plain')
        log = io.StringIO()
        with contextlib.redirect_stdout(log):
            torch.manual_seed(23)
            logged.fit(data, timestamp='logged')
        for key, value in base.state_dict().items():
            torch.testing.assert_close(logged.state_dict()[key], value,
                                       rtol=0, atol=0, msg=key)
        self.assertIn('阶段=周期测试评估', log.getvalue())
        self.assertIn('阶段=训练完成', log.getvalue())
        self.assertIsNone(logged._training_progress)

    def test_summary_recomputes_metrics_and_cli_saves_json(self):
        with tempfile.TemporaryDirectory(prefix='batlinet_results_') as temp:
            workspace = Path(temp)
            write_prediction(workspace, 0, errors=2.)
            write_prediction(workspace, 1, errors=4.)
            result = collect_results(workspace, 0, 1)
            self.assertTrue(result['complete'])
            self.assertAlmostEqual(result['aggregate']['RMSE']['mean'], 3.)
            self.assertAlmostEqual(result['aggregate']['RMSE']['sample_std'], math.sqrt(2))
            self.assertAlmostEqual(result['aggregate']['MAPE']['mean'], 0.225, places=6)
            output = workspace / 'summary.json'
            process = subprocess.run(
                [sys.executable, '-B', str(REPO_ROOT / 'scripts/summarize_latent_self_cross_attention.py'),
                 'mix_20', '--workspace', str(workspace), '--start-seed', '0',
                 '--end-seed', '1', '--output', str(output)],
                capture_output=True, encoding='utf-8',
                env=dict(os.environ, PYTHONIOENCODING='utf-8'))
            self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(saved['units']['MAPE'], 'fraction')
            self.assertEqual(saved['completed_seeds'], [0, 1])

    def test_missing_seeds_do_not_produce_final_aggregate(self):
        with tempfile.TemporaryDirectory(prefix='batlinet_missing_') as temp:
            workspace = Path(temp)
            write_prediction(workspace, 0)
            result = collect_results(workspace)
            self.assertFalse(result['complete'])
            self.assertEqual(result['missing_seeds'], list(range(1, 8)))
            self.assertIsNone(result['aggregate'])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                display_results(result)
            self.assertIn('不生成最终汇总', stream.getvalue())

    def test_duplicate_files_and_conflicting_scores_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix='batlinet_duplicate_') as temp:
            workspace = Path(temp)
            write_prediction(workspace, 0)
            (workspace / 'predictions_seed_0_duplicate.pkl').touch()
            with self.assertRaisesRegex(ValueError, '多份预测文件'):
                collect_results(workspace, 0, 0)
        with tempfile.TemporaryDirectory(prefix='batlinet_conflict_') as temp:
            workspace = Path(temp)
            write_prediction(workspace, 0, bad_score=True)
            with self.assertRaisesRegex(ValueError, '重算值不一致'):
                collect_results(workspace, 0, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
