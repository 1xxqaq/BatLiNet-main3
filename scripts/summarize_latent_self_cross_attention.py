"""Read fixed-protocol predictions and display seed metrics and sample SD."""

import argparse
import io
import json
import math
import pickle
import statistics
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

METRICS = ('RMSE', 'MAE', 'MAPE')


class CPUUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda data: torch.load(
                io.BytesIO(data), map_location='cpu', weights_only=False)
        return super().find_class(module, name)


def collect_results(workspace, start_seed=0, end_seed=7):
    if not 0 <= start_seed <= end_seed <= 7:
        raise ValueError('种子范围必须满足 0 <= 起始种子 <= 结束种子 <= 7。')
    workspace = Path(workspace)
    rows, missing = [], []
    for seed in range(start_seed, end_seed + 1):
        files = sorted(workspace.glob(f'predictions_seed_{seed}_*.pkl'))
        if not files:
            missing.append(seed)
            continue
        if len(files) != 1:
            raise ValueError(f'种子 {seed} 有多份预测文件，拒绝自动选择：{workspace}')
        with files[0].open('rb') as stream:
            payload = CPUUnpickler(stream).load()
        if payload.get('seed') != seed:
            raise ValueError(f'文件种子信息不一致：{files[0]}')
        if not payload.get('evaluation_protocol', {}).get('fixed_test_support_index_path'):
            raise ValueError(f'缺少固定协议记录：{files[0]}')
        data = payload['data'].to('cpu')
        prediction = payload['prediction'].detach().cpu()
        scores = {metric: data.evaluate(prediction, metric) for metric in METRICS}
        for metric, value in scores.items():
            saved = float(payload['scores'][metric])
            if not math.isfinite(value) or not math.isfinite(saved):
                raise ValueError(f'{files[0]} 的 {metric} 包含非有限数值。')
            if not math.isclose(value, saved, rel_tol=1e-5, abs_tol=1e-4):
                raise ValueError(f'{files[0]} 的 {metric} 与预测重算值不一致。')
        rows.append(dict(seed=seed, file=str(files[0]), scores=scores))
    aggregate = None
    if not missing:
        aggregate = {
            metric: dict(
                mean=statistics.mean(row['scores'][metric] for row in rows),
                sample_std=(statistics.stdev(row['scores'][metric] for row in rows)
                            if len(rows) > 1 else None))
            for metric in METRICS
        }
    return dict(workspace=str(workspace), expected_seeds=list(range(start_seed, end_seed + 1)),
                completed_seeds=[row['seed'] for row in rows], missing_seeds=missing,
                complete=not missing, rows=rows, aggregate=aggregate,
                units=dict(RMSE='cycles', MAE='cycles', MAPE='fraction'))


def display_results(result):
    print(f"结果目录：{result['workspace']}")
    print(f"已完成：{len(result['rows'])}/{len(result['expected_seeds'])} 个种子")
    print('种子       RMSE          MAE       MAPE（%）')
    for row in result['rows']:
        scores = row['scores']
        print(f"{row['seed']:>4} {scores['RMSE']:>12.4f} {scores['MAE']:>12.4f} "
              f"{100 * scores['MAPE']:>12.4f}")
    if not result['complete']:
        print(f"尚未完成的种子：{result['missing_seeds']}；不生成最终汇总。")
        return
    print('各种子指标均值 ± 样本标准差（不是多模型集成结果）：')
    for metric in METRICS:
        scale = 100 if metric == 'MAPE' else 1
        suffix = '%' if metric == 'MAPE' else ' 周期'
        values = result['aggregate'][metric]
        deviation = values['sample_std']
        spread = f'{scale * deviation:.4f}' if deviation is not None else '不适用'
        print(f"{metric}: {scale * values['mean']:.4f} ± {spread}{suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=('mix_20', 'mix_100'))
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--start-seed', type=int, default=0)
    parser.add_argument('--end-seed', type=int, default=7)
    parser.add_argument('--output', type=Path,
                        help='Optionally save a complete summary as JSON.')
    args = parser.parse_args()
    workspace = args.workspace or (
        REPO_ROOT / 'workspaces/fair_eval/batlinet_latent_self_cross_attention'
        / args.task / 'protocol_v1_formal')
    try:
        result = collect_results(workspace, args.start_seed, args.end_seed)
        display_results(result)
        if not result['complete']:
            return 1
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(dict(task=args.task, **result), ensure_ascii=False, indent=2)
                + '\n', encoding='utf-8')
            print(f'汇总已保存：{args.output}')
        return 0
    except (ValueError, KeyError, OSError, EOFError, pickle.UnpicklingError) as error:
        print(f'结果检查失败：{error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
