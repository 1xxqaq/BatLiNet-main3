import argparse
import os
from pathlib import Path

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_ANALYSIS_DIR = (
    "analysis/reference_selector/"
    "batlinet_original_mix20_trainS2_testS32/"
    "attention_reference_selector_v1"
)

STRATEGY_LABELS = {
    "all_32_mean": "32参考 mean",
    "batlinet_median_cache": "原始 median",
    "model_softmax_t1": "attention softmax",
    "model_keep_ratio_0.5_mean_top16": "attention top16",
    "model_top5_mean": "attention top5",
    "oracle_best_single": "oracle best",
    "oracle_top5_mean": "oracle top5",
}

STRATEGY_COLORS = {
    "all_32_mean": "#7A8DA4",
    "batlinet_median_cache": "#9AA7B7",
    "model_softmax_t1": "#4C9ED9",
    "model_keep_ratio_0.5_mean_top16": "#86BDE6",
    "model_top5_mean": "#B7D5EE",
    "oracle_best_single": "#51A36D",
    "oracle_top5_mean": "#8BC79B",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="绘制离线参考选择器实验的 PPT 汇报图。"
    )
    parser.add_argument(
        "--analysis-dir",
        default=DEFAULT_ANALYSIS_DIR,
        help="attention_reference_selector_v1 分析目录。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="汇总图输出目录；默认写入 analysis/reference_selector/ppt_summary。",
    )
    parser.add_argument(
        "--font-path",
        default=None,
        help="可选中文字体文件路径，例如 C:/Windows/Fonts/simsun.ttc。",
    )
    return parser.parse_args()


def setup_matplotlib(font_path: str = None):
    font_names = [
        "SimSun",
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    font_path = font_path or os.environ.get("BATLINET_FONT_PATH")
    if font_path:
        font_file = Path(font_path).expanduser()
        if not font_file.exists():
            raise FileNotFoundError(f"指定的字体文件不存在：{font_file}")
        font_manager.fontManager.addfont(str(font_file))
        font_prop = font_manager.FontProperties(fname=str(font_file))
        font_names.insert(0, font_prop.get_name())

    plt.rcParams["font.sans-serif"] = font_names
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["savefig.dpi"] = 220


def ensure_dirs(analysis_dir: Path, output_dir: Path):
    analysis_fig_dir = analysis_dir / "figures"
    summary_fig_dir = output_dir / "figures"
    analysis_fig_dir.mkdir(parents=True, exist_ok=True)
    summary_fig_dir.mkdir(parents=True, exist_ok=True)
    return analysis_fig_dir, summary_fig_dir


def plot_strategy_metrics(metric_summary: pd.DataFrame, analysis_fig_dir: Path, summary_fig_dir: Path):
    selected = [
        "all_32_mean",
        "batlinet_median_cache",
        "model_softmax_t1",
        "model_keep_ratio_0.5_mean_top16",
        "model_top5_mean",
        "oracle_best_single",
        "oracle_top5_mean",
    ]
    df = metric_summary[metric_summary["strategy"].isin(selected)].copy()
    df["strategy"] = pd.Categorical(df["strategy"], categories=selected, ordered=True)
    df = df.sort_values("strategy")

    metric_specs = [
        ("rmse_mean", "rmse_std", "RMSE"),
        ("mae_mean", "mae_std", "MAE"),
        ("mape_mean", "mape_std", "MAPE"),
    ]
    labels = [STRATEGY_LABELS[s] for s in df["strategy"]]
    colors = [STRATEGY_COLORS[s] for s in df["strategy"]]
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, (mean_col, std_col, title) in zip(axes, metric_specs):
        values = df[mean_col].to_numpy()
        stds = df[std_col].to_numpy()
        ax.bar(x, values, yerr=stds, color=colors, edgecolor="#333333", linewidth=0.6, capsize=3)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("数值（越低越好）")
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_axisbelow(True)

    fig.suptitle("attention_reference_selector_v1：离线聚合策略指标对比", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_name = "attention_strategy_metric_summary.png"
    fig.savefig(summary_fig_dir / out_name, bbox_inches="tight")
    fig.savefig(analysis_fig_dir / f"ppt_{out_name}", bbox_inches="tight")
    plt.close(fig)


def metric_row(rank_summary: pd.DataFrame, metric_name: str):
    row = rank_summary[rank_summary["metric"] == metric_name]
    if row.empty:
        raise KeyError(f"rank_alignment_summary.csv 缺少指标：{metric_name}")
    return row.iloc[0]


def plot_rank_alignment(rank_summary: pd.DataFrame, analysis_fig_dir: Path, summary_fig_dir: Path):
    overlap_metrics = [
        ("top1_overlap_rate", "top1", 1 / 32),
        ("top5_overlap_rate", "top5", 5 / 32),
        ("top16_overlap_rate", "top16", 16 / 32),
    ]
    labels = [item[1] for item in overlap_metrics]
    means = [metric_row(rank_summary, item[0])["mean"] * 100 for item in overlap_metrics]
    stds = [metric_row(rank_summary, item[0])["std"] * 100 for item in overlap_metrics]
    randoms = [item[2] * 100 for item in overlap_metrics]

    residual_metrics = [
        ("pred_top16_true_residual_mean", "预测 top16"),
        ("pred_bottom16_true_residual_mean", "丢弃 bottom16"),
        ("all32_true_residual_mean", "全部 32"),
    ]
    residual_labels = [item[1] for item in residual_metrics]
    residual_means = [metric_row(rank_summary, item[0])["mean"] for item in residual_metrics]
    residual_stds = [metric_row(rank_summary, item[0])["std"] for item in residual_metrics]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    x = np.arange(len(labels))
    axes[0].bar(x, means, yerr=stds, color="#4C9ED9", edgecolor="#333333", linewidth=0.6, capsize=3, label="模型")
    axes[0].scatter(x, randoms, color="#C84C4C", marker="D", zorder=3, label="随机期望")
    axes[0].set_title("排序 overlap 与随机水平对比", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("overlap（%）")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].grid(axis="y", linestyle="--", alpha=0.35)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=9)

    rx = np.arange(len(residual_labels))
    axes[1].bar(
        rx,
        residual_means,
        yerr=residual_stds,
        color=["#51A36D", "#C95F5F", "#7A8DA4"],
        edgecolor="#333333",
        linewidth=0.6,
        capsize=3,
    )
    axes[1].set_title("预测保留/丢弃参考的真实残差", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("真实融合残差均值")
    axes[1].set_xticks(rx)
    axes[1].set_xticklabels(residual_labels)
    axes[1].grid(axis="y", linestyle="--", alpha=0.35)
    axes[1].set_axisbelow(True)

    fig.suptitle("attention_reference_selector_v1：能学到弱排序信号，但不足以带来指标提升", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out_name = "attention_rank_alignment_summary.png"
    fig.savefig(summary_fig_dir / out_name, bbox_inches="tight")
    fig.savefig(analysis_fig_dir / f"ppt_{out_name}", bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    setup_matplotlib(args.font_path)

    analysis_dir = Path(args.analysis_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path("analysis/reference_selector/ppt_summary")
    analysis_fig_dir, summary_fig_dir = ensure_dirs(analysis_dir, output_dir)

    metric_path = analysis_dir / "tables" / "metric_summary.csv"
    rank_path = analysis_dir / "tables" / "rank_alignment_summary.csv"
    if not metric_path.exists():
        raise FileNotFoundError(f"未找到指标表：{metric_path}")
    if not rank_path.exists():
        raise FileNotFoundError(f"未找到排序诊断表：{rank_path}")

    metric_summary = pd.read_csv(metric_path)
    rank_summary = pd.read_csv(rank_path)

    plot_strategy_metrics(metric_summary, analysis_fig_dir, summary_fig_dir)
    plot_rank_alignment(rank_summary, analysis_fig_dir, summary_fig_dir)

    print(f"已输出离线选择器 PPT 汇报图到：{summary_fig_dir}")
    print(f"并同步写入实验 figures 目录：{analysis_fig_dir}")


if __name__ == "__main__":
    main()
