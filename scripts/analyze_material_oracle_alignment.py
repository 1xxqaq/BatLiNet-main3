import argparse
import csv
import math
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import torch


TOKENS = {
    "surface": "#FCFCFD",
    "panel": "#FFFFFF",
    "ink": "#1F2430",
    "muted": "#6F768A",
    "grid": "#E6E8F0",
    "axis": "#D7DBE7",
}

NEUTRAL_MARKS = {
    "open": TOKENS["panel"],
    "xlight": "#F4F5F7",
    "light": "#E2E5EA",
    "base": "#C5CAD3",
    "mid": "#7A828F",
    "dark": "#464C55",
}

COLOR_FAMILIES = {
    "blue": {
        "open": TOKENS["panel"],
        "xlight": "#EAF1FE",
        "light": "#CEDFFE",
        "base": "#A3BEFA",
        "mid": "#5477C4",
        "dark": "#2E4780",
    },
    "gold": {
        "open": TOKENS["panel"],
        "xlight": "#FFF4C2",
        "light": "#FFEA8F",
        "base": "#FFE15B",
        "mid": "#B8A037",
        "dark": "#736422",
    },
    "orange": {
        "open": TOKENS["panel"],
        "xlight": "#FFEDDE",
        "light": "#FFBDA1",
        "base": "#F0986E",
        "mid": "#CC6F47",
        "dark": "#804126",
    },
    "olive": {
        "open": TOKENS["panel"],
        "xlight": "#D8ECBD",
        "light": "#BEEB96",
        "base": "#A3D576",
        "mid": "#71B436",
        "dark": "#386411",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze cathode-material alignment with oracle reference quality."
    )
    parser.add_argument(
        "--cache-root",
        default="artifacts/frozen_reference_cache/batlinet_original_mix20_trainS2_testS32",
        help="Frozen reference cache root.",
    )
    parser.add_argument(
        "--test-subdir",
        default="protocol_v1",
        help="Test cache subdirectory under cache root.",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4,5,6,7",
        help="Comma-separated test seeds.",
    )
    parser.add_argument(
        "--output-dir",
        default=(
            "analysis/reference_selector/"
            "batlinet_original_mix20_trainS2_testS32/material_oracle_diagnostics"
        ),
        help="Output directory for tables, samples, figures, and summary.",
    )
    parser.add_argument(
        "--topk",
        default="1,3,5,8,16",
        help="Comma-separated oracle top-k values to summarize.",
    )
    parser.add_argument(
        "--font-path",
        default=r"C:\Windows\Fonts\msyh.ttc",
        help="Optional font path for Chinese chart text.",
    )
    return parser.parse_args()


def parse_int_list(value):
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_label_inverse_params(tensors):
    transformed = tensors["target_label"].float()
    original = tensors["target_label_original"].float().clamp_min(1e-8)
    log_original = torch.log(original)
    transformed_mean = transformed.mean()
    log_mean = log_original.mean()
    numerator = ((transformed - transformed_mean) * (log_original - log_mean)).sum()
    denominator = ((transformed - transformed_mean) ** 2).sum().clamp_min(1e-8)
    std = numerator / denominator
    mean = log_mean - std * transformed_mean
    return mean, std


def inverse_label_transform(transformed, mean, std):
    return torch.exp(transformed * std + mean)


def material_value(meta, key):
    value = meta.get(key, "")
    if value is None or value == "":
        return "unknown"
    return str(value)


def source_from_meta(meta):
    source_path = str(meta.get("source_path", ""))
    parts = source_path.replace("\\", "/").split("/")
    if "processed" in parts:
        idx = parts.index("processed")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    if len(parts) >= 2:
        return parts[-2]
    return "unknown"


def load_seed_pair_rows(cache_path, seed, topks):
    payload = torch.load(cache_path, map_location="cpu")
    tensors = payload["tensors"]
    target_meta = payload.get("target_metadata", [])
    support_meta = payload.get("support_metadata", [])
    alpha = float(payload.get("alpha", 0.5))

    label_mean, label_std = infer_label_inverse_params(tensors)
    y_true = tensors["target_label_original"].float()
    y_ori = tensors["y_ori"].float()
    y_sup = tensors["y_sup"].float()
    y_ori_original = tensors["y_ori_original"].float()
    y_sup_original = tensors["y_sup_original"].float()
    support_label_original = tensors["support_label_original"].float()
    support_index = tensors["support_index"].long()

    fused_pair = (1.0 - alpha) * y_ori[:, None] + alpha * y_sup
    fused_pair_original = inverse_label_transform(fused_pair, label_mean, label_std)
    true_residual = (fused_pair_original - y_true[:, None]).abs()
    true_order = torch.argsort(true_residual, dim=1)
    true_rank = torch.empty_like(true_order)
    rank_values = torch.arange(true_order.size(1))[None, :].expand_as(true_order)
    true_rank.scatter_(1, true_order, rank_values)

    rows = []
    num_targets, support_size = y_sup.size()
    for target_idx in range(num_targets):
        tm = target_meta[target_idx] if target_idx < len(target_meta) else {}
        target_cathode = material_value(tm, "cathode_material")
        target_anode = material_value(tm, "anode_material")
        target_source = source_from_meta(tm)
        target_cell = str(tm.get("cell_id", ""))
        target_key = f"{seed}:{target_idx}:{target_cell}"

        for support_pos in range(support_size):
            sm = {}
            if target_idx < len(support_meta) and support_pos < len(support_meta[target_idx]):
                sm = support_meta[target_idx][support_pos]
            support_cathode = material_value(sm, "cathode_material")
            support_anode = material_value(sm, "anode_material")
            support_source = source_from_meta(sm)
            support_cell = str(sm.get("cell_id", ""))
            rank = int(true_rank[target_idx, support_pos].item()) + 1

            row = {
                "seed": seed,
                "target_row": target_idx,
                "support_pos": support_pos,
                "target_key": target_key,
                "support_index": int(support_index[target_idx, support_pos].item()),
                "target_cell_id": target_cell,
                "support_cell_id": support_cell,
                "target_source": target_source,
                "support_source": support_source,
                "target_cathode": target_cathode,
                "support_cathode": support_cathode,
                "same_cathode": target_cathode == support_cathode,
                "target_anode": target_anode,
                "support_anode": support_anode,
                "same_anode": target_anode == support_anode,
                "y_true": float(y_true[target_idx].item()),
                "y_ori": float(y_ori_original[target_idx].item()),
                "y_sup": float(y_sup_original[target_idx, support_pos].item()),
                "support_label": float(support_label_original[target_idx, support_pos].item()),
                "fused_prediction": float(fused_pair_original[target_idx, support_pos].item()),
                "true_fused_abs_error": float(true_residual[target_idx, support_pos].item()),
                "true_rank": rank,
            }
            for topk in topks:
                row[f"is_oracle_top{topk}"] = rank <= topk
            rows.append(row)
    return rows


def safe_divide(numerator, denominator):
    if denominator == 0 or pd.isna(denominator):
        return math.nan
    return numerator / denominator


def summarize_material_pairs(pair_df, topks):
    group_cols = ["target_cathode", "support_cathode"]
    grouped = pair_df.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        num_pairs=("true_fused_abs_error", "size"),
        num_seed_targets=("target_key", "nunique"),
        mean_true_fused_abs_error=("true_fused_abs_error", "mean"),
        median_true_fused_abs_error=("true_fused_abs_error", "median"),
        mean_true_rank=("true_rank", "mean"),
        median_true_rank=("true_rank", "median"),
    ).reset_index()

    total_by_target = pair_df.groupby("target_cathode").agg(
        target_material_pairs=("true_fused_abs_error", "size"),
        target_material_seed_targets=("target_key", "nunique"),
    ).reset_index()
    summary = summary.merge(total_by_target, on="target_cathode", how="left")
    summary["support_share_within_target_material"] = (
        summary["num_pairs"] / summary["target_material_pairs"]
    )

    for topk in topks:
        col = f"is_oracle_top{topk}"
        hits = grouped[col].sum().reset_index(name=f"top{topk}_count")
        summary = summary.merge(hits, on=group_cols, how="left")
        total_hits = pair_df.groupby("target_cathode")[col].sum().reset_index(
            name=f"target_material_top{topk}_events"
        )
        summary = summary.merge(total_hits, on="target_cathode", how="left")
        summary[f"top{topk}_pair_hit_rate"] = summary[f"top{topk}_count"] / summary["num_pairs"]
        summary[f"top{topk}_event_share_within_target_material"] = summary.apply(
            lambda row: safe_divide(row[f"top{topk}_count"], row[f"target_material_top{topk}_events"]),
            axis=1,
        )
        summary[f"top{topk}_lift_vs_support_share"] = summary.apply(
            lambda row: safe_divide(
                row[f"top{topk}_event_share_within_target_material"],
                row["support_share_within_target_material"],
            ),
            axis=1,
        )

    return summary.sort_values(["target_cathode", "support_cathode"])


def summarize_same_vs_cross(pair_df, topks):
    rows = []
    total_pairs = len(pair_df)
    for same_value, part in pair_df.groupby("same_cathode"):
        row = {
            "comparison": "same_cathode" if bool(same_value) else "cross_cathode",
            "num_pairs": len(part),
            "support_share": len(part) / total_pairs,
            "num_seed_targets": part["target_key"].nunique(),
            "mean_true_fused_abs_error": part["true_fused_abs_error"].mean(),
            "median_true_fused_abs_error": part["true_fused_abs_error"].median(),
            "mean_true_rank": part["true_rank"].mean(),
            "median_true_rank": part["true_rank"].median(),
        }
        for topk in topks:
            col = f"is_oracle_top{topk}"
            total_events = pair_df[col].sum()
            count = part[col].sum()
            row[f"top{topk}_count"] = int(count)
            row[f"top{topk}_pair_hit_rate"] = count / len(part)
            row[f"top{topk}_event_share"] = safe_divide(count, total_events)
            row[f"top{topk}_lift_vs_support_share"] = safe_divide(
                row[f"top{topk}_event_share"],
                row["support_share"],
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("comparison")


def summarize_target_material(pair_df, topks):
    rows = []
    for target_cathode, part in pair_df.groupby("target_cathode"):
        same = part[part["same_cathode"]]
        row = {
            "target_cathode": target_cathode,
            "num_pairs": len(part),
            "num_seed_targets": part["target_key"].nunique(),
            "support_share_same_cathode": len(same) / len(part),
            "mean_true_fused_abs_error": part["true_fused_abs_error"].mean(),
            "median_true_fused_abs_error": part["true_fused_abs_error"].median(),
        }
        for topk in topks:
            col = f"is_oracle_top{topk}"
            total_events = part[col].sum()
            same_events = same[col].sum()
            row[f"top{topk}_same_cathode_event_share"] = safe_divide(same_events, total_events)
            row[f"top{topk}_same_cathode_lift"] = safe_divide(
                row[f"top{topk}_same_cathode_event_share"],
                row["support_share_same_cathode"],
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("target_cathode")


def use_chart_theme(font_path=None):
    font_family = ["Microsoft YaHei", "SimHei", "DejaVu Sans", "Arial", "sans-serif"]
    if font_path and Path(font_path).exists():
        fm.fontManager.addfont(font_path)
        font_family = [fm.FontProperties(fname=font_path).get_name()] + font_family
    sns.set_theme(
        style="whitegrid",
        rc={
            "figure.facecolor": TOKENS["surface"],
            "figure.edgecolor": "none",
            "savefig.facecolor": TOKENS["surface"],
            "savefig.edgecolor": "none",
            "axes.facecolor": TOKENS["panel"],
            "axes.edgecolor": TOKENS["axis"],
            "axes.labelcolor": TOKENS["ink"],
            "axes.grid": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": TOKENS["grid"],
            "grid.linewidth": 0.8,
            "font.family": "sans-serif",
            "font.sans-serif": font_family,
            "axes.unicode_minus": False,
            "patch.linewidth": 1.0,
        },
    )


def add_chart_header(fig, ax, title, subtitle):
    ax.set_title("")
    fig.subplots_adjust(top=0.82)
    left = ax.get_position().x0
    fig.text(
        left,
        0.97,
        title,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="semibold",
        color=TOKENS["ink"],
    )
    fig.text(
        left,
        0.915,
        subtitle,
        ha="left",
        va="top",
        fontsize=9.5,
        color=TOKENS["muted"],
    )
    sns.despine(ax=ax)


def save_figure(fig, path_base):
    path_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(path_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_material_heatmap(summary, value_col, title, subtitle, output_path, fmt=".2f"):
    matrix = summary.pivot(
        index="target_cathode",
        columns="support_cathode",
        values=value_col,
    ).sort_index()
    annot = matrix.map(lambda value: "" if pd.isna(value) else format(value, fmt))
    fig, ax = plt.subplots(figsize=(8.8, 6.6))
    family = COLOR_FAMILIES["gold"]
    cmap = sns.blend_palette(
        [TOKENS["panel"], family["xlight"], family["light"], family["base"], family["mid"]],
        as_cmap=True,
    )
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        linewidths=1.0,
        linecolor=TOKENS["panel"],
        annot=annot,
        fmt="",
        cbar_kws={"shrink": 0.82},
    )
    ax.set_xlabel("参考电池正极材料")
    ax.set_ylabel("目标电池正极材料")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    add_chart_header(fig, ax, title, subtitle)
    save_figure(fig, output_path)


def plot_same_cross_bar(same_df, topk, output_path):
    value_col = f"top{topk}_lift_vs_support_share"
    plot_df = same_df.copy()
    plot_df["comparison_label"] = plot_df["comparison"].map({
        "same_cathode": "同正极材料",
        "cross_cathode": "跨正极材料",
    })
    plot_df = plot_df.sort_values(value_col, ascending=True)

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    palette = {
        "同正极材料": COLOR_FAMILIES["olive"]["base"],
        "跨正极材料": COLOR_FAMILIES["orange"]["base"],
    }
    edge = {
        "同正极材料": COLOR_FAMILIES["olive"]["dark"],
        "跨正极材料": COLOR_FAMILIES["orange"]["dark"],
    }
    sns.barplot(
        data=plot_df,
        x=value_col,
        y="comparison_label",
        hue="comparison_label",
        palette=palette,
        legend=False,
        dodge=False,
        ax=ax,
        edgecolor=TOKENS["ink"],
        linewidth=1.0,
    )
    for patch, label in zip(ax.patches, plot_df["comparison_label"]):
        patch.set_edgecolor(edge[label])
        value = patch.get_width()
        ax.text(
            value + 0.03,
            patch.get_y() + patch.get_height() / 2,
            f"{value:.2f}x",
            va="center",
            ha="left",
            fontsize=9,
            color=TOKENS["ink"],
        )
    ax.axvline(1.0, color=TOKENS["ink"], linestyle=":", linewidth=1.0)
    ax.set_xlabel(f"Oracle top{topk} 富集倍数")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.1fx"))
    add_chart_header(
        fig,
        ax,
        f"同材料参考在 oracle top{topk} 中是否富集",
        "富集倍数 = oracle top-k 事件占比 / 参考池暴露占比；1x 表示与随机暴露一致。",
    )
    save_figure(fig, output_path)


def write_summary(path, pair_summary, same_df, target_df, topk):
    path.parent.mkdir(parents=True, exist_ok=True)
    best_pairs = pair_summary.sort_values(
        f"top{topk}_lift_vs_support_share",
        ascending=False,
    ).head(8)
    same_row = same_df[same_df["comparison"] == "same_cathode"].iloc[0]
    cross_row = same_df[same_df["comparison"] == "cross_cathode"].iloc[0]

    lines = [
        "# 材料体系与 oracle 参考质量诊断",
        "",
        f"- 分析对象：固定测试参考协议 `protocol_v1`，8 个 seed，按正极材料 `cathode_material` 汇总。",
        f"- `top{topk}` 富集倍数定义：oracle top-k 事件占比 / 该材料组合在参考池中的暴露占比。",
        f"- 同正极材料参考暴露占比：{same_row['support_share']:.4f}；oracle top{topk} 事件占比：{same_row[f'top{topk}_event_share']:.4f}；富集倍数：{same_row[f'top{topk}_lift_vs_support_share']:.3f}x。",
        f"- 跨正极材料参考暴露占比：{cross_row['support_share']:.4f}；oracle top{topk} 事件占比：{cross_row[f'top{topk}_event_share']:.4f}；富集倍数：{cross_row[f'top{topk}_lift_vs_support_share']:.3f}x。",
        "",
        f"## top{topk} 富集最高的材料组合",
        "",
        "| 目标正极 | 参考正极 | 参考暴露占比 | oracle top-k 占比 | 富集倍数 | 平均融合误差 | pair 数 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in best_pairs.iterrows():
        lines.append(
            "| {target} | {support} | {support_share:.4f} | {event_share:.4f} | "
            "{lift:.3f}x | {err:.2f} | {pairs:d} |".format(
                target=row["target_cathode"],
                support=row["support_cathode"],
                support_share=row["support_share_within_target_material"],
                event_share=row[f"top{topk}_event_share_within_target_material"],
                lift=row[f"top{topk}_lift_vs_support_share"],
                err=row["mean_true_fused_abs_error"],
                pairs=int(row["num_pairs"]),
            )
        )
    lines.extend([
        "",
        "## 目标材料同材料参考摘要",
        "",
        "| 目标正极 | 同材料参考暴露占比 | 同材料 oracle top-k 占比 | 同材料富集倍数 |",
        "| --- | ---: | ---: | ---: |",
    ])
    for _, row in target_df.iterrows():
        lines.append(
            "| {target} | {support_share:.4f} | {event_share:.4f} | {lift:.3f}x |".format(
                target=row["target_cathode"],
                support_share=row["support_share_same_cathode"],
                event_share=row[f"top{topk}_same_cathode_event_share"],
                lift=row[f"top{topk}_same_cathode_lift"],
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    cache_root = Path(args.cache_root)
    test_dir = cache_root / args.test_subdir
    output_dir = Path(args.output_dir)
    topks = parse_int_list(args.topk)

    all_rows = []
    for seed in parse_int_list(args.seeds):
        cache_path = test_dir / f"test_seed_{seed}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing test cache: {cache_path}")
        all_rows.extend(load_seed_pair_rows(cache_path, seed, topks))

    pair_df = pd.DataFrame(all_rows)
    pair_summary = summarize_material_pairs(pair_df, topks)
    same_df = summarize_same_vs_cross(pair_df, topks)
    target_df = summarize_target_material(pair_df, topks)

    samples_dir = output_dir / "samples"
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    summary_dir = output_dir / "summary"
    samples_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    pair_df.to_csv(samples_dir / "material_pair_details.csv", index=False, encoding="utf-8")
    pair_summary.to_csv(tables_dir / "material_pair_oracle_lift_summary.csv", index=False, encoding="utf-8")
    same_df.to_csv(tables_dir / "same_vs_cross_material_summary.csv", index=False, encoding="utf-8")
    target_df.to_csv(tables_dir / "target_material_oracle_summary.csv", index=False, encoding="utf-8")

    use_chart_theme(args.font_path)
    plot_material_heatmap(
        pair_summary,
        "top5_lift_vs_support_share",
        "材料组合在 oracle top5 参考中的富集倍数",
        "按目标正极材料和参考正极材料汇总；数值越高，说明该组合更常出现在真实融合误差前 5。",
        figures_dir / "material_pair_top5_lift_heatmap",
        fmt=".2f",
    )
    plot_material_heatmap(
        pair_summary,
        "mean_true_fused_abs_error",
        "材料组合的平均单参考融合误差",
        "误差单位为原始寿命尺度；数值越低，说明该目标-参考材料组合更接近 oracle 好参考。",
        figures_dir / "material_pair_mean_fused_abs_error_heatmap",
        fmt=".1f",
    )
    plot_same_cross_bar(
        same_df,
        topk=5,
        output_path=figures_dir / "same_vs_cross_top5_lift_bar",
    )
    write_summary(
        summary_dir / "material_oracle_diagnostics_summary.md",
        pair_summary,
        same_df,
        target_df,
        topk=5,
    )

    print(f"Wrote material oracle diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
