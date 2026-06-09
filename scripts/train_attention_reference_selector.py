import argparse
import copy
import csv
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset


class SetAttentionResidualSelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, ff_dim, dropout):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.score_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        h = self.input_proj(x)
        h = self.encoder(h)
        return self.score_head(h).squeeze(-1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an offline attention-based frozen-cache reference selector."
    )
    parser.add_argument(
        "--config",
        default="configs/reference_selector/attention_reference_selector_v1.yaml",
        help="Path to the reference selector config.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Optional comma-separated seed override, for example 0,1.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_cache(path):
    payload = torch.load(path, map_location="cpu")
    tensors = payload["tensors"]
    return payload, tensors


def set_features(tensors):
    x_ori = tensors["x_ori"].float()
    x_sup = tensors["x_sup"].float()
    y_ori = tensors["y_ori_original"].float()
    y_sup = tensors["y_sup_original"].float()
    support_label = tensors["support_label_original"].float()

    bsz, support_size, channels = x_sup.shape
    x_ori_expanded = x_ori[:, None, :].expand(bsz, support_size, channels)
    y_ori_expanded = y_ori[:, None].expand(bsz, support_size)
    fused_pair_original = 0.5 * y_ori_expanded + 0.5 * y_sup

    scalar_features = torch.stack(
        [
            y_ori_expanded,
            y_sup,
            support_label,
            y_sup - y_ori_expanded,
            (y_sup - y_ori_expanded).abs(),
            y_sup - support_label,
            (y_sup - support_label).abs(),
            fused_pair_original,
        ],
        dim=-1,
    )
    return torch.cat([x_ori_expanded, x_sup, scalar_features], dim=-1)


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


def residual_targets(tensors, alpha, target_transform):
    label_mean, label_std = infer_label_inverse_params(tensors)
    y_true = tensors["target_label_original"].float()
    y_ori = tensors["y_ori"].float()
    y_sup = tensors["y_sup"].float()
    fused_pair = (1.0 - alpha) * y_ori[:, None] + alpha * y_sup
    fused_pair_original = inverse_label_transform(fused_pair, label_mean, label_std)
    residual = (fused_pair_original - y_true[:, None]).abs()
    if target_transform == "log1p":
        return torch.log1p(residual)
    if target_transform in (None, "none"):
        return residual
    raise ValueError(f"Unsupported target transform: {target_transform}")


def inverse_residual_score(score, target_transform):
    if target_transform == "log1p":
        return torch.expm1(score).clamp_min(0.0)
    return score.clamp_min(0.0)


def split_targets(num_targets, val_fraction, seed):
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_targets, generator=generator)
    num_val = max(1, int(round(num_targets * val_fraction)))
    val_targets = set(perm[:num_val].tolist())
    train_mask = torch.tensor([idx not in val_targets for idx in range(num_targets)])
    val_mask = ~train_mask
    return train_mask, val_mask


def standardize_set_features(train_x, all_x):
    train_flat = train_x.reshape(-1, train_x.size(-1))
    mean = train_flat.mean(dim=0)
    std = train_flat.std(dim=0).clamp_min(1e-6)
    return (all_x - mean) / std, mean, std


def selector_loss(pred, target, train_cfg):
    teacher_temp = float(train_cfg["teacher_temperature"])
    score_temp = float(train_cfg["score_temperature"])
    teacher = torch.softmax(-target / teacher_temp, dim=1)
    student_log = F.log_softmax(-pred / score_temp, dim=1)
    rank_loss = F.kl_div(student_log, teacher, reduction="batchmean")
    residual_loss = F.mse_loss(pred, target)
    loss = (
        float(train_cfg["rank_loss_weight"]) * rank_loss
        + float(train_cfg["residual_loss_weight"]) * residual_loss
    )
    return loss, rank_loss, residual_loss


def train_model(features, targets, cfg, device, seed):
    train_cfg = cfg["training"]
    train_mask, val_mask = split_targets(
        features.size(0),
        train_cfg["val_fraction"],
        train_cfg["random_seed"] + seed,
    )
    features, feat_mean, feat_std = standardize_set_features(features[train_mask], features)
    train_ds = TensorDataset(features[train_mask], targets[train_mask])
    val_x = features[val_mask].to(device)
    val_y = targets[val_mask].to(device)

    model_cfg = cfg["model"]
    model = SetAttentionResidualSelector(
        input_dim=features.size(-1),
        hidden_dim=model_cfg["hidden_dim"],
        num_layers=model_cfg["num_layers"],
        num_heads=model_cfg["num_heads"],
        ff_dim=model_cfg["ff_dim"],
        dropout=model_cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg["weight_decay"],
    )
    loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        generator=torch.Generator().manual_seed(train_cfg["random_seed"] + seed),
    )

    best_state = None
    best_val = float("inf")
    best_epoch = -1
    patience_left = train_cfg["patience"]
    last_train_loss = None
    last_train_rank_loss = None
    last_train_residual_loss = None

    for epoch in range(1, train_cfg["epochs"] + 1):
        model.train()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_residual_loss = 0.0
        total_count = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss, rank_loss, residual_loss = selector_loss(pred, batch_y, train_cfg)
            optimizer.zero_grad()
            loss.backward()
            grad_clip = train_cfg.get("grad_clip_norm")
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            optimizer.step()

            batch_count = len(batch_x)
            total_loss += loss.item() * batch_count
            total_rank_loss += rank_loss.item() * batch_count
            total_residual_loss += residual_loss.item() * batch_count
            total_count += batch_count

        last_train_loss = total_loss / max(total_count, 1)
        last_train_rank_loss = total_rank_loss / max(total_count, 1)
        last_train_residual_loss = total_residual_loss / max(total_count, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss, val_rank_loss, val_residual_loss = selector_loss(
                val_pred,
                val_y,
                train_cfg,
            )
            val_loss = val_loss.item()
            val_rank_loss = val_rank_loss.item()
            val_residual_loss = val_residual_loss.item()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_val_rank_loss = val_rank_loss
            best_val_residual_loss = val_residual_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            patience_left = train_cfg["patience"]
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary = {
        "best_epoch": best_epoch,
        "train_loss": last_train_loss,
        "train_rank_loss": last_train_rank_loss,
        "train_residual_loss": last_train_residual_loss,
        "val_loss": best_val,
        "val_rank_loss": best_val_rank_loss,
        "val_residual_loss": best_val_residual_loss,
        "num_train_targets": int(train_mask.sum().item()),
        "num_val_targets": int(val_mask.sum().item()),
        "support_size": int(features.size(1)),
        "feature_dim": int(features.size(2)),
    }
    return model, feat_mean, feat_std, summary


def predict_scores(model, tensors, feat_mean, feat_std, device):
    features = set_features(tensors)
    features = (features - feat_mean) / feat_std
    model.eval()
    with torch.no_grad():
        return model(features.to(device)).cpu()


def metric_dict(seed, strategy, pred, y_true):
    err = pred - y_true
    rmse = torch.sqrt(torch.mean(err ** 2)).item()
    mae = torch.mean(err.abs()).item()
    mape = torch.mean(err.abs() / y_true.abs().clamp_min(1e-6)).item()
    return {
        "seed": seed,
        "strategy": strategy,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
    }


def aggregate_support_prediction(y_ori, y_sup, selected_indices, alpha, label_mean, label_std):
    rows = torch.arange(y_sup.size(0))[:, None]
    chosen = y_sup[rows, selected_indices]
    y_sup_agg = chosen.mean(dim=1)
    transformed_pred = (1.0 - alpha) * y_ori + alpha * y_sup_agg
    return inverse_label_transform(transformed_pred, label_mean, label_std)


def evaluate_strategies(seed, tensors, pred_scores, cfg):
    alpha = float(cfg.get("alpha", 0.5))
    strategies_cfg = cfg["strategies"]
    label_mean, label_std = infer_label_inverse_params(tensors)
    y_true = tensors["target_label_original"].float()
    y_ori = tensors["y_ori"].float()
    y_sup = tensors["y_sup"].float()
    y_ori_original = tensors["y_ori_original"].float()
    cached_prediction = tensors["prediction_original"].float()

    support_size = y_sup.size(1)
    fused_pair = (1.0 - alpha) * y_ori[:, None] + alpha * y_sup
    fused_pair_original = inverse_label_transform(fused_pair, label_mean, label_std)
    true_residual = (fused_pair_original - y_true[:, None]).abs()
    true_order = torch.argsort(true_residual, dim=1)
    pred_order = torch.argsort(pred_scores, dim=1)

    rows = []
    rows.append(metric_dict(seed, "batlinet_median_cache", cached_prediction, y_true))
    rows.append(metric_dict(seed, "ori_only", y_ori_original, y_true))
    mean_pred = inverse_label_transform(
        (1.0 - alpha) * y_ori + alpha * y_sup.mean(dim=1),
        label_mean,
        label_std,
    )
    median_pred = inverse_label_transform(
        (1.0 - alpha) * y_ori + alpha * y_sup.median(dim=1).values,
        label_mean,
        label_std,
    )
    rows.append(metric_dict(seed, "all_32_mean", mean_pred, y_true))
    rows.append(metric_dict(seed, "all_32_median", median_pred, y_true))

    oracle_best_idx = true_order[:, 0]
    rows.append(
        metric_dict(
            seed,
            "oracle_best_single",
            fused_pair_original[torch.arange(len(y_true)), oracle_best_idx],
            y_true,
        )
    )
    for topk in [3, 5, 8, 16]:
        k = min(topk, support_size)
        pred = aggregate_support_prediction(
            y_ori,
            y_sup,
            true_order[:, :k],
            alpha,
            label_mean,
            label_std,
        )
        rows.append(metric_dict(seed, f"oracle_top{k}_mean", pred, y_true))

    for ratio in strategies_cfg["filter_keep_ratios"]:
        k = max(1, min(support_size, int(round(support_size * ratio))))
        pred = aggregate_support_prediction(
            y_ori,
            y_sup,
            pred_order[:, :k],
            alpha,
            label_mean,
            label_std,
        )
        rows.append(metric_dict(seed, f"model_keep_ratio_{ratio:g}_mean_top{k}", pred, y_true))

    for topk in strategies_cfg["topk"]:
        k = max(1, min(support_size, int(topk)))
        pred = aggregate_support_prediction(
            y_ori,
            y_sup,
            pred_order[:, :k],
            alpha,
            label_mean,
            label_std,
        )
        rows.append(metric_dict(seed, f"model_top{k}_mean", pred, y_true))

    for temp in strategies_cfg["softmax_temperatures"]:
        weights = torch.softmax(-pred_scores / float(temp), dim=1)
        y_sup_agg = (weights * y_sup).sum(dim=1)
        pred = inverse_label_transform(
            (1.0 - alpha) * y_ori + alpha * y_sup_agg,
            label_mean,
            label_std,
        )
        rows.append(metric_dict(seed, f"model_softmax_t{temp:g}", pred, y_true))

    return rows, true_residual, pred_order, true_order


def summarize_metrics(metric_rows):
    grouped = {}
    for row in metric_rows:
        grouped.setdefault(row["strategy"], []).append(row)
    summary = []
    for strategy, rows in sorted(grouped.items()):
        item = {"strategy": strategy, "num_seeds": len(rows)}
        for metric in ["rmse", "mae", "mape"]:
            values = torch.tensor([float(row[metric]) for row in rows])
            item[f"{metric}_mean"] = values.mean().item()
            item[f"{metric}_std"] = values.std(unbiased=False).item()
        summary.append(item)
    return summary


def pearson_corr(x, y):
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)
    x = x - x.mean()
    y = y - y.mean()
    denom = torch.sqrt((x ** 2).sum() * (y ** 2).sum()).clamp_min(1e-12)
    return ((x * y).sum() / denom).item()


def ranks_from_order(order):
    rank = torch.empty_like(order)
    rank_values = torch.arange(order.size(1))[None, :].expand_as(order)
    rank.scatter_(1, order, rank_values)
    return rank


def mean_sample_spearman(pred_order, true_order):
    pred_rank = ranks_from_order(pred_order).float()
    true_rank = ranks_from_order(true_order).float()
    values = []
    for idx in range(pred_rank.size(0)):
        values.append(pearson_corr(pred_rank[idx], true_rank[idx]))
    return float(torch.tensor(values).mean().item())


def topk_overlap(pred_order, true_order, topk):
    k = min(topk, pred_order.size(1))
    hits = []
    for idx in range(pred_order.size(0)):
        pred_set = set(pred_order[idx, :k].tolist())
        true_set = set(true_order[idx, :k].tolist())
        hits.append(len(pred_set & true_set) / k)
    return float(torch.tensor(hits).mean().item())


def rank_alignment_row(seed, pred_scores, true_residual, pred_order, true_order, cfg):
    pred_residual = inverse_residual_score(
        pred_scores,
        cfg["training"]["target_transform"],
    )
    support_size = pred_order.size(1)
    row = {
        "seed": seed,
        "pearson_pred_residual_true_residual": pearson_corr(pred_residual, true_residual),
        "mean_sample_spearman_rank": mean_sample_spearman(pred_order, true_order),
    }
    for topk in [1, 3, 5, 8, 16]:
        k = min(topk, support_size)
        row[f"top{k}_overlap_rate"] = topk_overlap(pred_order, true_order, k)

    k = min(16, support_size)
    rows = torch.arange(true_residual.size(0))[:, None]
    selected = true_residual[rows, pred_order[:, :k]]
    rejected = true_residual[rows, pred_order[:, k:]]
    row["pred_top16_true_residual_mean"] = selected.mean().item()
    row["pred_bottom16_true_residual_mean"] = rejected.mean().item()
    row["all32_true_residual_mean"] = true_residual.mean().item()
    return row


def summarize_rank_alignment(rows):
    if not rows:
        return []
    fieldnames = [key for key in rows[0].keys() if key != "seed"]
    summary = []
    for field in fieldnames:
        values = torch.tensor([float(row[field]) for row in rows])
        summary.append(
            {
                "metric": field,
                "num_seeds": len(rows),
                "mean": values.mean().item(),
                "std": values.std(unbiased=False).item(),
            }
        )
    return summary


def save_support_scores(path, seed, tensors, pred_scores, true_residual, pred_order, true_order, cfg):
    pred_rank = ranks_from_order(pred_order)
    true_rank = ranks_from_order(true_order)

    target_meta = cfg.get("_target_metadata", [])
    support_meta = cfg.get("_support_metadata", [])
    y_true = tensors["target_label_original"].float()
    y_ori = tensors["y_ori_original"].float()
    y_sup = tensors["y_sup_original"].float()
    support_label = tensors["support_label_original"].float()
    support_index = tensors["support_index"].long()
    pred_residual = inverse_residual_score(pred_scores, cfg["training"]["target_transform"])

    rows = []
    for target_idx in range(y_sup.size(0)):
        target_cell = ""
        if target_meta:
            target_cell = target_meta[target_idx].get("cell_id", "")
        for support_pos in range(y_sup.size(1)):
            support_cell = ""
            if support_meta:
                support_cell = support_meta[target_idx][support_pos].get("cell_id", "")
            rows.append(
                {
                    "seed": seed,
                    "target_row": target_idx,
                    "support_pos": support_pos,
                    "support_index": int(support_index[target_idx, support_pos].item()),
                    "target_cell_id": target_cell,
                    "support_cell_id": support_cell,
                    "y_true": float(y_true[target_idx].item()),
                    "y_ori": float(y_ori[target_idx].item()),
                    "y_sup": float(y_sup[target_idx, support_pos].item()),
                    "support_label": float(support_label[target_idx, support_pos].item()),
                    "true_fused_abs_error": float(true_residual[target_idx, support_pos].item()),
                    "predicted_residual": float(pred_residual[target_idx, support_pos].item()),
                    "predicted_score": float(pred_scores[target_idx, support_pos].item()),
                    "predicted_rank": int(pred_rank[target_idx, support_pos].item()) + 1,
                    "true_rank": int(true_rank[target_idx, support_pos].item()) + 1,
                }
            )

    write_csv(
        path,
        rows,
        [
            "seed",
            "target_row",
            "support_pos",
            "support_index",
            "target_cell_id",
            "support_cell_id",
            "y_true",
            "y_ori",
            "y_sup",
            "support_label",
            "true_fused_abs_error",
            "predicted_residual",
            "predicted_score",
            "predicted_rank",
            "true_rank",
        ],
    )


def run_seed(seed, cfg, output_dir, device):
    cache_root = Path(cfg["cache_root"])
    train_path = cache_root / cfg["train_subdir"] / f"train_seed_{seed}.pt"
    test_path = cache_root / cfg["test_subdir"] / f"test_seed_{seed}.pt"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train cache: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test cache: {test_path}")

    train_payload, train_tensors = load_cache(train_path)
    test_payload, test_tensors = load_cache(test_path)
    alpha = float(train_payload["alpha"])
    cfg = copy.deepcopy(cfg)
    cfg["alpha"] = alpha

    train_x = set_features(train_tensors)
    train_y = residual_targets(
        train_tensors,
        alpha=alpha,
        target_transform=cfg["training"]["target_transform"],
    )

    model, feat_mean, feat_std, train_summary = train_model(
        train_x,
        train_y,
        cfg=cfg,
        device=device,
        seed=seed,
    )

    pred_scores = predict_scores(model, test_tensors, feat_mean, feat_std, device)
    metric_rows, true_residual, pred_order, true_order = evaluate_strategies(
        seed,
        test_tensors,
        pred_scores,
        cfg,
    )
    rank_row = rank_alignment_row(
        seed,
        pred_scores,
        true_residual,
        pred_order,
        true_order,
        cfg,
    )

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "seed": seed,
            "model_state": model.state_dict(),
            "feature_mean": feat_mean,
            "feature_std": feat_std,
            "config": cfg,
            "train_summary": train_summary,
        },
        model_dir / f"selector_seed_{seed}.pt",
    )

    samples_dir = output_dir / "samples"
    sample_cfg = copy.deepcopy(cfg)
    sample_cfg["_target_metadata"] = test_payload.get("target_metadata", [])
    sample_cfg["_support_metadata"] = test_payload.get("support_metadata", [])
    save_support_scores(
        samples_dir / f"test_seed_{seed}_support_scores.csv",
        seed,
        test_tensors,
        pred_scores,
        true_residual,
        pred_order,
        true_order,
        sample_cfg,
    )

    train_summary = {
        "seed": seed,
        **train_summary,
        "train_cache": str(train_path),
        "test_cache": str(test_path),
    }
    return metric_rows, train_summary, rank_row


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.seeds is not None:
        cfg["seeds"] = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    output_dir = Path(cfg["output_dir"])
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device_name = cfg.get("device", "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {device_name}, but CUDA is not available.")
    device = torch.device(device_name)

    all_metric_rows = []
    train_rows = []
    rank_rows = []
    for seed in cfg["seeds"]:
        print(f"Running seed {seed}", flush=True)
        metric_rows, train_summary, rank_row = run_seed(seed, cfg, output_dir, device)
        all_metric_rows.extend(metric_rows)
        train_rows.append(train_summary)
        rank_rows.append(rank_row)

    metric_fields = ["seed", "strategy", "rmse", "mae", "mape"]
    write_csv(output_dir / "tables" / "seed_strategy_metrics.csv", all_metric_rows, metric_fields)

    summary_rows = summarize_metrics(all_metric_rows)
    summary_fields = [
        "strategy",
        "num_seeds",
        "rmse_mean",
        "rmse_std",
        "mae_mean",
        "mae_std",
        "mape_mean",
        "mape_std",
    ]
    write_csv(output_dir / "tables" / "metric_summary.csv", summary_rows, summary_fields)

    train_fields = [
        "seed",
        "best_epoch",
        "train_loss",
        "train_rank_loss",
        "train_residual_loss",
        "val_loss",
        "val_rank_loss",
        "val_residual_loss",
        "num_train_targets",
        "num_val_targets",
        "support_size",
        "feature_dim",
        "train_cache",
        "test_cache",
    ]
    write_csv(output_dir / "tables" / "training_summary.csv", train_rows, train_fields)

    rank_fields = list(rank_rows[0].keys()) if rank_rows else []
    write_csv(output_dir / "tables" / "rank_alignment_by_seed.csv", rank_rows, rank_fields)

    rank_summary_rows = summarize_rank_alignment(rank_rows)
    write_csv(
        output_dir / "tables" / "rank_alignment_summary.csv",
        rank_summary_rows,
        ["metric", "num_seeds", "mean", "std"],
    )

    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
