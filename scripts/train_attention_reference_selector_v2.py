import argparse
import copy
import csv
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_attention_reference_selector import (  # noqa: E402
    SetAttentionResidualSelector,
    evaluate_strategies,
    infer_label_inverse_params,
    load_cache,
    predict_scores,
    rank_alignment_row,
    save_support_scores,
    selector_loss,
    split_targets,
    standardize_set_features,
    summarize_metrics,
    summarize_rank_alignment,
    write_csv,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train attention reference selector on sharded multi-repeat cache."
    )
    parser.add_argument(
        "--config",
        default="configs/reference_selector/attention_reference_selector_v2.yaml",
        help="Path to the reference selector config.",
    )
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def inverse_label_transform(transformed, mean, std):
    return torch.exp(transformed * std + mean)


def shard_paths(cache_root, train_subdir, seed):
    seed_dir = Path(cache_root) / train_subdir / f"seed_{seed}"
    paths = sorted(seed_dir.glob("shard_*.pt"))
    if not paths:
        raise FileNotFoundError(f"No sharded train cache found in {seed_dir}")
    return paths


def load_sharded_train_cache(cache_root, train_subdir, seed, target_transform):
    paths = shard_paths(cache_root, train_subdir, seed)
    base_tensors = None
    feature_chunks = []
    target_chunks = []
    repeat_counts = []

    for path in paths:
        payload = torch.load(path, map_location="cpu")
        if payload.get("format") != "train_random32_multi_shard":
            raise ValueError(f"Unsupported shard format in {path}")
        current_base = payload["base_tensors"]
        if base_tensors is None:
            base_tensors = current_base
        repeat_tensors = payload["repeat_tensors"]
        features = set_features_from_shard(current_base, repeat_tensors)
        targets = residual_targets_from_shard(
            current_base,
            repeat_tensors,
            alpha=float(payload["alpha"]),
            target_transform=target_transform,
        )
        feature_chunks.append(features)
        target_chunks.append(targets)
        repeat_counts.append(int(payload["num_repeats"]))

    features = torch.cat(feature_chunks, dim=0)
    targets = torch.cat(target_chunks, dim=0)
    summary = {
        "num_shards": len(paths),
        "num_repeats": sum(repeat_counts),
        "num_targets": int(base_tensors["target_label_original"].size(0)),
        "support_size": int(features.size(1)),
        "feature_dim": int(features.size(2)),
    }
    return features, targets, summary


def set_features_from_shard(base_tensors, repeat_tensors):
    x_ori = base_tensors["x_ori"].float()
    x_sup = repeat_tensors["x_sup"].float()
    y_ori = base_tensors["y_ori_original"].float()
    y_sup = repeat_tensors["y_sup_original"].float()
    support_label = repeat_tensors["support_label_original"].float()

    num_repeats, num_targets, support_size, channels = x_sup.shape
    x_ori_expanded = (
        x_ori[None, :, None, :]
        .expand(num_repeats, num_targets, support_size, channels)
    )
    y_ori_expanded = y_ori[None, :, None].expand(num_repeats, num_targets, support_size)
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
    features = torch.cat([x_ori_expanded, x_sup, scalar_features], dim=-1)
    return features.reshape(num_repeats * num_targets, support_size, -1)


def residual_targets_from_shard(base_tensors, repeat_tensors, alpha, target_transform):
    label_mean, label_std = infer_label_inverse_params(base_tensors)
    y_true = base_tensors["target_label_original"].float()
    y_ori = base_tensors["y_ori"].float()
    y_sup = repeat_tensors["y_sup"].float()
    fused_pair = (1.0 - alpha) * y_ori[None, :, None] + alpha * y_sup
    fused_pair_original = inverse_label_transform(fused_pair, label_mean, label_std)
    residual = (fused_pair_original - y_true[None, :, None]).abs()
    if target_transform == "log1p":
        residual = torch.log1p(residual)
    elif target_transform not in (None, "none"):
        raise ValueError(f"Unsupported target transform: {target_transform}")
    num_repeats, num_targets, support_size = residual.shape
    return residual.reshape(num_repeats * num_targets, support_size)


def target_ids_for_repeats(num_repeats, num_targets):
    return torch.arange(num_targets).repeat(num_repeats)


def train_model(features, targets, cache_summary, cfg, device, seed):
    train_cfg = cfg["training"]
    num_targets = cache_summary["num_targets"]
    num_repeats = cache_summary["num_repeats"]
    train_target_mask, val_target_mask = split_targets(
        num_targets,
        train_cfg["val_fraction"],
        train_cfg["random_seed"] + seed,
    )
    ids = target_ids_for_repeats(num_repeats, num_targets)
    train_mask = train_target_mask[ids]
    val_mask = val_target_mask[ids]

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
    best_val_rank_loss = None
    best_val_residual_loss = None
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
        "num_train_sets": int(train_mask.sum().item()),
        "num_val_sets": int(val_mask.sum().item()),
        **cache_summary,
    }
    return model, feat_mean, feat_std, summary


def run_seed(seed, cfg, output_dir, device):
    cache_root = Path(cfg["cache_root"])
    test_path = cache_root / cfg["test_subdir"] / f"test_seed_{seed}.pt"
    if not test_path.exists():
        raise FileNotFoundError(f"Missing test cache: {test_path}")

    train_x, train_y, cache_summary = load_sharded_train_cache(
        cache_root,
        cfg["train_subdir"],
        seed,
        target_transform=cfg["training"]["target_transform"],
    )
    model, feat_mean, feat_std, train_summary = train_model(
        train_x,
        train_y,
        cache_summary=cache_summary,
        cfg=cfg,
        device=device,
        seed=seed,
    )

    test_payload, test_tensors = load_cache(test_path)
    cfg = copy.deepcopy(cfg)
    cfg["alpha"] = float(test_payload["alpha"])
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
        "train_cache": str(cache_root / cfg["train_subdir"] / f"seed_{seed}"),
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

    write_csv(
        output_dir / "tables" / "seed_strategy_metrics.csv",
        all_metric_rows,
        ["seed", "strategy", "rmse", "mae", "mape"],
    )
    write_csv(
        output_dir / "tables" / "metric_summary.csv",
        summarize_metrics(all_metric_rows),
        [
            "strategy",
            "num_seeds",
            "rmse_mean",
            "rmse_std",
            "mae_mean",
            "mae_std",
            "mape_mean",
            "mape_std",
        ],
    )

    train_fields = [
        "seed",
        "best_epoch",
        "train_loss",
        "train_rank_loss",
        "train_residual_loss",
        "val_loss",
        "val_rank_loss",
        "val_residual_loss",
        "num_train_sets",
        "num_val_sets",
        "num_shards",
        "num_repeats",
        "num_targets",
        "support_size",
        "feature_dim",
        "train_cache",
        "test_cache",
    ]
    write_csv(output_dir / "tables" / "training_summary.csv", train_rows, train_fields)

    rank_fields = list(rank_rows[0].keys()) if rank_rows else []
    write_csv(output_dir / "tables" / "rank_alignment_by_seed.csv", rank_rows, rank_fields)
    write_csv(
        output_dir / "tables" / "rank_alignment_summary.csv",
        summarize_rank_alignment(rank_rows),
        ["metric", "num_seeds", "mean", "std"],
    )
    print(f"Saved results to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
