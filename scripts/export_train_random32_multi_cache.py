import argparse
import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_frozen_reference_cache import (  # noqa: E402
    load_cell_material,
    tensor_to_original_scale,
)
from scripts.pipeline import CONFIGS, build_dataset, set_seed  # noqa: E402
from src.builders import MODELS  # noqa: E402
from src.utils import import_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export repeated random-32 train caches from frozen BatLiNet checkpoints."
    )
    parser.add_argument(
        "--config",
        default="configs/frozen_reference_cache/train_random32_multi_v1.yaml",
        help="Path to the multi-cache export config.",
    )
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed override.")
    parser.add_argument("--repeat-start", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing cache files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned outputs without exporting cache tensors.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_seed_override(value):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def find_checkpoint(checkpoint_dir, seed):
    checkpoint_dir = Path(checkpoint_dir)
    matches = sorted(checkpoint_dir.glob(f"*_seed_{seed}_epoch_*.ckpt"))
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint found for seed {seed} in {checkpoint_dir}."
        )
    if len(matches) > 1:
        epoch_matches = [path for path in matches if path.name.endswith("_epoch_1000.ckpt")]
        if len(epoch_matches) == 1:
            return epoch_matches[0]
        raise ValueError(
            f"Multiple checkpoints found for seed {seed}: "
            + ", ".join(path.name for path in matches)
        )
    return matches[0]


def prepare_seed_context(cfg, seed, checkpoint, device):
    set_seed(seed)
    configs = import_config(Path(cfg["model_config"]), CONFIGS)
    configs["model"]["seed"] = seed
    configs["model"]["test_support_size"] = int(cfg["support_size"])

    dataset = build_dataset(configs, device)
    model = MODELS.build(configs["model"])
    model.load_checkpoint(str(checkpoint), device=device)
    model = model.to(device)
    model.eval()

    target_data = dataset.train_data
    target_dataset = model.build_cycle_diff_dataset(target_data)
    batch_size = cfg.get("batch_size") or model.test_batch_size
    loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)

    processed_root = cfg.get("processed_data_root")
    processed_root = Path(processed_root) if processed_root is not None else None
    if processed_root is not None and not processed_root.exists():
        processed_root = None

    return SimpleNamespace(
        configs=configs,
        dataset=dataset,
        model=model,
        target_data=target_data,
        loader=loader,
        processed_root=processed_root,
        checkpoint=checkpoint,
    )


def export_repeat_cache(ctx, cfg, seed, repeat_idx, output, cache_random_seed):
    set_seed(cache_random_seed)
    dataset = ctx.dataset
    model = ctx.model
    loader = ctx.loader
    target_data = ctx.target_data
    device = cfg["device"]
    material_cache = {}
    train_meta = getattr(dataset.train_data, "metadata", None)
    target_meta = getattr(target_data, "metadata", None)

    chunks = {
        "target_label": [],
        "support_label": [],
        "y_ori": [],
        "y_sup": [],
        "y_sup_agg": [],
        "prediction": [],
        "x_ori": [],
        "x_sup": [],
        "support_index": [],
        "target_label_original": [],
        "support_label_original": [],
        "y_ori_original": [],
        "y_sup_original": [],
        "y_sup_agg_original": [],
        "prediction_original": [],
    }
    target_metadata = []
    support_metadata = []

    offset = 0
    with torch.no_grad():
        for data_batch in loader:
            x, y, raw_x = data_batch.values()
            sup_x, sup_y, sup_idx = model.get_support_set(
                raw_x,
                dataset.train_data.feature,
                dataset.train_data.label,
                return_indices=True,
            )
            y_ori, y_sup, y_sup_agg, _, _, x_ori, x_sup = (
                model.compute_prediction_components(
                    x, sup_x, sup_y, return_features=True
                )
            )
            prediction = (1.0 - model.alpha) * y_ori + model.alpha * y_sup_agg

            chunks["target_label"].append(y.detach().cpu())
            chunks["support_label"].append(sup_y.detach().cpu())
            chunks["y_ori"].append(y_ori.detach().cpu())
            chunks["y_sup"].append(y_sup.detach().cpu())
            chunks["y_sup_agg"].append(y_sup_agg.detach().cpu())
            chunks["prediction"].append(prediction.detach().cpu())
            chunks["x_ori"].append(x_ori.detach().cpu().view(len(x), model.channels))
            chunks["x_sup"].append(x_sup.detach().cpu())
            chunks["support_index"].append(sup_idx.detach().cpu())

            chunks["target_label_original"].append(tensor_to_original_scale(dataset, y))
            chunks["support_label_original"].append(
                tensor_to_original_scale(dataset, sup_y)
            )
            chunks["y_ori_original"].append(tensor_to_original_scale(dataset, y_ori))
            chunks["y_sup_original"].append(tensor_to_original_scale(dataset, y_sup))
            chunks["y_sup_agg_original"].append(
                tensor_to_original_scale(dataset, y_sup_agg)
            )
            chunks["prediction_original"].append(
                tensor_to_original_scale(dataset, prediction)
            )

            if target_meta is not None:
                for meta in target_meta[offset:offset + len(x)]:
                    target_metadata.append(
                        load_cell_material(meta, ctx.processed_root, material_cache)
                    )
            if train_meta is not None:
                for row in sup_idx.detach().cpu().tolist():
                    support_metadata.append(
                        [
                            load_cell_material(
                                train_meta[i], ctx.processed_root, material_cache
                            )
                            for i in row
                        ]
                    )
            offset += len(x)

    payload = {
        "version": 1,
        "seed": seed,
        "repeat": repeat_idx,
        "split": "train",
        "config": cfg["model_config"],
        "checkpoint": str(ctx.checkpoint),
        "alpha": model.alpha,
        "support_size": int(cfg["support_size"]),
        "cache_random_seed": cache_random_seed,
        "fixed_support_index_path": None,
        "label_space": "transformed_and_original",
        "tensors": {
            key: torch.cat(value, dim=0)
            for key, value in chunks.items()
        },
        "target_metadata": target_metadata,
        "support_metadata": support_metadata,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "repeat",
        "cache_random_seed",
        "checkpoint",
        "output",
        "status",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.device is not None:
        cfg["device"] = args.device
    if args.seeds is not None:
        cfg["seeds"] = parse_seed_override(args.seeds)
    if args.repeat_start is not None:
        cfg["repeat_start"] = args.repeat_start
    if args.num_repeats is not None:
        cfg["num_repeats"] = args.num_repeats
    if args.overwrite:
        cfg["skip_existing"] = False

    cache_root = Path(cfg["cache_root"])
    output_root = cache_root / cfg["output_subdir"]
    repeat_start = int(cfg.get("repeat_start", 0))
    num_repeats = int(cfg["num_repeats"])
    random_seed_base = int(cfg["cache_random_seed_base"])
    skip_existing = bool(cfg.get("skip_existing", True))

    planned = []
    for seed in cfg["seeds"]:
        checkpoint = find_checkpoint(cfg["checkpoint_dir"], seed)
        for repeat_idx in range(repeat_start, repeat_start + num_repeats):
            output = output_root / f"repeat_{repeat_idx:03d}" / f"train_seed_{seed}.pt"
            cache_random_seed = random_seed_base + seed * 100000 + repeat_idx
            status = "planned"
            if output.exists() and skip_existing:
                status = "exists"
            planned.append(
                {
                    "seed": seed,
                    "repeat": repeat_idx,
                    "cache_random_seed": cache_random_seed,
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                    "status": status,
                }
            )

    if args.dry_run:
        for row in planned:
            print(
                f"[{row['status']}] seed={row['seed']} repeat={row['repeat']:03d} "
                f"-> {row['output']}",
                flush=True,
            )
        write_manifest(output_root / "export_manifest.csv", planned)
        return

    by_seed = {}
    for row in planned:
        by_seed.setdefault(row["seed"], []).append(row)

    manifest_rows = []
    for seed, rows in by_seed.items():
        checkpoint = Path(rows[0]["checkpoint"])
        print(f"Preparing seed {seed} from {checkpoint}", flush=True)
        ctx = prepare_seed_context(cfg, seed, checkpoint, cfg["device"])
        for row in rows:
            output = Path(row["output"])
            if row["status"] == "exists":
                print(
                    f"Skip existing seed={seed} repeat={row['repeat']:03d}: {output}",
                    flush=True,
                )
                manifest_rows.append(row)
                continue
            print(
                f"Export seed={seed} repeat={row['repeat']:03d} "
                f"cache_random_seed={row['cache_random_seed']}",
                flush=True,
            )
            export_repeat_cache(
                ctx,
                cfg,
                seed=seed,
                repeat_idx=int(row["repeat"]),
                output=output,
                cache_random_seed=int(row["cache_random_seed"]),
            )
            row = dict(row)
            row["status"] = "exported"
            manifest_rows.append(row)

    write_manifest(output_root / "export_manifest.csv", manifest_rows)
    print(f"Saved manifest to {output_root / 'export_manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
