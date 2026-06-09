import argparse
import csv
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.export_frozen_reference_cache import tensor_to_original_scale  # noqa: E402
from scripts.pipeline import CONFIGS, build_dataset, set_seed  # noqa: E402
from src.builders import MODELS  # noqa: E402
from src.utils import import_config  # noqa: E402


BASE_KEYS = [
    "target_label",
    "target_label_original",
    "y_ori",
    "y_ori_original",
    "x_ori",
]

REPEAT_KEYS = [
    "support_label",
    "support_label_original",
    "y_sup",
    "y_sup_original",
    "y_sup_agg",
    "y_sup_agg_original",
    "prediction",
    "prediction_original",
    "x_sup",
    "support_index",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export lightweight sharded repeated random-32 train caches."
    )
    parser.add_argument(
        "--config",
        default="configs/frozen_reference_cache/train_random32_multi_v2_sharded.yaml",
        help="Path to the sharded multi-cache export config.",
    )
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed override.")
    parser.add_argument("--repeat-start", type=int, default=None)
    parser.add_argument("--num-repeats", type=int, default=None)
    parser.add_argument("--repeats-per-shard", type=int, default=None)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing shard files instead of skipping them.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned shard outputs without exporting cache tensors.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_seed_override(value):
    if value is None:
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def normalize_numpy_seed(seed):
    return int(seed) % (2 ** 32 - 1)


def set_random_seed_quiet(seed):
    seed = normalize_numpy_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
    return SimpleNamespace(
        dataset=dataset,
        model=model,
        target_data=target_data,
        loader=loader,
        checkpoint=checkpoint,
    )


def empty_chunks(keys):
    return {key: [] for key in keys}


def cat_chunks(chunks):
    return {key: torch.cat(value, dim=0) for key, value in chunks.items()}


def collect_repeat(ctx, cache_random_seed, collect_base):
    set_random_seed_quiet(cache_random_seed)
    dataset = ctx.dataset
    model = ctx.model
    base_chunks = empty_chunks(BASE_KEYS) if collect_base else None
    repeat_chunks = empty_chunks(REPEAT_KEYS)

    with torch.no_grad():
        for data_batch in ctx.loader:
            x, y, raw_x = data_batch.values()
            sup_x, sup_y, sup_idx = model.get_support_set(
                raw_x,
                dataset.train_data.feature,
                dataset.train_data.label,
                return_indices=True,
            )
            y_ori, y_sup, y_sup_agg, _, _, x_ori, x_sup = (
                model.compute_prediction_components(
                    x,
                    sup_x,
                    sup_y,
                    return_features=True,
                )
            )
            prediction = (1.0 - model.alpha) * y_ori + model.alpha * y_sup_agg

            if collect_base:
                base_chunks["target_label"].append(y.detach().cpu())
                base_chunks["target_label_original"].append(
                    tensor_to_original_scale(dataset, y)
                )
                base_chunks["y_ori"].append(y_ori.detach().cpu())
                base_chunks["y_ori_original"].append(
                    tensor_to_original_scale(dataset, y_ori)
                )
                base_chunks["x_ori"].append(
                    x_ori.detach().cpu().view(len(x), model.channels)
                )

            repeat_chunks["support_label"].append(sup_y.detach().cpu())
            repeat_chunks["support_label_original"].append(
                tensor_to_original_scale(dataset, sup_y)
            )
            repeat_chunks["y_sup"].append(y_sup.detach().cpu())
            repeat_chunks["y_sup_original"].append(
                tensor_to_original_scale(dataset, y_sup)
            )
            repeat_chunks["y_sup_agg"].append(y_sup_agg.detach().cpu())
            repeat_chunks["y_sup_agg_original"].append(
                tensor_to_original_scale(dataset, y_sup_agg)
            )
            repeat_chunks["prediction"].append(prediction.detach().cpu())
            repeat_chunks["prediction_original"].append(
                tensor_to_original_scale(dataset, prediction)
            )
            repeat_chunks["x_sup"].append(x_sup.detach().cpu())
            repeat_chunks["support_index"].append(sup_idx.detach().cpu())

    base_tensors = cat_chunks(base_chunks) if collect_base else None
    repeat_tensors = cat_chunks(repeat_chunks)
    return base_tensors, repeat_tensors


def export_shard(ctx, cfg, seed, shard_index, repeat_indices, output, random_seed_base):
    start = time.perf_counter()
    base_tensors = None
    repeat_rows = []
    stacked_repeat = {key: [] for key in REPEAT_KEYS}

    for pos, repeat_idx in enumerate(repeat_indices):
        cache_random_seed = normalize_numpy_seed(random_seed_base + seed * 100000 + repeat_idx)
        collect_base = pos == 0
        current_base, repeat_tensors = collect_repeat(
            ctx,
            cache_random_seed=cache_random_seed,
            collect_base=collect_base,
        )
        if collect_base:
            base_tensors = current_base
        for key in REPEAT_KEYS:
            stacked_repeat[key].append(repeat_tensors[key])
        repeat_rows.append(
            {
                "repeat": repeat_idx,
                "cache_random_seed": cache_random_seed,
            }
        )

    payload = {
        "version": 2,
        "format": "train_random32_multi_shard",
        "seed": seed,
        "split": "train",
        "shard_index": shard_index,
        "repeat_start": repeat_indices[0],
        "repeat_end": repeat_indices[-1],
        "num_repeats": len(repeat_indices),
        "repeats": repeat_rows,
        "config": cfg["model_config"],
        "checkpoint": str(ctx.checkpoint),
        "alpha": ctx.model.alpha,
        "support_size": int(cfg["support_size"]),
        "label_space": "transformed_and_original",
        "base_tensors": base_tensors,
        "repeat_tensors": {
            key: torch.stack(value, dim=0)
            for key, value in stacked_repeat.items()
        },
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, tmp_output)
    tmp_output.replace(output)
    elapsed = time.perf_counter() - start
    return elapsed, output.stat().st_size


def shard_repeat_ranges(repeat_start, num_repeats, repeats_per_shard):
    repeats = list(range(repeat_start, repeat_start + num_repeats))
    for shard_index, start in enumerate(range(0, len(repeats), repeats_per_shard)):
        yield shard_index, repeats[start:start + repeats_per_shard]


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "shard_index",
        "repeat_start",
        "repeat_end",
        "num_repeats",
        "checkpoint",
        "output",
        "status",
        "elapsed_seconds",
        "file_size_bytes",
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
    if args.repeats_per_shard is not None:
        cfg["repeats_per_shard"] = args.repeats_per_shard
    if args.overwrite:
        cfg["skip_existing"] = False

    cache_root = Path(cfg["cache_root"])
    output_root = cache_root / cfg["output_subdir"]
    repeat_start = int(cfg.get("repeat_start", 0))
    num_repeats = int(cfg["num_repeats"])
    repeats_per_shard = int(cfg["repeats_per_shard"])
    random_seed_base = int(cfg["cache_random_seed_base"])
    skip_existing = bool(cfg.get("skip_existing", True))

    planned = []
    for seed in cfg["seeds"]:
        checkpoint = find_checkpoint(cfg["checkpoint_dir"], seed)
        for shard_index, repeat_indices in shard_repeat_ranges(
            repeat_start,
            num_repeats,
            repeats_per_shard,
        ):
            output = output_root / f"seed_{seed}" / f"shard_{shard_index:03d}.pt"
            status = "planned"
            if output.exists() and skip_existing:
                status = "exists"
            planned.append(
                {
                    "seed": seed,
                    "shard_index": shard_index,
                    "repeat_start": repeat_indices[0],
                    "repeat_end": repeat_indices[-1],
                    "num_repeats": len(repeat_indices),
                    "checkpoint": str(checkpoint),
                    "output": str(output),
                    "status": status,
                    "elapsed_seconds": "",
                    "file_size_bytes": "",
                }
            )

    if args.dry_run:
        for row in planned:
            print(
                f"[{row['status']}] seed={row['seed']} shard={row['shard_index']:03d} "
                f"repeat={row['repeat_start']:03d}-{row['repeat_end']:03d} "
                f"-> {row['output']}",
                flush=True,
            )
        write_manifest(output_root / "export_manifest.csv", planned)
        return

    by_seed = {}
    for row in planned:
        by_seed.setdefault(row["seed"], []).append(row)

    manifest_rows = []
    total_shards = len(planned)
    completed_shards = 0
    run_start = time.perf_counter()
    for seed, rows in by_seed.items():
        checkpoint = Path(rows[0]["checkpoint"])
        print(f"Preparing seed {seed} from {checkpoint}", flush=True)
        ctx = prepare_seed_context(cfg, seed, checkpoint, cfg["device"])
        for row in rows:
            if row["status"] == "exists":
                completed_shards += 1
                print(
                    f"Skip existing seed={seed} shard={row['shard_index']:03d} "
                    f"({completed_shards}/{total_shards})",
                    flush=True,
                )
                manifest_rows.append(row)
                continue
            output = Path(row["output"])
            repeat_indices = list(
                range(int(row["repeat_start"]), int(row["repeat_end"]) + 1)
            )
            print(
                f"Export seed={seed} shard={row['shard_index']:03d} "
                f"repeat={row['repeat_start']:03d}-{row['repeat_end']:03d}",
                flush=True,
            )
            elapsed, size_bytes = export_shard(
                ctx,
                cfg,
                seed=seed,
                shard_index=int(row["shard_index"]),
                repeat_indices=repeat_indices,
                output=output,
                random_seed_base=random_seed_base,
            )
            completed_shards += 1
            total_elapsed = time.perf_counter() - run_start
            avg_per_shard = total_elapsed / max(completed_shards, 1)
            remaining = max(total_shards - completed_shards, 0) * avg_per_shard
            row = dict(row)
            row["status"] = "exported"
            row["elapsed_seconds"] = f"{elapsed:.3f}"
            row["file_size_bytes"] = str(size_bytes)
            manifest_rows.append(row)
            print(
                f"Saved {output} ({size_bytes / 1024 / 1024:.2f} MB), "
                f"shard {completed_shards}/{total_shards}, "
                f"elapsed {elapsed:.1f}s, ETA {remaining / 60:.1f} min",
                flush=True,
            )

    write_manifest(output_root / "export_manifest.csv", manifest_rows)
    print(f"Saved manifest to {output_root / 'export_manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
