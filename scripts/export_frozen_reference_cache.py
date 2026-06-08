import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline import CONFIGS, build_dataset, set_seed  # noqa: E402
from src.builders import MODELS  # noqa: E402
from src.data import BatteryData  # noqa: E402
from src.utils import import_config  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export frozen BatLiNet reference-selection cache."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--support-size", type=int, default=32)
    parser.add_argument("--fixed-support-index-path", default=None)
    parser.add_argument("--processed-data-root", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def tensor_to_original_scale(data_bundle, value):
    tensor = value.detach().float()
    if data_bundle.label_transformation is not None:
        tensor = data_bundle.label_transformation.inverse_transform(tensor)
    return tensor.detach().cpu()


def resolve_processed_path(source_path, processed_root):
    path = Path(source_path)
    if path.exists():
        return path
    if processed_root is None:
        return None
    parts = path.parts
    if "processed" in parts:
        rel_parts = parts[parts.index("processed") + 1:]
    else:
        rel_parts = parts[-2:]
    candidate = processed_root.joinpath(*rel_parts)
    return candidate if candidate.exists() else None


def load_cell_material(meta, processed_root, cache):
    source_path = meta.get("source_path")
    if source_path is None:
        return dict(meta)
    key = str(Path(source_path).as_posix()).lower()
    if key in cache:
        return {**meta, **cache[key]}

    material = {
        "cathode_material": None,
        "anode_material": None,
        "nominal_capacity_in_Ah": None,
        "metadata_status": "not_loaded",
    }
    resolved = resolve_processed_path(source_path, processed_root)
    if resolved is not None:
        try:
            cell = BatteryData.load(str(resolved))
            material = {
                "cathode_material": getattr(cell, "cathode_material", None),
                "anode_material": getattr(cell, "anode_material", None),
                "nominal_capacity_in_Ah": getattr(
                    cell, "nominal_capacity_in_Ah", None),
                "metadata_status": "loaded_processed",
            }
        except Exception as exc:  # noqa: BLE001
            material["metadata_status"] = f"load_failed:{type(exc).__name__}"
    cache[key] = material
    return {**meta, **material}


def load_fixed_indices(path):
    if path is None:
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        payload = payload.get("indices")
    if payload is None:
        raise ValueError(f"No support indices found in {path}.")
    if payload.dim() != 2:
        raise ValueError("Fixed support indices must be a 2D tensor.")
    return payload.long().contiguous()


def build_cache(args):
    set_seed(args.seed)
    configs = import_config(Path(args.config), CONFIGS)
    configs["model"]["seed"] = args.seed
    configs["model"]["test_support_size"] = args.support_size

    dataset = build_dataset(configs, args.device)
    model = MODELS.build(configs["model"])
    model.load_checkpoint(args.checkpoint, device=args.device)
    model = model.to(args.device)
    model.eval()

    target_data = dataset.test_data if args.split == "test" else dataset.train_data
    target_dataset = model.build_cycle_diff_dataset(target_data)
    batch_size = args.batch_size or model.test_batch_size
    loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=False)
    fixed_indices = load_fixed_indices(args.fixed_support_index_path)
    if fixed_indices is not None and fixed_indices.size(0) != len(target_data):
        raise ValueError(
            "Fixed support protocol does not match the selected split length.")

    processed_root = (
        Path(args.processed_data_root)
        if args.processed_data_root is not None
        else None
    )
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
            batch_fixed_indices = None
            if fixed_indices is not None:
                batch_fixed_indices = fixed_indices[offset:offset + len(x)]
            sup_x, sup_y, sup_idx = model.get_support_set(
                raw_x,
                dataset.train_data.feature,
                dataset.train_data.label,
                fixed_indices=batch_fixed_indices,
                return_indices=True,
            )
            y_ori, y_sup, y_sup_agg, _, _, x_ori, x_sup = \
                model.compute_prediction_components(
                    x, sup_x, sup_y, return_features=True)
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

            chunks["target_label_original"].append(
                tensor_to_original_scale(dataset, y))
            chunks["support_label_original"].append(
                tensor_to_original_scale(dataset, sup_y))
            chunks["y_ori_original"].append(
                tensor_to_original_scale(dataset, y_ori))
            chunks["y_sup_original"].append(
                tensor_to_original_scale(dataset, y_sup))
            chunks["y_sup_agg_original"].append(
                tensor_to_original_scale(dataset, y_sup_agg))
            chunks["prediction_original"].append(
                tensor_to_original_scale(dataset, prediction))

            if target_meta is not None:
                for meta in target_meta[offset:offset + len(x)]:
                    target_metadata.append(
                        load_cell_material(meta, processed_root, material_cache))
            if train_meta is not None:
                for row in sup_idx.detach().cpu().tolist():
                    support_metadata.append([
                        load_cell_material(train_meta[i], processed_root, material_cache)
                        for i in row
                    ])
            offset += len(x)

    payload = {
        "version": 1,
        "seed": args.seed,
        "split": args.split,
        "config": args.config,
        "checkpoint": args.checkpoint,
        "alpha": model.alpha,
        "support_size": args.support_size,
        "fixed_support_index_path": args.fixed_support_index_path,
        "label_space": "transformed_and_original",
        "tensors": {
            key: torch.cat(value, dim=0)
            for key, value in chunks.items()
        },
        "target_metadata": target_metadata,
        "support_metadata": support_metadata,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved frozen reference cache: {output}", flush=True)


if __name__ == "__main__":
    build_cache(parse_args())
