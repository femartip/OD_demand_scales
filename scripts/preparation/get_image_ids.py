import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import fiftyone as fo
import fiftyone.zoo as foz


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import (PARTITIONS,_load_full_dataset,dataset_seed,load_split_config,manifest_path,resolve_repo_path,)


def source_group_id(dataset, image_id, source_split):
    if dataset == "driving" and "_jpg.rf." in image_id:
        return image_id.split("_jpg.rf.", maxsplit=1)[0]
    return f"{source_split}:{image_id}"



def rem_inventory(dataset_name, source_split, split_config):
    config = split_config["datasets"][dataset_name]
    with open(resolve_repo_path(config[f"{source_split}_annotations"])) as file:
        data = json.load(file)
    annotations = {}
    for annotation in data["annotations"]:
        annotations.setdefault(annotation["image_id"], []).append(annotation)
    image_dir = Path(fo.config.dataset_zoo_dir) / config["name"] / source_split / "data"
    rows = []
    overlap_threshold = config.get("calibration_sampling", {}).get("overlap_threshold", 0.5)
    for image in sorted(data["images"], key=lambda image: image["id"]):
        image_id = f"{image['id']:012d}"
        targets = annotations.get(image["id"], [])
        objects = [annotation for annotation in targets if not annotation.get("iscrowd", 0)]
        boxes = np.asarray([annotation["bbox"] for annotation in objects], dtype=float).reshape(-1, 4)
        areas = boxes[:, 2] * boxes[:, 3]
        overlap_fraction = 0.0
        if len(boxes) > 1:
            starts = np.maximum(boxes[:, None, :2], boxes[None, :, :2])
            ends = np.minimum((boxes[:, :2] + boxes[:, 2:])[:, None, :], (boxes[:, :2] + boxes[:, 2:])[None, :, :])
            intersection = np.maximum(ends - starts, 0).prod(axis=2)
            overlap = intersection / np.minimum(areas[:, None], areas[None, :])
            np.fill_diagonal(overlap, 0)
            overlap_fraction = float((overlap >= overlap_threshold).any(axis=1).mean())
        rows.append({"dataset": dataset_name, "image_id": image_id,"filepath": str(image_dir / image["file_name"]), "source_split": source_split,"source_group_id": image_id, "object_count": len(objects),"median_relative_box_area": float(np.median(areas) / (image["height"] * image["width"])) if len(areas) else np.nan,"overlap_fraction": overlap_fraction,"category_count": len({annotation["category_id"] for annotation in objects}),"crowd_present": any(annotation.get("iscrowd", 0) for annotation in targets),})
    return pd.DataFrame(rows)


def split_rem_calibration(train, split_config, dataset_name):
    config = split_config["datasets"][dataset_name]["calibration_sampling"]
    seed = dataset_seed(dataset_name, split_config)
    rng = random.Random(seed)
    indices = list(train.index)
    rng.shuffle(indices)
    selection_count = int(split_config["partitions"]["mllm_selection"]["samples_per_dataset"])
    excluded_path = config.get("mllm_excluded_image_ids")
    excluded_ids = set(resolve_repo_path(excluded_path).read_text().splitlines()) if excluded_path else set()
    selection_indices = [index for index in indices if train.at[index, "image_id"] not in excluded_ids][:selection_count]
    selection = train.loc[selection_indices].copy()
    selection["selection_group"] = "random"
    reserved = set(selection_indices)
    pool_indices = [index for index in indices if index not in reserved]
    rng.shuffle(pool_indices)
    pool = train.loc[pool_indices]
    random_count = int(config["random_samples"])
    calibration_random = pool.iloc[:random_count].copy()
    calibration_random["selection_group"] = "random"
    fraction = float(config["tail_fraction"])
    thresholds = {"median_relative_box_area": float(pool["median_relative_box_area"].quantile(fraction)),"object_count": float(pool["object_count"].quantile(1 - fraction)),"overlap_fraction": float(pool["overlap_fraction"].quantile(1 - fraction)),"category_count": float(pool["category_count"].quantile(1 - fraction)),}
    eligible = {"small_objects": pool["median_relative_box_area"] <= thresholds["median_relative_box_area"],"many_objects": pool["object_count"] >= thresholds["object_count"],"high_overlap": (pool["overlap_fraction"] >= thresholds["overlap_fraction"]) & (pool["overlap_fraction"] > 0),"many_categories": pool["category_count"] >= thresholds["category_count"],"crowd": pool["crowd_present"],}
    used = set(calibration_random.index)
    candidates = {}
    for name, mask in eligible.items():
        candidates[name] = sorted(set(pool.index[mask]) - used)
        rng.shuffle(candidates[name])
    available_counts = {name: len(rows) for name, rows in candidates.items()}
    selected, groups = [], []

    # Round-robin quotas avoid filling one overlapping property group first.
    for _ in range(int(config["samples_per_property"])):
        for name, rows in candidates.items():
            while rows and rows[-1] in used:
                rows.pop()
            if not rows:
                raise ValueError(f"Insufficient distinct images for the frozen {name} quota")
            index = rows.pop()
            selected.append(index)
            groups.append(name)
            used.add(index)
    supplementary = train.loc[selected].copy()
    supplementary["selection_group"] = groups
    calibration = pd.concat([calibration_random, supplementary], ignore_index=True)
    target = int(split_config["partitions"]["calibration"]["samples_per_dataset"])
    if len(calibration) != target or len(selection) != selection_count:
        raise ValueError("Sampling allocations do not match configured partition sizes")
    report = {"seed": seed, "training_pool": len(train), "threshold_reference_pool": len(pool), "criteria": config, "thresholds": thresholds, "eligible_supplementary_counts": available_counts,"calibration_counts": calibration["selection_group"].value_counts().to_dict(),"mllm_selection_count": len(selection), "prior_calibration_excluded_from_mllm": len(excluded_ids)}
    return calibration, selection, report


def inventory_dataset(dataset_name, source_split, split_config):
    if split_config["datasets"][dataset_name]["type"] == "coco_rem":
        return rem_inventory(dataset_name, source_split, split_config)
    dataset = _load_full_dataset(dataset_name,source_split,split_config,limit_zoo_train=True,)
    rows = []
    for filepath in dataset.values("filepath"):
        image_id = os.path.splitext(os.path.basename(filepath))[0]
        rows.append({"dataset": dataset_name,"image_id": image_id,"filepath": str(Path(filepath).resolve()),"source_split": source_split,"source_group_id": source_group_id(dataset_name, image_id, source_split),})
    inventory = pd.DataFrame(rows)
    if inventory["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image IDs found in {dataset_name} {source_split}")
    return inventory


def take_grouped_rows(inventory, target, seed, excluded_groups=None):
    excluded_groups = set(excluded_groups or ())
    groups = [(group_id, group.copy()) for group_id, group in inventory.groupby("source_group_id", sort=False) if group_id not in excluded_groups]
    random.Random(seed).shuffle(groups)

    selected = []
    count = 0
    for _, group in groups:
        if count + len(group) <= target:
            selected.append(group)
            count += len(group)
        if count == target:
            break
    if count != target:
        raise ValueError(f"Could select only {count} of {target} requested images without splitting source groups" )
    result = pd.concat(selected, ignore_index=True)
    return result, set(result["source_group_id"])


def build_manifests(split_config):
    manifests = {partition: [] for partition in PARTITIONS}
    reports = {}

    for dataset_name in split_config["datasets"]:
        train = inventory_dataset(dataset_name, "train", split_config)
        validation = inventory_dataset(dataset_name, "validation", split_config)

        overlap = set(train["source_group_id"]) & set(validation["source_group_id"])
        if overlap:
            raise ValueError(f"{dataset_name} has {len(overlap)} source groups shared by train and validation")

        if "calibration_sampling" in split_config["datasets"][dataset_name]:
            calibration, selection, reports[dataset_name] = split_rem_calibration(train, split_config, dataset_name)
            calibration["partition"] = "calibration"
            selection["partition"] = "mllm_selection"
            manifests["calibration"].append(calibration)
            manifests["mllm_selection"].append(selection)
        else:
            used_groups = set()
            for partition in ("calibration", "mllm_selection"):
                target = int(split_config["partitions"][partition]["samples_per_dataset"])
                selected, selected_groups = take_grouped_rows(train, target, dataset_seed(dataset_name, split_config), excluded_groups=used_groups,)
                selected["partition"] = partition
                manifests[partition].append(selected)
                used_groups.update(selected_groups)

        validation = validation.copy()
        validation["partition"] = "locked_confirmation"
        validation["selection_group"] = "all_validation"
        manifests["locked_confirmation"].append(validation)

    columns = ["dataset","image_id","filepath","source_split","partition","source_group_id",]
    result = {}
    for partition, parts in manifests.items():
        frame = pd.concat(parts, ignore_index=True)
        extra_columns = [column for column in frame if column not in columns]
        result[partition] = frame[columns + extra_columns]
    return result, reports


def validate_local_dataset_paths(split_config):
    for dataset_name, dataset_config in split_config["datasets"].items():
        if dataset_config["type"] != "coco":
            continue
        for source_split in ("train", "validation"):
            data_path = resolve_repo_path(dataset_config[f"{source_split}_path"])
            labels_path = data_path / "_annotations.coco.json"
            if not data_path.is_dir() or not labels_path.is_file():
                raise FileNotFoundError(f"Expected {dataset_name} {source_split} COCO dataset at {data_path} with annotations at {labels_path}")


def validate_manifests(manifests):
    datasets = set().union(*(set(df["dataset"]) for df in manifests.values()))
    for dataset in datasets:
        group_sets = {partition: set(df[df["dataset"] == dataset]["source_group_id"]) for partition, df in manifests.items()}
        for index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[index + 1 :]:
                overlap = group_sets[left] & group_sets[right]
                if overlap:
                    raise ValueError(f"{dataset} has {len(overlap)} source groups in both {left} and {right}")



parser = argparse.ArgumentParser(description="Generate stable global calibration, MLLM-selection, and confirmation manifests")
parser.add_argument("--partition",required=True,choices=(*PARTITIONS, "all"),help="Manifest to write; use 'all' for the one-time generation of all three",)
parser.add_argument("--config",default="configs/splits.toml",help="Global split configuration",)
parser.add_argument("--overwrite", action="store_true", help="Replace existing manifest files",)
args = parser.parse_args()

split_config = load_split_config(args.config)
validate_local_dataset_paths(split_config)
selected_partitions = PARTITIONS if args.partition == "all" else (args.partition,)

for partition in selected_partitions:
    output_path = manifest_path(partition, split_config)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Manifest already exists: {output_path}. Use --overwrite to replace it.")

manifests, reports = build_manifests(split_config)
validate_manifests(manifests)

# Download only selected ReM images, after sampling from the full annotation inventory.
for dataset_name, config in split_config["datasets"].items():
    if config["type"] != "coco_rem":
        continue
    chosen = pd.concat([manifests[partition] for partition in selected_partitions])
    chosen = chosen[chosen["dataset"] == dataset_name]
    for source_split, rows in chosen.groupby("source_split"):
        foz.download_zoo_dataset(config["name"], split=source_split, image_ids=[int(image_id) for image_id in rows["image_id"]])
    missing = [filepath for filepath in chosen["filepath"] if not Path(filepath).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} selected images missing; first: {missing[0]}")

for partition in selected_partitions:
    output_path = manifest_path(partition, split_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".csv.tmp")
    manifests[partition].to_csv(temporary_path, index=False)
    os.replace(temporary_path, output_path)
    counts = manifests[partition].groupby("dataset").size().to_dict()
    print(f"Saved {output_path}: {counts}")

if reports:
    for dataset_name, report in reports.items():
        config = split_config["datasets"][dataset_name]
        report["annotation_sha256"] = {}
        for source_split in ("train", "validation"):
            with open(resolve_repo_path(config[f"{source_split}_annotations"]), "rb") as file:
                report["annotation_sha256"][source_split] = hashlib.file_digest(file, "sha256").hexdigest()
    report_path = resolve_repo_path(split_config["manifest_dir"]) / "sampling_summary.json"
    report_path.write_text(json.dumps(reports, indent=2) + "\n")
