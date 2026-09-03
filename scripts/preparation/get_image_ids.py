import argparse
import os
import random
import sys
from pathlib import Path

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import (  # noqa: E402
    PARTITIONS,
    _load_full_dataset,
    dataset_seed,
    load_split_config,
    manifest_path,
    resolve_repo_path,
)


def source_group_id(dataset, image_id, source_split):
    if dataset == "driving" and "_jpg.rf." in image_id:
        return image_id.split("_jpg.rf.", maxsplit=1)[0]
    # FiftyOne prepares VOC files with split-local numeric filenames, so the
    # same basename can refer to different official train and validation images.
    return f"{source_split}:{image_id}"


def inventory_dataset(dataset_name, source_split, split_config):
    dataset = _load_full_dataset(
        dataset_name,
        source_split,
        split_config,
        limit_zoo_train=True,
    )
    rows = []
    for filepath in dataset.values("filepath"):
        image_id = os.path.splitext(os.path.basename(filepath))[0]
        rows.append(
            {
                "dataset": dataset_name,
                "image_id": image_id,
                "filepath": str(Path(filepath).resolve()),
                "source_split": source_split,
                "source_group_id": source_group_id(dataset_name, image_id, source_split),
            }
        )
    inventory = pd.DataFrame(rows)
    if inventory["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image IDs found in {dataset_name} {source_split}")
    return inventory


def take_grouped_rows(inventory, target, seed, excluded_groups=None):
    excluded_groups = set(excluded_groups or ())
    groups = [
        (group_id, group.copy())
        for group_id, group in inventory.groupby("source_group_id", sort=False)
        if group_id not in excluded_groups
    ]
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
        raise ValueError(
            f"Could select only {count} of {target} requested images without splitting source groups"
        )
    result = pd.concat(selected, ignore_index=True)
    return result, set(result["source_group_id"])


def build_manifests(split_config):
    manifests = {partition: [] for partition in PARTITIONS}

    for dataset_name in split_config["datasets"]:
        train = inventory_dataset(dataset_name, "train", split_config)
        validation = inventory_dataset(dataset_name, "validation", split_config)

        overlap = set(train["source_group_id"]) & set(validation["source_group_id"])
        if overlap:
            raise ValueError(
                f"{dataset_name} has {len(overlap)} source groups shared by train and validation"
            )

        used_groups = set()
        for partition in ("calibration", "mllm_selection"):
            target = int(split_config["partitions"][partition]["samples_per_dataset"])
            selected, selected_groups = take_grouped_rows(
                train,
                target,
                dataset_seed(dataset_name, split_config),
                excluded_groups=used_groups,
            )
            selected["partition"] = partition
            manifests[partition].append(selected)
            used_groups.update(selected_groups)

        validation = validation.copy()
        validation["partition"] = "locked_confirmation"
        manifests["locked_confirmation"].append(validation)

    columns = [
        "dataset",
        "image_id",
        "filepath",
        "source_split",
        "partition",
        "source_group_id",
    ]
    return {
        partition: pd.concat(parts, ignore_index=True)[columns]
        for partition, parts in manifests.items()
    }


def validate_local_dataset_paths(split_config):
    for dataset_name, dataset_config in split_config["datasets"].items():
        if dataset_config["type"] != "coco":
            continue
        for source_split in ("train", "validation"):
            data_path = resolve_repo_path(dataset_config[f"{source_split}_path"])
            labels_path = data_path / "_annotations.coco.json"
            if not data_path.is_dir() or not labels_path.is_file():
                raise FileNotFoundError(
                    f"Expected {dataset_name} {source_split} COCO dataset at {data_path} "
                    f"with annotations at {labels_path}"
                )


def validate_manifests(manifests):
    datasets = set().union(*(set(df["dataset"]) for df in manifests.values()))
    for dataset in datasets:
        group_sets = {
            partition: set(df[df["dataset"] == dataset]["source_group_id"])
            for partition, df in manifests.items()
        }
        for index, left in enumerate(PARTITIONS):
            for right in PARTITIONS[index + 1 :]:
                overlap = group_sets[left] & group_sets[right]
                if overlap:
                    raise ValueError(
                        f"{dataset} has {len(overlap)} source groups in both {left} and {right}"
                    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate stable global calibration, MLLM-selection, and confirmation manifests"
    )
    parser.add_argument(
        "--partition",
        required=True,
        choices=(*PARTITIONS, "all"),
        help="Manifest to write; use 'all' for the one-time generation of all three",
    )
    parser.add_argument(
        "--config",
        default="configs/splits.toml",
        help="Global split configuration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing manifest files",
    )
    args = parser.parse_args()

    split_config = load_split_config(args.config)
    validate_local_dataset_paths(split_config)
    selected_partitions = PARTITIONS if args.partition == "all" else (args.partition,)

    for partition in selected_partitions:
        output_path = manifest_path(partition, split_config)
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Manifest already exists: {output_path}. Use --overwrite to replace it."
            )

    manifests = build_manifests(split_config)
    validate_manifests(manifests)
    for partition in selected_partitions:
        output_path = manifest_path(partition, split_config)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_suffix(".csv.tmp")
        manifests[partition].to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
        counts = manifests[partition].groupby("dataset").size().to_dict()
        print(f"Saved {output_path}: {counts}")


if __name__ == "__main__":
    main()
