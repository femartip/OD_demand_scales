import os
import tomllib
from pathlib import Path

import fiftyone as fo
import fiftyone.zoo as foz
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
PARTITIONS = ("calibration", "mllm_selection", "locked_confirmation")
PROMPT_STRATEGIES = ("zeroshot", "fewshot")
DEFAULT_SPLIT_CONFIG = REPO_ROOT / "configs/splits.toml"


def resolve_repo_path(path):
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def load_toml(path):
    with open(resolve_repo_path(path), "rb") as file:
        return tomllib.load(file)


def load_split_config(path=DEFAULT_SPLIT_CONFIG):
    config = load_toml(path)
    missing = set(PARTITIONS) - set(config.get("partitions", {}))
    if missing:
        raise ValueError(f"Missing partition definitions: {', '.join(sorted(missing))}")
    return config


def load_experiment_config(version):
    path = REPO_ROOT / f"configs/experiments/v{version}.toml"
    if not path.is_file():
        raise FileNotFoundError(f"Experiment configuration not found: {path}")
    config = load_toml(path)
    if int(config.get("version", -1)) != int(version):
        raise ValueError(f"Experiment version in {path} does not match v{version}")
    return config


def split_config_for_version(version):
    experiment = load_experiment_config(version)
    return load_split_config(experiment["split_config"])


def manifest_path(partition, split_config=None):
    validate_partition(partition)
    config = split_config or load_split_config()
    return resolve_repo_path(config["manifest_dir"]) / f"{partition}.csv"


def load_manifest(partition, dataset=None, split_config=None):
    path = manifest_path(partition, split_config)
    if not path.is_file():
        raise FileNotFoundError(
            f"Partition manifest not found: {path}. Generate it with "
            f"`poetry run python scripts/preparation/get_image_ids.py --partition {partition}`."
        )
    manifest = pd.read_csv(path, dtype={"image_id": str, "source_group_id": str})
    required = {"dataset", "image_id", "filepath", "source_split", "partition", "source_group_id"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    if set(manifest["partition"].unique()) != {partition}:
        raise ValueError(f"{path} contains rows outside partition {partition}")
    if dataset is not None:
        manifest = manifest[manifest["dataset"] == dataset].copy()
        if manifest.empty:
            raise ValueError(f"No {dataset} rows found in {path}")
    return manifest


def validate_partition(partition):
    if partition not in PARTITIONS:
        raise ValueError(f"Partition must be one of: {', '.join(PARTITIONS)}")


def object_detection_root(partition):
    validate_partition(partition)
    return REPO_ROOT / "outputs/object_detection" / partition


def object_detection_dataset_dir(partition, dataset):
    return object_detection_root(partition) / dataset


def annotations_dir(version, partition):
    validate_partition(partition)
    return REPO_ROOT / "outputs/annotations" / f"v{version}" / partition


def figures_dir(version, partition):
    validate_partition(partition)
    return REPO_ROOT / "outputs/figures" / f"v{version}" / partition


def training_pool_size(split_config):
    return sum(
        int(split_config["partitions"][partition]["samples_per_dataset"])
        for partition in ("calibration", "mllm_selection")
    )


def dataset_seed(dataset_name, split_config):
    dataset_names = list(split_config["datasets"])
    return int(split_config["seed"]) + dataset_names.index(dataset_name)


def _load_full_dataset(dataset_name, source_split, split_config, limit_zoo_train=False):
    dataset_config = split_config["datasets"][dataset_name]
    dataset_type = dataset_config["type"]
    if dataset_type == "fiftyone_zoo":
        options = {}
        if source_split == "train" and limit_zoo_train:
            options = {
                "max_samples": training_pool_size(split_config),
                "shuffle": True,
                "seed": dataset_seed(dataset_name, split_config),
            }
        return foz.load_zoo_dataset(dataset_config["name"], split=source_split, **options)
    if dataset_type == "coco":
        data_path = resolve_repo_path(dataset_config[f"{source_split}_path"])
        labels_path = data_path / "_annotations.coco.json"
        if not data_path.is_dir() or not labels_path.is_file():
            raise FileNotFoundError(
                f"Expected {dataset_name} {source_split} COCO dataset at {data_path} "
                f"with annotations at {labels_path}"
            )
        return fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=str(data_path),
            labels_path=str(labels_path),
            include_id=True,
        )
    raise ValueError(f"Unsupported dataset type for {dataset_name}: {dataset_type}")


def load_partition_dataset(dataset_name, partition, version):
    split_config = split_config_for_version(version)
    partition_config = split_config["partitions"][partition]
    source_split = partition_config["source_split"]
    manifest = load_manifest(partition, dataset_name, split_config)
    dataset = _load_full_dataset(
        dataset_name,
        source_split,
        split_config,
        limit_zoo_train=True,
    )

    wanted_ids = set(manifest["image_id"].astype(str))
    sample_ids = []
    found_ids = set()
    for sample_id, filepath in zip(dataset.values("id"), dataset.values("filepath")):
        image_id = os.path.splitext(os.path.basename(filepath))[0]
        if image_id in wanted_ids:
            sample_ids.append(sample_id)
            found_ids.add(image_id)

    missing = wanted_ids - found_ids
    if missing:
        preview = ", ".join(sorted(missing)[:5])
        raise RuntimeError(
            f"{len(missing)} manifest images were not found in {dataset_name} {source_split}; "
            f"first missing IDs: {preview}"
        )
    view = dataset.select(sample_ids)
    if len(view) != len(manifest):
        raise RuntimeError(
            f"Loaded {len(view)} {dataset_name} samples but manifest contains {len(manifest)} rows"
        )
    return view
