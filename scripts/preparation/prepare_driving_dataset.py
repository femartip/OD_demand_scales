import argparse
import itertools
import json
import os
import re
import shutil
import uuid
from collections import defaultdict
from pathlib import Path


CANONICAL_SUFFIX = re.compile(r"[0-9a-f]{32}")


def source_id(file_name):
    return file_name.split("_jpg.rf.", maxsplit=1)[0]


def canonical_images(images):
    grouped = defaultdict(list)
    for image in images:
        grouped[source_id(image["file_name"])].append(image)

    selected = {}
    for group_id, variants in grouped.items():
        canonical = [
            image
            for image in variants
            if CANONICAL_SUFFIX.fullmatch(
                image["file_name"].split(".rf.", maxsplit=1)[1].rsplit(".", maxsplit=1)[0]
            )
        ]
        if len(canonical) != 1:
            raise ValueError(
                f"Expected one canonical Roboflow variant for {group_id}, found {len(canonical)}"
            )
        selected[group_id] = canonical[0]
    return selected


def timestamp_sequences(group_ids, maximum_gap):
    timestamps = sorted(int(group_id) for group_id in group_ids)
    sequences = [[timestamps[0]]]
    for previous, current in zip(timestamps, timestamps[1:]):
        if current - previous > maximum_gap:
            sequences.append([])
        sequences[-1].append(current)
    return sequences


def choose_development_sequences(sequences, minimum_size):
    best = None
    for count in range(1, len(sequences) + 1):
        for indices in itertools.combinations(range(len(sequences)), count):
            size = sum(len(sequences[index]) for index in indices)
            if size < minimum_size:
                continue
            candidate = (size - minimum_size, count, indices)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError(f"No complete sequence combination contains {minimum_size} images")
    return best[2]


def link_or_copy(source, destination):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_split(data, images_by_source, selected_source_ids, source_dir, destination):
    images = [images_by_source[group_id] for group_id in sorted(selected_source_ids, key=int)]
    image_ids = {image["id"] for image in images}
    annotations = [
        annotation for annotation in data["annotations"] if annotation["image_id"] in image_ids
    ]

    destination.mkdir(parents=True)
    for image in images:
        link_or_copy(source_dir / image["file_name"], destination / image["file_name"])

    split_data = {
        "info": dict(data.get("info", {})),
        "licenses": data.get("licenses", []),
        "categories": data["categories"],
        "images": images,
        "annotations": annotations,
    }
    split_data["info"]["description"] = (
        f"{split_data['info'].get('description', '')} "
        "Canonical images with sequence-disjoint OD_pred_rubrics split."
    ).strip()
    with open(destination / "_annotations.coco.json", "w") as file:
        json.dump(split_data, file)

    return len(images), len(annotations)


def main():
    parser = argparse.ArgumentParser(
        description="Deduplicate and sequence-split the flat Roboflow driving COCO export"
    )
    parser.add_argument(
        "--source-dir",
        default="../vision_datasets/self-driving-car-v2-fixed-large/export",
    )
    parser.add_argument(
        "--output-dir",
        default="../vision_datasets/self-driving-car-v2-split",
    )
    parser.add_argument(
        "--development-min-size",
        type=int,
        default=2500,
        help="Minimum number of complete-sequence images reserved for calibration and selection",
    )
    parser.add_argument(
        "--sequence-gap-ns",
        type=int,
        default=1_000_000_000,
        help="Timestamp gap that starts a new capture sequence",
    )
    args = parser.parse_args()

    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    annotations_path = source_dir / "_annotations.coco.json"
    if not annotations_path.is_file():
        raise FileNotFoundError(f"COCO annotations not found: {annotations_path}")
    if output_dir.exists():
        raise FileExistsError(f"Output already exists: {output_dir}")

    with open(annotations_path) as file:
        data = json.load(file)

    images_by_source = canonical_images(data["images"])
    sequences = timestamp_sequences(images_by_source, args.sequence_gap_ns)
    development_indices = choose_development_sequences(sequences, args.development_min_size)
    development_ids = {
        str(timestamp)
        for index in development_indices
        for timestamp in sequences[index]
    }
    confirmation_ids = set(images_by_source) - development_ids
    if development_ids & confirmation_ids:
        raise RuntimeError("Driving source images overlap between train and validation")

    temporary_dir = output_dir.with_name(f"{output_dir.name}.tmp-{uuid.uuid4().hex}")
    try:
        train_counts = write_split(
            data,
            images_by_source,
            development_ids,
            source_dir,
            temporary_dir / "train",
        )
        validation_counts = write_split(
            data,
            images_by_source,
            confirmation_ids,
            source_dir,
            temporary_dir / "validation",
        )
        os.replace(temporary_dir, output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"Canonical source images: {len(images_by_source)}")
    print(
        f"Train: {train_counts[0]} images, {train_counts[1]} annotations, "
        f"sequences {list(development_indices)}"
    )
    print(
        f"Validation: {validation_counts[0]} images, "
        f"{validation_counts[1]} annotations"
    )
    print(f"Saved sequence-disjoint dataset to {output_dir}")


if __name__ == "__main__":
    main()

