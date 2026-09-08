"""Download COCO-ReM and derive detection boxes from its refined instance masks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

import requests
from pycocotools import mask as mask_utils

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import load_split_config, resolve_repo_path


def prepare_split(source_split, config):
    output = resolve_repo_path(config[f"{source_split}_annotations"])
    output.parent.mkdir(parents=True, exist_ok=True)
    release_name = "instances_trainrem.json" if source_split == "train" else "instances_valrem.json"
    archive = output.parent / f"{release_name}.zip"
    url = f"https://huggingface.co/datasets/kdexd/coco-rem/resolve/main/{release_name}.zip"
    if not archive.exists():
        print(f"Downloading {url}", flush=True)
        temporary = archive.with_suffix(".zip.tmp")
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with temporary.open("wb") as target:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    target.write(chunk)
        os.replace(temporary, archive)
    print(f"Preparing {source_split} boxes from masks", flush=True)
    with zipfile.ZipFile(archive) as source:
        with source.open(release_name) as file:
            data = json.load(file)
    image_sizes = {im["id"]: (im["height"], im["width"]) for im in data["images"]}
    skipped, changed = [], 0
    kept = []
    for annotation in data["annotations"]:
        height, width = image_sizes[annotation["image_id"]]
        # Some training annotations omit the optional non-crowd flag.
        annotation.setdefault("iscrowd", 0)
        segmentation = annotation.pop("segmentation")
        if isinstance(segmentation, list):
            rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, height, width))
        elif isinstance(segmentation["counts"], list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        if list(rle["size"]) != [height, width]:
            raise ValueError(f"Mask size mismatch for annotation {annotation['id']}")
        area = int(mask_utils.area(rle))
        if area == 0:
            skipped.append(annotation["id"])
            continue
        bbox = mask_utils.toBbox(rle).tolist()
        changed += bbox != annotation["bbox"]
        annotation["bbox"] = bbox
        annotation["area"] = area
        kept.append(annotation)
    data["annotations"] = kept
    temporary = output.with_suffix(".json.tmp")
    with temporary.open("w") as file:
        json.dump(data, file, separators=(",", ":"))
    os.replace(temporary, output)
    with archive.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    report = {
        "source_url": url,
        "source_sha256": digest,
        "images": len(data["images"]),
        "categories": len(data["categories"]),
        "annotations": len(kept),
        "changed_boxes": changed,
        "omitted_zero_area_annotation_ids": skipped,
        "box_policy": "Tight COCO xywh boxes and area derived from ReM masks; IDs, categories and iscrowd preserved. Original masks remain in the release ZIP.",
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}: {len(kept)} targets, {changed} corrected boxes, {len(skipped)} empty masks omitted", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "validation", "all"), default="all")
    parser.add_argument("--config", default="configs/splits.toml")
    args = parser.parse_args()
    config = load_split_config(args.config)["datasets"]["coco-rem"]
    for source_split in (("train", "validation") if args.split == "all" else (args.split,)):
        prepare_split(source_split, config)


if __name__ == "__main__":
    main()
