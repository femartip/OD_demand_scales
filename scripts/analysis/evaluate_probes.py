"""Score annotator probes against COCO-ReM, stratified by the known policy divergences.

The prompt counts depicted objects and individuals inside crowd regions; the reference does
neither, so agreement is reported on the subset where the two policies agree as well as overall.

python scripts/analysis/evaluate_probes.py --partition calibration --annotators qwen3.8-27b-q5
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, REPO_ROOT, load_manifest, load_split_config, load_toml, resolve_repo_path

SMALL_AREA = 0.01
BORDER = 2
# probe -> (answer key, reference column, whether the quantity is a fraction)
PROBES = {"count": ("target_count", "gt_count", False),
          "small_fraction": ("small_fraction", "gt_small_fraction", True),
          "truncation": ("truncation_fraction", "gt_truncation_fraction", True),
          "category_diversity": ("category_count", "gt_category_count", False)}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--annotators", nargs="+", required=True)
parser.add_argument("--dataset", default="coco-rem")
parser.add_argument("--probes", nargs="+", choices=tuple(PROBES), default=tuple(PROBES))
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()

config = load_toml("configs/probes.toml")
split_config = load_split_config(config["split_config"])
manifest = load_manifest(args.partition, args.dataset, split_config)

source_split = split_config["partitions"][args.partition]["source_split"]
annotation_path = resolve_repo_path(split_config["datasets"][args.dataset][f"{source_split}_annotations"])
if not annotation_path.is_file():
    annotation_path = REPO_ROOT / "data/datasets/coco-rem" / annotation_path.name
with annotation_path.open() as file:
    raw = json.load(file)
wanted = {int(image_id) for image_id in manifest.image_id}
area = {image["id"]: image["width"] * image["height"] for image in raw["images"] if image["id"] in wanted}
shape = {image["id"]: (image["width"], image["height"]) for image in raw["images"] if image["id"] in wanted}
count, small, truncated, categories = defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(set)
for target in raw["annotations"]:
    image_id = target["image_id"]
    if image_id not in area or target.get("iscrowd", 0):
        continue
    x, y, w, h = target["bbox"]
    width, height = shape[image_id]
    count[image_id] += 1
    small[image_id] += int(w * h / area[image_id] < SMALL_AREA)
    truncated[image_id] += int(x <= BORDER or y <= BORDER or x + w >= width - BORDER or y + h >= height - BORDER)
    categories[image_id].add(target["category_id"])
del raw

reference = pd.DataFrame({"image_id": manifest.image_id})
key = reference.image_id.astype(int)
reference["gt_count"] = [count[i] for i in key]
reference["gt_small_fraction"] = [small[i] / count[i] if count[i] else np.nan for i in key]
reference["gt_truncation_fraction"] = [truncated[i] / count[i] if count[i] else np.nan for i in key]
reference["gt_category_count"] = [len(categories[i]) for i in key]
reference = reference.merge(manifest[["image_id", "crowd_present", "selection_group"]], on="image_id")
reference["cohort"] = np.where(reference.selection_group.eq("random"), "random", "properties")

rows = []
for annotator in args.annotators:
    for probe in args.probes:
        answer_key, gt_column, is_fraction = PROBES[probe]
        path = REPO_ROOT / "outputs/probes" / args.partition / f"{probe}_{args.dataset}_{annotator}.jsonl"
        if not path.is_file():
            print(f"missing: {path}")
            continue
        records = [json.loads(line) for line in path.read_text().splitlines()]
        frame = pd.DataFrame([{"image_id": r["image_id"], "predicted": (r["answer"] or {}).get(answer_key),
                               "depictions": tuple((r["answer"] or {}).get("depiction_types") or [])}
                              for r in records if r.get("answer")])
        frame = frame.drop_duplicates("image_id", keep="last")
        merged = reference.merge(frame, on="image_id").dropna(subset=["predicted", gt_column])
        merged["physical_only"] = merged.depictions.map(lambda types: types == ("physical_object",))
        clean = ~merged.crowd_present & merged.physical_only
        strata = {"all": merged, "clean policy": merged[clean],
                  "clean, dense (GT count >= 20)": merged[clean & merged.gt_count.ge(20)]}
        for cohort in ("random", "properties"):
            for name, subset in strata.items():
                subset = subset[subset.cohort.eq(cohort)]
                if len(subset) < 25:
                    continue
                error = subset.predicted - subset[gt_column]
                rows.append(dict(annotator=annotator, probe=probe, cohort=cohort, stratum=name,
                                 n=len(subset), spearman=spearmanr(subset.predicted, subset[gt_column]).statistic,
                                 mae=error.abs().mean(), bias=error.mean(),
                                 mean_predicted=subset.predicted.mean(), mean_reference=subset[gt_column].mean()))

results = pd.DataFrame(rows)
output_dir = args.output_dir or REPO_ROOT / "outputs/analysis" / args.partition / "probes"
output_dir.mkdir(parents=True, exist_ok=True)
results.to_csv(output_dir / "probe_agreement.csv", index=False)
print(results.round(3).to_string(index=False))
print(f"\nSaved {output_dir / 'probe_agreement.csv'}")
