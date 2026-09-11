"""Ionescu et al. (CVPR 2016) difficulty prediction, retargeted to detector performance.

Frozen ImageNet features plus ridge regression, trained on the detector score instead of
their human visual-search time, using the folds evaluate_rubrics.py evaluates on. This is
a supervised reference: unlike a rubric it sees detector outcomes during training.

python scripts/baselines/ionescu_features.py --partition calibration
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, baselines_dir, load_manifest, load_split_config, object_detection_root

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--task", choices=("detection", "localization"), default="detection")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--name", default="ionescu")
args = parser.parse_args()

config = load_split_config()
images = load_manifest(args.partition, split_config=config).sort_values(["dataset", "image_id"]).reset_index(drop=True)
random_ids = np.flatnonzero(images["selection_group"].eq("random")) if args.partition == "calibration" else np.arange(len(images))

evaluation = pd.read_csv(object_detection_root(args.partition) / "detection_difficulty.csv", dtype={"image_id": str})
scores = evaluation.pivot(index=["dataset", "image_id"], columns="model", values=f"{args.task}_quality")
target = scores.loc[pd.MultiIndex.from_frame(images[["dataset", "image_id"]])].to_numpy().mean(axis=1)

weights = ResNet50_Weights.IMAGENET1K_V2
backbone = resnet50(weights=weights)
backbone.fc = torch.nn.Identity()
backbone = backbone.eval().cuda()
transform = weights.transforms()

features = []
with torch.no_grad():
    for start in range(0, len(images), args.batch_size):
        paths = images["filepath"].iloc[start:start + args.batch_size]
        pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).cuda()
        features.append(backbone(pixels).cpu().numpy())
        print(f"features {min(start + args.batch_size, len(images))}/{len(images)}")
features = np.concatenate(features)

alphas = np.logspace(-1, 4, 12)
prediction = np.empty(len(images))
for train, test in KFold(n_splits=5, shuffle=True, random_state=config["seed"]).split(random_ids):
    fitted = RidgeCV(alphas=alphas).fit(features[random_ids[train]], target[random_ids[train]])
    prediction[random_ids[test]] = fitted.predict(features[random_ids[test]])
held_out = np.setdiff1d(np.arange(len(images)), random_ids)
if len(held_out):
    prediction[held_out] = RidgeCV(alphas=alphas).fit(features[random_ids], target[random_ids]).predict(features[held_out])

# A high predicted score is an easy image, so the highest bin becomes level 1.
edges = np.quantile(prediction[random_ids], [0.2, 0.4, 0.6, 0.8])
images["predicted_score"] = prediction
images["level"] = 5 - np.searchsorted(edges, prediction)

output_path = baselines_dir(args.partition)
output_path.mkdir(parents=True, exist_ok=True)
output_path = output_path / f"{args.name}.csv"
images[["dataset", "image_id", "level", "predicted_score"]].to_csv(output_path, index=False)
print(f"Out-of-fold Spearman against the target: {spearmanr(prediction[random_ids], target[random_ids]).statistic:.3f}")
print(f"Saved {output_path}: {images['level'].value_counts().sort_index().to_dict()}")
