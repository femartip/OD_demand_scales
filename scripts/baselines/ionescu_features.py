import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, baselines_dir, load_manifest, load_split_config

NAME="ionescu"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--device", default="cuda", help="Device for frozen feature extraction")
args = parser.parse_args()

config = load_split_config()
images = load_manifest(args.partition, split_config=config).sort_values(["dataset", "image_id"]).reset_index(drop=True)
weights = ResNet50_Weights.IMAGENET1K_V2
backbone = resnet50(weights=weights)
backbone.fc = torch.nn.Identity()
backbone = backbone.eval().to(args.device)
transform = weights.transforms()

features = []
with torch.no_grad():
    for start in range(0, len(images), 32):
        paths = images["filepath"].iloc[start:start + 32]
        pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).to(args.device)
        features.append(backbone(pixels).cpu().numpy())
        print(f"features {min(start + 32, len(images))}/{len(images)}")
features = np.concatenate(features)

output_path = baselines_dir(args.partition)
output_path.mkdir(parents=True, exist_ok=True)
output_path = output_path / f"{NAME}_features.npz"
np.savez_compressed(output_path, dataset=images["dataset"].to_numpy(dtype=str),
                    image_id=images["image_id"].to_numpy(dtype=str), features=features)
print(f"Saved {output_path}: {features.shape}; supervised fitting is done by evaluate_rubrics.py")
