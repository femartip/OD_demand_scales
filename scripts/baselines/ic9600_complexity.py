"""IC9600 image complexity (Feng et al., TPAMI 2023) as a zero-shot difficulty predictor.

Needs a clone of https://github.com/tinglyfeng/IC9600 for ICNet.py and its Google Drive
checkpoint. Complexity is predicted from the image alone, with no detector outcomes,
so this is the closest peer to a zero-shot rubric.

python scripts/baselines/ic9600_complexity.py --partition calibration
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, baselines_dir, load_manifest, load_split_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--ic9600-dir", type=Path, default=Path("../data/models/ic9600"), help="Clone providing ICNet.py")
parser.add_argument("--checkpoint", type=Path, default=Path("data/models/ic9600/ck.pth"))
parser.add_argument("--batch-size", type=int, default=8)
parser.add_argument("--name", default="ic9600")
args = parser.parse_args()

sys.path.insert(0, str(args.ic9600_dir.resolve()))
from ICNet import ICNet

config = load_split_config()
images = load_manifest(args.partition, split_config=config).sort_values(["dataset", "image_id"]).reset_index(drop=True)
random_ids = np.flatnonzero(images["selection_group"].eq("random")) if args.partition == "calibration" else np.arange(len(images))

model = ICNet()
model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
model = model.eval().cuda()
# The resize and normalisation match gene.py in the IC9600 release.
transform = transforms.Compose([transforms.Resize((512, 512)), transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

complexity = []
with torch.no_grad():
    for start in range(0, len(images), args.batch_size):
        paths = images["filepath"].iloc[start:start + args.batch_size]
        pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).cuda()
        complexity.append(model(pixels)[0].flatten().cpu().numpy())
        print(f"complexity {min(start + args.batch_size, len(images))}/{len(images)}")
complexity = np.concatenate(complexity)

edges = np.quantile(complexity[random_ids], [0.2, 0.4, 0.6, 0.8])
images["complexity"] = complexity
images["level"] = 1 + np.searchsorted(edges, complexity)

output_path = baselines_dir(args.partition)
output_path.mkdir(parents=True, exist_ok=True)
output_path = output_path / f"{args.name}.csv"
images[["dataset", "image_id", "level", "complexity"]].to_csv(output_path, index=False)
print(f"Saved {output_path}: {images['level'].value_counts().sort_index().to_dict()}")
