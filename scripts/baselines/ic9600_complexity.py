import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from ICNet import ICNet

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, baselines_dir, load_manifest, load_split_config

IC9600_DIR=Path("../data/models/ic9600")
NAME="ic9600"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--checkpoint", type=Path, default=Path("data/models/ic9600/ck.pth"))
args = parser.parse_args()

sys.path.insert(0, str(IC9600_DIR.resolve()))


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
    for start in range(0, len(images), 8):
        paths = images["filepath"].iloc[start:start + 8]
        pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).cuda()
        complexity.append(model(pixels)[0].flatten().cpu().numpy())
        print(f"complexity {min(start + 8, len(images))}/{len(images)}")
complexity = np.concatenate(complexity)

images["complexity"] = complexity

output_path = baselines_dir(args.partition)
output_path.mkdir(parents=True, exist_ok=True)
output_path = output_path / f"{NAME}.csv"
images[["dataset", "image_id", "complexity"]].to_csv(output_path, index=False)
print(f"Saved {output_path}: complexity in [{complexity.min():.3f}, {complexity.max():.3f}]")
