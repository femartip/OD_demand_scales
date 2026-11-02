"""Frozen image embeddings mapped directly to detector performance, with no interpretable stage.

This is the high-capacity reference: if a generic visual representation predicts performance as well
as the rubric, the rubric's dimensions are not carrying information of their own. Two backbones are
provided, a supervised ImageNet one and a self-supervised one, because the strength of the reference
depends on how modern the representation is. Supervised fitting is done by evaluate_rubrics.py.

python scripts/baselines/embedding_features.py --partition calibration --backbone dinov2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import PARTITIONS, baselines_dir, load_manifest, load_split_config

BACKBONES = ("resnet50", "dinov2")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--backbone", required=True, choices=BACKBONES)
parser.add_argument("--device", default="cuda", help="Device for frozen feature extraction")
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--image-root", type=Path, help="Resolve manifest images by basename under this "
                    "directory; the manifest stores absolute paths from whichever machine wrote it")
args = parser.parse_args()

config = load_split_config()
images = load_manifest(args.partition, split_config=config).sort_values(["dataset", "image_id"]).reset_index(drop=True)

# Both backbones default to a centre crop, which would discard roughly a third of the width and hide
# exactly the peripheral small and frame-touching targets the rubric scores. The whole image is
# resized to the native input resolution instead, so the representation sees what the rubric describes.
SIZE = 224
if args.backbone == "resnet50":
    from torchvision.models import ResNet50_Weights, resnet50
    from torchvision.transforms import Compose, Normalize, Resize, ToTensor
    weights = ResNet50_Weights.IMAGENET1K_V2
    backbone = resnet50(weights=weights)
    backbone.fc = torch.nn.Identity()
    reference = weights.transforms()
    transform = Compose([Resize((SIZE, SIZE)), ToTensor(), Normalize(reference.mean, reference.std)])
    embed = lambda pixels: backbone(pixels)
else:
    from transformers import AutoImageProcessor, AutoModel
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
    backbone = AutoModel.from_pretrained("facebook/dinov2-base")
    transform = lambda image: processor(images=image, do_center_crop=False,
                                        size={"height": SIZE, "width": SIZE},
                                        return_tensors="pt")["pixel_values"][0]
    embed = lambda pixels: backbone(pixel_values=pixels).pooler_output

backbone = backbone.eval().to(args.device)

features = []
with torch.no_grad():
    for start in range(0, len(images), args.batch_size):
        paths = images["filepath"].iloc[start:start + args.batch_size]
        if args.image_root:
            paths = [args.image_root / Path(path).name for path in paths]
        pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths]).to(args.device)
        features.append(embed(pixels).cpu().numpy())
        print(f"features {min(start + args.batch_size, len(images))}/{len(images)}", flush=True)
features = np.concatenate(features).astype(np.float32)

output_path = baselines_dir(args.partition)
output_path.mkdir(parents=True, exist_ok=True)
output_path = output_path / f"{args.backbone}_features.npz"
np.savez_compressed(output_path, dataset=images["dataset"].to_numpy(dtype=str),
                    image_id=images["image_id"].to_numpy(dtype=str), features=features)
print(f"Saved {output_path}: {features.shape}")
