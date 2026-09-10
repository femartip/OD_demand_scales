import pandas as pd
import re
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS, PROMPT_STRATEGIES, annotations_dir, figures_dir, object_detection_root,)



parser = argparse.ArgumentParser()
parser.add_argument("version")
parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("prompt_strategy", choices=PROMPT_STRATEGIES)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
args = parser.parse_args()

version = str(args.version)
task = args.task
prompt_strategy = args.prompt_strategy
strategy_display = "Zero-shot" if prompt_strategy == "zeroshot" else "Few-shot"
figure_dir = figures_dir(version, args.partition)
figure_dir.mkdir(parents=True, exist_ok=True)
annotation_dir = annotations_dir(version, args.partition)
evaluation_dir = object_detection_root(args.partition)



if task == "detection":
    display_name = "Object Detection"
    df = pd.read_csv(evaluation_dir / "detection_difficulty.csv", dtype={'image_id': object},usecols=["dataset", "image_id", "detection_quality", "model"],)
    df = df.rename(columns={"detection_quality": "score"})
elif task == "localization":
    display_name = "Localization"
    df = pd.read_csv(evaluation_dir / "detection_difficulty.csv", dtype={'image_id': object},usecols=["dataset", "image_id", "localization_quality", "model"],)
    df = df.rename(columns={"localization_quality": "score"})
else:
    sys.exit(1)


gpt_diff = pd.read_csv(annotation_dir / f"v{version}_{task}_{prompt_strategy}_dataset_gpt_difficulty.csv",dtype={'image_id': object},)

gpt_diff = gpt_diff[gpt_diff['level'] != 'error']

try:
    gpt_diff["level"] = gpt_diff["level"].apply(lambda x: int(re.search(r'\d+', x).group()))
except TypeError:
    pass

print(gpt_diff["level"].value_counts())

model_families = {
    "yolov5": [f"yolov5{size}-coco-torch" for size in ("n", "s", "m", "l", "x")],
    "yolov8": [f"yolov8{size}-coco-torch" for size in ("n", "s", "m", "l", "x")],
    "yolov9": [f"yolov9{size}-coco-torch" for size in ("c", "e")],
    "yolov10": [f"yolov10{size}-coco-torch" for size in ("n", "s", "m", "l", "x")],
    "yolo11": [f"yolo11{size}-coco-torch" for size in ("n", "s", "m", "l", "x")],
    "dfine": [f"dfine-{size}-coco-torch" for size in ("nano", "small", "medium", "large", "xlarge")],
    "rfdetr": [f"rfdetr-{size}-coco-torch" for size in ("nano", "small", "medium", "base", "large")],
    "rtdetr": [f"rtdetr-{size}-coco-torch" for size in ("l", "x")],
    "rtdetr-v2": [f"rtdetr-v2-{size}-coco-torch" for size in ("s", "m", "l")],
    "detr": ["detection-transformer-torch"],
    "faster-rcnn": ["faster-rcnn-resnet50-fpn-coco-torch"],
    "retinanet": ["retinanet-resnet50-fpn-coco-torch"],
}

family_display_names = {
    "yolov5": "YOLOv5",
    "yolov8": "YOLOv8",
    "yolov9": "YOLOv9",
    "yolov10": "YOLOv10",
    "yolo11": "YOLO11",
    "dfine": "D-FINE",
    "rfdetr": "RF-DETR",
    "rtdetr": "RT-DETR",
    "rtdetr-v2": "RT-DETRv2",
    "detr": "DETR",
    "faster-rcnn": "Faster R-CNN",
    "retinanet": "RetinaNet",
}


color_dict = {
    "yolov5l-coco-torch": "#5fa2d5",
    "yolov5m-coco-torch": "#93c5ec",
    "yolov5n-coco-torch": "#cfe2f5",
    "yolov5s-coco-torch": "#aec7e8",
    "yolov5x-coco-torch": "#1f77b4",
    "yolov8l-coco-torch": "#137547",
    "yolov8m-coco-torch": "#2a9134",
    "yolov8n-coco-torch": "#5bba6f",
    "yolov8s-coco-torch": "#3fa34d",
    "yolov8x-coco-torch": "#054a29",
    "yolov9c-coco-torch": "#ffbb78",
    "yolov9e-coco-torch": "#ff7f0e",
    "detection-transformer-torch": "#ff9896",
    "faster-rcnn-resnet50-fpn-coco-torch": "#7f7f7f",
    "retinanet-resnet50-fpn-coco-torch": "#c5b0d5",
}

available_models = set(df["model"].unique())

for family, family_models in model_families.items():
    missing_models = [model for model in family_models if model not in available_models]
    if missing_models:
        print(f"Warning: no evaluation rows found for: {', '.join(missing_models)}")
    models_to_plot = [model for model in family_models if model in available_models]
    if not models_to_plot:
        print(f"Skipping {family_display_names[family]}: no evaluation rows found")
        continue

    plt.figure(figsize=(11, 6))

    for model in models_to_plot:
        temp_df = df[df["model"] == model]
        del temp_df["model"]
        grouped = temp_df.groupby(['dataset', 'image_id'], as_index=False).mean()
        final = pd.merge(grouped, gpt_diff, on=['dataset', 'image_id'])
        final = final.dropna()
        final = final[(final['level'] >= 1) & (final['level'] <= 5)]

        avg_trend = final.groupby('level')['score'].mean().reset_index()
        avg_trend['level'] = avg_trend['level'].astype(str)
        print(f"{family_display_names[family]} / {model}\n{avg_trend}")

        plt.plot(avg_trend['level'], avg_trend['score'], color=color_dict.get(model), linewidth=2, label=f'{model}',marker='o')

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.xlabel(f'{display_name} demand level')
    plt.ylabel(f'{display_name} quality')
    plt.title(f'{family_display_names[family]} {display_name} Quality by Demand Level — {strategy_display} ({args.partition})')
    plt.subplots_adjust(right=0.7)
    plt.savefig(figure_dir / f"v{version}_{family}_{task}_{prompt_strategy}_accuracy_curves.pdf")
    plt.close()
