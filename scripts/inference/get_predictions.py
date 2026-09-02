import json
import fiftyone as fo
import fiftyone.zoo as foz
import sys
import pandas as pd
import random
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--skip-existing",action="store_true",help="Skip model/dataset evaluations whose output JSON already exists",)
parser.add_argument(
    "--transformer-confidence-threshold",
    type=float,
    default=0.5,
    help="Confidence threshold for Hugging Face transformer detectors",
)
args = parser.parse_args()

datasets = ["coco-2017","voc-2007","driving"]

for dataset_name in datasets:
    if dataset_name in ["voc-2007", "coco-2017"]:
        predefined = True
    else:
        predefined = False

    print(f"Dataset Name: {dataset_name}, Predefined: {predefined}")

    if predefined:
        ground_truth = "ground_truth"
        dataset = foz.load_zoo_dataset(dataset_name,split="validation",)
    else:
        ground_truth = "detections"
        dataset = fo.Dataset.from_dir(dataset_type=fo.types.COCODetectionDataset,data_path=f"../vision_datasets/{dataset_name}-validation",labels_path=f"../vision_datasets/{dataset_name}-validation/_annotations.coco.json",include_id=True,)



    model_list = [
        "yolov5n-coco-torch",
        "yolov5s-coco-torch",
        "yolov5m-coco-torch",
        "yolov5l-coco-torch",
        "yolov5x-coco-torch",
        "yolov8n-coco-torch",
        "yolov8s-coco-torch",
        "yolov8m-coco-torch",
        "yolov8l-coco-torch",
        "yolov8x-coco-torch",
        "yolov9c-coco-torch",
        "yolov9e-coco-torch",
        "yolov10n-coco-torch",
        "yolov10s-coco-torch",
        "yolov10m-coco-torch",
        "yolov10l-coco-torch",
        "yolov10x-coco-torch",
        "yolo11n-coco-torch",
        "yolo11s-coco-torch",
        "yolo11m-coco-torch",
        "yolo11l-coco-torch",
        "yolo11x-coco-torch",
        "dfine-nano-coco-torch",
        "dfine-small-coco-torch",
        "dfine-medium-coco-torch",
        "dfine-large-coco-torch",
        "dfine-xlarge-coco-torch",
        "rfdetr-nano-coco-torch",
        "rfdetr-small-coco-torch",
        "rfdetr-medium-coco-torch",
        "rfdetr-base-coco-torch",
        "rfdetr-large-coco-torch",
        "rtdetr-l-coco-torch",
        "rtdetr-x-coco-torch",
        "rtdetr-v2-s-coco-torch",
        "rtdetr-v2-m-coco-torch",
        "rtdetr-v2-l-coco-torch",
        "detection-transformer-torch",
        "faster-rcnn-resnet50-fpn-coco-torch",
        "retinanet-resnet50-fpn-coco-torch",
    ]


    for model_name in model_list:
        print(model_name)

        output_dir = f'./outputs/object_detection/{dataset_name}'
        output_path = f'{output_dir}/{model_name}_object_detection_evaluation.json'
        if args.skip_existing and os.path.isfile(output_path):
            print(f"Skipping existing result: {output_path}")
            continue

        model = foz.load_zoo_model(model_name)
        transformer_model = (model_name.startswith(("dfine-", "rtdetr-v2-")) or model_name == "detection-transformer-torch")
        confidence_thresh = (args.transformer_confidence_threshold if transformer_model else None)
        if confidence_thresh is not None:
            print(f"Using confidence threshold {confidence_thresh}")

        dataset.apply_model(model,label_field="predictions",confidence_thresh=confidence_thresh,batch_size=8,num_workers=12,pin_memory=True,)

        # Dont need to evaluate here, this is done on the evaluation/detection.py
        #results = dataset.evaluate_detections(pred_field="predictions",gt_field=ground_truth,eval_key="eval_coco",compute_mAP=False,)

        dataset_dict = dataset.to_dict()

        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(dataset_dict, f, indent=4)
