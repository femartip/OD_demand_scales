import json
import fiftyone as fo
import fiftyone.zoo as foz
import sys
import pandas as pd
import random

datasets = ["coco-2017","voc-2007","driving"]

for dataset_name in datasets:
    if dataset_name in ["voc-2007", "coco-2017"]:
        predefined = True
    else:
        predefined = False

    print(f"Dataset Name: {dataset_name}, Predefined: {predefined}")

    if predefined:
        ground_truth = "ground_truth"
        dataset = foz.load_zoo_dataset(
            dataset_name,
            split="validation",
        )
    else:
        ground_truth = "detections"
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            data_path=f"../vision_datasets/{dataset_name}-validation",
            labels_path=f"../vision_datasets/{dataset_name}-validation/_annotations.coco.json",
            include_id=True,
        )



    model_list = [
        "yolov5l-coco-torch",
        "yolov5m-coco-torch",
        "yolov5n-coco-torch",
        "yolov5s-coco-torch",
        "yolov5x-coco-torch",
        "yolov8l-coco-torch",
        "yolov8m-coco-torch",
        "yolov8n-coco-torch",
        "yolov8s-coco-torch",
        "yolov8x-coco-torch",
        "yolov9c-coco-torch",
        "yolov9e-coco-torch",
        "zero-shot-detection-transformer-torch",
        "detection-transformer-torch",
        "faster-rcnn-resnet50-fpn-coco-torch",
        "retinanet-resnet50-fpn-coco-torch",
        "yolo-nas-torch"
    ]


    for model_name in model_list:
        print(model_name)

        model = foz.load_zoo_model(model_name)
        dataset.apply_model(model, label_field="predictions")



        results = dataset.evaluate_detections(
            pred_field="predictions",
            gt_field=ground_truth,
            eval_key="eval_coco",
            compute_mAP=True,
        )


        dataset_dict = dataset.to_dict()

        with open(f'./results_{dataset_name}/{model_name}_results.json', 'w') as f:
            json.dump(dataset_dict, f, indent=4)