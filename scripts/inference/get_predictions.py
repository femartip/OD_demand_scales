import json
import fiftyone as fo
import fiftyone.zoo as foz
import os
import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import (PARTITIONS, load_experiment_config, load_partition_dataset, object_detection_dataset_dir, split_config_for_version,)

def patch_rfdetr_class_name_conversion():
    """Missmatch between libraries creates an error where they are not in the same format 
    RF-DETR expose class names as a list but is expected as a dict."""
    import fiftyone.utils.rfdetr as four

    original_converter = four._sv_detections_to_fo
    if getattr(original_converter, "_supports_rfdetr_list_class_names", False):
        return

    def compatible_converter(sv_dets, width, height, class_names, has_masks):
        if isinstance(class_names, (list, tuple)):
            detection_data = getattr(sv_dets, "data", {}) or {}
            detection_names = detection_data.get("class_name")
            class_ids = getattr(sv_dets, "class_id", None)

            if detection_names is not None and class_ids is not None:
                class_names = {int(class_id): str(class_name) for class_id, class_name in zip(class_ids, detection_names)}
            else:
                class_names = dict(enumerate(class_names))

        return original_converter(sv_dets, width, height, class_names, has_masks)

    compatible_converter._supports_rfdetr_list_class_names = True
    four._sv_detections_to_fo = compatible_converter

parser = argparse.ArgumentParser()
parser.add_argument("--version",type=int,required=True,help="Experiment version",)
parser.add_argument("--partition",required=True,choices=PARTITIONS,help="Experimental partition to run",)
parser.add_argument("--skip-existing",action="store_true",help="Skip model/dataset predictions whose output JSON already exists",)
parser.add_argument("--transformer-confidence-threshold",type=float,help="Confidence threshold for Hugging Face transformer detectors; defaults to the experiment configuration",)
parser.add_argument("--models",nargs="+", help="Only run the specified model names",)
args = parser.parse_args()

patch_rfdetr_class_name_conversion()

split_config = split_config_for_version(args.version)
experiment_config = load_experiment_config(args.version)
transformer_confidence_threshold = (args.transformer_confidence_threshold if args.transformer_confidence_threshold is not None else float(experiment_config["inference"]["transformer_confidence_threshold"]))
datasets = list(split_config["datasets"])

for dataset_name in datasets:
    if dataset_name in ["voc-2007", "coco-2017"]:
        predefined = True
    else:
        predefined = False

    print(f"Dataset Name: {dataset_name}, Predefined: {predefined}")

    dataset = load_partition_dataset(dataset_name, args.partition, args.version)



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

    if args.models:
        unknown_models = sorted(set(args.models) - set(model_list))
        if unknown_models:
            parser.error(f"Unknown model names: {', '.join(unknown_models)}")
        model_list = args.models


    for model_name in model_list:
        print(model_name)

        output_dir = object_detection_dataset_dir(args.partition, dataset_name)
        output_path = output_dir / f'{model_name}_predictions.json'
        if args.skip_existing and os.path.isfile(output_path):
            print(f"Skipping existing result: {output_path}")
            continue

        model = foz.load_zoo_model(model_name)
        transformer_model = (model_name.startswith(("dfine-", "rtdetr-v2-")) or model_name == "detection-transformer-torch")
        confidence_thresh = (transformer_confidence_threshold if transformer_model else None)
        if confidence_thresh is not None:
            print(f"Using confidence threshold {confidence_thresh}")

        if "predictions" in dataset.get_field_schema():
            dataset.clear_sample_field("predictions")

        dataset.apply_model(model,label_field="predictions",confidence_thresh=confidence_thresh,batch_size=8,num_workers=12,pin_memory=True,skip_failures=False,)

        missing_predictions = len(dataset) - dataset.exists("predictions").count()
        if missing_predictions:
            raise RuntimeError(f"{model_name} did not write predictions for {missing_predictions/len(dataset)} samples")

        # Dont need to evaluate here, this is done on the evaluation/detection.py
        #results = dataset.evaluate_detections(pred_field="predictions",gt_field=ground_truth,eval_key="eval_coco",compute_mAP=False,)

        dataset_dict = dataset.to_dict()

        os.makedirs(output_dir, exist_ok=True)
        temporary_output_path = Path(f"{output_path}.tmp")
        with open(temporary_output_path, 'w') as f:
            json.dump(dataset_dict, f, indent=4)
        os.replace(temporary_output_path, output_path)
    print("All models finished for this dataset")

print(f"All datasets and models finished, predicions saved to {output_path}")
