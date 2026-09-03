import json
import os
import uuid
import argparse
import sys
from pathlib import Path

import fiftyone as fo


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import (PARTITIONS, load_experiment_config, object_detection_dataset_dir,)


CLASS_MAPS = {
    "voc-2007": {
        "aeroplane": "airplane",
        "diningtable": "dining table",
        "motorbike": "motorcycle",
        "pottedplant": "potted plant",
        "sofa": "couch",
        "tvmonitor": "tv",
    },
    "driving": {
        "biker": "bicycle",
        "pedestrian": "person",
        "trafficLight": "traffic light",
        "trafficLight-Green": "traffic light",
        "trafficLight-GreenLeft": "traffic light",
        "trafficLight-Red": "traffic light",
        "trafficLight-RedLeft": "traffic light",
        "trafficLight-Yellow": "traffic light",
        "trafficLight-YellowLeft": "traffic light",
    },
}


def normalize_label(dataset, label):
    return CLASS_MAPS.get(dataset, {}).get(label, label)


def set_labels(sample, field_name, label_fn):
    detections = sample[field_name]
    if detections is None:
        return

    for detection in detections.detections:
        detection.label = label_fn(detection.label)


def evaluate_with_fiftyone(data, dataset_name, ground_truth_field, iou_threshold, compute_map):
    temporary_name = f"od-pred-rubrics-evaluation-{uuid.uuid4().hex}"
    dataset = fo.Dataset.from_dict(data,name=temporary_name,persistent=False,progress=False,)

    try:
        for sample in dataset.iter_samples(autosave=True, progress=False):
            set_labels(sample,ground_truth_field,lambda label: normalize_label(dataset_name, label),)
            set_labels(sample,"predictions",lambda label: normalize_label(dataset_name, label),)

        dataset.evaluate_detections("predictions",gt_field=ground_truth_field,eval_key="mapped_detection",method="coco",iou=iou_threshold,classwise=True,compute_mAP=compute_map,progress=False,)

        for sample in dataset.iter_samples(autosave=True, progress=False):
            set_labels(sample, ground_truth_field, lambda _: "object")
            set_labels(sample, "predictions", lambda _: "object")

        dataset.evaluate_detections("predictions",gt_field=ground_truth_field,eval_key="localization",method="coco",iou=iou_threshold,classwise=True,compute_mAP=compute_map,progress=False,)

        counts_by_filepath = {}
        for sample in dataset.iter_samples(progress=False):
            counts_by_filepath[os.path.normpath(sample.filepath)] = {
                "eval_coco_tp": sample.mapped_detection_tp,
                "eval_coco_fp": sample.mapped_detection_fp,
                "eval_coco_fn": sample.mapped_detection_fn,
                "detection_tp": sample.localization_tp,
                "detection_fp": sample.localization_fp,
                "detection_fn": sample.localization_fn,
            }

        for sample in data["samples"]:
            counts = counts_by_filepath[os.path.normpath(sample["filepath"])]
            sample.update(counts)
    finally:
        dataset.delete()


def list_files_in_directory(directory_path):
    return [file_name for file_name in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file_name))]


def save_json_to_folder(data, folder_path, file_name):
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, file_name)
    temporary_path = f"{file_path}.tmp"

    with open(temporary_path, "w") as json_file:
        json.dump(data, json_file, indent=4, default=str)
    os.replace(temporary_path, file_path)

    print(f"JSON file saved at {file_path}")


parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--version",type=int,required=True,help="Experiment version",)
parser.add_argument("--partition",required=True,choices=PARTITIONS,help="Experimental partition to evaluate",)
parser.add_argument("--skip-existing",action="store_true",help="Skip predictions whose combined evaluation JSON already exists",)
parser.add_argument("--models",nargs="+",help="Only evaluate the specified model names",)
args = parser.parse_args()

dataset_name = args.dataset
experiment_config = load_experiment_config(args.version)
evaluation_config = experiment_config["evaluation"]
ground_truth_field = ("ground_truth" if dataset_name in {"coco-2017", "voc-2007"} else "detections")
directory_path = object_detection_dataset_dir(args.version, args.partition, dataset_name)
files_list = [file_name for file_name in list_files_in_directory(directory_path) if file_name.endswith("_predictions.json")]
if args.models:
    selected_files = {f"{model_name}_predictions.json" for model_name in args.models}
    files_list = [file_name for file_name in files_list if file_name in selected_files]

for file_name in files_list:
    output_file = file_name.replace("_predictions.json", "_detection_and_localization_evaluation.json")
    output_path = os.path.join(directory_path, output_file)
    if args.skip_existing and os.path.isfile(output_path):
        print(f"Skipping existing result: {output_path}")
        continue

    with open(os.path.join(directory_path, file_name)) as json_file:
        data = json.load(json_file)

    evaluate_with_fiftyone(data, dataset_name, ground_truth_field, float(evaluation_config["iou_threshold"]), bool(evaluation_config["compute_map"]),)

    save_json_to_folder(data, directory_path, output_file)

print(f"Evaluations saved to {directory_path}, contains both localization and object detection!")
