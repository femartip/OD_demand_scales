import pandas as pd
import json
import os
import argparse
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.experiment import (PARTITIONS, load_experiment_config, load_manifest, object_detection_root, split_config_for_version,)


def get_all_file_paths(folder):
    file_paths = []

    for root, _, files in os.walk(folder):
        for file in files:
            if file.endswith("_detection_and_localization_evaluation.json"):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)

    return file_paths


parser = argparse.ArgumentParser()
parser.add_argument("--version",type=int,required=True,help="Experiment version")
parser.add_argument("--partition",required=True,choices=PARTITIONS,help="Experimental partition to aggregate")
args = parser.parse_args()

output_root = object_detection_root(args.version, args.partition)
experiment_config = load_experiment_config(args.version)
datasets = list(split_config_for_version(args.version)["datasets"])
all_paths = {}
for dataset in datasets:
    print(dataset)
    folder = output_root / dataset
    all_paths[dataset] = get_all_file_paths(folder)

print(all_paths)

df = pd.DataFrame()
for dataset in all_paths:
    expected_image_ids = set(load_manifest(args.partition, dataset, split_config_for_version(args.version))["image_id"])
    for file_path in all_paths[dataset]:
        with open(file_path, 'r') as file:
            data = json.load(file)
            temp_df = pd.DataFrame()
            for i,image in enumerate(data['samples']):
                image_name_with_extension = os.path.basename(image['filepath'])
                temp_df.loc[i, "image_id"] = os.path.splitext(image_name_with_extension)[0]
                temp_df.loc[i, "tp"] = image['eval_coco_tp']
                temp_df.loc[i, "fp"] = image['eval_coco_fp']
                temp_df.loc[i, "fn"] = image['eval_coco_fn']
                temp_df.loc[i, "tp_detection"] = image['detection_tp']
                temp_df.loc[i, "fp_detection"] = image['detection_fp']
                temp_df.loc[i, "fn_detection"] = image['detection_fn']

            actual_image_ids = set(temp_df["image_id"].astype(str))
            if actual_image_ids != expected_image_ids or len(temp_df) != len(expected_image_ids):
                raise ValueError(f"{file_path} does not exactly match the {args.partition} manifest for {dataset}")
                
            temp_df["model"] = os.path.basename(file_path).replace("_detection_and_localization_evaluation.json", "")
            temp_df["dataset"] = dataset
            temp_df["version"] = args.version
            temp_df["partition"] = args.partition
            df = pd.concat([df, temp_df])

if df.empty:
    raise RuntimeError(f"No evaluation files found under {output_root}")

duplicate_key = ["dataset", "image_id", "model"]
if df.duplicated(duplicate_key).any():
    raise ValueError(f"Duplicate evaluation rows found for {duplicate_key}")

df["accuracy"] = df["tp"] / (df["tp"] + df["fn"] + df["fp"])
df["accuracy_detection"] = df["tp_detection"] / (df["tp_detection"] + df["fn_detection"] + df["fp_detection"])

df = df.fillna(float(experiment_config["evaluation"]["empty_union_score"]))


os.makedirs(output_root, exist_ok=True)
output_path = output_root / "detection_difficulty.csv"
df.to_csv(output_path, index = False)
print(f"Merged all object detection results to {output_path}")
