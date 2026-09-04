import pandas as pd
import glob
import os
import re
import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS,PROMPT_STRATEGIES,annotations_dir,load_manifest,)


parser = argparse.ArgumentParser()
parser.add_argument("version")
parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("prompt_strategy", choices=PROMPT_STRATEGIES)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
args = parser.parse_args()

version = str(args.version)
task = args.task
prompt_strategy = args.prompt_strategy

folder_path = annotations_dir(version, args.partition)
os.makedirs(folder_path, exist_ok=True)

excluded_datasets = []
datasets = ["coco-2017", "voc-2007", "driving"]

csv_files = sorted(glob.glob(os.path.join(folder_path,f'v{version}_{task}_{prompt_strategy}_labelled_images_*.csv',)))

data_frames = []

for file in csv_files:
    file_name_with_extension = os.path.basename(file)
    file_name = os.path.splitext(file_name_with_extension)[0]
    dataset = next((name for name in datasets if f"_images_{name}_" in file_name or file_name.endswith(f"_images_{name}")), None)
    print(file_name)

    if dataset is None:
        raise ValueError(f"Could not determine dataset from {file_name_with_extension}")

    if dataset not in excluded_datasets:
        df = pd.read_csv(file, delimiter=";", dtype={'image_id': object})
        allowed_ids = set(load_manifest(args.partition, dataset)["image_id"])
        outside_partition = set(df["image_id"].astype(str)) - allowed_ids
        if outside_partition:
            raise ValueError(f"{file} contains {len(outside_partition)} images outside {args.partition}")
        df["dataset"] = dataset
        data_frames.append(df)

if not data_frames:
    raise RuntimeError(f"No {task} annotation files found in {folder_path}")

combined_df = pd.concat(data_frames, ignore_index=True)
combined_df["partition"] = args.partition
combined_df["prompt_strategy"] = prompt_strategy

def extract_level(text):
    print(text)
    if not pd.isna(text):
        match = re.search(r'\b\d+\b', text)
        if match:
            return int(match.group())
    return None

if int(version) < 4:
    combined_df['level'] = combined_df['level'].apply(extract_level)

print(combined_df)

combined_df.to_csv(folder_path / f"v{version}_{task}_{prompt_strategy}_dataset_gpt_difficulty.csv",index=False,)
