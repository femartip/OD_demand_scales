import pandas as pd
#import matplotlib.pyplot as plt
import glob
import os
import re
#import numpy as np
import sys

# Configuration
os.makedirs("./outputs/annotations", exist_ok=True)
if len(sys.argv) < 3:
    print("Usage: python script_name.py <version> <task>")
    sys.exit(1)


version = str(sys.argv[1])

task = str(sys.argv[2])


folder_path = './outputs/annotations'

excluded_datasets = []
datasets = ["coco-2017", "voc-2007", "driving"]



if task == "localization" or task == "detection":
    csv_files = glob.glob(os.path.join(folder_path, f'v{version}_{task}_fewshot_labelled*.csv'))
else:
    csv_files = glob.glob(os.path.join(folder_path, f'v{version}_fewshot_labelled*.csv'))




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
        df["dataset"] = dataset
        data_frames.append(df)

combined_df = pd.concat(data_frames, ignore_index=True)

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

if task == "localization" or task == "detection":
    combined_df.to_csv(f"./outputs/annotations/v{version}_{task}_fewshot_dataset_gpt_difficulty.csv", index = False)
else:
    combined_df.to_csv(f"./outputs/annotations/v{version}_fewshot_dataset_gpt_difficulty.csv", index = False)
