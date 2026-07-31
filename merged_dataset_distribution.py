import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np
import sys

# Configuration
if len(sys.argv) < 3:
    print("Usage: python script_name.py <version> <task>")
    sys.exit(1)


version = str(sys.argv[1])

task = str(sys.argv[2])


MAX_SAMPLES = 800

folder_path = '.'

excluded_datasets = []



if task == "localization" or task == "detection":
    csv_files = glob.glob(os.path.join(folder_path, f'v{version}_{task}_fewshot_labelled*.csv'))
else:
    csv_files = glob.glob(os.path.join(folder_path, f'v{version}_fewshot_labelled*.csv'))




data_frames = []

for file in csv_files:
    file_name_with_extension = os.path.basename(file)
    file_name = os.path.splitext(file_name_with_extension)[0]
    file_name = file_name.replace(f"v{version}_labelled_images_", "")
    print(file_name)

    if file_name not in excluded_datasets:
        df = pd.read_csv(file, delimiter=";", dtype={'image_id': object})
        data_frames.append(df[:MAX_SAMPLES])

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
    combined_df.to_csv(f"v{version}_{task}_fewshot_dataset_gpt_difficulty.csv", index = False)
else:
    combined_df.to_csv(f"v{version}_fewshot_dataset_gpt_difficulty.csv", index = False)

