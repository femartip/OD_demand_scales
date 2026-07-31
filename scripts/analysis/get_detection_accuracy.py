import pandas as pd
import json
import os


def get_all_file_paths(folder):
    file_paths = []

    for root, _, files in os.walk(folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)

    return file_paths


datasets = ["coco-2017","voc-2007","driving"]
all_paths = {}
for dataset in datasets:
    print(dataset)
    folder = f'./detections_{dataset}'  # Replace with your folder path
    all_paths[dataset] = get_all_file_paths(folder)

print(all_paths)

df = pd.DataFrame()
for dataset in all_paths:
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
                
            temp_df["model"] = os.path.basename(file_path).replace("_results.json", "")
            temp_df["dataset"] = dataset
            df = pd.concat([df, temp_df])

df["accuracy"] = df["tp"] / (df["tp"] + df["fn"] + df["fp"])
df["accuracy_detection"] = df["tp_detection"] / (df["tp_detection"] + df["fn_detection"] + df["fp_detection"])

df = df.fillna(1)


df.to_csv("detection_difficulty.csv", index = False)