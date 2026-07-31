import pandas as pd
import json
import os
import sys

if len(sys.argv) != 2:
    print("Usage: python script_name.py <dataset>")
    sys.exit(1)

dataset = str(sys.argv[1])
file_path = f'./outputs/object_detection/{dataset}/yolov5l-coco-torch_object_detection_evaluation.json'  # Replace with your file path
with open(file_path, 'r') as file:
    data = json.load(file)

file_names = []
for i,image in enumerate(data['samples']):
    file_name_with_extension = os.path.basename(image['filepath'])

    file_name = os.path.splitext(file_name_with_extension)[0]

    file_names.append(file_name)


df = pd.DataFrame({"image_id": file_names})

os.makedirs("./outputs/object_detection", exist_ok=True)
df.to_csv(f"./outputs/object_detection/images_experiment_{dataset}.csv", index=False)

