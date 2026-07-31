import pandas as pd
import re
import matplotlib.pyplot as plt
import sys

# Configuration
if len(sys.argv) < 4:
    print("Usage: python script_name.py <version> <task> <prefix>")
    sys.exit(1)


version = str(sys.argv[1])

task = str(sys.argv[2])

prefix = str(sys.argv[3])



if task == "detection":
    df = pd.read_csv("detection_difficulty.csv",dtype={'image_id': object}, usecols= ["image_id","accuracy","model"])
elif task == "localization":
    df = pd.read_csv("detection_difficulty.csv",dtype={'image_id': object}, usecols= ["image_id","accuracy_detection","model"])
    df = df.rename(columns={"accuracy_detection":"accuracy"})
else:
    sys.exit(1)


if task == "localization" or task == "detection":
    gpt_diff = pd.read_csv(f"v{version}_{task}_fewshot_dataset_gpt_difficulty.csv", dtype={'image_id': object})
else:
    gpt_diff = pd.read_csv(f"v{version}_fewshot_dataset_gpt_difficulty.csv", dtype={'image_id': object})

gpt_diff = gpt_diff[gpt_diff['level'] != 'error']

try:
    gpt_diff["level"] = gpt_diff["level"].apply(lambda x: int(re.search(r'\d+', x).group()))
except TypeError:
    pass

print(gpt_diff["level"].value_counts())

def filter_by_prefix(input_list, prefix):
    return [element for element in input_list if element.startswith(prefix)]


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
        "yolo-nas-torch",
        "zero-shot-detection-transformer-torch",
        "detection-transformer-torch",
        "faster-rcnn-resnet50-fpn-coco-torch",
        "retinanet-resnet50-fpn-coco-torch",
    ]

large_models = ["yolov5x-coco-torch",
        "yolov8x-coco-torch",
        "yolov9e-coco-torch"]

if prefix == "yolo":
    model_list = large_models
else:
    model_list = filter_by_prefix(model_list,prefix)


color_dict = {
    "yolov5l-coco-torch": "#5fa2d5",
    "yolov5m-coco-torch": "#93c5ec",
    "yolov5n-coco-torch": "#cfe2f5",
    "yolov5s-coco-torch": "#aec7e8",
    "yolov5x-coco-torch": "#1f77b4",
    "yolov8l-coco-torch": "#137547",
    "yolov8m-coco-torch": "#2a9134",
    "yolov8n-coco-torch": "#5bba6f",
    "yolov8s-coco-torch": "#3fa34d",
    "yolov8x-coco-torch": "#054a29",
    "yolov9c-coco-torch": "#ffbb78",
    "yolov9e-coco-torch": "#ff7f0e",
    "zero-shot-detection-transformer-torch": "#9467bd",
    "detection-transformer-torch": "#ff9896",
    "faster-rcnn-resnet50-fpn-coco-torch": "#7f7f7f",
    "retinanet-resnet50-fpn-coco-torch": "#c5b0d5",
    "yolo-nas-torch": "#d62728"
}

plt.figure(figsize=(11, 6))

for model in model_list:
    temp_df = df[df["model"] == model]
    del temp_df["model"]
    grouped = temp_df.groupby('image_id', as_index=False).mean()
    final = pd.merge(grouped, gpt_diff, on='image_id')
    final = final.dropna()
    final = final[(final['level'] >= 1) & (final['level'] <= 4)]
    print(final)

    avg_trend = final.groupby('level')['accuracy'].mean().reset_index()

    avg_trend['level'] = avg_trend['level'].astype(str)


    print(avg_trend)

    plt.plot(avg_trend['level'], avg_trend['accuracy'], color=color_dict[model], linewidth=2, label=f'{model}',marker='o')

plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.xlabel('Level')
plt.ylabel('Accuracy')
plt.subplots_adjust(right=0.7)


if task == "localization" or task == "detection":
    plt.savefig(f"v{version}_{prefix}_{task}_fewshot_detection_scatterplot.pdf")
else:
    plt.savefig(f"v{version}_fewshot_detection_scatterplot.pdf")

plt.show()