import pandas as pd
import re
import matplotlib.pyplot as plt
import sys
import os

# Configuration
os.makedirs("./outputs/figures", exist_ok=True)
if len(sys.argv) < 3:
    print("Usage: python script_name.py <version> <task>")
    sys.exit(1)


version = str(sys.argv[1])

task = str(sys.argv[2])


if task == "detection":
    metric = "accuracy"
    display_name = "Object Detection"
    color = "#FF6859"
elif task == "localization":
    metric = "accuracy_detection"
    display_name = "Localization"
    color = "skyblue"
else:
    print("Task must be 'localization' or 'detection'")
    sys.exit(1)

df = pd.read_csv("./outputs/object_detection/detection_difficulty.csv",dtype={'image_id': object}, usecols= ["image_id", metric])

if task == "localization" or task == "detection":
    gpt_diff = pd.read_csv(f"./outputs/annotations/v{version}_{task}_fewshot_dataset_gpt_difficulty.csv", dtype={'image_id': object})
else:
    gpt_diff = pd.read_csv(f"./outputs/annotations/v{version}_fewshot_dataset_gpt_difficulty.csv", dtype={'image_id': object})


gpt_diff = gpt_diff[gpt_diff['level'] != 'error']


try:
    gpt_diff["level"] = gpt_diff["level"].apply(lambda x: int(re.search(r'\d+', x).group()))
except TypeError:
    pass

level_counts = gpt_diff["level"].value_counts().sort_index()
print(level_counts)

level_counts.plot(kind='bar', color = "#31005c")

plt.xticks(rotation=0)
plt.xlabel('Level')
plt.ylabel('Count')
plt.title(f'{display_name} Demand-Level Distribution')

plt.savefig(f"./outputs/figures/v{version}_{task}_level_distribution.pdf")


grouped = df.groupby('image_id', as_index=False).mean()
print(grouped)

final = pd.merge(grouped, gpt_diff, on='image_id')
final = final.dropna()
final = final[(final['level'] >= 1) & (final['level'] <= 5)]


print(final)

plt.figure(figsize=(9, 6))

avg_trend = final.groupby('level')[metric].mean().reset_index()

avg_trend['level'] = avg_trend['level'].astype(str)


print(avg_trend)

plt.plot(avg_trend['level'], avg_trend[metric], color=color, linewidth=2, label=f'{display_name} Accuracy',marker='o')





plt.xlabel(f'{display_name} demand level')
plt.ylabel(f'{display_name} accuracy')
plt.title(f'Mean {display_name} Accuracy by {display_name} Demand Level')
plt.legend()
if task == "localization" or task == "detection":
    plt.savefig(f"./outputs/figures/v{version}_{task}_fewshot_accuracy_curve.pdf")
else:
    plt.savefig(f"./outputs/figures/v{version}_fewshot_detection_scatterplot.pdf")

plt.show()