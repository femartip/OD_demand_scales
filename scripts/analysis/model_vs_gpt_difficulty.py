import pandas as pd
import re
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS, PROMPT_STRATEGIES, annotations_dir, figures_dir, object_detection_root,)



parser = argparse.ArgumentParser()
parser.add_argument("version")
parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("prompt_strategy", choices=PROMPT_STRATEGIES)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
args = parser.parse_args()

version = str(args.version)
task = args.task
prompt_strategy = args.prompt_strategy
strategy_display = "Zero-shot" if prompt_strategy == "zeroshot" else "Few-shot"
figure_dir = figures_dir(version, args.partition)
figure_dir.mkdir(parents=True, exist_ok=True)
annotation_dir = annotations_dir(version, args.partition)
evaluation_dir = object_detection_root(version, args.partition)


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

df = pd.read_csv(evaluation_dir / "detection_difficulty.csv", dtype={'image_id': object}, usecols=["dataset", "image_id", metric],)

gpt_diff = pd.read_csv(annotation_dir / f"v{version}_{task}_{prompt_strategy}_dataset_gpt_difficulty.csv",dtype={'image_id': object},)


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
plt.title(f'{display_name} Demand-Level Distribution — {strategy_display} ({args.partition})')

plt.savefig(figure_dir / f"v{version}_{task}_{prompt_strategy}_level_distribution.pdf")


grouped = df.groupby(['dataset', 'image_id'], as_index=False).mean()
print(grouped)

final = pd.merge(grouped, gpt_diff, on=['dataset', 'image_id'])
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
plt.title(f'Mean {display_name} Accuracy by Demand Level — {strategy_display} ({args.partition})')
plt.legend()
plt.savefig(figure_dir / f"v{version}_{task}_{prompt_strategy}_accuracy_curve.pdf")

plt.show()
