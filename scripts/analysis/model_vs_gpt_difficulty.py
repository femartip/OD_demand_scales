import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS, PROMPT_STRATEGIES, annotations_dir, figures_dir, object_detection_root, load_manifest, split_config_for_version,)



parser = argparse.ArgumentParser()
parser.add_argument("version")
parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("prompt_strategy", choices=PROMPT_STRATEGIES)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--columns", nargs="+", default=["level"], help="Annotation columns, or dimensions for all six levels")
parser.add_argument("--annotator", help="Select one annotator from merged annotations")
args = parser.parse_args()

version = str(args.version)
task = args.task
prompt_strategy = args.prompt_strategy
strategy_display = "Zero-shot" if prompt_strategy == "zeroshot" else "Few-shot"
figure_dir = figures_dir(version, args.partition)
figure_dir.mkdir(parents=True, exist_ok=True)
annotation_dir = annotations_dir(version, args.partition)
evaluation_dir = object_detection_root(args.partition)


if task == "detection":
    metric = "detection_quality"
    display_name = "Object Detection"
    color = "#FF6859"
elif task == "localization":
    metric = "localization_quality"
    display_name = "Localization"
    color = "skyblue"
else:
    print("Task must be 'localization' or 'detection'")
    sys.exit(1)

df = pd.read_csv(evaluation_dir / "detection_difficulty.csv", dtype={'image_id': object}, usecols=["dataset", "image_id", metric],)

gpt_diff = pd.read_csv(annotation_dir / f"v{version}_{task}_{prompt_strategy}_dataset_gpt_difficulty.csv",dtype={'image_id': object},)


if args.annotator:
    gpt_diff = gpt_diff[gpt_diff["annotator"].eq(args.annotator)]
assert not gpt_diff.duplicated(["dataset", "image_id"]).any(), "Select one annotation per image with --annotator"
columns = args.columns
if columns == ["dimensions"]:
    columns = [f"{name}_level" for name in ("target_abundance", "target_scale", "target_visibility", "instance_separation", "image_degradation", "appearance_atypicality")]
if columns != ["level"]:
    gpt_diff[columns] = gpt_diff[columns].apply(pd.to_numeric, errors="coerce")
    manifest = load_manifest(args.partition, split_config=split_config_for_version(version))
    manifest["cohort"] = np.where(manifest["selection_group"].eq("random"), "random", "properties") if args.partition == "calibration" else "random"
    gpt_diff = gpt_diff.merge(manifest[["dataset", "image_id", "cohort"]], on=["dataset", "image_id"], validate="one_to_one")

if columns != ["level"]:
    grouped = df.groupby(["dataset", "image_id"], as_index=False)[metric].mean()
    final = gpt_diff.merge(grouped, on=["dataset", "image_id"], validate="one_to_one")
    layout = (int(np.ceil(len(columns) / 3)), min(3, len(columns)))
    for cohort, frame in final.groupby("cohort"):
        suffix = "_".join(args.columns)
        for plot in ("distribution", "performance"):
            fig, axes = plt.subplots(*layout, figsize=(5 * layout[1], 4 * layout[0]), squeeze=False)
            for ax, column in zip(axes.flat, columns):
                data = frame[frame[column].isin(range(1, 6))]
                stats = data.groupby(column)[metric].agg(["mean", "sem", "count"]).reindex(range(1, 6))
                count = stats["count"].fillna(0).astype(int)
                if plot == "distribution":
                    ax.bar(range(1, 6), count, color="#31005c")
                    ax.set_ylabel("Images")
                else:
                    ax.errorbar(range(1, 6), stats["mean"], yerr=1.96 * stats["sem"], marker="o", color=color, capsize=3)
                    ax.set(ylabel=f"Mean {task} S", ylim=(0, 1))
                ax.set(title=column.removesuffix("_level").replace("_", " ").title(),
                       xticks=range(1, 6), xticklabels=[f"{level}\nn={count.loc[level]}" for level in range(1, 6)])
            for ax in list(axes.flat)[len(columns):]:
                ax.set_visible(False)
            fig.suptitle(f"v{version} / {cohort} / {plot}" + (" (approx. 95% mean intervals)" if plot == "performance" else ""))
            fig.tight_layout()
            fig.savefig(figure_dir / f"v{version}_{task}_{prompt_strategy}_{cohort}_{suffix}_{plot}.pdf")
            plt.close(fig)
        fractions = [column for column in frame if column.endswith("_severe_fraction")]
        if fractions:
            fig, axes = plt.subplots(2, 3, figsize=(15, 8), squeeze=False)
            for ax, column in zip(axes.flat, fractions):
                data = frame[[column, metric]].dropna()
                ax.scatter(data[column], data[metric], alpha=0.15, s=12, color=color)
                bins = pd.cut(data[column], [0, 0.25, 0.5, 0.75, 1], include_lowest=True)
                trend = data.groupby(bins, observed=True)[[column, metric]].mean()
                ax.plot(trend[column], trend[metric], 'o-', color="#31005c")
                ax.set(xlabel="Fraction at levels 4–5", ylabel=f"Mean {task} S", xlim=(0, 1), ylim=(0, 1),
                       title=column.removesuffix("_severe_fraction").replace("_", " ").title())
            for ax in list(axes.flat)[len(fractions):]:
                ax.set_visible(False)
            fig.suptitle(f"v{version} / {cohort}")
            fig.tight_layout()
            fig.savefig(figure_dir / f"v{version}_{task}_{prompt_strategy}_{cohort}_{suffix}_severe_fractions.pdf")
            plt.close(fig)
    sys.exit(0)

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

plt.plot(avg_trend['level'], avg_trend[metric], color=color, linewidth=2, label=f'{display_name} Quality',marker='o')





plt.xlabel(f'{display_name} demand level')
plt.ylabel(f'{display_name} quality')
plt.title(f'Mean {display_name} Quality by Demand Level — {strategy_display} ({args.partition})')
plt.legend()
plt.savefig(figure_dir / f"v{version}_{task}_{prompt_strategy}_accuracy_curve.pdf")

plt.show()
