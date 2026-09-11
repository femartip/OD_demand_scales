import argparse
from itertools import combinations
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS, PROMPT_STRATEGIES, REPO_ROOT, annotations_dir, baselines_dir, load_manifest, load_split_config, object_detection_root,split_config_for_version)


def display(name):
    return f"v{name}" if str(name).isdigit() else str(name)


def level_path(name, partition, prompt_strategy):
    # A numeric name is a rubric version; anything else is a baseline from scripts/baselines.
    if name.isdigit():
        return annotations_dir(name, partition) / f"v{name}_detection_{prompt_strategy}_dataset_gpt_difficulty.csv"
    return baselines_dir(partition) / f"{name}.csv"


def fit_mapping(levels, scores):
    return np.column_stack([
        IsotonicRegression(increasing=False, out_of_bounds="clip")
        .fit(levels, scores[:, m]).predict(np.arange(1, 6))
        for m in range(scores.shape[1])
    ])


def auc(levels, failures):
    # Pair-count AUROC for five ordinal scores, including half credit for ties.
    # Leading dimensions allow the same calculation on bootstrap samples.
    positive = np.stack([np.sum((levels == l) & failures, axis=-1) for l in range(1, 6)], axis=-1)
    negative = np.stack([np.sum((levels == l) & ~failures, axis=-1) for l in range(1, 6)], axis=-1)
    pairs = positive.sum(axis=-1) * negative.sum(axis=-1)
    wins = (positive * (negative.cumsum(axis=-1) - negative / 2)).sum(axis=-1)
    return np.divide(wins, pairs, out=np.full(np.shape(pairs), np.nan), where=pairs > 0)


def skill(mse, baseline_mse):
    return 1 - np.divide(mse, baseline_mse, out=np.full(np.shape(mse), np.nan), where=baseline_mse > 0)


def interval(values):
    values = values[np.isfinite(values)]
    return np.quantile(values, [0.025, 0.975]) if len(values) else (np.nan, np.nan)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("versions", nargs="+", help="Rubric versions and/or baseline names")
parser.add_argument("task", choices=("detection", "localization"))
parser.add_argument("prompt_strategy", choices=PROMPT_STRATEGIES)
parser.add_argument("--partition", required=True, choices=PARTITIONS)
parser.add_argument("--failure-thresholds", nargs="+", type=float, default=[0.25, 0.5, 0.75])
parser.add_argument("--bootstrap", type=int, default=1000)
parser.add_argument("--mappings", type=Path, help="Frozen development mappings.csv; no refitting")
parser.add_argument("--output-dir", type=Path)
args = parser.parse_args()
if args.partition == "locked_confirmation" and args.mappings is None:
    parser.error("locked_confirmation requires --mappings from development")

metric = f"{args.task}_quality"
run_name = "_".join(f"v{v}" if v.isdigit() else v for v in args.versions) + f"_{args.task}_{args.prompt_strategy}"
output_dir = args.output_dir or REPO_ROOT / "outputs/analysis" / args.partition / run_name
output_dir.mkdir(parents=True, exist_ok=True)
numeric = [v for v in args.versions if v.isdigit()]
config = split_config_for_version(numeric[0]) if numeric else load_split_config()
images = load_manifest(args.partition, split_config=config)
if args.partition == "calibration":
    images["cohort"] = np.where(images["selection_group"].eq("random"), "random", "properties")
else:
    images["cohort"] = "random"

for version in args.versions:
    # Detection and localization share the same image annotation.
    annotations = pd.read_csv(level_path(version, args.partition, args.prompt_strategy), dtype={"image_id": str})
    annotations["level"] = pd.to_numeric(annotations["level"], errors="coerce")
    annotations = annotations[annotations["level"].isin(range(1, 6))]
    print(f"{version}: {len(annotations)} valid annotations")
    annotations = annotations[["dataset", "image_id", "level"]].rename(columns={"level": version})
    images = images.merge(annotations, on=["dataset", "image_id"], validate="one_to_one")

images = images.sort_values(["dataset", "image_id"]).reset_index(drop=True)
print(f"Shared images: {len(images)}; cohorts: {images['cohort'].value_counts().to_dict()}")
evaluation = pd.read_csv(object_detection_root(args.partition) / "detection_difficulty.csv", dtype={"image_id": str})
scores = evaluation.pivot(index=["dataset", "image_id"], columns="model", values=metric)
scores = scores.loc[pd.MultiIndex.from_frame(images[["dataset", "image_id"]])]
assert scores.notna().all().all(), "Every shared image must have a score for every detector"
models = scores.columns
y = scores.to_numpy()
random_ids = np.flatnonzero(images["cohort"].eq("random"))
frozen = pd.read_csv(args.mappings) if args.mappings else None
if frozen is not None:
    assert frozen["metric"].eq(metric).all() and frozen["prompt_strategy"].eq(args.prompt_strategy).all()
    folds = []
else:
    folds = [(random_ids[train], random_ids[test]) for train, test in
             KFold(n_splits=5, shuffle=True, random_state=config["seed"]).split(random_ids)]

rng = np.random.default_rng(config["seed"])
cohorts = {name: np.flatnonzero(images["cohort"].eq(name)) for name in images["cohort"].unique()}
resamples = {name: rng.integers(len(ids), size=(args.bootstrap, len(ids))) for name, ids in cohorts.items()}
summary, per_model, failures_table, predictions, mappings, level_table = [], [], [], [], [], []
estimates, bootstraps = {}, {}

for version in args.versions:
    levels = images[version].to_numpy(dtype=int)
    if frozen is None:
        mapping = fit_mapping(levels[random_ids], y[random_ids])
        baseline_mean = y[random_ids].mean(axis=0)
        support = np.bincount(levels[random_ids], minlength=6)[1:]
    else:
        saved = frozen[frozen["version"].eq(version)]
        mapping = saved.pivot(index="level", columns="model", values="prediction").loc[range(1, 6), models].to_numpy()
        baseline_mean = saved.groupby("model")["baseline"].first().loc[models].to_numpy()
        support = saved.groupby("level")["n_train"].first().loc[range(1, 6)].to_numpy()
    predicted = mapping[levels - 1].copy()
    baseline = np.broadcast_to(baseline_mean, y.shape).copy()
    fold_ids = np.full(len(images), -1)
    n_train_level = support[levels - 1].copy()
    for fold, (train, test) in enumerate(folds):
        predicted[test] = fit_mapping(levels[train], y[train])[levels[test] - 1]
        baseline[test] = y[train].mean(axis=0)
        fold_ids[test] = fold
        n_train_level[test] = np.bincount(levels[train], minlength=6)[levels[test]]
    for m, model in enumerate(models):
        for level in range(1, 6):
            mappings.append(dict(version=version, model=model, level=level, prediction=mapping[level - 1, m],
                                 baseline=baseline_mean[m], n_train=support[level - 1], metric=metric,
                                 prompt_strategy=args.prompt_strategy))
        frame = images[["dataset", "image_id", "cohort"]].copy()
        frame = frame.assign(version=version, model=model, level=levels, score=y[:, m],
                             prediction=predicted[:, m], baseline=baseline[:, m], fold=fold_ids, n_train_level=n_train_level)
        predictions.append(frame)

    for cohort, ids in cohorts.items():
        sample = resamples[cohort]
        level = levels[ids]
        observed, prediction, reference = y[ids], predicted[ids], baseline[ids]
        errors = (observed - prediction) ** 2
        baseline_errors = (observed - reference) ** 2
        mse, baseline_mse = errors.mean(axis=1), baseline_errors.mean(axis=1)
        boot_mse, boot_baseline = mse[sample].mean(axis=1), baseline_mse[sample].mean(axis=1)
        values = dict(rmse=np.sqrt(mse.mean()), mse_skill=skill(mse.mean(), baseline_mse.mean()).item())
        boots = dict(rmse=np.sqrt(boot_mse), mse_skill=skill(boot_mse, boot_baseline))
        row = dict(version=version, cohort=cohort, n_images=len(ids), n_models=len(models),
                   baseline_rmse=np.sqrt(baseline_mse.mean()), mae=np.abs(observed - prediction).mean())
        for m, model in enumerate(models):
            per_model.append(dict(version=version, cohort=cohort, model=model, n_images=len(ids),
                                  rmse=np.sqrt(errors[:, m].mean()), baseline_rmse=np.sqrt(baseline_errors[:, m].mean()),
                                  mse_skill=skill(errors[:, m].mean(), baseline_errors[:, m].mean()).item(),
                                  mae=np.abs(observed[:, m] - prediction[:, m]).mean()))
        for threshold in args.failure_thresholds:
            aucs, auc_boot = [], []
            for m, model in enumerate(models):
                failed = observed[:, m] < threshold
                value = auc(level, failed).item()
                boot = auc(level[sample], failed[sample])
                low, high = interval(boot)
                failures_table.append(dict(version=version, cohort=cohort, model=model, threshold=threshold,
                                           n_images=len(ids), n_failures=failed.sum(), prevalence=failed.mean(),
                                           auroc=value, ci_low=low, ci_high=high))
                if np.isfinite(value):
                    aucs.append(value)
                    auc_boot.append(boot)
            key = f"auroc_{threshold:g}"
            values[key] = np.mean(aucs) if aucs else np.nan
            boots[key] = np.mean(auc_boot, axis=0) if aucs else np.full(args.bootstrap, np.nan)
            row[f"{key}_n_models"] = len(aucs)
        for key, value in values.items():
            low, high = interval(boots[key])
            row.update({key: value, f"{key}_ci_low": low, f"{key}_ci_high": high})
            estimates[version, cohort, key] = value
            bootstraps[version, cohort, key] = boots[key]
        summary.append(row)
        for l in range(1, 6):
            scores_at_level = observed[level == l].mean(axis=1)
            level_table.append(dict(version=version, cohort=cohort, level=l, n_images=len(scores_at_level),
                                    mean=np.mean(scores_at_level) if len(scores_at_level) else np.nan))
        print(f"{version} / {cohort}: RMSE {values['rmse']:.4f}, MSE skill {values['mse_skill']:.3f}")

comparisons = []
for first, second in combinations(args.versions, 2):
    for cohort in cohorts:
        for key in ["rmse", "mse_skill"] + [f"auroc_{t:g}" for t in args.failure_thresholds]:
            difference = estimates[second, cohort, key] - estimates[first, cohort, key]
            boot = bootstraps[second, cohort, key] - bootstraps[first, cohort, key]
            low, high = interval(boot)
            comparisons.append(dict(version_a=first, version_b=second, cohort=cohort, metric=key,
                                    difference_b_minus_a=difference, ci_low=low, ci_high=high,
                                    n_valid_bootstrap=np.isfinite(boot).sum()))

summary = pd.DataFrame(summary)
predictions = pd.concat(predictions, ignore_index=True)
for name, table in [("summary", summary), ("per_model", pd.DataFrame(per_model)),
                    ("failure_detection", pd.DataFrame(failures_table)),
                    ("comparisons", pd.DataFrame(comparisons, columns=["version_a", "version_b", "cohort", "metric",
                                                                      "difference_b_minus_a", "ci_low", "ci_high", "n_valid_bootstrap"])),
                    ("predictions", predictions), ("mappings", pd.DataFrame(mappings)), ("levels", pd.DataFrame(level_table))]:
    table.to_csv(output_dir / f"{name}.csv", index=False)

for cohort in cohorts:
    data = summary[summary["cohort"].eq(cohort)]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - 0.2, data["baseline_rmse"], width=0.4, color="lightgray", label="Detector-average baseline")
    ax.bar(x + 0.2, data["rmse"], width=0.4, color="#31005c", label="Rubric")
    ax.set(xticks=x, xticklabels=[display(v) for v in data["version"]], ylabel=f"{args.task.capitalize()} S RMSE", title=cohort)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{cohort}_rmse.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for _, row in data.iterrows():
        keys = [f"auroc_{t:g}" for t in args.failure_thresholds]
        ax.plot(args.failure_thresholds, [row[k] for k in keys], marker="o", label=display(row['version']))
        ax.fill_between(args.failure_thresholds, [row[f"{k}_ci_low"] for k in keys],
                        [row[f"{k}_ci_high"] for k in keys], alpha=0.1)
    ax.axhline(0.5, color="gray", linestyle="--")
    ax.set(xlabel="Failure definition: S < threshold", ylabel="Mean per-detector AUROC", ylim=(0, 1), title=cohort)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{cohort}_auroc.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(args.versions), figsize=(4 * len(args.versions), 4), squeeze=False, sharey=True)
    for ax, version in zip(axes[0], args.versions):
        ids = cohorts[cohort]
        level = images.loc[ids, version].to_numpy()
        present = np.unique(level).astype(int)
        ax.boxplot([y[ids][level == l].mean(axis=1) for l in present], positions=present, showfliers=False)
        ax.set(xticks=range(1, 6), xticklabels=[f"{l}\nn={(level == l).sum()}" for l in range(1, 6)],
               xlabel="Rubric level", title=f"{display(version)} / {cohort}", ylim=(0, 1))
    axes[0, 0].set_ylabel(f"Mean {args.task} S across detectors")
    fig.tight_layout()
    fig.savefig(output_dir / f"{cohort}_levels.pdf")
    plt.close(fig)

print(f"Saved evaluation to {output_dir}")
print("Intervals use paired image bootstraps conditional on the fitted predictions; calibration results are developmental.")
