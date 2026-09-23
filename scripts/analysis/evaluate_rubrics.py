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
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata

from ionescu import fit_ionescu

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.experiment import (PARTITIONS, PROMPT_STRATEGIES, REPO_ROOT, annotations_dir, baselines_dir, load_manifest, load_split_config, object_detection_root,split_config_for_version)


def display(name):
    return f"v{name}".replace(":", "\n").replace("-", " ") if str(name).split(":")[0].isdigit() else str(name)


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


DIMENSIONS = ("target_abundance", "target_scale", "target_visibility", "instance_separation", "image_degradation", "appearance_atypicality")


def ridge_score(estimator, features, scores):
    return -np.mean((np.clip(estimator.predict(features), 0, 1) - scores) ** 2)


def fit_ridge(features, scores, seed):
    coefficients, intercepts, alphas = [], [], []
    inner = KFold(n_splits=5, shuffle=True, random_state=seed)
    for m in range(scores.shape[1]):
        search = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                              {"ridge__alpha": [0.1, 1, 10, 100, 1000]},
                              scoring=ridge_score, cv=inner)
        search.fit(features, scores[:, m])
        scaler, ridge = search.best_estimator_.steps[0][1], search.best_estimator_.steps[1][1]
        # Save coefficients in original feature units, including the scaler's offset.
        coefficient = ridge.coef_ / scaler.scale_
        coefficients.append(coefficient)
        intercepts.append(ridge.intercept_ - scaler.mean_ @ coefficient)
        alphas.append(ridge.alpha)
    return np.array(coefficients).T, np.array(intercepts), np.array(alphas)


def risk_auc(risk, failures):
    positive = failures.sum(axis=-1)
    pairs = positive * (failures.shape[-1] - positive)
    wins = (rankdata(risk, axis=-1) * failures).sum(axis=-1) - positive * (positive + 1) / 2
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
parser.add_argument("--predictor", choices=("isotonic", "ridge"), default="isotonic")
parser.add_argument("--features", nargs="+", choices=("levels", "levels-and-fractions", "abundance-and-scale", *DIMENSIONS), default=["levels"])
parser.add_argument("--ablations", action="store_true", help="Add leave-one-dimension-out and abundance/scale-plus-one ridge assessors")
parser.add_argument("--auc-score", choices=("level", "prediction"), help="Default: level for isotonic runs, prediction for ridge comparisons")
parser.add_argument("--annotator", help="Select one annotator from merged annotation tables")
args = parser.parse_args()
if args.partition == "locked_confirmation" and args.mappings is None:
    parser.error("locked_confirmation requires --mappings from development")

if args.ablations and args.predictor != "ridge":
    parser.error("--ablations requires --predictor ridge")

auc_score = args.auc_score or ("prediction" if args.predictor == "ridge" else "level")
if args.predictor == "ridge" and auc_score == "level":
    parser.error("ridge comparisons require --auc-score prediction")
metric = f"{args.task}_quality"
run_name = "_".join(f"v{v}" if v.isdigit() else v.replace(":", "-") for v in args.versions) + f"_{args.task}_{args.prompt_strategy}"
if args.predictor != "isotonic" or args.features != ["levels"] or auc_score != "level":
    run_name += f"_{args.predictor}_{'_'.join(args.features)}_{auc_score}"
if args.ablations:
    run_name += "_ablations"
output_dir = REPO_ROOT / "outputs/analysis" / args.partition / run_name
output_dir.mkdir(parents=True, exist_ok=True)
numeric = [v for v in args.versions if v.isdigit()]
config = split_config_for_version(numeric[0]) if numeric else load_split_config()
images = load_manifest(args.partition, split_config=config)
if args.partition == "calibration":
    images["cohort"] = np.where(images["selection_group"].eq("random"), "random", "properties")
else:
    images["cohort"] = "random"

candidates, coverage = {}, []
for entry in args.versions:
    version, _, requested = entry.partition(":")
    if version.isdigit():
        version, requested = entry, ""
    # Detection and localization share the same image annotation.
    if version == "ionescu":
        if args.predictor != "ridge" or requested not in ("", "predicted_score"):
            parser.error("ionescu requires --predictor ridge and uses predicted_score")
        with np.load(baselines_dir(args.partition) / "ionescu_features.npz") as cached:
            annotations = pd.DataFrame({key: cached[key] for key in ("dataset", "image_id")})
            embeddings = pd.DataFrame(cached["features"], index=pd.MultiIndex.from_frame(annotations))
        # The scalar is generated inside each training fold, never read from ionescu.csv.
        annotations["predicted_score"] = 0.
    else:
        annotations = pd.read_csv(level_path(version, args.partition, args.prompt_strategy), dtype={"image_id": str})
    if args.annotator and version != "ionescu":
        annotations = annotations[annotations["annotator"].eq(args.annotator)]
    assert not annotations.duplicated(["dataset", "image_id"]).any(), "Select one annotation per image with --annotator"
    scalar = "level" in annotations
    baseline = not version.split(":")[0].isdigit()
    selections = ["level"] if scalar else list(dict.fromkeys(args.features))
    if baseline:
        # Default to every numeric feature the baseline wrote; name:feature picks a subset.
        available = [c for c in annotations.columns if c not in ("dataset", "image_id", "partition")
                     and pd.api.types.is_numeric_dtype(pd.to_numeric(annotations[c], errors="coerce"))]
        selections = [requested] if requested else [",".join(available)]
    if args.ablations and not scalar and not baseline:
        selections = list(dict.fromkeys([*selections, "levels", "abundance-and-scale",
                                        *[f"without-{name}" for name in DIMENSIONS],
                                        *[f"abundance-and-scale-plus-{name}" for name in DIMENSIONS[2:]]]))
    selected = []
    for selection in selections:
        if baseline:
            columns, kind = selection.split(","), args.predictor
        elif scalar:
            columns, kind = ["level"], "isotonic"
        elif selection in DIMENSIONS:
            columns, kind = [f"{selection}_level"], args.predictor
        else:
            columns = [f"{name}_level" for name in DIMENSIONS]
            if selection.startswith("without-"):
                columns.remove(f"{selection.removeprefix('without-')}_level")
            elif selection.startswith("abundance-and-scale-plus-"):
                columns = columns[:2] + [f"{selection.removeprefix('abundance-and-scale-plus-')}_level"]
            elif selection == "abundance-and-scale":
                columns = columns[:2]
            elif selection == "levels-and-fractions":
                columns += [f"{name}_severe_fraction" for name in DIMENSIONS[1:]]
            kind = args.predictor
            if kind != "ridge":
                parser.error("Multiple dimensions require --predictor ridge; select one dimension for isotonic")
        name = version if (scalar or baseline) else f"{version}:{selection}"
        candidates[name] = dict(version=version, columns=columns, predictor=kind,
                                inputs=[version if scalar else f"{version}:{column}" for column in columns])
        if baseline:
            candidates[name]["inputs"] = [f"{version}:{column}" for column in columns]
        selected.extend(columns)
    selected = list(dict.fromkeys(selected))
    annotations[selected] = annotations[selected].apply(pd.to_numeric, errors="coerce")
    valid = pd.Series(True, index=annotations.index)
    for column in selected:
        if column.endswith("_fraction"):
            valid &= annotations[column].between(0, 1)
        elif column == "level" or column.endswith("_level"):
            valid &= annotations[column].isin(range(1, 6))
        else:
            valid &= np.isfinite(annotations[column])
    coverage.append(dict(version=version, n_manifest=len(load_manifest(args.partition, split_config=config)),
                         n_annotations=len(annotations), n_valid=int(valid.sum()), n_excluded=int((~valid).sum())))
    print(f"{version}: {valid.sum()} usable / {len(annotations)} annotations")
    annotations = annotations.loc[valid, ["dataset", "image_id", *selected]]
    annotations = annotations.rename(columns={column: version if (scalar and not baseline) else f"{version}:{column}" for column in selected})
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
frozen = pd.read_csv(args.mappings, dtype={"version": str}) if args.mappings else None
if frozen is not None:
    assert frozen["metric"].eq(metric).all() and frozen["prompt_strategy"].eq(args.prompt_strategy).all()
    folds = []
else:
    folds = [(random_ids[train], random_ids[test]) for train, test in
             KFold(n_splits=5, shuffle=True, random_state=config["seed"]).split(random_ids)]

rng = np.random.default_rng(config["seed"])
cohorts = {name: np.flatnonzero(images["cohort"].eq(name)) for name in images["cohort"].unique()}
resamples = {name: rng.integers(len(ids), size=(args.bootstrap, len(ids))) for name, ids in cohorts.items()}
summary, per_model, failures_table, predictions, mappings, level_table, selected_alphas = [], [], [], [], [], [], []
estimates, bootstraps = {}, {}

for version, candidate in candidates.items():
    x = images[candidate["inputs"]].to_numpy(dtype=float)
    is_ridge = candidate["predictor"] == "ridge"
    is_ionescu = version == "ionescu"
    if is_ionescu:
        features = embeddings.loc[pd.MultiIndex.from_frame(images[["dataset", "image_id"]])].to_numpy(dtype=float)
    levels = np.full(len(images), np.nan) if is_ridge else x[:, 0].astype(int)
    if frozen is None:
        baseline_mean = y[random_ids].mean(axis=0)
        if is_ionescu:
            stage1, coefficients, intercepts, alphas = fit_ionescu(features[random_ids], y[random_ids], config["seed"])
            stage1_coef, stage1_intercept, stage1_alpha = stage1.coef_, stage1.intercept_, stage1.alpha_
        elif is_ridge:
            coefficients, intercepts, alphas = fit_ridge(x[random_ids], y[random_ids], config["seed"])
        else:
            mapping = fit_mapping(levels[random_ids], y[random_ids])
            support = np.bincount(levels[random_ids], minlength=6)[1:]
    else:
        saved = frozen[frozen["version"].eq(version)]
        assert not saved.empty, f"No frozen assessor for {version}"
        if "predictor" in saved:
            assert saved["predictor"].eq(candidate["predictor"]).all()
        baseline_mean = saved.groupby("model")["baseline"].first().loc[models].to_numpy()
        if is_ridge:
            assert set(saved["feature"]) == set(candidate["columns"])
            coefficients = saved.pivot(index="feature", columns="model", values="coefficient").loc[candidate["columns"], models].to_numpy()
            intercepts = saved.groupby("model")["intercept"].first().loc[models].to_numpy()
            alphas = saved.groupby("model")["alpha"].first().loc[models].to_numpy()
        else:
            mapping = saved.pivot(index="level", columns="model", values="prediction").loc[range(1, 6), models].to_numpy()
            support = saved.groupby("level")["n_train"].first().loc[range(1, 6)].to_numpy()
    if is_ionescu:
        if frozen is not None:
            with np.load(args.mappings.parent / "ionescu_stage1.npz") as stage1:
                stage1_coef, stage1_intercept, stage1_alpha = (stage1[key] for key in ("coefficient", "intercept", "alpha"))
        x = (features @ stage1_coef + stage1_intercept)[:, None]
        np.savez(output_dir / "ionescu_stage1.npz", coefficient=stage1_coef,
                 intercept=stage1_intercept, alpha=stage1_alpha)
    predicted = np.clip(x @ coefficients + intercepts, 0, 1) if is_ridge else mapping[levels - 1].copy()
    baseline = np.broadcast_to(baseline_mean, y.shape).copy()
    fold_ids = np.full(len(images), -1)
    n_train_level = np.full(len(images), np.nan) if is_ridge else support[levels - 1].copy()
    for fold, (train, test) in enumerate(folds):
        if is_ridge:
            if is_ionescu:
                stage1, coef, intercept, alpha = fit_ionescu(features[train], y[train], config["seed"])
                x[test, 0] = stage1.predict(features[test])
                print(f"ionescu: fitted both stages for fold {fold + 1}/{len(folds)}", flush=True)
            else:
                coef, intercept, alpha = fit_ridge(x[train], y[train], config["seed"])
            predicted[test] = np.clip(x[test] @ coef + intercept, 0, 1)
            selected_alphas.extend(dict(version=version, model=model, fold=fold, alpha=alpha[m], n_train=len(train), first_stage_alpha=stage1.alpha_ if is_ionescu else np.nan) for m, model in enumerate(models))
        else:
            predicted[test] = fit_mapping(levels[train], y[train])[levels[test] - 1]
            n_train_level[test] = np.bincount(levels[train], minlength=6)[levels[test]]
        baseline[test] = y[train].mean(axis=0)
        fold_ids[test] = fold
    for m, model in enumerate(models):
        metadata = dict(version=version, rubric_version=candidate["version"], model=model,
                        predictor=candidate["predictor"], baseline=baseline_mean[m], metric=metric,
                        prompt_strategy=args.prompt_strategy)
        if frozen is not None:
            mappings.extend(saved[saved["model"].eq(model)].to_dict("records"))
        elif is_ridge:
            for j, feature in enumerate(candidate["columns"]):
                mappings.append(dict(**metadata, feature=feature, coefficient=coefficients[j, m],
                                     intercept=intercepts[m], alpha=alphas[m], n_train=len(random_ids)))
            selected_alphas.append(dict(version=version, model=model, fold=-1, alpha=alphas[m], n_train=len(random_ids), first_stage_alpha=stage1_alpha if is_ionescu else np.nan))
        else:
            for level in range(1, 6):
                mappings.append(dict(**metadata, level=level, prediction=mapping[level - 1, m], n_train=support[level - 1]))
        frame = images[["dataset", "image_id", "cohort"]].copy()
        for j, column in enumerate(candidate["columns"]):
            frame[column] = x[:, j]
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
        row = dict(version=version, predictor=candidate["predictor"], auc_score=auc_score, cohort=cohort, n_images=len(ids), n_models=len(models),
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
                risk = -prediction[:, m] if auc_score == "prediction" else level
                calculate_auc = risk_auc if auc_score == "prediction" else auc
                value = calculate_auc(risk, failed).item()
                boot = calculate_auc(risk[sample], failed[sample])
                low, high = interval(boot)
                failures_table.append(dict(version=version, cohort=cohort, model=model, threshold=threshold, auc_score=auc_score,
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
        for j, column in enumerate(candidate["columns"]):
            if column.endswith("_fraction"):
                continue
            for l in range(1, 6):
                scores_at_level = observed[x[ids, j] == l].mean(axis=1)
                level_table.append(dict(version=version, dimension=column, cohort=cohort, level=l, n_images=len(scores_at_level),
                                        mean=np.mean(scores_at_level) if len(scores_at_level) else np.nan))
        print(f"{version} / {cohort}: RMSE {values['rmse']:.4f}, MSE skill {values['mse_skill']:.3f}")

comparisons = []
for first, second in combinations(candidates, 2):
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
                    ("predictions", predictions), ("mappings", pd.DataFrame(mappings)), ("levels", pd.DataFrame(level_table)),
                    ("selected_alphas", pd.DataFrame(selected_alphas)), ("coverage", pd.DataFrame(coverage).assign(n_shared=len(images)))]:
    table.to_csv(output_dir / f"{name}.csv", index=False)

for cohort in cohorts:
    data = summary[summary["cohort"].eq(cohort)]
    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(7 if len(data) <= 3 else 2.5 * len(data), 4))
    ax.bar(x - 0.2, data["baseline_rmse"], width=0.4, color="lightgray", label="Detector-average baseline")
    ax.bar(x + 0.2, data["rmse"], width=0.4, color="#31005c", label="Rubric")
    ax.set(xticks=x, xticklabels=[display(v) for v in data["version"]], ylabel=f"{args.task.capitalize()} S RMSE", title=cohort)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{cohort}_rmse.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7 if len(data) <= 3 else 2.5 * len(data), 4))
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

    # Only single-level predictors have an overall level distribution.
    scalar_candidates = [name for name, candidate in candidates.items() if candidate["predictor"] == "isotonic"]
    if scalar_candidates:
        fig, axes = plt.subplots(1, len(scalar_candidates), figsize=(4 * len(scalar_candidates), 4), squeeze=False, sharey=True)
        for ax, version in zip(axes[0], scalar_candidates):
            ids = cohorts[cohort]
            level = images.loc[ids, candidates[version]["inputs"][0]].to_numpy()
            present = np.unique(level).astype(int)
            ax.boxplot([y[ids][level == l].mean(axis=1) for l in present], positions=present, showfliers=False)
            ax.set(xticks=range(1, 6), xticklabels=[f"{l}\nn={(level == l).sum()}" for l in range(1, 6)],
                   xlabel="Rubric level", title=f"{display(version)} / {cohort}", ylim=(0, 1))
        axes[0, 0].set_ylabel(f"Mean {args.task} S across detectors")
        fig.tight_layout()
        fig.savefig(output_dir / f"{cohort}_levels.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7 if len(data) <= 3 else 2.5 * len(data), 4))
    ax.bar(x, data["mse_skill"], color="#31005c")
    ax.axhline(0, color="gray", linestyle="--")
    ax.set(xticks=x, xticklabels=[display(v) for v in data["version"]], ylabel="MSE skill", title=cohort)
    fig.tight_layout()
    fig.savefig(output_dir / f"{cohort}_skill.pdf")
    plt.close(fig)

    for version in candidates:
        frame = predictions[predictions["version"].eq(version) & predictions["cohort"].eq(cohort)]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.hexbin(frame["prediction"], frame["score"], gridsize=30, mincnt=1, cmap="Purples")
        bins = pd.cut(frame["prediction"], np.linspace(0, 1, 11), include_lowest=True)
        trend = frame.groupby(bins, observed=True)[["prediction", "score"]].mean()
        ax.plot(trend["prediction"], trend["score"], 'o-', color="orange", label="Observed mean per prediction bin")
        ax.plot([0, 1], [0, 1], '--', color="gray")
        ax.set(xlabel="Predicted S", ylabel="Observed S", xlim=(0, 1), ylim=(0, 1),
               title=f"{display(version)} / {cohort} / all detectors")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{cohort}_{version.replace(':', '_')}_predicted_vs_observed.pdf")
        plt.close(fig)

print(f"Saved evaluation to {output_dir}")
print("Intervals use paired image bootstraps conditional on the fitted predictions; calibration results are developmental.")
