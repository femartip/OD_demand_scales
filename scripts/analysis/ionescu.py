import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler


def first_stage(features, scores, seed):
    # Each training scalar is predicted without that image's outcome.
    target = scores.mean(axis=1)
    ridge = RidgeCV(alphas=np.logspace(-1, 4, 12))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    training = cross_val_predict(ridge, features, target, cv=folds)[:, None]
    fitted = ridge.fit(features, target)
    return fitted, training


def second_stage(features, scores, alphas):
    scaler = StandardScaler().fit(features)
    ridge = Ridge(alpha=alphas).fit(scaler.transform(features), scores)
    coefficients = ridge.coef_.T / scaler.scale_[:, None]
    intercepts = ridge.intercept_ - scaler.mean_ @ coefficients
    return coefficients, intercepts


def fit_ionescu(features, scores, seed):
    # Select second-stage penalties by validating the entire two-stage pipeline.
    # All first-stage fitting and its own alpha selection stay inside each split.
    alphas = np.array([0.1, 1, 10, 100, 1000])
    errors = np.zeros((len(alphas), scores.shape[1]))
    folds = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train, test in folds.split(features):
        fitted, training = first_stage(features[train], scores[train], seed)
        testing = fitted.predict(features[test])[:, None]
        for i, alpha in enumerate(alphas):
            coefficients, intercepts = second_stage(training, scores[train], alpha)
            prediction = np.clip(testing @ coefficients + intercepts, 0, 1)
            errors[i] += ((prediction - scores[test]) ** 2).sum(axis=0)
    selected = alphas[errors.argmin(axis=0)]
    fitted, training = first_stage(features, scores, seed)
    coefficients, intercepts = second_stage(training, scores, selected)
    return fitted, coefficients, intercepts, selected
