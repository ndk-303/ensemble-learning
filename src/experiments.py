"""Comparable ensemble experiments on the same stratified Digits splits."""

import time

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from src.data import build_model, load_data
from src.methods import AdaBoost, Bagging, Stacking


MODEL_ORDER = ("Stump", "Tree", "KNN", "LR", "SVM", "Bagging", "AdaBoost", "Stacking")
BASELINES = {"Bagging": "Tree", "AdaBoost": "Stump"}


def build_comparison_models(seed=42, bag_t=50, ada_t=100, learning_rate=1.0,
                            stacking_models=("dt", "knn", "svm")):
    """Build fresh models; baselines match the ensemble components."""
    return {
        "Stump": build_model("dt", mode="boosting", seed=seed),
        "Tree": build_model("dt", mode="bagging", seed=seed),
        "KNN": build_model("knn", seed=seed),
        "LR": build_model("lr", seed=seed),
        "SVM": build_model("svm", seed=seed),
        "Bagging": Bagging(build_model("dt", mode="bagging", seed=seed), T=bag_t, seed=seed),
        "AdaBoost": AdaBoost(build_model("dt", mode="boosting", seed=seed),
                             T=ada_t, learning_rate=learning_rate, random_state=seed),
        "Stacking": Stacking(
            [build_model(name, mode="stacking", seed=seed) for name in stacking_models],
            build_model("lr", seed=seed), k=5, seed=seed,
        ),
    }


def build_reference_models(seed, bag_t, ada_t, learning_rate, stacking_models):
    """Equivalent scikit-learn ensembles for a first-split reference check."""
    return {
        "Bagging": BaggingClassifier(
            estimator=build_model("dt", mode="bagging", seed=seed),
            n_estimators=bag_t, random_state=seed,
        ),
        "AdaBoost": AdaBoostClassifier(
            estimator=build_model("dt", mode="boosting", seed=seed),
            n_estimators=ada_t, learning_rate=learning_rate, random_state=seed,
        ),
        "Stacking": StackingClassifier(
            estimators=[(f"base_{i}_{name}", build_model(name, mode="stacking", seed=seed))
                        for i, name in enumerate(stacking_models)],
            final_estimator=build_model("lr", seed=seed),
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            stack_method="predict_proba",
        ),
    }


def evaluate_references(first, bag_t, ada_t, learning_rate, stacking_models):
    """Score library references on the first comparison split only."""
    Xtr, ytr, Xte, yte = first["data"]
    refs = build_reference_models(first["seed"], bag_t, ada_t, learning_rate, stacking_models)
    scores = {}
    for name, model in refs.items():
        model.fit(Xtr, ytr)
        scores[name] = accuracy_score(yte, model.predict(Xte))
    return scores


def evaluate_split(seed, test_size, bag_t, ada_t, learning_rate, stacking_models):
    """Fit all models on one shared split, returning scores, times and first-run artifacts."""
    Xtr, ytr, Xte, yte, _ = load_data(test_size=test_size, seed=seed)
    models = build_comparison_models(seed, bag_t, ada_t, learning_rate, stacking_models)
    scores, train_scores, times = {}, {}, {}
    for name, model in models.items():
        start = time.perf_counter()
        model.fit(Xtr, ytr)
        times[name] = time.perf_counter() - start
        train_scores[name] = accuracy_score(ytr, model.predict(Xtr))
        scores[name] = accuracy_score(yte, model.predict(Xte))
    return {
        "seed": seed, "scores": scores, "train_scores": train_scores,
        "times": times, "models": models,
        "data": (Xtr, ytr, Xte, yte),
    }


def summarize_scores(runs):
    """Summarize scores and paired gains against matching baselines."""
    if not runs:
        raise ValueError("At least one run is required")
    summary = {}
    for name in MODEL_ORDER:
        values = np.array([run[name] for run in runs], dtype=float)
        baseline = None
        if name == "Stacking":
            baseline = max(("Tree", "KNN", "LR", "SVM"),
                           key=lambda candidate: np.mean([run[candidate] for run in runs]))
        elif name in BASELINES:
            baseline = BASELINES[name]
        delta = None
        if baseline:
            delta = float(np.mean([run[name] - run[baseline] for run in runs]))
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "baseline": baseline,
            "paired_delta": delta,
        }
    return summary
