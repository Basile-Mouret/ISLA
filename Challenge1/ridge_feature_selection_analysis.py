from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_selection import f_regression

from pipelines import build_final_model, ensure_target, make_cv_splits
from ridge_feature_models import (
    BaggedScoreRidgeRegressor,
    StableScoreRidgeRegressor,
    _numeric_columns,
    _safe_abs_corr,
    compute_feature_scores,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="results/ridge_feature_selection",
        help="Directory for ridge feature selection outputs.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def resolve_training_paths(data_dir: Path) -> tuple[Path, Path]:
    direct_X = data_dir / "X_train.csv"
    direct_y = data_dir / "y_train.csv"
    nested_X = data_dir / "train" / "X_train.csv"
    nested_y = data_dir / "train" / "y_train.csv"

    if direct_X.exists() and direct_y.exists():
        return direct_X, direct_y
    if nested_X.exists() and nested_y.exists():
        return nested_X, nested_y
    raise FileNotFoundError(f"Could not find training data under {data_dir}")


def load_training_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    X_path, y_path = resolve_training_paths(data_dir)
    X = pd.read_csv(X_path)
    y = ensure_target(pd.read_csv(y_path))
    return X, y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def evaluate_estimator_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    keep_oof: bool = False,
) -> dict[str, object]:
    fold_rmses = []
    oof_sum = np.zeros(X.shape[0], dtype=float) if keep_oof else None
    oof_count = np.zeros(X.shape[0], dtype=int) if keep_oof else None

    for train_idx, valid_idx in cv_splits:
        estimator_fold = clone(estimator)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].to_numpy()

        estimator_fold.fit(X_train, y_train)
        predictions = np.asarray(estimator_fold.predict(X_valid)).reshape(-1)
        fold_rmses.append(rmse(y_valid, predictions))

        if keep_oof:
            oof_sum[valid_idx] += predictions
            oof_count[valid_idx] += 1

    result = {
        "rmse": float(np.mean(fold_rmses)),
        "rmse_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
    }
    if keep_oof:
        result["oof_predictions"] = oof_sum / np.maximum(oof_count, 1)
    return result


def build_candidates() -> list[tuple[str, str, dict[str, object], object]]:
    candidates = [("baseline", "global_ridge_best", {}, None)]

    for k in [3000, 3500, 5000]:
        params = {
            "k": k,
            "alpha": 0.01,
            "score_method": "f_score",
            "n_resamples": 15,
            "sample_fraction": 0.8,
            "random_state": 42,
        }
        candidates.append(
            (
                "stable_fscore",
                f"stable_fscore_k{k}",
                params,
                StableScoreRidgeRegressor(**params),
            )
        )

    for k in [3000, 3500, 5000]:
        for gap_penalty in [0.25, 0.5]:
            params = {
                "k": k,
                "alpha": 0.01,
                "score_method": "gender_stable",
                "n_resamples": 15,
                "sample_fraction": 0.8,
                "gap_penalty": gap_penalty,
                "random_state": 42,
            }
            candidates.append(
                (
                    "stable_gender",
                    f"stable_gender_k{k}_gp{gap_penalty}",
                    params,
                    StableScoreRidgeRegressor(**params),
                )
            )

    for k in [3500, 5000]:
        params = {
            "k": k,
            "alpha": 0.01,
            "score_method": "f_score",
            "n_estimators": 7,
            "sample_fraction": 0.8,
            "random_state": 42,
        }
        candidates.append(
            (
                "bagged_fscore",
                f"bagged_fscore_k{k}",
                params,
                BaggedScoreRidgeRegressor(**params),
            )
        )

    for k in [3500, 5000]:
        for gap_penalty in [0.25, 0.5]:
            params = {
                "k": k,
                "alpha": 0.01,
                "score_method": "gender_stable",
                "n_estimators": 7,
                "sample_fraction": 0.8,
                "gap_penalty": gap_penalty,
                "random_state": 42,
            }
            candidates.append(
                (
                    "bagged_gender",
                    f"bagged_gender_k{k}_gp{gap_penalty}",
                    params,
                    BaggedScoreRidgeRegressor(**params),
                )
            )

    return candidates


def make_estimator(name: str, params: dict[str, object], columns: list[str]):
    if name == "global_ridge_best":
        return build_final_model(columns)
    if name.startswith("stable_"):
        return StableScoreRidgeRegressor(**params)
    if name.startswith("bagged_"):
        return BaggedScoreRidgeRegressor(**params)
    raise ValueError(f"Unknown estimator label: {name}")


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + body)


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def fit_baseline_features(X: pd.DataFrame, y: pd.Series, k: int = 3500) -> list[str]:
    feature_columns = _numeric_columns(X)
    X_numeric = X[feature_columns].to_numpy()
    scores, _ = f_regression(X_numeric, y.to_numpy())
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
    indices = np.argsort(scores)[-k:]
    return [feature_columns[idx] for idx in indices]


def plot_feature_score_diagnostics(
    X: pd.DataFrame,
    y: pd.Series,
    output_dir: Path,
) -> None:
    feature_columns = _numeric_columns(X)
    X_numeric = X[feature_columns].to_numpy()
    genders = X["gender"].astype(str).to_numpy()

    f_scores = compute_feature_scores(X_numeric, y.to_numpy(), genders, "f_score", gap_penalty=0.5)
    gender_scores = compute_feature_scores(
        X_numeric,
        y.to_numpy(),
        genders,
        "gender_stable",
        gap_penalty=0.5,
    )
    abs_corr = _safe_abs_corr(X_numeric, y.to_numpy())

    top_idx = np.argsort(f_scores)[-1500:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].scatter(
        abs_corr[top_idx],
        gender_scores[top_idx],
        s=10,
        alpha=0.3,
        color="#4c78a8",
        linewidths=0,
    )
    axes[0].set_title("Overall |corr| vs Gender-Stable Score")
    axes[0].set_xlabel("Overall absolute correlation with age")
    axes[0].set_ylabel("Gender-stable score")

    male_mask = genders == "m"
    female_mask = genders == "f"
    male_corr = _safe_abs_corr(X_numeric[male_mask], y.to_numpy()[male_mask])
    female_corr = _safe_abs_corr(X_numeric[female_mask], y.to_numpy()[female_mask])
    corr_gap = np.abs(male_corr - female_corr)
    axes[1].scatter(
        corr_gap[top_idx],
        gender_scores[top_idx],
        s=10,
        alpha=0.3,
        color="#f28e2b",
        linewidths=0,
    )
    axes[1].set_title("Gender Correlation Gap vs Gender-Stable Score")
    axes[1].set_xlabel("| male corr - female corr |")
    axes[1].set_ylabel("Gender-stable score")

    plt.tight_layout()
    plt.savefig(output_dir / "feature_score_diagnostics.png", dpi=180, bbox_inches="tight")
    plt.close()


def plot_overlap_and_frequency(
    best_models: dict[str, object],
    baseline_features: list[str],
    output_dir: Path,
) -> list[dict[str, object]]:
    baseline_set = set(baseline_features)
    overlap_rows = []
    for model_name, fitted in best_models.items():
        if hasattr(fitted, "selected_features_"):
            feature_set = set(fitted.selected_features_)
        elif hasattr(fitted, "consensus_features_"):
            feature_set = set(fitted.consensus_features_)
        else:
            continue
        overlap_rows.append(
            {
                "model": model_name,
                "selected_features": len(feature_set),
                "overlap_with_baseline": len(feature_set & baseline_set),
                "jaccard_with_baseline": jaccard(feature_set, baseline_set),
            }
        )

    if overlap_rows:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        model_names = [row["model"] for row in overlap_rows]
        axes[0].bar(model_names, [row["jaccard_with_baseline"] for row in overlap_rows], color="#4c78a8")
        axes[0].set_title("Feature Overlap vs Baseline Ridge")
        axes[0].set_ylabel("Jaccard overlap")
        axes[0].tick_params(axis="x", rotation=25)

        bagged_name = next((name for name, model in best_models.items() if hasattr(model, "selection_frequency_")), None)
        if bagged_name is not None:
            freq = best_models[bagged_name].selection_frequency_
            axes[1].hist(freq, bins=np.arange(freq.max() + 2) - 0.5, color="#f28e2b", rwidth=0.9)
            axes[1].set_title(f"Selection Frequency - {bagged_name}")
            axes[1].set_xlabel("Times selected across ensemble members")
            axes[1].set_ylabel("Number of CpGs")
        else:
            axes[1].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "overlap_and_frequency.png", dpi=180, bbox_inches="tight")
        plt.close()

    return overlap_rows


def plot_final_model_scores(final_results: list[dict[str, object]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    model_names = [row["model"] for row in final_results]
    ax.bar(model_names, [row["rmse"] for row in final_results], color="#59a14f")
    ax.set_ylabel("Repeated CV RMSE")
    ax.set_title("Final Ridge Feature Selection Comparison")
    ax.tick_params(axis="x", rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "final_model_scores.png", dpi=180, bbox_inches="tight")
    plt.close()


def build_report(
    output_dir: Path,
    quick_rows: list[dict[str, object]],
    final_results: list[dict[str, object]],
    overlap_rows: list[dict[str, object]],
    best_model_name: str,
) -> None:
    family_best_rows = []
    seen = set()
    for row in quick_rows:
        if row["family"] in seen:
            continue
        family_best_rows.append(row)
        seen.add(row["family"])

    report = f"""# Ridge Feature Selection Analysis

- Search CV: `1 x 3` repeated stratified folds on age bins
- Final comparison CV: `2 x 5` repeated stratified folds on age bins
- Goal: improve Ridge primarily through better CpG feature selection rather than changing the core linear model

## Plot Files

- `feature_score_diagnostics.png`
- `overlap_and_frequency.png`
- `final_model_scores.png`

## Best Search Candidate Per Family

{markdown_table(family_best_rows, [('family', 'Family'), ('model', 'Model'), ('rmse', 'Search RMSE'), ('rmse_std', 'Search Std'), ('params', 'Params')])}

## Final Repeated-CV Comparison

{markdown_table(final_results, [('model', 'Model'), ('rmse', 'Final CV RMSE'), ('rmse_std', 'Final CV Std')])}

## Feature Overlap With Baseline

{markdown_table(overlap_rows, [('model', 'Model'), ('selected_features', 'Selected CpGs'), ('overlap_with_baseline', 'Overlap'), ('jaccard_with_baseline', 'Jaccard')])}

## Interpretation

- The current best model from this ridge-only feature-selection analysis is `{best_model_name}`
- If the best stable or bagged selector beats the baseline, that means selection variance matters and more robust CpG ranking is helping
- If the gain is small, then the baseline `f_regression` selector was already close to optimal and future gains likely require model combination rather than another single selector tweak
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data(Path(args.data_dir))
    columns = list(X.columns)

    quick_cv = make_cv_splits(y, n_splits=3, n_repeats=1)
    final_cv = make_cv_splits(y, n_splits=5, n_repeats=2)

    quick_rows = []
    for family, model_name, params, estimator in build_candidates():
        log(f"Search evaluating {model_name}...")
        estimator_obj = build_final_model(columns) if estimator is None else estimator
        evaluation = evaluate_estimator_cv(estimator_obj, X, y, quick_cv, keep_oof=False)
        quick_rows.append(
            {
                "family": family,
                "model": model_name,
                "rmse": evaluation["rmse"],
                "rmse_std": evaluation["rmse_std"],
                "params": json.dumps(params, sort_keys=True),
                "param_dict": params,
            }
        )

    quick_rows.sort(key=lambda row: (row["family"], row["rmse"], row["rmse_std"]))

    family_best = {}
    for row in quick_rows:
        family_best.setdefault(row["family"], row)

    finalists = [
        family_best["baseline"],
        family_best["stable_fscore"],
        family_best["stable_gender"],
        family_best["bagged_fscore"],
        family_best["bagged_gender"],
    ]

    final_results = []
    fitted_models = {}
    for row in finalists:
        model_name = row["model"]
        params = row["param_dict"]
        estimator = make_estimator(model_name, params, columns)
        log(f"Final evaluating {model_name}...")
        evaluation = evaluate_estimator_cv(estimator, X, y, final_cv, keep_oof=False)
        final_results.append(
            {
                "model": model_name,
                "rmse": evaluation["rmse"],
                "rmse_std": evaluation["rmse_std"],
            }
        )
        fitted = make_estimator(model_name, params, columns)
        fitted.fit(X, y)
        fitted_models[model_name] = fitted

    final_results.sort(key=lambda row: (row["rmse"], row["rmse_std"]))
    best_model_name = final_results[0]["model"]

    baseline_features = fit_baseline_features(X, y, k=3500)
    plot_feature_score_diagnostics(X, y, output_dir)
    overlap_rows = plot_overlap_and_frequency(fitted_models, baseline_features, output_dir)
    plot_final_model_scores(final_results, output_dir)

    build_report(output_dir, quick_rows, final_results, overlap_rows, best_model_name)

    summary = {
        "quick_rows": quick_rows,
        "final_results": final_results,
        "overlap_rows": overlap_rows,
        "best_model_name": best_model_name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(f"Wrote ridge feature selection analysis to {output_dir}")


if __name__ == "__main__":
    main()
