from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone

from pipelines import build_final_model, ensure_target, make_cv_splits
from ridge_feature_models import BaggedScoreRidgeRegressor, StableScoreRidgeRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="results/ridge_feature_refine",
        help="Directory for refined ridge feature selection results.",
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


def load_training_data(data_dir: Path):
    X_path, y_path = resolve_training_paths(data_dir)
    X = pd.read_csv(X_path)
    y = ensure_target(pd.read_csv(y_path))
    return X, y


def rmse(y_true, y_pred) -> float:
    return float((((y_true - y_pred) ** 2).mean()) ** 0.5)


def evaluate_estimator_cv(estimator, X, y, cv_splits):
    scores = []
    for train_idx, valid_idx in cv_splits:
        est = clone(estimator)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].to_numpy()
        est.fit(X_train, y_train)
        pred = est.predict(X_valid)
        scores.append(rmse(y_valid, pred))
    return float(pd.Series(scores).mean()), float(pd.Series(scores).std(ddof=0))


def build_candidates():
    candidates = [("baseline", "global_ridge_best", {}, None)]

    for k in [2800, 3000, 3200, 3500]:
        for n_resamples in [15, 25]:
            for sample_fraction in [0.7, 0.8]:
                params = {
                    "k": k,
                    "alpha": 0.01,
                    "score_method": "f_score",
                    "n_resamples": n_resamples,
                    "sample_fraction": sample_fraction,
                    "random_state": 42,
                }
                label = f"stable_fscore_k{k}_r{n_resamples}_sf{sample_fraction}"
                candidates.append(("stable_fscore_refine", label, params, StableScoreRidgeRegressor(**params)))

    for k in [3200, 3500, 3800, 4200]:
        for n_estimators in [5, 7, 9]:
            for sample_fraction in [0.7, 0.8, 0.9]:
                params = {
                    "k": k,
                    "alpha": 0.01,
                    "score_method": "f_score",
                    "n_estimators": n_estimators,
                    "sample_fraction": sample_fraction,
                    "random_state": 42,
                }
                label = f"bagged_fscore_k{k}_e{n_estimators}_sf{sample_fraction}"
                candidates.append(("bagged_fscore_refine", label, params, BaggedScoreRidgeRegressor(**params)))

    return candidates


def make_estimator(model_name: str, params: dict[str, object], columns: list[str]):
    if model_name == "global_ridge_best":
        return build_final_model(columns)
    if model_name.startswith("stable_fscore"):
        return StableScoreRidgeRegressor(**params)
    if model_name.startswith("bagged_fscore"):
        return BaggedScoreRidgeRegressor(**params)
    raise ValueError(model_name)


def markdown_table(rows, columns):
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


def plot_bagged_heatmap(rows, output_dir: Path):
    bagged_rows = [row for row in rows if row["family"] == "bagged_fscore_refine"]
    if not bagged_rows:
        return

    bagged_df = pd.DataFrame(
        [
            {
                "k": row["param_dict"]["k"],
                "n_estimators": row["param_dict"]["n_estimators"],
                "sample_fraction": row["param_dict"]["sample_fraction"],
                "rmse": row["rmse"],
            }
            for row in bagged_rows
        ]
    )

    for sample_fraction in sorted(bagged_df["sample_fraction"].unique()):
        subset = bagged_df[bagged_df["sample_fraction"] == sample_fraction]
        pivot = subset.pivot(index="n_estimators", columns="k", values="rmse")
        plt.figure(figsize=(6, 4))
        plt.imshow(pivot.values, aspect="auto", cmap="viridis")
        plt.colorbar(label="Search RMSE")
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xlabel("k")
        plt.ylabel("n_estimators")
        plt.title(f"Bagged F-score Ridge Search RMSE (sample_fraction={sample_fraction})")
        plt.tight_layout()
        plt.savefig(output_dir / f"bagged_heatmap_sf_{str(sample_fraction).replace('.', '_')}.png", dpi=180, bbox_inches="tight")
        plt.close()


def build_report(output_dir: Path, quick_rows, final_rows, best_model_name: str):
    family_best = []
    seen = set()
    for row in quick_rows:
        if row["family"] in seen:
            continue
        family_best.append(row)
        seen.add(row["family"])

    report = f"""# Refined Ridge Feature Selection Sweep

- Search CV: `1 x 3` repeated stratified folds on age bins
- Final comparison CV: `2 x 5` repeated stratified folds on age bins
- Focus: refine the strongest Ridge-only selector family found earlier (`bagged_fscore`) and compare it against a tightened `stable_fscore` sweep

## Best Search Candidate Per Family

{markdown_table(family_best, [('family', 'Family'), ('model', 'Model'), ('rmse', 'Search RMSE'), ('rmse_std', 'Search Std'), ('params', 'Params')])}

## Final Repeated-CV Comparison

{markdown_table(final_rows, [('model', 'Model'), ('rmse', 'Final CV RMSE'), ('rmse_std', 'Final CV Std')])}

## Interpretation

- The best refined Ridge-only model from this sweep is `{best_model_name}`
- If this beats the earlier `bagged_fscore_k3500`, the bagging hyperparameters still matter and the Ridge-only path has a little more room
- If it ties the earlier result, Ridge feature selection is likely close to saturated and future gains should come from model combination rather than more selector tuning
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
    for family, label, params, estimator in build_candidates():
        log(f"Search evaluating {label}...")
        est = build_final_model(columns) if estimator is None else estimator
        mean_rmse, std_rmse = evaluate_estimator_cv(est, X, y, quick_cv)
        quick_rows.append(
            {
                "family": family,
                "model": label,
                "rmse": mean_rmse,
                "rmse_std": std_rmse,
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
        family_best["stable_fscore_refine"],
        family_best["bagged_fscore_refine"],
    ]

    final_rows = []
    for row in finalists:
        log(f"Final evaluating {row['model']}...")
        est = make_estimator(row["model"], row["param_dict"], columns)
        mean_rmse, std_rmse = evaluate_estimator_cv(est, X, y, final_cv)
        final_rows.append({"model": row["model"], "rmse": mean_rmse, "rmse_std": std_rmse})

    final_rows.sort(key=lambda row: (row["rmse"], row["rmse_std"]))
    best_model_name = final_rows[0]["model"]

    plot_bagged_heatmap(quick_rows, output_dir)
    build_report(output_dir, quick_rows, final_rows, best_model_name)

    payload = {
        "quick_rows": quick_rows,
        "final_rows": final_rows,
        "best_model_name": best_model_name,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"Wrote refined ridge sweep to {output_dir}")


if __name__ == "__main__":
    main()
