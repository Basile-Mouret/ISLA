from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from pipelines import (
    build_best_ridge_feature_model,
    build_final_model,
    ensure_target,
    make_age_bins,
)
from ridge_feature_models import StableScoreRidgeRegressor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="results/elasticnet_public_like_screen",
        help="Directory where markdown and JSON summaries are written.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=12,
        help="Number of repeated public-like holdout splits.",
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=100,
        help="Validation rows per split, matching the public leaderboard size.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


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
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def beta_to_m_values(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[1] > 1:
        beta = np.clip(arr[:, 1:], 1e-5, 1 - 1e-5)
        out = arr.copy()
        out[:, 1:] = np.log2(beta / (1.0 - beta))
        return out
    return arr


class FlexibleElasticNetCVRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        l1_ratio_grid=(0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0),
        n_alphas: int = 25,
        cv: int = 5,
        max_iter: int = 2500,
        tol: float = 1e-3,
        use_m_value: bool = False,
    ):
        self.l1_ratio_grid = l1_ratio_grid
        self.n_alphas = n_alphas
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.use_m_value = use_m_value

    def preprocess_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X.copy()
        if "gender" in X_num.columns:
            X_num["gender"] = (
                X_num["gender"].map({"m": 1, "f": 0, "M": 1, "F": 0}).fillna(0.5)
            )
        return X_num.astype(np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        X_num = self.preprocess_df(X)
        steps: list[tuple[str, object]] = [("variance_filter", VarianceThreshold(threshold=1e-5))]

        if self.use_m_value:
            steps.append(
                (
                    "beta_to_m_value",
                    FunctionTransformer(beta_to_m_values, validate=False),
                )
            )

        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNetCV(
                        l1_ratio=list(self.l1_ratio_grid),
                        n_alphas=self.n_alphas,
                        cv=self.cv,
                        n_jobs=-1,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=42,
                    ),
                ),
            ]
        )

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X_num, np.asarray(y).reshape(-1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(self.preprocess_df(X))


def evaluate(estimator, X: pd.DataFrame, y: pd.Series, splits) -> dict[str, object]:
    scores = []
    chosen = []

    for train_idx, valid_idx in splits:
        est = clone(estimator)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].to_numpy()

        est.fit(X_train, y_train)
        pred = est.predict(X_valid)
        scores.append(rmse(y_valid, pred))

        if hasattr(est, "pipeline_"):
            elasticnet = est.pipeline_.named_steps["elasticnet"]
            chosen.append(
                {
                    "alpha": float(elasticnet.alpha_),
                    "l1_ratio": float(elasticnet.l1_ratio_),
                }
            )

    arr = np.asarray(scores)
    result = {
        "mean_rmse": float(arr.mean()),
        "std_rmse": float(arr.std()),
        "median_rmse": float(np.quantile(arr, 0.5)),
        "best_rmse": float(arr.min()),
        "worst_rmse": float(arr.max()),
    }

    if chosen:
        l1_counts = pd.Series([row["l1_ratio"] for row in chosen]).value_counts().sort_index()
        result["chosen_alpha_mean"] = float(np.mean([row["alpha"] for row in chosen]))
        result["chosen_l1_ratio_counts"] = {str(key): int(value) for key, value in l1_counts.items()}

    return result


def build_candidates(columns: list[str]) -> dict[str, object]:
    return {
        "plain_ridge_base": build_final_model(columns),
        "bagged_ridge_base": build_best_ridge_feature_model(columns),
        "stable_fscore_k2800_r15_sf0.7": StableScoreRidgeRegressor(
            k=2800,
            alpha=0.01,
            score_method="f_score",
            n_resamples=15,
            sample_fraction=0.7,
            random_state=42,
        ),
        "elasticnetcv_model1_baseline": FlexibleElasticNetCVRegressor(),
        "elasticnetcv_low_l1_dense": FlexibleElasticNetCVRegressor(
            l1_ratio_grid=(0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3),
            n_alphas=50,
            cv=5,
            max_iter=8000,
            tol=1e-4,
            use_m_value=False,
        ),
        "elasticnetcv_low_l1_mvalue": FlexibleElasticNetCVRegressor(
            l1_ratio_grid=(0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3),
            n_alphas=50,
            cv=5,
            max_iter=8000,
            tol=1e-4,
            use_m_value=True,
        ),
    }


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = "| Model | Mean RMSE | Std | Median | Best | Worst | Chosen l1 ratios |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for row in rows:
        l1_counts = row.get("chosen_l1_ratio_counts", {})
        l1_repr = ", ".join(f"{k}:{v}" for k, v in l1_counts.items()) if l1_counts else "-"
        lines.append(
            f"| {row['model']} | {row['mean_rmse']:.4f} | {row['std_rmse']:.4f} | {row['median_rmse']:.4f} | {row['best_rmse']:.4f} | {row['worst_rmse']:.4f} | {l1_repr} |"
        )
    return "\n".join(lines)


def write_outputs(output_dir: Path, ranked_rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps({"ranked": ranked_rows}, indent=2),
        encoding="utf-8",
    )

    report = f"""# ElasticNet Public-Like Screen

- Validation strategy: `{args.n_splits}` repeated stratified shuffle splits
- Validation size per split: `{args.test_size}` rows
- Purpose: compare current Ridge baselines against ElasticNetCV variants on leaderboard-sized holdouts

## Ranked Results

{markdown_table(ranked_rows)}

## Notes

- `elasticnetcv_model1_baseline` matches the public `3.9` model structure you added in `model_1.py`
- `elasticnetcv_low_l1_dense` explores denser low-`l1_ratio` settings with a wider alpha path and stricter optimization
- `elasticnetcv_low_l1_mvalue` applies the same search on M-values instead of raw beta values
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    X, y = load_training_data(Path(args.data_dir))
    columns = list(X.columns)

    age_bins = make_age_bins(y)
    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    splits = list(splitter.split(np.zeros((len(y), 1)), age_bins))

    results = []
    for model_name, estimator in build_candidates(columns).items():
        print(f"evaluating {model_name}...", flush=True)
        results.append({"model": model_name, **evaluate(estimator, X, y, splits)})

    results.sort(key=lambda row: (row["mean_rmse"], row["worst_rmse"]))
    write_outputs(Path(args.output_dir), results, args)
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
