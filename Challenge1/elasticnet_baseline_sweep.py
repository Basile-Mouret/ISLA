from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedShuffleSplit

from elasticnet_models import build_elasticnet_from_preset, get_baseline_elasticnet_presets
from pipelines import ensure_target, make_age_bins


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="results/elasticnet_baseline_sweep",
        help="Directory where markdown and JSON summaries are written.",
    )
    parser.add_argument(
        "--presets",
        default=None,
        help="Comma-separated preset names to run. Defaults to all baseline presets.",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=8,
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


def select_presets(presets_arg: str | None):
    presets = get_baseline_elasticnet_presets()
    if not presets_arg:
        return presets

    requested = [name.strip() for name in presets_arg.split(",") if name.strip()]
    preset_map = {preset.name: preset for preset in presets}
    unknown = [name for name in requested if name not in preset_map]
    if unknown:
        available = ", ".join(sorted(preset_map))
        raise ValueError(f"Unknown preset(s): {', '.join(unknown)}. Available presets: {available}")
    return [preset_map[name] for name in requested]


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

        elasticnet = est.pipeline_.named_steps["elasticnet"]
        chosen.append(
            {
                "alpha": float(elasticnet.alpha_),
                "l1_ratio": float(elasticnet.l1_ratio_),
            }
        )

    arr = np.asarray(scores)
    l1_counts = pd.Series([row["l1_ratio"] for row in chosen]).value_counts().sort_index()
    return {
        "mean_rmse": float(arr.mean()),
        "std_rmse": float(arr.std()),
        "median_rmse": float(np.quantile(arr, 0.5)),
        "best_rmse": float(arr.min()),
        "worst_rmse": float(arr.max()),
        "chosen_alpha_mean": float(np.mean([row["alpha"] for row in chosen])),
        "chosen_l1_ratio_counts": {str(key): int(value) for key, value in l1_counts.items()},
    }


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = "| Preset | Mean RMSE | Std | Median | Best | Worst | Mean alpha | Chosen l1 ratios |"
    sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for row in rows:
        l1_repr = ", ".join(f"{k}:{v}" for k, v in row["chosen_l1_ratio_counts"].items())
        lines.append(
            f"| {row['model']} | {row['mean_rmse']:.4f} | {row['std_rmse']:.4f} | {row['median_rmse']:.4f} | {row['best_rmse']:.4f} | {row['worst_rmse']:.4f} | {row['chosen_alpha_mean']:.4f} | {l1_repr} |"
        )
    return "\n".join(lines)


def write_outputs(output_dir: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps({"ranked": rows}, indent=2), encoding="utf-8")

    report = f"""# ElasticNet Baseline Sweep

- Validation strategy: `{args.n_splits}` repeated stratified shuffle splits
- Validation size per split: `{args.test_size}` rows
- Focus: small set of baseline-like ElasticNetCV presets derived from `model_1.py`

## Ranked Results

{markdown_table(rows)}

## Presets Evaluated

- `baseline_v1`: exact `model_1.py`
- `baseline_v2_more_alphas`: same l1 ratios with denser alpha path and stricter optimization
- `baseline_v3_low_l1`: concentrates search near low `l1_ratio` values, since the fitted baseline tends to choose `0.1`
- `baseline_v4_low_l1_mvalue`: same low-l1 search with M-values
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    X, y = load_training_data(Path(args.data_dir))
    age_bins = make_age_bins(y)
    splitter = StratifiedShuffleSplit(
        n_splits=args.n_splits,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    splits = list(splitter.split(np.zeros((len(y), 1)), age_bins))

    rows = []
    for preset in select_presets(args.presets):
        print(f"evaluating {preset.name}...", flush=True)
        estimator = build_elasticnet_from_preset(preset)
        rows.append(
            {
                "model": preset.name,
                "notes": preset.notes,
                "alpha_count": len(preset.l1_ratio_grid) * preset.n_alphas,
                **evaluate(estimator, X, y, splits),
            }
        )

    rows.sort(key=lambda row: (row["mean_rmse"], row["worst_rmse"]))
    write_outputs(Path(args.output_dir), rows, args)
    print(f"Wrote results to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
