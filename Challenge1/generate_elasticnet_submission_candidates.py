from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from elasticnet_models import ElasticNetPreset, build_elasticnet_from_preset
from pipelines import ensure_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="submissions/elasticnet_candidates",
        help="Directory where candidate submission folders are written.",
    )
    parser.add_argument(
        "--presets",
        default="candidate_baseline_v1,candidate_fixed_l1_0p1,candidate_fixed_l1_0p5",
        help="Comma-separated candidate preset names to generate.",
    )
    return parser.parse_args()


def resolve_paths(data_dir: Path) -> tuple[Path, Path, Path]:
    direct_train_X = data_dir / "X_train.csv"
    direct_train_y = data_dir / "y_train.csv"
    direct_test_X = data_dir / "X_test.csv"

    nested_train_X = data_dir / "train" / "X_train.csv"
    nested_train_y = data_dir / "train" / "y_train.csv"
    nested_test_X = data_dir / "test" / "X_test.csv"

    if direct_train_X.exists() and direct_train_y.exists() and direct_test_X.exists():
        return direct_train_X, direct_train_y, direct_test_X
    if nested_train_X.exists() and nested_train_y.exists() and nested_test_X.exists():
        return nested_train_X, nested_train_y, nested_test_X
    raise FileNotFoundError(f"Could not resolve train/test CSVs under {data_dir}")


def build_submission_presets() -> list[ElasticNetPreset]:
    return [
        ElasticNetPreset(
            name="candidate_baseline_v1",
            l1_ratio_grid=(0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0),
            n_alphas=25,
            cv=5,
            max_iter=2500,
            tol=1e-3,
            use_m_value=False,
            notes="Exact `model_1.py` baseline reproduction.",
        ),
        ElasticNetPreset(
            name="candidate_baseline_v2_more_alphas",
            l1_ratio_grid=(0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0),
            n_alphas=40,
            cv=5,
            max_iter=6000,
            tol=5e-4,
            use_m_value=False,
            notes="Same ratio grid, denser alpha path, stricter optimization.",
        ),
        ElasticNetPreset(
            name="candidate_fixed_l1_0p1",
            l1_ratio_grid=(0.1,),
            n_alphas=50,
            cv=5,
            max_iter=8000,
            tol=1e-4,
            use_m_value=False,
            notes="Anchored to the low-l1 regime often chosen by the baseline.",
        ),
        ElasticNetPreset(
            name="candidate_fixed_l1_0p5",
            l1_ratio_grid=(0.5,),
            n_alphas=50,
            cv=5,
            max_iter=8000,
            tol=1e-4,
            use_m_value=False,
            notes="Anchored to the medium-sparsity regime often chosen by the baseline.",
        ),
    ]


def select_presets(presets_arg: str) -> list[ElasticNetPreset]:
    preset_map = {preset.name: preset for preset in build_submission_presets()}
    requested = [name.strip() for name in presets_arg.split(",") if name.strip()]
    unknown = [name for name in requested if name not in preset_map]
    if unknown:
        available = ", ".join(sorted(preset_map))
        raise ValueError(f"Unknown preset(s): {', '.join(unknown)}. Available presets: {available}")
    return [preset_map[name] for name in requested]


def fit_and_summarize(preset: ElasticNetPreset, X_train: pd.DataFrame, y_train: pd.Series):
    estimator = build_elasticnet_from_preset(preset)
    estimator.fit(X_train, y_train)
    elasticnet = estimator.pipeline_.named_steps["elasticnet"]
    coef = np.asarray(elasticnet.coef_)
    return estimator, {
        "chosen_alpha": float(elasticnet.alpha_),
        "chosen_l1_ratio": float(elasticnet.l1_ratio_),
        "nonzero_coef_count": int(np.sum(np.abs(coef) > 1e-12)),
        "variance_kept": int(estimator.pipeline_.named_steps["variance_filter"].get_support().sum()),
    }


def markdown_table(rows: list[dict[str, object]]) -> str:
    header = "| Candidate | Chosen alpha | Chosen l1_ratio | Nonzero coefs | Notes |"
    sep = "| --- | ---: | ---: | ---: | --- |"
    lines = [header, sep]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['chosen_alpha']:.6f} | {row['chosen_l1_ratio']:.4f} | {row['nonzero_coef_count']} | {row['notes']} |"
        )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_X_path, train_y_path, test_X_path = resolve_paths(data_dir)
    X_train = pd.read_csv(train_X_path)
    y_train = ensure_target(pd.read_csv(train_y_path))
    X_test = pd.read_csv(test_X_path)

    summary_rows = []
    metadata = []
    for preset in select_presets(args.presets):
        print(f"training {preset.name}...", flush=True)
        estimator, summary = fit_and_summarize(preset, X_train, y_train)
        prediction = estimator.predict(X_test)

        candidate_dir = output_dir / preset.name
        candidate_dir.mkdir(parents=True, exist_ok=True)
        submission_path = candidate_dir / "y_pred.csv"
        pd.DataFrame({"age": pd.Series(prediction, dtype=float)}).to_csv(submission_path, index=False)

        row = {
            "name": preset.name,
            "notes": preset.notes,
            **summary,
        }
        summary_rows.append(row)
        metadata.append(
            {
                "preset": asdict(preset),
                "summary": row,
                "submission_path": str(submission_path),
            }
        )

    (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = f"""# ElasticNet Submission Candidates

- Source script: `generate_elasticnet_submission_candidates.py`
- Base inspiration: `model_1.py` plus nearby stable variants

## Generated Candidates

{markdown_table(summary_rows)}
"""
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(f"Wrote candidate submissions to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
