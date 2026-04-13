from __future__ import annotations

import argparse
import json
import math
import warnings
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import ParameterGrid, cross_validate
from sklearn.pipeline import Pipeline

from pipelines import (
    ensure_target,
    get_benchmark_specs,
    get_finalist_specs,
    make_cv_splits,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing either train/*.csv or X_train.csv/y_train.csv.",
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory where markdown reports are written.",
    )
    parser.add_argument(
        "--suite",
        choices=["benchmark", "finalists"],
        default="benchmark",
        help="Experiment suite to run.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated list of model names to run. Defaults to all models in the suite.",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--sklearn-verbose",
        type=int,
        default=0,
        help="Sklearn search verbosity. Keep at 0 for model-level logs only; use 1 for coarse search summaries.",
    )
    parser.add_argument(
        "--detailed-step-logs",
        action="store_true",
        help="Enable verbose inner pipeline and per-fit sklearn logs.",
    )
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
    raise FileNotFoundError(
        f"Could not find training files under {data_dir}. Expected either X_train.csv/y_train.csv or train/X_train.csv/train/y_train.csv."
    )


def load_training_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    X_path, y_path = resolve_training_paths(data_dir)
    X = pd.read_csv(X_path)
    y = ensure_target(pd.read_csv(y_path))
    return X, y


def filter_specs(specs, models_arg: str | None):
    if not models_arg:
        return specs

    requested_names = [name.strip() for name in models_arg.split(",") if name.strip()]
    if not requested_names:
        return specs

    spec_by_name = {spec.name: spec for spec in specs}
    unknown = [name for name in requested_names if name not in spec_by_name]
    if unknown:
        available = ", ".join(sorted(spec_by_name))
        raise ValueError(
            f"Unknown model(s): {', '.join(unknown)}. Available models: {available}"
        )

    return [spec_by_name[name] for name in requested_names]


def format_params(params: dict[str, Any] | None) -> str:
    if not params:
        return "-"
    pairs = []
    for key in sorted(params):
        value = params[key]
        if isinstance(value, float):
            value_repr = f"{value:.6g}"
        else:
            value_repr = str(value)
        pairs.append(f"`{key}`={value_repr}")
    return ", ".join(pairs)


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def enable_step_verbosity(estimator: Any, sklearn_verbose: int) -> Any:
    if isinstance(estimator, Pipeline):
        estimator.verbose = True
        for _, step in estimator.steps:
            enable_step_verbosity(step, sklearn_verbose)
        return estimator

    if isinstance(estimator, ColumnTransformer):
        estimator.verbose = True
        for _, transformer, _ in estimator.transformers:
            if transformer not in ("drop", "passthrough"):
                enable_step_verbosity(transformer, sklearn_verbose)
        return estimator

    if hasattr(estimator, "verbose"):
        current_value = getattr(estimator, "verbose")
        if isinstance(current_value, bool):
            setattr(estimator, "verbose", True)
        else:
            setattr(estimator, "verbose", sklearn_verbose)

    return estimator


def choose_progress_interval(total_candidates: int, target_updates: int = 8) -> int:
    if total_candidates <= 10:
        return 1

    raw_interval = max(1.0, total_candidates / target_updates)
    magnitude = 10 ** int(math.floor(math.log10(raw_interval)))
    normalized = raw_interval / magnitude

    if normalized <= 1:
        nice_step = 1
    elif normalized <= 2:
        nice_step = 2
    elif normalized <= 5:
        nice_step = 5
    else:
        nice_step = 10

    return int(nice_step * magnitude)


def evaluate_spec(
    spec,
    X,
    y,
    cv_splits,
    n_jobs: int,
    sklearn_verbose: int,
    detailed_step_logs: bool,
    model_index: int,
    total_models: int,
) -> dict[str, Any]:
    estimator = clone(spec.estimator)
    if detailed_step_logs:
        estimator = enable_step_verbosity(estimator, sklearn_verbose)

    n_candidates = len(ParameterGrid(spec.param_grid)) if spec.param_grid else 1
    n_cv_fits = n_candidates * len(cv_splits)
    remaining_after_current = total_models - model_index

    log(
        f"[{model_index}/{total_models}] {spec.name}: {n_candidates} candidate(s) x {len(cv_splits)} split(s) = {n_cv_fits} CV fit(s), {remaining_after_current} model(s) remaining after this one"
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        started = perf_counter()

        if spec.param_grid:
            all_params = list(ParameterGrid(spec.param_grid))
            progress_interval = choose_progress_interval(n_candidates)
            if n_candidates > 1:
                log(
                    f"{spec.name}: reporting progress every {progress_interval} candidate(s)"
                )

            candidate_results = []
            for candidate_index, params in enumerate(all_params, start=1):
                candidate_estimator = clone(estimator)
                candidate_estimator.set_params(**params)
                candidate_started = perf_counter()
                scores = cross_validate(
                    estimator=candidate_estimator,
                    X=X,
                    y=y,
                    scoring="neg_root_mean_squared_error",
                    cv=cv_splits,
                    n_jobs=n_jobs,
                    return_train_score=False,
                    verbose=sklearn_verbose if detailed_step_logs else 0,
                )
                candidate_elapsed = perf_counter() - candidate_started
                rmse_scores = -scores["test_score"]
                candidate_results.append(
                    {
                        "name": spec.name,
                        "notes": spec.notes,
                        "rmse": float(rmse_scores.mean()),
                        "rmse_std": float(rmse_scores.std()),
                        "fit_time": float(scores["fit_time"].mean()),
                        "params": params,
                        "candidate_elapsed_seconds": float(candidate_elapsed),
                    }
                )

                should_log_progress = (
                    candidate_index % progress_interval == 0
                    or candidate_index == n_candidates
                )
                if should_log_progress and n_candidates > 1:
                    elapsed_so_far = perf_counter() - started
                    avg_candidate_time = elapsed_so_far / candidate_index
                    eta_seconds = avg_candidate_time * (n_candidates - candidate_index)
                    completed_fits = candidate_index * len(cv_splits)
                    log(
                        f"{spec.name} progress: {candidate_index}/{n_candidates} candidates complete ({completed_fits}/{n_cv_fits} CV fits), elapsed={elapsed_so_far:.1f}s, eta={eta_seconds:.1f}s"
                    )

            ranked_candidates = sorted(
                candidate_results,
                key=lambda row: (row["rmse"], row["rmse_std"]),
            )
            best_candidate = ranked_candidates[0]

            refit_started = perf_counter()
            best_estimator = clone(estimator)
            best_estimator.set_params(**best_candidate["params"])
            best_estimator.fit(X, y)
            refit_time = perf_counter() - refit_started
            elapsed = perf_counter() - started
            top_candidates = [
                {
                    "rmse": row["rmse"],
                    "rmse_std": row["rmse_std"],
                    "params": row["params"],
                }
                for row in ranked_candidates[:5]
            ]

            return {
                "name": spec.name,
                "notes": spec.notes,
                "rmse": best_candidate["rmse"],
                "rmse_std": best_candidate["rmse_std"],
                "fit_time": best_candidate["fit_time"],
                "params": best_candidate["params"],
                "top_candidates": top_candidates,
                "n_candidates": n_candidates,
                "n_cv_fits": n_cv_fits,
                "elapsed_seconds": float(elapsed),
                "refit_time": float(refit_time),
            }

        scores = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring="neg_root_mean_squared_error",
            cv=cv_splits,
            n_jobs=n_jobs,
            return_train_score=False,
            verbose=sklearn_verbose if detailed_step_logs else 0,
        )
        elapsed = perf_counter() - started
        rmse_scores = -scores["test_score"]
        return {
            "name": spec.name,
            "notes": spec.notes,
            "rmse": float(rmse_scores.mean()),
            "rmse_std": float(rmse_scores.std()),
            "fit_time": float(scores["fit_time"].mean()),
            "params": None,
            "top_candidates": [],
            "n_candidates": n_candidates,
            "n_cv_fits": n_cv_fits,
            "elapsed_seconds": float(elapsed),
            "refit_time": 0.0,
        }


def render_markdown(
    suite: str,
    generated_at: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
    run_status: str,
    completed_models: int,
    total_models: int,
    results: list[dict[str, Any]],
) -> str:
    lines = [
        f"# {suite.title()} Experiments",
        "",
        f"- Generated at: `{generated_at}`",
        f"- Training rows: `{X.shape[0]}`",
        f"- Training columns: `{X.shape[1]}`",
        f"- Target mean age: `{y.mean():.3f}`",
        f"- CV strategy: `{n_repeats} x {n_splits}` repeated stratified folds on age bins",
        f"- Run status: `{run_status}`",
        f"- Completed models: `{completed_models}/{total_models}`",
        "",
        "## Ranked Results",
        "",
        "| Rank | Model | CV RMSE | CV Std | Mean Fit Time (s) | Wall Time (s) | CV Fits | Best Params | Notes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for rank, result in enumerate(results, start=1):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    result["name"],
                    f"{result['rmse']:.4f}",
                    f"{result['rmse_std']:.4f}",
                    f"{result['fit_time']:.2f}",
                    f"{result['elapsed_seconds']:.2f}",
                    str(result["n_cv_fits"]),
                    format_params(result["params"]),
                    result["notes"],
                ]
            )
            + " |"
        )

    lines.extend(["", "## Top Candidate Details", ""])
    for result in results[:5]:
        lines.append(f"### {result['name']}")
        lines.append("")
        lines.append(f"- Best CV RMSE: `{result['rmse']:.4f}`")
        lines.append(f"- Fold-to-fold std: `{result['rmse_std']:.4f}`")
        lines.append(f"- Mean fit time: `{result['fit_time']:.2f}` seconds")
        lines.append(f"- Total wall time: `{result['elapsed_seconds']:.2f}` seconds")
        lines.append(f"- CV fits: `{result['n_cv_fits']}`")
        lines.append(f"- Grid candidates: `{result['n_candidates']}`")
        lines.append(f"- Refit time: `{result['refit_time']:.2f}` seconds")
        lines.append(f"- Best params: {format_params(result['params'])}")
        lines.append(f"- Notes: {result['notes']}")
        if result["top_candidates"]:
            lines.append("")
            lines.append("Top grid candidates:")
            lines.append("")
            lines.append("| RMSE | Std | Params |")
            lines.append("| ---: | ---: | --- |")
            for candidate in result["top_candidates"][:3]:
                lines.append(
                    f"| {candidate['rmse']:.4f} | {candidate['rmse_std']:.4f} | {format_params(candidate['params'])} |"
                )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    results_dir: Path,
    suite: str,
    markdown: str,
    results: list[dict[str, Any]],
    archive: bool,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    latest_md = results_dir / f"latest_{suite}.md"
    latest_json = results_dir / f"latest_{suite}.json"

    latest_md.write_text(markdown, encoding="utf-8")
    latest_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if archive:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ")
        archive_md = results_dir / f"{timestamp}_{suite}.md"
        archive_md.write_text(markdown, encoding="utf-8")


def persist_results(
    results_dir: Path,
    suite: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int,
    n_repeats: int,
    total_models: int,
    results: list[dict[str, Any]],
    run_status: str,
    archive: bool,
) -> None:
    sorted_results = sorted(results, key=lambda row: (row["rmse"], row["rmse_std"]))
    generated_at = datetime.now(timezone.utc).isoformat()
    markdown = render_markdown(
        suite=suite,
        generated_at=generated_at,
        X=X,
        y=y,
        n_splits=n_splits,
        n_repeats=n_repeats,
        run_status=run_status,
        completed_models=len(results),
        total_models=total_models,
        results=sorted_results,
    )
    write_outputs(results_dir, suite, markdown, sorted_results, archive=archive)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)

    X, y = load_training_data(data_dir)
    cv_splits = make_cv_splits(
        y,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
    )

    if args.suite == "benchmark":
        specs = get_benchmark_specs(list(X.columns))
    else:
        specs = get_finalist_specs(list(X.columns))
    specs = filter_specs(specs, args.models)

    total_models = len(specs)
    results = []
    try:
        for index, spec in enumerate(specs, start=1):
            result = evaluate_spec(
                spec,
                X,
                y,
                cv_splits,
                n_jobs=args.n_jobs,
                sklearn_verbose=args.sklearn_verbose,
                detailed_step_logs=args.detailed_step_logs,
                model_index=index,
                total_models=total_models,
            )
            results.append(result)
            log(
                f"Completed [{index}/{total_models}] {spec.name}: RMSE={result['rmse']:.4f}, std={result['rmse_std']:.4f}, wall_time={result['elapsed_seconds']:.2f}s, {total_models - index} model(s) remaining"
            )
            persist_results(
                results_dir=results_dir,
                suite=args.suite,
                X=X,
                y=y,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                total_models=total_models,
                results=results,
                run_status=f"in_progress ({index}/{total_models} models complete)",
                archive=False,
            )
            log(f"Saved partial results to {results_dir}")
    except KeyboardInterrupt:
        if results:
            persist_results(
                results_dir=results_dir,
                suite=args.suite,
                X=X,
                y=y,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                total_models=total_models,
                results=results,
                run_status=f"interrupted ({len(results)}/{total_models} models complete)",
                archive=False,
            )
            log(f"Saved interrupted partial results to {results_dir}")
        raise
    except Exception:
        if results:
            persist_results(
                results_dir=results_dir,
                suite=args.suite,
                X=X,
                y=y,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                total_models=total_models,
                results=results,
                run_status=f"failed ({len(results)}/{total_models} models complete)",
                archive=False,
            )
            log(f"Saved failed partial results to {results_dir}")
        raise

    persist_results(
        results_dir=results_dir,
        suite=args.suite,
        X=X,
        y=y,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        total_models=total_models,
        results=results,
        run_status="completed",
        archive=True,
    )
    log(f"Wrote final results to {results_dir}")


if __name__ == "__main__":
    main()
