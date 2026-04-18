import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from utils.file_manager import (
    append_result_row,
    candidate_key,
    cached_result_keys,
    ensure_parent_dir,
    load_results,
    load_oof_lookup,
    load_subject_data,
    oof_cache_path,
    package_submission_dir,
    write_prediction_dir,
)
from utils.helpers import (
    SUBJECTS,
    estimate_fit_counts,
    make_window_grid,
    run_tag,
    set_thread_env,
    setup_logging,
)
from utils.models import MODEL_SPECS, evaluate_candidate_task, predict_weighted_ensemble
from utils.selection import (
    select_diverse_combo,
    top_candidate_pool,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heavy subject-wise model search with diversity-aware top-k voting selection."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--sample-rate-hz", type=float, default=256.0)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=summary only, 1=per-candidate, 2=per-fold.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel candidate workers.")
    parser.add_argument(
        "--inner-threads",
        type=int,
        default=1,
        help="Number of BLAS/OpenMP threads per worker (set to 1 to avoid oversubscription).",
    )

    parser.add_argument("--start-min", type=int, default=192)
    parser.add_argument("--start-max", type=int, default=960)
    parser.add_argument("--start-step", type=int, default=64)
    parser.add_argument("--stop-min", type=int, default=1088)
    parser.add_argument("--stop-max", type=int, default=1537)
    parser.add_argument("--stop-step", type=int, default=64)
    parser.add_argument("--min-window-len", type=int, default=384)

    parser.add_argument("--top-per-model", type=int, default=8)
    parser.add_argument("--max-candidates-per-subject", type=int, default=48)

    parser.add_argument("--min-ensemble-size", type=int, default=2)
    parser.add_argument("--max-ensemble-size", type=int, default=3)
    parser.add_argument("--max-same-model", type=int, default=1)
    parser.add_argument("--max-same-family", type=int, default=2)
    parser.add_argument("--min-window-shift", type=int, default=96)
    parser.add_argument("--min-disagreement", type=float, default=0.04)
    parser.add_argument("--diversity-weight", type=float, default=0.03)
    parser.add_argument("--double-fault-weight", type=float, default=0.02)

    parser.add_argument("--estimate-only", action="store_true")

    parser.add_argument("--results-csv", default="submissions/heavy_search_results.csv")
    parser.add_argument("--selection-csv", default="submissions/heavy_search_selection.csv")
    parser.add_argument("--models-csv", default="submissions/heavy_subject_models.csv")
    parser.add_argument("--cache-dir", default="submissions/heavy_oof_cache")
    parser.add_argument("--output-model-name", default="heavy_top3_diverse")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.submissions_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    set_thread_env(args.inner_threads)

    tag = run_tag(args)
    log_file = args.log_file or os.path.join(args.submissions_dir, f"heavy_search_{tag}.log")
    logger = setup_logging(log_file, args.verbose)

    logger.info("Run tag: %s", tag)
    logger.info("Results CSV: %s", args.results_csv)
    logger.info("Selection CSV: %s", args.selection_csv)
    logger.info("Models CSV: %s", args.models_csv)
    logger.info("Output model name: %s", args.output_model_name)
    logger.info("Verbose level: %d", args.verbose)
    logger.info("Parallel workers (n_jobs): %d", args.n_jobs)
    logger.info("Inner threads per worker: %d", args.inner_threads)

    X_a, _, _ = load_subject_data(args.data_dir, "A")
    windows = make_window_grid(X_a.shape[-1], args)
    logger.info("Window grid size: %d", len(windows))

    total_candidates, total_cv_fits = estimate_fit_counts(SUBJECTS, MODEL_SPECS, windows, args.cv_folds)
    existing = load_results(args.results_csv)
    done_keys = cached_result_keys(existing, args.cache_dir) if not existing.empty else set()
    remaining_configs = total_candidates - len(done_keys)

    logger.info("Total candidate configs: %d", total_candidates)
    logger.info("Total CV fits (cold run): %d", total_cv_fits)
    logger.info("Cached configs with OOF present: %d", len(done_keys))
    logger.info("Remaining configs: %d", remaining_configs)
    logger.info("Remaining CV fits: %d", remaining_configs * args.cv_folds)

    if args.estimate_only:
        return

    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    processed = 0

    for subject in SUBJECTS:
        X_train, y_train, _ = load_subject_data(args.data_dir, subject)
        split_list = list(splitter.split(X_train, y_train))
        logger.info("Searching subject %s (%d train samples)", subject, X_train.shape[0])

        pending = []
        for spec in MODEL_SPECS:
            model_name = spec["model"]
            family = spec["family"]
            for start, stop in windows:
                key = candidate_key(subject, model_name, start, stop)
                if key in done_keys:
                    logger.info("[cache skip] %s | %s | samples_%d_%d", subject, model_name, start, stop)
                    continue
                cache_path = oof_cache_path(args.cache_dir, subject, model_name, start, stop)
                pending.append((
                    model_name,
                    family,
                    start,
                    stop,
                    key,
                    cache_path,
                ))

        logger.info("Subject %s pending candidates: %d", subject, len(pending))
        if not pending:
            continue

        if args.n_jobs <= 1:
            for model_name, family, start, stop, key, cache_path in pending:
                processed += 1
                label = f"{subject} | {model_name} | samples_{start}_{stop}"
                logger.info("[search %d/%d] START %s", processed, remaining_configs, label)
                row = evaluate_candidate_task(
                    subject=subject,
                    model_name=model_name,
                    family=family,
                    start=start,
                    stop=stop,
                    sample_rate_hz=args.sample_rate_hz,
                    X_train=X_train,
                    y_train=y_train,
                    split_list=split_list,
                    cache_path=cache_path,
                    inner_threads=args.inner_threads,
                    logger=logger,
                    verbose=args.verbose,
                    log_label=f"[search {processed}/{remaining_configs}] {label}",
                )
                append_result_row(args.results_csv, row)
                done_keys.add(key)
                logger.info(
                    "[search %d/%d] DONE %s mean=%.4f std=%.4f",
                    processed,
                    remaining_configs,
                    label,
                    row["mean_accuracy"],
                    row["std_accuracy"],
                )
        else:
            with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
                future_map = {}
                for model_name, family, start, stop, key, cache_path in pending:
                    label = f"{subject} | {model_name} | samples_{start}_{stop}"
                    logger.info("[submit] %s", label)
                    future = executor.submit(
                        evaluate_candidate_task,
                        subject,
                        model_name,
                        family,
                        start,
                        stop,
                        args.sample_rate_hz,
                        X_train,
                        y_train,
                        split_list,
                        cache_path,
                        args.inner_threads,
                        logger,
                        args.verbose,
                        label,
                    )
                    future_map[future] = (model_name, start, stop, key)

                for future in as_completed(future_map):
                    model_name, start, stop, key = future_map[future]
                    processed += 1
                    label = f"{subject} | {model_name} | samples_{start}_{stop}"
                    row = future.result()
                    append_result_row(args.results_csv, row)
                    done_keys.add(key)
                    logger.info(
                        "[search %d/%d] DONE %s mean=%.4f std=%.4f",
                        processed,
                        remaining_configs,
                        label,
                        row["mean_accuracy"],
                        row["std_accuracy"],
                    )

    results = load_results(args.results_csv)
    selection_rows = []
    model_rows = []
    final_predictions = {}

    for subject in SUBJECTS:
        X_train, y_train, X_test = load_subject_data(args.data_dir, subject)
        subject_results = results[results["subject"] == subject]
        if subject_results.empty:
            raise ValueError(f"No search results found for subject {subject}")

        pool = top_candidate_pool(subject_results, args)
        logger.info("Subject %s pool size: %d", subject, len(pool))

        oof_lookup = load_oof_lookup(pool, args.cache_dir)

        selected_combo, metrics = select_diverse_combo(pool, oof_lookup, y_train, args)
        logger.info(
            "Subject %s selected combo size=%d acc=%.4f disagree=%.4f double_fault=%.4f score=%.4f",
            subject,
            len(selected_combo),
            metrics["ensemble_accuracy"],
            metrics["avg_disagreement"],
            metrics["avg_double_fault"],
            metrics["score"],
        )

        subject_rows = [
            {
                "subject": subject,
                "rank": rank,
                "model": row.model,
                "family": row.family,
                "start": int(row.start),
                "stop": int(row.stop),
                "weight": float(row.mean_accuracy),
            }
            for rank, row in enumerate(selected_combo, start=1)
        ]

        final_predictions[subject] = predict_weighted_ensemble(subject_rows, X_train, y_train, X_test, args.sample_rate_hz)
        model_rows.extend(subject_rows)
        selection_rows.append(
            {
                "subject": subject,
                "ensemble_size": len(selected_combo),
                "ensemble_oof_accuracy": metrics["ensemble_accuracy"],
                "avg_disagreement": metrics["avg_disagreement"],
                "avg_double_fault": metrics["avg_double_fault"],
                "selection_score": metrics["score"],
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values("subject")
    ensure_parent_dir(args.selection_csv)
    selection_df.to_csv(args.selection_csv, index=False)
    ensure_parent_dir(args.models_csv)
    pd.DataFrame(model_rows).sort_values(["subject", "rank"]).to_csv(args.models_csv, index=False)

    output_dir = os.path.join(args.submissions_dir, args.output_model_name)
    write_prediction_dir(output_dir, final_predictions)
    zip_path = package_submission_dir(output_dir, os.path.join(args.submissions_dir, f"{args.output_model_name}.zip"))
    logger.info("Generated submission zip: %s", zip_path)
    logger.info("Selection summary:\n%s", selection_df.to_string(index=False))


if __name__ == "__main__":
    main()
