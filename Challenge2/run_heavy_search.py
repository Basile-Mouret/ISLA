import argparse
import hashlib
import itertools
import logging
import os

import numpy as np
import pandas as pd
from mi_models import SUBJECTS, build_model, load_subject_data, weighted_vote
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


MODEL_SPECS = [
    {"model": "mne_fbcsp_lda", "family": "fbcsp"},
    {"model": "fbcsp_broad_c6_k16_lda", "family": "fbcsp"},
    {"model": "fbcsp_dense_c6_k20_lda", "family": "fbcsp"},
    {"model": "mne_csp_8_30_lda", "family": "csp"},
    {"model": "riemann_ts_lr_6_35", "family": "riemann_ts"},
    {"model": "riemann_ts_lr_8_30", "family": "riemann_ts"},
    {"model": "riemann_fgmdm_8_30", "family": "riemann_fgmdm"},
    {"model": "riemann_fgmdm_8_30_lwf", "family": "riemann_fgmdm"},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heavy subject-wise model search with diversity-aware top-k voting selection."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--sample-rate-hz", type=float, default=256.0)
    parser.add_argument("--cv-folds", type=int, default=4)

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
    parser.add_argument("--models-md", default="submissions/heavy_subject_models.md")
    parser.add_argument("--cache-dir", default="submissions/heavy_oof_cache")
    parser.add_argument("--output-model-name", default="heavy_top3_diverse")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def setup_logging(log_file_path):
    logger = logging.getLogger("heavy_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def run_tag(args):
    payload = repr(
        (
            args.cv_folds,
            args.start_min,
            args.start_max,
            args.start_step,
            args.stop_min,
            args.stop_max,
            args.stop_step,
            args.min_window_len,
            args.top_per_model,
            args.max_candidates_per_subject,
            args.min_ensemble_size,
            args.max_ensemble_size,
            args.max_same_model,
            args.max_same_family,
            args.min_window_shift,
            args.min_disagreement,
            args.diversity_weight,
            args.double_fault_weight,
            args.sample_rate_hz,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def make_window_grid(n_times, args):
    starts = list(range(args.start_min, args.start_max + 1, args.start_step))
    stops = list(range(args.stop_min, args.stop_max + 1, args.stop_step))
    if n_times not in stops:
        stops.append(n_times)

    windows = []
    for start in starts:
        for stop in stops:
            if stop > n_times:
                continue
            if stop - start < args.min_window_len:
                continue
            if stop <= start:
                continue
            windows.append((start, stop))

    windows.append((0, n_times))
    windows = sorted(set(windows))
    return windows


def candidate_key(subject, model_name, start, stop):
    return (subject, model_name, int(start), int(stop))


def oof_cache_path(cache_dir, subject, model_name, start, stop):
    key = f"{subject}__{model_name}__{int(start)}__{int(stop)}".replace("/", "_")
    return os.path.join(cache_dir, f"{key}.npy")


def load_results(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = ["subject", "model", "family", "start", "stop", "mean_accuracy", "std_accuracy"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in existing results file: {path}")
    df["start"] = df["start"].astype(int)
    df["stop"] = df["stop"].astype(int)
    df = df.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="last")
    return df


def append_result_row(path, row):
    row_df = pd.DataFrame([row])
    write_header = not os.path.exists(path)
    row_df.to_csv(path, mode="a", header=write_header, index=False)


def estimate_fit_counts(subjects, windows, cv_folds):
    candidates = len(subjects) * len(MODEL_SPECS) * len(windows)
    cv_fits = candidates * cv_folds
    return candidates, cv_fits


def top_candidate_pool(subject_results, args):
    parts = []
    for model_name in subject_results["model"].unique():
        model_rows = (
            subject_results[subject_results["model"] == model_name]
            .sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
            .head(args.top_per_model)
        )
        parts.append(model_rows)

    if not parts:
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="first")
    pool = pool.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
    pool = pool.head(args.max_candidates_per_subject).reset_index(drop=True)
    return pool


def disagreement(pred_a, pred_b):
    return float(np.mean(pred_a != pred_b))


def double_fault(pred_a, pred_b, y_true):
    return float(np.mean((pred_a != y_true) & (pred_b != y_true)))


def combo_valid(combo_rows, args):
    model_counts = {}
    family_counts = {}
    for row in combo_rows:
        model_counts[row.model] = model_counts.get(row.model, 0) + 1
        family_counts[row.family] = family_counts.get(row.family, 0) + 1
        if model_counts[row.model] > args.max_same_model:
            return False
        if family_counts[row.family] > args.max_same_family:
            return False

    for left, right in itertools.combinations(combo_rows, 2):
        if left.family != right.family:
            continue
        if abs(int(left.start) - int(right.start)) < args.min_window_shift and abs(int(left.stop) - int(right.stop)) < args.min_window_shift:
            return False
    return True


def combo_metrics(combo_rows, oof_lookup, y_true, args):
    oof_rows = [oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] for row in combo_rows]
    weights = [float(row.mean_accuracy) for row in combo_rows]
    ensemble_pred = weighted_vote(oof_rows, weights)
    ensemble_acc = accuracy_score(y_true, ensemble_pred)

    if len(combo_rows) == 1:
        avg_disagreement = 0.0
        avg_double_fault = 0.0
    else:
        disagreements = []
        double_faults = []
        for left_index in range(len(combo_rows)):
            for right_index in range(left_index + 1, len(combo_rows)):
                left_pred = oof_rows[left_index]
                right_pred = oof_rows[right_index]
                disagreements.append(disagreement(left_pred, right_pred))
                double_faults.append(double_fault(left_pred, right_pred, y_true))
        avg_disagreement = float(np.mean(disagreements))
        avg_double_fault = float(np.mean(double_faults))

    score = (
        ensemble_acc
        + args.diversity_weight * avg_disagreement
        - args.double_fault_weight * avg_double_fault
    )
    return {
        "ensemble_accuracy": float(ensemble_acc),
        "avg_disagreement": avg_disagreement,
        "avg_double_fault": avg_double_fault,
        "score": float(score),
    }


def select_diverse_combo(subject_pool, oof_lookup, y_true, args):
    rows = list(subject_pool.itertuples(index=False))
    best_combo = None
    best_metrics = None

    sizes = list(range(args.min_ensemble_size, args.max_ensemble_size + 1))
    if args.min_ensemble_size <= 1:
        sizes = [1] + sizes

    for size in sizes:
        for combo_idx in itertools.combinations(range(len(rows)), size):
            combo_rows = [rows[index] for index in combo_idx]
            if not combo_valid(combo_rows, args):
                continue

            metrics = combo_metrics(combo_rows, oof_lookup, y_true, args)
            if size > 1 and metrics["avg_disagreement"] < args.min_disagreement:
                continue

            if best_combo is None:
                best_combo = combo_rows
                best_metrics = metrics
                continue

            left = (metrics["score"], metrics["ensemble_accuracy"], metrics["avg_disagreement"], -metrics["avg_double_fault"])
            right = (best_metrics["score"], best_metrics["ensemble_accuracy"], best_metrics["avg_disagreement"], -best_metrics["avg_double_fault"])
            if left > right:
                best_combo = combo_rows
                best_metrics = metrics

    if best_combo is None:
        fallback = [rows[0]]
        best_combo = fallback
        best_metrics = combo_metrics(fallback, oof_lookup, y_true, args)

    return best_combo, best_metrics


def write_models_csv(path, selected_rows):
    out_rows = []
    for subject in SUBJECTS:
        subject_rows = [row for row in selected_rows if row["subject"] == subject]
        subject_rows = sorted(subject_rows, key=lambda r: r["rank"])
        for row in subject_rows:
            out_rows.append(
                {
                    "subject": row["subject"],
                    "rank": row["rank"],
                    "model": row["model"],
                    "start": row["start"],
                    "stop": row["stop"],
                    "weight": row["weight"],
                    "note": "heavy-diverse",
                }
            )
    pd.DataFrame(out_rows).to_csv(path, index=False)


def write_models_md(path, selected_rows):
    lines = []
    lines.append("# Heavy Search Subject Models")
    lines.append("")
    lines.append("Top-k weighted vote selected with diversity constraints (disagreement + double-fault aware selection).")
    lines.append("")
    lines.append("| subject | rank | model | start | stop | weight | note |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")

    for subject in SUBJECTS:
        subject_rows = [row for row in selected_rows if row["subject"] == subject]
        subject_rows = sorted(subject_rows, key=lambda r: r["rank"])
        for row in subject_rows:
            lines.append(
                f"| {row['subject']} | {row['rank']} | {row['model']} | {row['start']} | {row['stop']} | {row['weight']:.12f} | heavy-diverse |"
            )

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.submissions_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    tag = run_tag(args)
    log_file = args.log_file or os.path.join(args.submissions_dir, f"heavy_search_{tag}.log")
    logger = setup_logging(log_file)

    logger.info("Run tag: %s", tag)
    logger.info("Results CSV: %s", args.results_csv)
    logger.info("Selection CSV: %s", args.selection_csv)
    logger.info("Models CSV: %s", args.models_csv)
    logger.info("Models MD: %s", args.models_md)
    logger.info("Output model name: %s", args.output_model_name)

    X_a, _, _ = load_subject_data(args.data_dir, "A")
    n_times = X_a.shape[-1]
    windows = make_window_grid(n_times, args)
    logger.info("Window grid size: %d", len(windows))

    total_candidates, total_cv_fits = estimate_fit_counts(SUBJECTS, windows, args.cv_folds)

    existing = load_results(args.results_csv)
    done_keys = set()
    if not existing.empty:
        done_keys = {
            candidate_key(row.subject, row.model, row.start, row.stop)
            for row in existing.itertuples(index=False)
            if os.path.exists(oof_cache_path(args.cache_dir, row.subject, row.model, row.start, row.stop))
        }
    remaining_configs = total_candidates - len(done_keys)
    remaining_cv_fits = remaining_configs * args.cv_folds

    logger.info("Total candidate configs: %d", total_candidates)
    logger.info("Total CV fits (cold run): %d", total_cv_fits)
    logger.info("Cached configs with OOF present: %d", len(done_keys))
    logger.info("Remaining configs: %d", remaining_configs)
    logger.info("Remaining CV fits: %d", remaining_cv_fits)

    if args.estimate_only:
        return

    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    processed = 0

    for subject in SUBJECTS:
        X_train, y_train, _ = load_subject_data(args.data_dir, subject)
        split_list = list(splitter.split(X_train, y_train))
        logger.info("Searching subject %s (%d train samples)", subject, X_train.shape[0])

        for spec in MODEL_SPECS:
            model_name = spec["model"]
            family = spec["family"]
            for start, stop in windows:
                key = candidate_key(subject, model_name, start, stop)
                cache_path = oof_cache_path(args.cache_dir, subject, model_name, start, stop)
                if key in done_keys:
                    logger.info("[cache skip] %s | %s | samples_%d_%d", subject, model_name, start, stop)
                    continue

                processed += 1
                label = f"{subject} | {model_name} | samples_{start}_{stop}"
                logger.info("[search %d/%d] START %s", processed, remaining_configs, label)

                oof_pred = np.empty(y_train.shape[0], dtype="<U16")
                fold_scores = []
                for fold_index, (train_idx, valid_idx) in enumerate(split_list, start=1):
                    logger.info("[search %d/%d] FOLD %d/%d %s", processed, remaining_configs, fold_index, len(split_list), label)
                    model = build_model(model_name=model_name, sample_rate_hz=args.sample_rate_hz, start=start, stop=stop)
                    model.fit(X_train[train_idx], y_train[train_idx])
                    pred = model.predict(X_train[valid_idx])
                    oof_pred[valid_idx] = pred
                    score = accuracy_score(y_train[valid_idx], pred)
                    fold_scores.append(score)
                    logger.info("[search %d/%d] FOLD %d/%d SCORE %.4f %s", processed, remaining_configs, fold_index, len(split_list), score, label)

                mean_score = float(np.mean(fold_scores))
                std_score = float(np.std(fold_scores))
                np.save(cache_path, oof_pred)

                row = {
                    "subject": subject,
                    "model": model_name,
                    "family": family,
                    "start": start,
                    "stop": stop,
                    "mean_accuracy": mean_score,
                    "std_accuracy": std_score,
                }
                append_result_row(args.results_csv, row)
                done_keys.add(key)
                logger.info("[search %d/%d] DONE %s mean=%.4f std=%.4f", processed, remaining_configs, label, mean_score, std_score)

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

        oof_lookup = {}
        for row in pool.itertuples(index=False):
            cache_path = oof_cache_path(args.cache_dir, row.subject, row.model, row.start, row.stop)
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Missing OOF cache for {row.subject}/{row.model}/{row.start}:{row.stop}: {cache_path}")
            oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] = np.load(cache_path)

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

        pred_rows = []
        weights = []
        for rank, row in enumerate(selected_combo, start=1):
            model = build_model(model_name=row.model, sample_rate_hz=args.sample_rate_hz, start=int(row.start), stop=int(row.stop))
            model.fit(X_train, y_train)
            pred_rows.append(model.predict(X_test))
            weights.append(float(row.mean_accuracy))
            model_rows.append(
                {
                    "subject": subject,
                    "rank": rank,
                    "model": row.model,
                    "start": int(row.start),
                    "stop": int(row.stop),
                    "weight": float(row.mean_accuracy),
                    "family": row.family,
                }
            )

        final_predictions[subject] = weighted_vote(pred_rows, weights)
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
    selection_df.to_csv(args.selection_csv, index=False)

    write_models_csv(args.models_csv, model_rows)
    write_models_md(args.models_md, model_rows)

    output_dir = os.path.join(args.submissions_dir, args.output_model_name)
    os.makedirs(output_dir, exist_ok=True)
    for subject in SUBJECTS:
        pd.DataFrame({"y_pred": final_predictions[subject]}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"),
            index=False,
        )

    from package_submission import package_submission_dir

    zip_path = package_submission_dir(output_dir, os.path.join(args.submissions_dir, f"{args.output_model_name}.zip"))
    logger.info("Generated submission zip: %s", zip_path)
    logger.info("Selection summary:\n%s", selection_df.to_string(index=False))


if __name__ == "__main__":
    main()
