import argparse
import hashlib
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from eeg_models import SUBJECTS, load_subject_data, write_submission
from mne_pyriemann_models import build_model as build_base_model
from run_mne_pyriemann_finetune import (
    build_finetune_model,
    family_from_model_name,
    make_cv_splitter,
    prepare_result_rows,
    subject_model_specs,
)
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build subject-wise ensembles from saved base and fine-tune MNE/pyRiemann results."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Directory where logs, caches, reports, and submission archives are written.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate used when rebuilding candidate models.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=SUBJECTS,
        choices=SUBJECTS,
        help="Subjects to include in the ensemble search.",
    )
    parser.add_argument(
        "--base-results-path",
        default="submissions/mne_pyriemann_cv_results.csv",
        help="CSV from the base MNE/pyRiemann search.",
    )
    parser.add_argument(
        "--finetune-results-path",
        default="submissions/mne_pyriemann_finetune_cv_results.csv",
        help="CSV from the fine-tune search.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of stratified folds used to generate OOF predictions.",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=1,
        help="Number of CV repeats. Keep at 1 unless you explicitly want extra cost.",
    )
    parser.add_argument(
        "--max-candidates-per-subject",
        type=int,
        default=6,
        help="Maximum candidate configs retained per subject before ensemble search.",
    )
    parser.add_argument(
        "--per-family-limit",
        type=int,
        default=2,
        help="Maximum candidates per model family retained in the pool.",
    )
    parser.add_argument(
        "--min-window-shift",
        type=int,
        default=64,
        help="Minimum start and stop difference required before treating two same-model windows as distinct.",
    )
    parser.add_argument(
        "--max-ensemble-size",
        type=int,
        default=5,
        help="Maximum number of members selected by the greedy ensemble search per subject.",
    )
    parser.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum OOF accuracy gain required to add a new member to the ensemble.",
    )
    parser.add_argument(
        "--fallback-prediction-dir",
        default="submissions/mne_pyriemann_finetune_subject_top3_vote",
        help="Directory containing subject_A_y_pred.csv ... subject_F_y_pred.csv for subjects not being re-ensembled in this run.",
    )
    parser.add_argument(
        "--output-model-name",
        default=None,
        help="Optional submission folder/zip base name. Defaults to a run-tagged ensemble name.",
    )
    return parser.parse_args()


def make_run_tag(args):
    payload = repr(
        (
            tuple(args.subjects),
            args.sample_rate_hz,
            args.cv_folds,
            args.cv_repeats,
            args.max_candidates_per_subject,
            args.per_family_limit,
            args.min_window_shift,
            args.max_ensemble_size,
            args.min_improvement,
            args.base_results_path,
            args.finetune_results_path,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def setup_logging(log_file_path):
    logger = logging.getLogger("mne_pyriemann_ensemble_search")
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


def load_result_csv(csv_path, source):
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    return prepare_result_rows(pd.read_csv(csv_path), source)


def load_fallback_predictions(fallback_dir, subject):
    csv_path = os.path.join(fallback_dir, f"subject_{subject}_y_pred.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing fallback prediction file: {csv_path}")
    return pd.read_csv(csv_path)["y_pred"].to_numpy()


def candidate_key(row):
    return (row.subject, row.source, row.model, int(row.start), int(row.stop))


def dedupe_result_rows(df):
    if df.empty:
        return df
    return (
        df.sort_values(["subject", "mean_accuracy", "std_accuracy"], ascending=[True, False, True])
        .drop_duplicates(subset=["subject", "source", "model", "start", "stop"], keep="first")
        .reset_index(drop=True)
    )


def window_is_too_similar(left, right, min_window_shift):
    return abs(int(left.start) - int(right.start)) < min_window_shift and abs(int(left.stop) - int(right.stop)) < min_window_shift


def select_subject_candidates(subject_rows, max_candidates, per_family_limit, min_window_shift):
    if subject_rows.empty:
        return subject_rows

    ranked_rows = subject_rows.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True]).reset_index(drop=True)
    selected_indices = []
    family_counts = Counter()

    def can_add(candidate):
        if family_counts[candidate.family] >= per_family_limit:
            return False
        for selected_index in selected_indices:
            selected = ranked_rows.iloc[selected_index]
            if candidate.model == selected.model and window_is_too_similar(candidate, selected, min_window_shift):
                return False
        return True

    selected_indices.append(0)
    family_counts[ranked_rows.iloc[0].family] += 1

    for family in ranked_rows["family"].drop_duplicates():
        family_rows = ranked_rows[ranked_rows["family"] == family]
        for candidate in family_rows.itertuples(index=True):
            if candidate.Index in selected_indices:
                break
            if can_add(candidate):
                selected_indices.append(candidate.Index)
                family_counts[candidate.family] += 1
                break
        if len(selected_indices) >= max_candidates:
            break

    if len(selected_indices) < max_candidates:
        for candidate in ranked_rows.itertuples(index=True):
            if candidate.Index in selected_indices:
                continue
            if can_add(candidate):
                selected_indices.append(candidate.Index)
                family_counts[candidate.family] += 1
            if len(selected_indices) >= max_candidates:
                break

    return ranked_rows.loc[selected_indices].sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True]).reset_index(drop=True)


def cache_root(args):
    return os.path.join(args.submissions_dir, "mne_pyriemann_ensemble_cache")


def candidate_cache_id(row, args):
    payload = repr((candidate_key(row), args.sample_rate_hz, args.cv_folds, args.cv_repeats, "ensemble_v1"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def candidate_cache_path(row, args):
    return os.path.join(cache_root(args), f"{candidate_cache_id(row, args)}.npz")


def resolve_finetune_spec(model_name):
    specs = subject_model_specs([model_name], profile="full")
    for spec in specs:
        if spec["name"] == model_name:
            return spec
    raise ValueError(f"Could not resolve fine-tune model spec for {model_name}")


def build_model_from_row(row, sample_rate_hz):
    if row.source == "base":
        return build_base_model(row.model, sample_rate_hz=sample_rate_hz, start=int(row.start), stop=int(row.stop))

    if row.source == "fine_tuned":
        model_spec = resolve_finetune_spec(row.model)
        return build_finetune_model(
            model_spec=model_spec,
            sample_rate_hz=sample_rate_hz,
            start=int(row.start),
            stop=int(row.stop),
        )

    raise ValueError(f"Unsupported source: {row.source}")


def save_candidate_cache(cache_path, oof_predictions, test_predictions):
    temp_path = f"{cache_path}.tmp.npz"
    np.savez_compressed(
        temp_path,
        oof=np.asarray(oof_predictions),
        test=np.asarray(test_predictions),
    )
    os.replace(temp_path, cache_path)


def load_candidate_cache(cache_path):
    cache = np.load(cache_path, allow_pickle=False)
    return cache["oof"], cache["test"]


def generate_candidate_predictions(row, args, splitter, logger, progress_index, progress_total):
    cache_path = candidate_cache_path(row, args)
    config_label = f"{row.subject} | {row.model} | {row.window} | {row.source}"
    if os.path.exists(cache_path):
        logger.info("[candidate %d/%d] CACHE %s -> %s", progress_index, progress_total, config_label, os.path.basename(cache_path))
        return load_candidate_cache(cache_path)

    X_train, y_train, X_test = load_subject_data(args.data_dir, row.subject)
    oof_predictions = np.empty(y_train.shape[0], dtype="<U16")
    split_iterator = list(splitter.split(X_train, y_train))

    logger.info("[candidate %d/%d] START %s", progress_index, progress_total, config_label)
    for fold_index, (train_indices, valid_indices) in enumerate(split_iterator, start=1):
        logger.info("[candidate %d/%d] FOLD %d/%d %s", progress_index, progress_total, fold_index, len(split_iterator), config_label)
        model = build_model_from_row(row, args.sample_rate_hz)
        model.fit(X_train[train_indices], y_train[train_indices])
        oof_predictions[valid_indices] = model.predict(X_train[valid_indices])

    full_model = build_model_from_row(row, args.sample_rate_hz)
    full_model.fit(X_train, y_train)
    test_predictions = np.asarray(full_model.predict(X_test), dtype="<U16")
    save_candidate_cache(cache_path, oof_predictions, test_predictions)
    logger.info("[candidate %d/%d] DONE %s -> %s", progress_index, progress_total, config_label, os.path.basename(cache_path))
    return oof_predictions, test_predictions


def vote_predictions(prediction_rows, weights):
    labels = sorted({label for predictions in prediction_rows for label in predictions.tolist()})
    label_to_index = {label: index for index, label in enumerate(labels)}
    vote_matrix = np.zeros((prediction_rows[0].shape[0], len(labels)), dtype=float)

    for predictions, weight in zip(prediction_rows, weights):
        for index, label in enumerate(predictions):
            vote_matrix[index, label_to_index[label]] += weight

    return np.asarray([labels[index] for index in vote_matrix.argmax(axis=1)])


def subject_ensemble_score(candidate_rows, prediction_lookup, y_true, weight_mode):
    prediction_rows = [prediction_lookup[candidate_key(row)]["oof"] for row in candidate_rows]
    if weight_mode == "weighted":
        weights = [prediction_lookup[candidate_key(row)]["oof_accuracy"] for row in candidate_rows]
    else:
        weights = [1.0 for _ in candidate_rows]
    voted_predictions = vote_predictions(prediction_rows, weights)
    return accuracy_score(y_true, voted_predictions)


def choose_best_vote_mode(candidate_rows, prediction_lookup, y_true):
    equal_score = subject_ensemble_score(candidate_rows, prediction_lookup, y_true, "equal")
    weighted_score = subject_ensemble_score(candidate_rows, prediction_lookup, y_true, "weighted")
    if weighted_score > equal_score:
        return "weighted", weighted_score
    return "equal", equal_score


def greedy_subject_ensemble(subject, candidate_rows, prediction_lookup, y_true, max_ensemble_size, min_improvement, logger):
    ordered_rows = list(candidate_rows.itertuples(index=False))
    selected = [ordered_rows[0]]
    current_weight_mode, current_score = choose_best_vote_mode(selected, prediction_lookup, y_true)
    logger.info(
        "Subject %s ensemble seed: %s @ %s (%s) OOF=%.4f mode=%s",
        subject,
        selected[0].model,
        selected[0].window,
        selected[0].source,
        current_score,
        current_weight_mode,
    )

    while len(selected) < max_ensemble_size:
        best_candidate = None
        best_weight_mode = current_weight_mode
        best_score = current_score
        best_tiebreak = None

        for candidate in ordered_rows:
            if any(candidate_key(candidate) == candidate_key(selected_row) for selected_row in selected):
                continue

            trial_members = selected + [candidate]
            trial_weight_mode, trial_score = choose_best_vote_mode(trial_members, prediction_lookup, y_true)
            family_new = int(candidate.family not in {row.family for row in selected})
            window_distance = min(
                abs(int(candidate.start) - int(row.start)) + abs(int(candidate.stop) - int(row.stop))
                for row in selected
            )
            tiebreak = (trial_score, family_new, window_distance, prediction_lookup[candidate_key(candidate)]["oof_accuracy"])

            if trial_score > best_score + min_improvement or (
                abs(trial_score - best_score) <= 1e-12 and best_tiebreak is not None and tiebreak > best_tiebreak
            ):
                if trial_score > current_score + min_improvement:
                    best_candidate = candidate
                    best_weight_mode = trial_weight_mode
                    best_score = trial_score
                    best_tiebreak = tiebreak

        if best_candidate is None:
            break

        selected.append(best_candidate)
        current_weight_mode = best_weight_mode
        current_score = best_score
        logger.info(
            "Subject %s ensemble add: %s @ %s (%s) -> OOF=%.4f mode=%s size=%d",
            subject,
            best_candidate.model,
            best_candidate.window,
            best_candidate.source,
            current_score,
            current_weight_mode,
            len(selected),
        )

    return selected, current_weight_mode, current_score


def build_final_predictions(selected_rows, prediction_lookup, weight_mode):
    prediction_rows = [prediction_lookup[candidate_key(row)]["test"] for row in selected_rows]
    if weight_mode == "weighted":
        weights = [prediction_lookup[candidate_key(row)]["oof_accuracy"] for row in selected_rows]
    else:
        weights = [1.0 for _ in selected_rows]
    return vote_predictions(prediction_rows, weights)


def main():
    args = parse_args()
    os.makedirs(args.submissions_dir, exist_ok=True)
    os.makedirs(cache_root(args), exist_ok=True)

    run_tag = make_run_tag(args)
    output_model_name = args.output_model_name or f"mne_pyriemann_subject_greedy_ensemble_{run_tag}"
    logger = setup_logging(os.path.join(args.submissions_dir, f"mne_pyriemann_ensemble_{run_tag}.log"))
    logger.info("Run tag: %s", run_tag)
    logger.info("Cache dir: %s", cache_root(args))
    logger.info("Output model name: %s", output_model_name)

    base_results = load_result_csv(args.base_results_path, "base")
    finetune_results = load_result_csv(args.finetune_results_path, "fine_tuned")
    combined_results = dedupe_result_rows(pd.concat([base_results, finetune_results], ignore_index=True))
    combined_results = combined_results[combined_results["subject"].isin(args.subjects)].reset_index(drop=True)
    if combined_results.empty:
        raise ValueError("No candidate results were found. Run the base and/or fine-tune searches first.")

    candidate_pools = []
    for subject in args.subjects:
        subject_rows = combined_results[combined_results["subject"] == subject]
        candidate_pool = select_subject_candidates(
            subject_rows,
            max_candidates=args.max_candidates_per_subject,
            per_family_limit=args.per_family_limit,
            min_window_shift=args.min_window_shift,
        )
        candidate_pools.append(candidate_pool)
        logger.info(
            "Subject %s candidate pool: %d selected from %d available",
            subject,
            len(candidate_pool),
            len(subject_rows),
        )
        logger.info("Subject %s candidates:\n%s", subject, candidate_pool[["model", "window", "family", "mean_accuracy", "source"]].to_string(index=False))

    candidate_pool_df = pd.concat(candidate_pools, ignore_index=True)
    candidate_pool_path = os.path.join(args.submissions_dir, f"mne_pyriemann_ensemble_candidates_{run_tag}.csv")
    candidate_pool_df.to_csv(candidate_pool_path, index=False)

    splitter = make_cv_splitter(args.cv_folds, args.cv_repeats)
    prediction_lookup = {}
    candidate_rows = list(candidate_pool_df.itertuples(index=False))
    logger.info("Generating or loading predictions for %d candidates.", len(candidate_rows))

    candidate_metrics = []
    for progress_index, row in enumerate(candidate_rows, start=1):
        oof_predictions, test_predictions = generate_candidate_predictions(
            row=row,
            args=args,
            splitter=splitter,
            logger=logger,
            progress_index=progress_index,
            progress_total=len(candidate_rows),
        )
        _, y_train, _ = load_subject_data(args.data_dir, row.subject)
        oof_accuracy = accuracy_score(y_train, oof_predictions)
        prediction_lookup[candidate_key(row)] = {
            "oof": oof_predictions,
            "test": test_predictions,
            "oof_accuracy": oof_accuracy,
        }
        candidate_metrics.append(
            {
                "subject": row.subject,
                "source": row.source,
                "model": row.model,
                "window": row.window,
                "start": row.start,
                "stop": row.stop,
                "family": row.family,
                "previous_mean_accuracy": row.mean_accuracy,
                "previous_std_accuracy": row.std_accuracy,
                "oof_accuracy": oof_accuracy,
                "cache_path": candidate_cache_path(row, args),
            }
        )

    candidate_metrics_df = pd.DataFrame(candidate_metrics)
    candidate_metrics_path = os.path.join(args.submissions_dir, f"mne_pyriemann_ensemble_candidate_metrics_{run_tag}.csv")
    candidate_metrics_df.to_csv(candidate_metrics_path, index=False)

    selected_member_rows = []
    final_predictions = {}
    summary_rows = []
    for subject in args.subjects:
        subject_pool = candidate_pool_df[candidate_pool_df["subject"] == subject].copy()
        subject_pool = subject_pool.merge(
            candidate_metrics_df[["subject", "source", "model", "start", "stop", "oof_accuracy"]],
            on=["subject", "source", "model", "start", "stop"],
            how="left",
        )
        subject_pool = subject_pool.sort_values(["oof_accuracy", "mean_accuracy", "std_accuracy"], ascending=[False, False, True]).reset_index(drop=True)
        _, y_train, _ = load_subject_data(args.data_dir, subject)
        selected_rows, weight_mode, ensemble_oof_accuracy = greedy_subject_ensemble(
            subject=subject,
            candidate_rows=subject_pool,
            prediction_lookup=prediction_lookup,
            y_true=y_train,
            max_ensemble_size=args.max_ensemble_size,
            min_improvement=args.min_improvement,
            logger=logger,
        )
        final_predictions[subject] = build_final_predictions(selected_rows, prediction_lookup, weight_mode)
        summary_rows.append(
            {
                "subject": subject,
                "ensemble_size": len(selected_rows),
                "weight_mode": weight_mode,
                "ensemble_oof_accuracy": ensemble_oof_accuracy,
                "best_single_oof_accuracy": float(subject_pool.iloc[0]["oof_accuracy"]),
                "best_single_model": subject_pool.iloc[0]["model"],
                "best_single_window": subject_pool.iloc[0]["window"],
            }
        )
        for member_order, row in enumerate(selected_rows, start=1):
            selected_member_rows.append(
                {
                    "subject": subject,
                    "member_order": member_order,
                    "ensemble_size": len(selected_rows),
                    "weight_mode": weight_mode,
                    "model": row.model,
                    "window": row.window,
                    "start": row.start,
                    "stop": row.stop,
                    "family": row.family,
                    "source": row.source,
                    "oof_accuracy": prediction_lookup[candidate_key(row)]["oof_accuracy"],
                }
            )
        logger.info(
            "Subject %s final ensemble: size=%d mode=%s OOF=%.4f",
            subject,
            len(selected_rows),
            weight_mode,
            ensemble_oof_accuracy,
        )

    fallback_subjects = [subject for subject in SUBJECTS if subject not in args.subjects]
    for subject in fallback_subjects:
        final_predictions[subject] = load_fallback_predictions(args.fallback_prediction_dir, subject)
        summary_rows.append(
            {
                "subject": subject,
                "ensemble_size": 0,
                "weight_mode": "fallback",
                "ensemble_oof_accuracy": np.nan,
                "best_single_oof_accuracy": np.nan,
                "best_single_model": "fallback",
                "best_single_window": "fallback",
            }
        )
        logger.info("Subject %s fallback predictions loaded from %s", subject, args.fallback_prediction_dir)

    selected_members_df = pd.DataFrame(selected_member_rows)
    selected_members_path = os.path.join(args.submissions_dir, f"mne_pyriemann_ensemble_selected_members_{run_tag}.csv")
    selected_members_df.to_csv(selected_members_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.submissions_dir, f"mne_pyriemann_ensemble_summary_{run_tag}.csv")
    summary_df.to_csv(summary_path, index=False)

    zip_path = write_submission(
        output_model_name,
        final_predictions,
        submissions_dir=args.submissions_dir,
    )
    logger.info("Generated submission: %s", zip_path)
    logger.info("Candidate pool CSV: %s", candidate_pool_path)
    logger.info("Candidate metrics CSV: %s", candidate_metrics_path)
    logger.info("Selected members CSV: %s", selected_members_path)
    logger.info("Summary CSV: %s", summary_path)
    logger.info("Summary:\n%s", summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
