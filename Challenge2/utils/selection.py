import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .file_manager import candidate_key
from .models import weighted_vote


def top_candidate_pool(subject_results, args):
    parts = []
    for model_name in subject_results["model"].unique():
        parts.append(
            subject_results[subject_results["model"] == model_name]
            .sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
            .head(args.top_per_model)
        )

    if not parts:
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="first")
    return (
        pool.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
        .head(args.max_candidates_per_subject)
        .reset_index(drop=True)
    )


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

    score = ensemble_acc + args.diversity_weight * avg_disagreement - args.double_fault_weight * avg_double_fault
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

            left = (
                metrics["score"],
                metrics["ensemble_accuracy"],
                metrics["avg_disagreement"],
                -metrics["avg_double_fault"],
            )
            right = (
                best_metrics["score"],
                best_metrics["ensemble_accuracy"],
                best_metrics["avg_disagreement"],
                -best_metrics["avg_double_fault"],
            )
            if left > right:
                best_combo = combo_rows
                best_metrics = metrics

    if best_combo is None:
        best_combo = [rows[0]]
        best_metrics = combo_metrics(best_combo, oof_lookup, y_true, args)

    return best_combo, best_metrics
