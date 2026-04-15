import argparse
import os

import numpy as np
import pandas as pd
from eeg_models import SUBJECTS, load_subject_data, write_submission
from mne_pyriemann_models import build_model, default_model_names, default_window_specs
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run subject-specific MNE/CSP and pyRiemann model search and generate submission archives."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Directory where CV reports and submission archives are written.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate used for filtering. Override if the dataset uses a different rate.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Number of stratified folds for subject-specific cross-validation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model names to evaluate.",
    )
    parser.add_argument(
        "--top-vote-models",
        type=int,
        default=3,
        help="Number of top CV configurations per subject to include in the weighted vote submission.",
    )
    return parser.parse_args()


def evaluate_subject_configs(subject, X_train, y_train, sample_rate_hz, cv_folds, model_names):
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rows = []
    window_specs = default_window_specs(X_train.shape[-1])

    for window_spec in window_specs:
        for model_name in model_names:
            fold_scores = []
            for train_indices, valid_indices in splitter.split(X_train, y_train):
                model = build_model(
                    model_name,
                    sample_rate_hz=sample_rate_hz,
                    start=window_spec["start"],
                    stop=window_spec["stop"],
                )
                model.fit(X_train[train_indices], y_train[train_indices])
                predictions = model.predict(X_train[valid_indices])
                fold_scores.append(accuracy_score(y_train[valid_indices], predictions))

            rows.append(
                {
                    "subject": subject,
                    "model": model_name,
                    "window": window_spec["name"],
                    "start": window_spec["start"],
                    "stop": window_spec["stop"],
                    "mean_accuracy": float(np.mean(fold_scores)),
                    "std_accuracy": float(np.std(fold_scores)),
                    "fold_scores": repr(fold_scores),
                }
            )

    return pd.DataFrame(rows)


def fit_and_predict(row, data_dir, sample_rate_hz):
    X_train, y_train, X_test = load_subject_data(data_dir, row.subject)
    model = build_model(row.model, sample_rate_hz=sample_rate_hz, start=int(row.start), stop=int(row.stop))
    model.fit(X_train, y_train)
    return model.predict(X_test)


def weighted_vote(prediction_rows, weights):
    labels = sorted({label for predictions in prediction_rows for label in predictions.tolist()})
    label_to_index = {label: index for index, label in enumerate(labels)}
    vote_matrix = np.zeros((prediction_rows[0].shape[0], len(labels)), dtype=float)

    for predictions, weight in zip(prediction_rows, weights):
        for index, label in enumerate(predictions):
            vote_matrix[index, label_to_index[label]] += weight

    return np.asarray([labels[index] for index in vote_matrix.argmax(axis=1)])


def main():
    args = parse_args()
    os.makedirs(args.submissions_dir, exist_ok=True)
    model_names = args.models or default_model_names()

    subject_results = []
    for subject in SUBJECTS:
        X_train, y_train, _ = load_subject_data(args.data_dir, subject)
        print(f"Evaluating subject {subject}...", flush=True)
        subject_df = (
            evaluate_subject_configs(
                subject=subject,
                X_train=X_train,
                y_train=y_train,
                sample_rate_hz=args.sample_rate_hz,
                cv_folds=args.cv_folds,
                model_names=model_names,
            )
        )
        subject_results.append(subject_df)
        pd.concat(subject_results, ignore_index=True).to_csv(
            os.path.join(args.submissions_dir, "mne_pyriemann_cv_results.partial.csv"),
            index=False,
        )

    cv_results = pd.concat(subject_results, ignore_index=True)
    cv_results_path = os.path.join(args.submissions_dir, "mne_pyriemann_cv_results.csv")
    cv_results.to_csv(cv_results_path, index=False)

    subject_best = (
        cv_results.sort_values(["subject", "mean_accuracy", "std_accuracy"], ascending=[True, False, True])
        .groupby("subject", as_index=False)
        .first()
    )
    subject_best_path = os.path.join(args.submissions_dir, "mne_pyriemann_subject_best_configs.csv")
    subject_best.to_csv(subject_best_path, index=False)

    summary = (
        cv_results.groupby("model", as_index=False)
        .agg(mean_accuracy=("mean_accuracy", "mean"), std_accuracy=("mean_accuracy", "std"))
        .sort_values("mean_accuracy", ascending=False)
        .reset_index(drop=True)
    )
    summary_path = os.path.join(args.submissions_dir, "mne_pyriemann_model_summary.csv")
    summary.to_csv(summary_path, index=False)

    best_predictions = {}
    for row in subject_best.itertuples(index=False):
        best_predictions[row.subject] = fit_and_predict(row, args.data_dir, args.sample_rate_hz)
    best_zip_path = write_submission(
        "mne_pyriemann_subject_best",
        best_predictions,
        submissions_dir=args.submissions_dir,
    )

    vote_rows = (
        cv_results.sort_values(["subject", "mean_accuracy", "std_accuracy"], ascending=[True, False, True])
        .groupby("subject", as_index=False)
        .head(args.top_vote_models)
        .reset_index(drop=True)
    )
    vote_rows_path = os.path.join(args.submissions_dir, "mne_pyriemann_subject_top_vote_configs.csv")
    vote_rows.to_csv(vote_rows_path, index=False)

    vote_predictions = {}
    for subject in SUBJECTS:
        subject_vote_rows = vote_rows[vote_rows["subject"] == subject]
        prediction_rows = []
        weights = []
        for row in subject_vote_rows.itertuples(index=False):
            prediction_rows.append(fit_and_predict(row, args.data_dir, args.sample_rate_hz))
            weights.append(max(row.mean_accuracy, 1e-6))
        vote_predictions[subject] = weighted_vote(prediction_rows, weights)

    vote_zip_path = write_submission(
        "mne_pyriemann_subject_top3_vote",
        vote_predictions,
        submissions_dir=args.submissions_dir,
    )

    print(f"Generated submission: {best_zip_path}")
    print(f"Generated submission: {vote_zip_path}")
    print(subject_best[["subject", "model", "window", "mean_accuracy", "std_accuracy"]].to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
