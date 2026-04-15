import argparse
import os

import numpy as np
import pandas as pd
from eeg_models import SUBJECTS, build_model, default_model_names, load_subject_data, write_submission
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cross-validate EEG models, rank them, and generate ready-to-submit prediction archives."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Directory where submission folders, zips, and CV reports are written.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate used by models that depend on filtering or PSD estimation.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified folds for per-subject cross-validation.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Optional subset of model names to evaluate. Defaults to the full candidate list.",
    )
    return parser.parse_args()


def evaluate_model_on_subject(model_name, subject, X_train, y_train, sample_rate_hz, cv_folds):
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    fold_scores = []

    for train_indices, valid_indices in splitter.split(X_train, y_train):
        model = build_model(model_name, sample_rate_hz=sample_rate_hz)
        model.fit(X_train[train_indices], y_train[train_indices])
        predictions = model.predict(X_train[valid_indices])
        fold_scores.append(accuracy_score(y_train[valid_indices], predictions))

    return {
        "model": model_name,
        "subject": subject,
        "mean_accuracy": float(np.mean(fold_scores)),
        "std_accuracy": float(np.std(fold_scores)),
        "fold_scores": fold_scores,
    }


def evaluate_models(model_names, data_dir, sample_rate_hz, cv_folds):
    rows = []
    for model_name in model_names:
        for subject in SUBJECTS:
            X_train, y_train, _ = load_subject_data(data_dir, subject)
            rows.append(
                evaluate_model_on_subject(
                    model_name=model_name,
                    subject=subject,
                    X_train=X_train,
                    y_train=y_train,
                    sample_rate_hz=sample_rate_hz,
                    cv_folds=cv_folds,
                )
            )
    return pd.DataFrame(rows)


def summarize_results(cv_results):
    summary = (
        cv_results.groupby("model", as_index=False)
        .agg(mean_accuracy=("mean_accuracy", "mean"), std_accuracy=("mean_accuracy", "std"))
        .sort_values("mean_accuracy", ascending=False)
        .reset_index(drop=True)
    )

    per_subject = cv_results.pivot(index="model", columns="subject", values="mean_accuracy")
    per_subject = per_subject.reset_index()

    return summary.merge(per_subject, on="model", how="left")


def train_full_and_predict(model_name, data_dir, sample_rate_hz):
    predictions_by_subject = {}
    for subject in SUBJECTS:
        X_train, y_train, X_test = load_subject_data(data_dir, subject)
        model = build_model(model_name, sample_rate_hz=sample_rate_hz)
        model.fit(X_train, y_train)
        predictions_by_subject[subject] = model.predict(X_test)
    return predictions_by_subject


def build_subject_best_submission(cv_results, data_dir, sample_rate_hz, submissions_dir):
    best_rows = (
        cv_results.sort_values(["subject", "mean_accuracy", "std_accuracy"], ascending=[True, False, True])
        .groupby("subject", as_index=False)
        .first()
    )

    predictions_by_subject = {}
    for row in best_rows.itertuples(index=False):
        X_train, y_train, X_test = load_subject_data(data_dir, row.subject)
        model = build_model(row.model, sample_rate_hz=sample_rate_hz)
        model.fit(X_train, y_train)
        predictions_by_subject[row.subject] = model.predict(X_test)

    zip_path = write_submission("subject_best_cv_blend", predictions_by_subject, submissions_dir=submissions_dir)
    return best_rows, zip_path


def main():
    args = parse_args()
    model_names = args.models or default_model_names()
    os.makedirs(args.submissions_dir, exist_ok=True)

    cv_results = evaluate_models(
        model_names=model_names,
        data_dir=args.data_dir,
        sample_rate_hz=args.sample_rate_hz,
        cv_folds=args.cv_folds,
    )
    summary = summarize_results(cv_results)

    cv_results_path = os.path.join(args.submissions_dir, "cv_results_by_subject.csv")
    ranking_path = os.path.join(args.submissions_dir, "cv_model_ranking.csv")
    cv_results.to_csv(cv_results_path, index=False)
    summary.to_csv(ranking_path, index=False)

    for model_name in summary["model"]:
        predictions_by_subject = train_full_and_predict(
            model_name=model_name,
            data_dir=args.data_dir,
            sample_rate_hz=args.sample_rate_hz,
        )
        zip_path = write_submission(model_name, predictions_by_subject, submissions_dir=args.submissions_dir)
        print(f"Generated submission: {zip_path}")

    best_rows, blend_zip_path = build_subject_best_submission(
        cv_results=cv_results,
        data_dir=args.data_dir,
        sample_rate_hz=args.sample_rate_hz,
        submissions_dir=args.submissions_dir,
    )
    best_rows.to_csv(os.path.join(args.submissions_dir, "subject_best_cv_models.csv"), index=False)
    print(f"Generated submission: {blend_zip_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
