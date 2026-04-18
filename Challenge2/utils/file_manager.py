import csv
import os
import zipfile

import numpy as np
import pandas as pd

from .helpers import SUBJECTS


REQUIRED_RESULTS_COLUMNS = ["subject", "model", "family", "start", "stop", "mean_accuracy", "std_accuracy"]


def ensure_parent_dir(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)


def load_subject_data(data_dir, subject):
    X_train = np.load(os.path.join(data_dir, f"subject_{subject}_X_train.npy"))
    y_train = np.load(os.path.join(data_dir, f"subject_{subject}_y_train.npy"))
    X_test = np.load(os.path.join(data_dir, f"subject_{subject}_X_test.npy"))
    return X_train, y_train, X_test


def validate_prediction_csv(csv_path):
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"{csv_path} is empty")
    if rows[0] != ["y_pred"]:
        raise ValueError(f"{csv_path} must contain exactly one header column named 'y_pred'")
    if len(rows) != 61:
        raise ValueError(f"{csv_path} must contain exactly 60 predictions")

    valid_labels = {"left_hand", "right_hand"}
    for row in rows[1:]:
        if len(row) != 1 or row[0] not in valid_labels:
            raise ValueError(f"{csv_path} contains an invalid prediction value: {row}")


def write_prediction_dir(output_dir, predictions_by_subject):
    os.makedirs(output_dir, exist_ok=True)
    for subject in SUBJECTS:
        pd.DataFrame({"y_pred": predictions_by_subject[subject]}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"),
            index=False,
        )


def package_submission_dir(model_dir, output_path=None):
    model_dir = os.path.abspath(model_dir)
    model_name = os.path.basename(os.path.normpath(model_dir))
    output_path = os.path.abspath(output_path or os.path.join("submissions", f"{model_name}.zip"))

    csv_paths = []
    for subject in SUBJECTS:
        csv_path = os.path.join(model_dir, f"subject_{subject}_y_pred.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing required file: {csv_path}")
        validate_prediction_csv(csv_path)
        csv_paths.append(csv_path)

    ensure_parent_dir(output_path)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_path in csv_paths:
            archive.write(csv_path, arcname=os.path.basename(csv_path))

    return output_path


def candidate_key(subject, model_name, start, stop):
    return (subject, model_name, int(start), int(stop))


def oof_cache_path(cache_dir, subject, model_name, start, stop):
    key = f"{subject}__{model_name}__{int(start)}__{int(stop)}".replace("/", "_")
    return os.path.join(cache_dir, f"{key}.npy")


def load_results(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    for column in REQUIRED_RESULTS_COLUMNS:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in existing results file: {path}")
    df["start"] = df["start"].astype(int)
    df["stop"] = df["stop"].astype(int)
    return df.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="last")


def cached_result_keys(results, cache_dir):
    return {
        candidate_key(row.subject, row.model, row.start, row.stop)
        for row in results.itertuples(index=False)
        if os.path.exists(oof_cache_path(cache_dir, row.subject, row.model, row.start, row.stop))
    }


def load_oof_lookup(subject_pool, cache_dir):
    oof_lookup = {}
    for row in subject_pool.itertuples(index=False):
        cache_path = oof_cache_path(cache_dir, row.subject, row.model, row.start, row.stop)
        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Missing OOF cache for {row.subject}/{row.model}/{row.start}:{row.stop}: {cache_path}"
            )
        oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] = np.load(cache_path)
    return oof_lookup


def append_result_row(path, row):
    ensure_parent_dir(path)
    row_df = pd.DataFrame([row])
    row_df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)
