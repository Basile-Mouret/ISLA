import argparse
import os

import numpy as np
import pandas as pd
from package_submission import package_submission_dir
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


SUBJECTS = ["A", "B", "C", "D", "E", "F"]


def bandpass_filter(X, sample_rate_hz, low_cut_hz, high_cut_hz, order):
    sos = butter(
        order,
        [low_cut_hz, high_cut_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return sosfiltfilt(sos, X, axis=-1)


def temporal_variance_log(X):
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


def format_hz(value):
    return f"{value:g}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Band-pass EEG trials, extract log-variance features, and train an LDA per subject."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing the challenge .npy files.")
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Root directory where model-specific submission folders are written.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate in Hz used for the Butterworth filter. Override if the dataset uses a different rate.",
    )
    parser.add_argument("--low-cut-hz", type=float, default=8.0, help="Lower band-pass cutoff in Hz.")
    parser.add_argument("--high-cut-hz", type=float, default=30.0, help="Upper band-pass cutoff in Hz.")
    parser.add_argument("--filter-order", type=int, default=4, help="Butterworth filter order.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional output folder name under submissions/. Defaults to a name derived from the filter band.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_name = args.model_name or (
        f"bandpass_{format_hz(args.low_cut_hz)}_{format_hz(args.high_cut_hz)}_logvar_lda"
    )
    output_dir = os.path.join(args.submissions_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    pipeline = Pipeline([
        (
            "bandpass",
            FunctionTransformer(
                bandpass_filter,
                kw_args={
                    "sample_rate_hz": args.sample_rate_hz,
                    "low_cut_hz": args.low_cut_hz,
                    "high_cut_hz": args.high_cut_hz,
                    "order": args.filter_order,
                },
            ),
        ),
        ("log_var", FunctionTransformer(temporal_variance_log)),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

    for subject in SUBJECTS:
        X_train = np.load(os.path.join(args.data_dir, f"subject_{subject}_X_train.npy"))
        y_train = np.load(os.path.join(args.data_dir, f"subject_{subject}_y_train.npy"))
        X_test = np.load(os.path.join(args.data_dir, f"subject_{subject}_X_test.npy"))

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        pd.DataFrame({"y_pred": y_pred}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"),
            index=False,
        )

    output_zip_path = package_submission_dir(
        output_dir,
        os.path.join(args.submissions_dir, f"{model_name}.zip"),
    )
    print(f"Packaged submission: {output_zip_path}")


if __name__ == "__main__":
    main()
