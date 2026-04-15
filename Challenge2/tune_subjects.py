import argparse
import os

import numpy as np
import pandas as pd
from mi_models import SUBJECTS, build_model, load_subject_data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def parse_args():
    parser = argparse.ArgumentParser(
        description="Lightweight subject fine-tuning around anchor windows for a small model list."
    )
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS, choices=SUBJECTS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--sample-rate-hz", type=float, default=256.0)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument(
        "--base-results",
        default="submissions/mne_pyriemann_cv_results.csv",
        help="Base search CSV used for anchor windows.",
    )
    parser.add_argument(
        "--output",
        default="submissions/fine_tune_results.csv",
        help="Output CSV for fine-tune CV results.",
    )
    return parser.parse_args()


def focused_specs_for_subject(anchor_model):
    if "fbcsp" in anchor_model:
        return ["fbcsp_broad_c6_k16_lda", "fbcsp_dense_c6_k20_lda"]
    if "riemann" in anchor_model or "fgmdm" in anchor_model:
        return ["riemann_fgmdm_8_30_lwf", "riemann_ts_lr_8_30"]
    return ["mne_csp_8_30_lda", "riemann_ts_lr_6_35"]


def window_grid(anchor_start, anchor_stop, n_times):
    starts = [anchor_start - 128, anchor_start - 64, anchor_start, anchor_start + 64, anchor_start + 128]
    stops = [anchor_stop - 64, anchor_stop, anchor_stop + 64]
    windows = []
    for start in starts:
        for stop in stops:
            s = max(0, int(start))
            t = min(n_times, int(stop))
            if t - s < 384:
                continue
            windows.append((s, t))
    deduped = []
    seen = set()
    for window in windows:
        if window in seen:
            continue
        seen.add(window)
        deduped.append(window)
    return deduped


def main():
    args = parse_args()
    base = pd.read_csv(args.base_results)
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

    rows = []
    for subject in args.subjects:
        subject_base = base[base["subject"] == subject].sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
        if subject_base.empty:
            raise ValueError(f"No base rows for subject {subject}")

        anchor = subject_base.iloc[0]
        models = focused_specs_for_subject(anchor["model"])

        X_train, y_train, _ = load_subject_data(args.data_dir, subject)
        windows = window_grid(int(anchor["start"]), int(anchor["stop"]), X_train.shape[-1])

        for model_name in models:
            for start, stop in windows:
                fold_scores = []
                for train_idx, valid_idx in splitter.split(X_train, y_train):
                    model = build_model(
                        model_name=model_name,
                        sample_rate_hz=args.sample_rate_hz,
                        start=start,
                        stop=stop,
                    )
                    model.fit(X_train[train_idx], y_train[train_idx])
                    y_pred = model.predict(X_train[valid_idx])
                    fold_scores.append(accuracy_score(y_train[valid_idx], y_pred))

                rows.append(
                    {
                        "subject": subject,
                        "model": model_name,
                        "start": start,
                        "stop": stop,
                        "mean_accuracy": float(np.mean(fold_scores)),
                        "std_accuracy": float(np.std(fold_scores)),
                    }
                )

    result = pd.DataFrame(rows).sort_values(["subject", "mean_accuracy", "std_accuracy"], ascending=[True, False, True])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    result.to_csv(args.output, index=False)
    print(args.output)


if __name__ == "__main__":
    main()
