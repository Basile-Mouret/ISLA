import argparse
import os

from utils.file_manager import (
    load_subject_data,
    package_submission_dir,
    write_prediction_dir,
)
from utils.helpers import SUBJECTS
from utils.models import predict_weighted_ensemble


TOP3_DIVERSE = {
    "A": [
        {"model": "fbcsp_broad_c6_k16_lda", "start": 320, "stop": 1344, "weight": 0.8785714285714286},
        {"model": "mne_fbcsp_lda", "start": 384, "stop": 1088, "weight": 0.8357142857142857},
        {"model": "riemann_fgmdm_8_30_lwf", "start": 768, "stop": 1152, "weight": 0.65},
    ],
    "B": [
        {"model": "riemann_ts_lr_8_30", "start": 832, "stop": 1344, "weight": 0.7928571428571429},
        {"model": "mne_csp_8_30_lda", "start": 512, "stop": 1536, "weight": 0.7785714285714287},
        {"model": "fbcsp_broad_c6_k16_lda", "start": 192, "stop": 1408, "weight": 0.6785714285714286},
    ],
    "C": [
        {"model": "mne_fbcsp_lda", "start": 256, "stop": 1280, "weight": 0.9571428571428572},
        {"model": "riemann_ts_lr_6_35", "start": 832, "stop": 1536, "weight": 0.9285714285714284},
        {"model": "fbcsp_dense_c6_k20_lda", "start": 704, "stop": 1536, "weight": 0.9071428571428573},
    ],
    "D": [
        {"model": "fbcsp_dense_c6_k20_lda", "start": 640, "stop": 1280, "weight": 0.9428571428571428},
        {"model": "mne_csp_8_30_lda", "start": 320, "stop": 1152, "weight": 0.9428571428571428},
        {"model": "fbcsp_broad_c6_k16_lda", "start": 512, "stop": 1408, "weight": 0.9142857142857144},
    ],
    "E": [
        {"model": "riemann_ts_lr_6_35", "start": 640, "stop": 1408, "weight": 0.8785714285714287},
        {"model": "fbcsp_broad_c6_k16_lda", "start": 320, "stop": 1216, "weight": 0.8571428571428571},
        {"model": "mne_fbcsp_lda", "start": 512, "stop": 1472, "weight": 0.8428571428571429},
    ],
    "F": [
        {"model": "mne_fbcsp_lda", "start": 512, "stop": 1472, "weight": 0.8571428571428572},
        {"model": "fbcsp_broad_c6_k16_lda", "start": 256, "stop": 1537, "weight": 0.7857142857142857},
        {"model": "riemann_ts_lr_6_35", "start": 640, "stop": 1216, "weight": 0.75},
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the frozen heavy top3_diverse ensemble and package a submission.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--output-model-name", default="heavy_top3_diverse")
    parser.add_argument("--sample-rate-hz", type=float, default=256.0)
    return parser.parse_args()


def main():
    args = parse_args()
    predictions = {}

    for subject in SUBJECTS:
        X_train, y_train, X_test = load_subject_data(args.data_dir, subject)
        predictions[subject] = predict_weighted_ensemble(
            TOP3_DIVERSE[subject],
            X_train,
            y_train,
            X_test,
            args.sample_rate_hz,
        )

    output_dir = os.path.join(args.submissions_dir, args.output_model_name)
    write_prediction_dir(output_dir, predictions)
    zip_path = package_submission_dir(output_dir, os.path.join(args.submissions_dir, f"{args.output_model_name}.zip"))
    print(zip_path)


if __name__ == "__main__":
    main()
