import argparse
import os
import shutil

import pandas as pd
from eeg_models import SUBJECTS, write_submission
from run_mne_pyriemann_finetune import predict_from_base_row, predict_from_finetune_row, weighted_vote


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a hybrid submission that keeps legacy top3 predictions for A/B/C/D/F "
            "and recomputes E using the current top3 vote strategy."
        )
    )
    parser.add_argument(
        "--legacy-top3-dir",
        default="submissions/mne_pyriemann_finetune_subject_top3_vote",
        help="Directory containing the previous strong top3 prediction CSVs.",
    )
    parser.add_argument(
        "--vote-configs",
        default="submissions/mne_pyriemann_finetune_subject_top_vote_configs.csv",
        help="Vote-config CSV used to rebuild subject E with the current top3 strategy.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate used when rebuilding subject E vote members.",
    )
    parser.add_argument(
        "--output-model-name",
        default="mne_pyriemann_finetune_top3_keepABCDF_newEtop3",
        help="Output folder/zip base name under submissions/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = os.path.join("submissions", args.output_model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Keep legacy predictions exactly for all subjects except E.
    for subject in SUBJECTS:
        if subject == "E":
            continue
        source_csv = os.path.join(args.legacy_top3_dir, f"subject_{subject}_y_pred.csv")
        if not os.path.exists(source_csv):
            raise FileNotFoundError(f"Missing legacy prediction file: {source_csv}")
        destination_csv = os.path.join(output_dir, f"subject_{subject}_y_pred.csv")
        shutil.copyfile(source_csv, destination_csv)

    # Recompute E with the current top3 vote strategy from the vote-config table.
    vote_rows = pd.read_csv(args.vote_configs)
    subject_e_rows = vote_rows[vote_rows["subject"] == "E"]
    if subject_e_rows.empty:
        raise ValueError(f"No E rows found in vote-config CSV: {args.vote_configs}")

    prediction_rows = []
    weights = []
    for row in subject_e_rows.itertuples(index=False):
        if getattr(row, "source", "fine_tuned") == "base":
            prediction_rows.append(predict_from_base_row(row, args.data_dir, args.sample_rate_hz))
        else:
            prediction_rows.append(predict_from_finetune_row(row, args.data_dir, args.sample_rate_hz))
        weights.append(max(row.mean_accuracy, 1e-6))

    subject_e_prediction = weighted_vote(prediction_rows, weights)
    pd.DataFrame({"y_pred": subject_e_prediction}).to_csv(
        os.path.join(output_dir, "subject_E_y_pred.csv"),
        index=False,
    )

    zip_path = write_submission(args.output_model_name, {
        subject: pd.read_csv(os.path.join(output_dir, f"subject_{subject}_y_pred.csv"))["y_pred"].to_numpy()
        for subject in SUBJECTS
    })
    print(output_dir)
    print(zip_path)


if __name__ == "__main__":
    main()
