import argparse

import pandas as pd
from eeg_models import SUBJECTS, write_submission
from run_mne_pyriemann_finetune import predict_from_base_row, predict_from_finetune_row, weighted_vote


def parse_args():
    parser = argparse.ArgumentParser(
        description="Materialize a submission zip from a vote-config CSV produced by the fine-tune search."
    )
    parser.add_argument(
        "--vote-configs",
        required=True,
        help="CSV containing per-subject vote members and their scores.",
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Directory where the submission folder and zip are written.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate used when rebuilding model predictions.",
    )
    parser.add_argument(
        "--output-model-name",
        required=True,
        help="Output folder and zip base name under submissions/.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    vote_rows = pd.read_csv(args.vote_configs)

    vote_predictions = {}
    for subject in SUBJECTS:
        subject_vote_rows = vote_rows[vote_rows["subject"] == subject]
        if subject_vote_rows.empty:
            raise ValueError(f"No vote rows found for subject {subject} in {args.vote_configs}")

        prediction_rows = []
        weights = []
        for row in subject_vote_rows.itertuples(index=False):
            if getattr(row, "source", "fine_tuned") == "base":
                prediction_rows.append(predict_from_base_row(row, args.data_dir, args.sample_rate_hz))
            else:
                prediction_rows.append(predict_from_finetune_row(row, args.data_dir, args.sample_rate_hz))
            weights.append(max(row.mean_accuracy, 1e-6))

        vote_predictions[subject] = weighted_vote(prediction_rows, weights)

    zip_path = write_submission(
        args.output_model_name,
        vote_predictions,
        submissions_dir=args.submissions_dir,
    )
    print(zip_path)


if __name__ == "__main__":
    main()
