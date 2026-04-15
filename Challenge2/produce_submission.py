import argparse
import os

import pandas as pd
from mi_models import SUBJECTS, build_model, load_subject_data, weighted_vote
from package_submission import package_submission_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the final subject-specific top3 submission from a config CSV."
    )
    parser.add_argument("--config", default="subject_models.md", help="CSV/Markdown file with subject model configs.")
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument("--sample-rate-hz", type=float, default=256.0, help="Sampling rate for filtering.")
    parser.add_argument(
        "--output-model-name",
        default="final_top3_submission",
        help="Output folder/zip base name under submissions/.",
    )
    return parser.parse_args()


def load_config_table(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)

    with open(path, "r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]

    table_lines = [line for line in lines if line.startswith("|")]
    if len(table_lines) < 3:
        raise ValueError("Could not find a markdown table in config file.")

    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    rows = []
    for line in table_lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))

    df = pd.DataFrame(rows)
    required = ["subject", "rank", "model", "start", "stop", "weight"]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in config table: {missing}")

    df["start"] = df["start"].astype(int)
    df["stop"] = df["stop"].astype(int)
    df["weight"] = df["weight"].astype(float)
    df["rank"] = df["rank"].astype(int)
    return df


def main():
    args = parse_args()
    config = load_config_table(args.config)

    output_dir = os.path.join("submissions", args.output_model_name)
    os.makedirs(output_dir, exist_ok=True)

    for subject in SUBJECTS:
        rows = config[config["subject"] == subject].sort_values("rank")
        if rows.empty:
            raise ValueError(f"No config rows for subject {subject}")

        X_train, y_train, X_test = load_subject_data(args.data_dir, subject)
        prediction_rows = []
        weights = []
        for row in rows.itertuples(index=False):
            model = build_model(
                model_name=row.model,
                sample_rate_hz=args.sample_rate_hz,
                start=int(row.start),
                stop=int(row.stop),
            )
            model.fit(X_train, y_train)
            prediction_rows.append(model.predict(X_test))
            weights.append(float(row.weight))

        y_pred = weighted_vote(prediction_rows, weights)
        pd.DataFrame({"y_pred": y_pred}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"),
            index=False,
        )

    zip_path = package_submission_dir(output_dir, os.path.join("submissions", f"{args.output_model_name}.zip"))
    print(zip_path)


if __name__ == "__main__":
    main()
