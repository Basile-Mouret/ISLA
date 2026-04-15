import argparse
import csv
import os
import zipfile


SUBJECTS = ["A", "B", "C", "D", "E", "F"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package per-subject prediction CSVs into a flat Codabench submission zip."
    )
    parser.add_argument(
        "model_dir",
        help="Directory containing subject_A_y_pred.csv through subject_F_y_pred.csv.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output zip path. Defaults to submissions/<model_name>.zip.",
    )
    return parser.parse_args()


def validate_csv(csv_path):
    with open(csv_path, newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

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


def package_submission_dir(model_dir, output_path=None):
    model_dir = os.path.abspath(model_dir)
    model_name = os.path.basename(os.path.normpath(model_dir))
    output_path = os.path.abspath(output_path or os.path.join("submissions", f"{model_name}.zip"))

    csv_paths = []
    for subject in SUBJECTS:
        csv_path = os.path.join(model_dir, f"subject_{subject}_y_pred.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing required file: {csv_path}")
        validate_csv(csv_path)
        csv_paths.append(csv_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_path in csv_paths:
            archive.write(csv_path, arcname=os.path.basename(csv_path))

    return output_path


def main():
    args = parse_args()
    output_path = package_submission_dir(args.model_dir, args.output)
    print(output_path)


if __name__ == "__main__":
    main()
