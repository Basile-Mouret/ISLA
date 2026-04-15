import argparse
import os
import shutil

from package_submission import SUBJECTS, package_submission_dir, validate_csv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a hybrid submission by taking a base prediction directory and overriding selected subjects from other directories."
    )
    parser.add_argument("--base-dir", required=True, help="Directory providing default subject_A_y_pred.csv ... subject_F_y_pred.csv files.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Override in the form SUBJECT=DIR, for example E=submissions/mne_pyriemann_finetune_subject_best.",
    )
    parser.add_argument(
        "--output-model-name",
        required=True,
        help="Output folder and zip base name under submissions/.",
    )
    return parser.parse_args()


def parse_overrides(override_args):
    overrides = {}
    for override_arg in override_args:
        if "=" not in override_arg:
            raise ValueError(f"Invalid override '{override_arg}'. Expected SUBJECT=DIR.")
        subject, directory = override_arg.split("=", 1)
        subject = subject.strip().upper()
        directory = directory.strip()
        if subject not in SUBJECTS:
            raise ValueError(f"Invalid subject '{subject}' in override '{override_arg}'.")
        overrides[subject] = directory
    return overrides


def main():
    args = parse_args()
    overrides = parse_overrides(args.override)

    output_dir = os.path.join("submissions", args.output_model_name)
    os.makedirs(output_dir, exist_ok=True)

    for subject in SUBJECTS:
        source_dir = overrides.get(subject, args.base_dir)
        source_csv = os.path.join(source_dir, f"subject_{subject}_y_pred.csv")
        if not os.path.exists(source_csv):
            raise FileNotFoundError(f"Missing source CSV for subject {subject}: {source_csv}")
        validate_csv(source_csv)
        destination_csv = os.path.join(output_dir, f"subject_{subject}_y_pred.csv")
        shutil.copyfile(source_csv, destination_csv)

    zip_path = package_submission_dir(output_dir, os.path.join("submissions", f"{args.output_model_name}.zip"))
    print(output_dir)
    print(zip_path)


if __name__ == "__main__":
    main()
