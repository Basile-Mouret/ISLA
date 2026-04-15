import os

import numpy as np
import pandas as pd
from package_submission import package_submission_dir
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


MODEL_NAME = "baseline_logvar_lda"
SUBJECTS = ["A", "B", "C", "D", "E", "F"]


def temporal_variance_log(X):
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


def main():
    pipeline = Pipeline([
        ("log_var", FunctionTransformer(temporal_variance_log)),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

    output_dir = os.path.join("submissions", MODEL_NAME)
    os.makedirs(output_dir, exist_ok=True)

    for subject in SUBJECTS:
        X_train = np.load(f"data/subject_{subject}_X_train.npy")
        y_train = np.load(f"data/subject_{subject}_y_train.npy")
        X_test = np.load(f"data/subject_{subject}_X_test.npy")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        pd.DataFrame({"y_pred": y_pred}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"), index=False
        )

    output_zip_path = package_submission_dir(
        output_dir,
        os.path.join("submissions", f"{MODEL_NAME}.zip"),
    )
    print(f"Packaged submission: {output_zip_path}")


if __name__ == "__main__":
    main()
