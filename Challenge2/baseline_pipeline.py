import os

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def temporal_variance_log(X):
    var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
    return np.log(np.clip(var, 1e-10, None))


def main():
    pipeline = Pipeline([
        ("log_var", FunctionTransformer(temporal_variance_log)),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

    os.makedirs("submissions", exist_ok=True)

    for subject in ["A", "B", "C", "D", "E", "F"]:
        X_train = np.load(f"data/subject_{subject}_X_train.npy")
        y_train = np.load(f"data/subject_{subject}_y_train.npy")
        X_test = np.load(f"data/subject_{subject}_X_test.npy")

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        pd.DataFrame({"y_pred": y_pred}).to_csv(
            f"submissions/subject_{subject}_y_pred.csv", index=False
        )


if __name__ == "__main__":
    main()
