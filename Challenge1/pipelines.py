from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVR

from ridge_feature_models import BaggedScoreRidgeRegressor


RANDOM_STATE = 42
AGE_BIN_COUNT = 5


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator: BaseEstimator
    param_grid: dict[str, list[Any]] | None = None
    notes: str = ""


def ensure_target(y: pd.Series | pd.DataFrame | np.ndarray) -> pd.Series:
    if isinstance(y, pd.DataFrame):
        if y.shape[1] != 1:
            raise ValueError("Expected a single target column.")
        return y.iloc[:, 0]
    if isinstance(y, pd.Series):
        return y
    array = np.asarray(y).reshape(-1)
    return pd.Series(array, name="age")


def make_age_bins(y: pd.Series, n_bins: int = AGE_BIN_COUNT) -> np.ndarray:
    usable_bins = max(2, min(n_bins, y.nunique()))
    bins = pd.qcut(y, q=usable_bins, labels=False, duplicates="drop")
    return np.asarray(bins)


def make_cv_splits(
    y: pd.Series | pd.DataFrame | np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 2,
    random_state: int = RANDOM_STATE,
):
    target = ensure_target(y)
    age_bins = make_age_bins(target)
    splitter = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    dummy_X = np.zeros((len(target), 1))
    return list(splitter.split(dummy_X, age_bins))


def beta_to_m_values(X: np.ndarray) -> np.ndarray:
    clipped = np.clip(X, 1e-5, 1 - 1e-5)
    return np.log2(clipped / (1.0 - clipped))


def build_preprocessor(
    columns: list[str],
    methylation_transform: str = "raw",
    scale_numeric: bool = True,
) -> ColumnTransformer:
    numeric_features = [column for column in columns if column != "gender"]
    numeric_steps: list[tuple[str, Any]] = []

    if methylation_transform == "m_value":
        numeric_steps.append(
            (
                "beta_to_m_value",
                FunctionTransformer(beta_to_m_values, feature_names_out="one-to-one"),
            )
        )
    elif methylation_transform != "raw":
        raise ValueError(f"Unsupported methylation transform: {methylation_transform}")

    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))

    numeric_transformer: Any
    if numeric_steps:
        numeric_transformer = Pipeline(numeric_steps)
    else:
        numeric_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            (
                "gender",
                OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=False),
                ["gender"],
            ),
            ("numeric", numeric_transformer, numeric_features),
        ],
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def build_ridge_pipeline(
    columns: list[str],
    methylation_transform: str = "raw",
    use_feature_selection: bool = False,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        ("preprocess", build_preprocessor(columns, methylation_transform=methylation_transform)),
    ]
    if use_feature_selection:
        steps.append(("select", SelectKBest(score_func=f_regression, k=1000)))
    steps.append(("model", Ridge()))
    return Pipeline(steps)


def build_elasticnet_pipeline(
    columns: list[str],
    methylation_transform: str = "raw",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(columns, methylation_transform=methylation_transform)),
            ("select", SelectKBest(score_func=f_regression, k=1000)),
            (
                "model",
                ElasticNet(max_iter=20000, random_state=RANDOM_STATE),
            ),
        ]
    )


def build_pca_ridge_pipeline(
    columns: list[str],
    methylation_transform: str = "raw",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(columns, methylation_transform=methylation_transform)),
            ("reduce", PCA(svd_solver="randomized", random_state=RANDOM_STATE)),
            ("model", Ridge()),
        ]
    )


def build_pls_pipeline(
    columns: list[str],
    methylation_transform: str = "raw",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(columns, methylation_transform=methylation_transform)),
            ("model", PLSRegression(scale=False)),
        ]
    )


def build_linearsvr_pipeline(
    columns: list[str],
    methylation_transform: str = "raw",
) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(columns, methylation_transform=methylation_transform)),
            ("select", SelectKBest(score_func=f_regression, k=1000)),
            (
                "model",
                LinearSVR(
                    dual="auto",
                    max_iter=20000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def get_benchmark_specs(columns: list[str]) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="dummy_mean",
            estimator=Pipeline(
                steps=[
                    ("preprocess", build_preprocessor(columns, scale_numeric=False)),
                    ("model", DummyRegressor(strategy="mean")),
                ]
            ),
            param_grid=None,
            notes="Sanity-check baseline.",
        ),
        ModelSpec(
            name="ridge_all_raw",
            estimator=build_ridge_pipeline(columns, methylation_transform="raw"),
            param_grid={"model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0]},
            notes="All features, scaled beta values, Ridge.",
        ),
        ModelSpec(
            name="ridge_select_raw",
            estimator=build_ridge_pipeline(
                columns,
                methylation_transform="raw",
                use_feature_selection=True,
            ),
            param_grid={
                "select__k": [300, 1000, 3000, 5000],
                "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
            },
            notes="Univariate filter plus Ridge on raw beta values.",
        ),
        ModelSpec(
            name="ridge_select_mvalue",
            estimator=build_ridge_pipeline(
                columns,
                methylation_transform="m_value",
                use_feature_selection=True,
            ),
            param_grid={
                "select__k": [300, 1000, 3000, 5000],
                "model__alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
            },
            notes="Univariate filter plus Ridge on M-values.",
        ),
        ModelSpec(
            name="elasticnet_select_raw",
            estimator=build_elasticnet_pipeline(columns, methylation_transform="raw"),
            param_grid={
                "select__k": [300, 1000, 3000],
                "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            notes="Univariate filter plus ElasticNet on raw beta values.",
        ),
        ModelSpec(
            name="elasticnet_select_mvalue",
            estimator=build_elasticnet_pipeline(columns, methylation_transform="m_value"),
            param_grid={
                "select__k": [300, 1000, 3000],
                "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            },
            notes="Univariate filter plus ElasticNet on M-values.",
        ),
        ModelSpec(
            name="pca_ridge_raw",
            estimator=build_pca_ridge_pipeline(columns, methylation_transform="raw"),
            param_grid={
                "reduce__n_components": [10, 25, 50, 100, 150],
                "model__alpha": [0.1, 1.0, 10.0, 100.0],
            },
            notes="Dimensionality reduction with PCA followed by Ridge.",
        ),
        ModelSpec(
            name="pls_raw",
            estimator=build_pls_pipeline(columns, methylation_transform="raw"),
            param_grid={"model__n_components": [2, 5, 10, 15, 25, 40]},
            notes="Supervised latent factors with PLS.",
        ),
        ModelSpec(
            name="linearsvr_select_raw",
            estimator=build_linearsvr_pipeline(columns, methylation_transform="raw"),
            param_grid={
                "select__k": [300, 1000, 3000],
                "model__C": [0.1, 1.0, 10.0],
                "model__epsilon": [0.0, 0.1, 0.5],
            },
            notes="Linear SVR after scaling and filtering.",
        ),
    ]


def get_finalist_specs(columns: list[str]) -> list[ModelSpec]:
    return [
        ModelSpec(
            name="ridge_select_raw_refined",
            estimator=build_ridge_pipeline(
                columns,
                methylation_transform="raw",
                use_feature_selection=True,
            ),
            param_grid={
                "select__k": [2000, 2500, 3000, 3500, 4000, 5000],
                "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            },
            notes="Refined Ridge on raw beta values.",
        ),
        ModelSpec(
            name="ridge_select_mvalue_refined",
            estimator=build_ridge_pipeline(
                columns,
                methylation_transform="m_value",
                use_feature_selection=True,
            ),
            param_grid={
                "select__k": [2000, 2500, 3000, 3500, 4000, 5000],
                "model__alpha": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
            },
            notes="Refined Ridge on M-values.",
        ),
        ModelSpec(
            name="elasticnet_select_raw_refined",
            estimator=build_elasticnet_pipeline(columns, methylation_transform="raw"),
            param_grid={
                "select__k": [2000, 3000, 4000, 5000],
                "model__alpha": [0.003, 0.01, 0.02, 0.03, 0.05],
                "model__l1_ratio": [0.05, 0.1, 0.2, 0.3, 0.4],
            },
            notes="Refined ElasticNet on raw beta values.",
        ),
        ModelSpec(
            name="elasticnet_select_raw_refined_lite",
            estimator=build_elasticnet_pipeline(columns, methylation_transform="raw"),
            param_grid={
                "select__k": [3000, 4000, 5000],
                "model__alpha": [0.01, 0.02, 0.03],
                "model__l1_ratio": [0.05, 0.1, 0.2],
            },
            notes="Smaller refined ElasticNet sweep on raw beta values.",
        ),
        ModelSpec(
            name="elasticnet_select_mvalue_refined",
            estimator=build_elasticnet_pipeline(columns, methylation_transform="m_value"),
            param_grid={
                "select__k": [2000, 3000, 4000, 5000],
                "model__alpha": [0.003, 0.01, 0.02, 0.03, 0.05],
                "model__l1_ratio": [0.05, 0.1, 0.2, 0.3, 0.4],
            },
            notes="Refined ElasticNet on M-values.",
        ),
        ModelSpec(
            name="pca_ridge_raw_refined",
            estimator=build_pca_ridge_pipeline(columns, methylation_transform="raw"),
            param_grid={
                "reduce__n_components": [15, 25, 40, 60, 80, 120],
                "model__alpha": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            },
            notes="Refined PCA plus Ridge on raw beta values.",
        ),
    ]


def build_final_model(columns: list[str]) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(columns, methylation_transform="raw")),
            ("select", SelectKBest(score_func=f_regression, k=3500)),
            ("model", Ridge(alpha=0.01)),
        ]
    )


def build_best_ridge_feature_model(columns: list[str]) -> BaggedScoreRidgeRegressor:
    return BaggedScoreRidgeRegressor(
        k=3200,
        alpha=0.01,
        score_method="f_score",
        n_estimators=7,
        sample_fraction=0.7,
        random_state=42,
    )
