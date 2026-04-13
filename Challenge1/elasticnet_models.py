from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def beta_to_m_values(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[1] > 1:
        beta = np.clip(arr[:, 1:], 1e-5, 1 - 1e-5)
        out = arr.copy()
        out[:, 1:] = np.log2(beta / (1.0 - beta))
        return out
    return arr


@dataclass(frozen=True)
class ElasticNetPreset:
    name: str
    l1_ratio_grid: tuple[float, ...]
    n_alphas: int
    cv: int
    max_iter: int
    tol: float
    use_m_value: bool = False
    variance_threshold: float = 1e-5
    notes: str = ""


class ElasticNetCVRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        l1_ratio_grid=(0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0),
        n_alphas: int = 25,
        cv: int = 5,
        max_iter: int = 2500,
        tol: float = 1e-3,
        use_m_value: bool = False,
        variance_threshold: float = 1e-5,
    ):
        self.l1_ratio_grid = l1_ratio_grid
        self.n_alphas = n_alphas
        self.cv = cv
        self.max_iter = max_iter
        self.tol = tol
        self.use_m_value = use_m_value
        self.variance_threshold = variance_threshold

    def preprocess_df(self, X: pd.DataFrame) -> pd.DataFrame:
        X_num = X.copy()
        if "gender" in X_num.columns:
            X_num["gender"] = (
                X_num["gender"].map({"m": 1, "f": 0, "M": 1, "F": 0}).fillna(0.5)
            )
        return X_num.astype(np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        X_num = self.preprocess_df(X)
        steps: list[tuple[str, object]] = [
            ("variance_filter", VarianceThreshold(threshold=self.variance_threshold))
        ]

        if self.use_m_value:
            steps.append(
                (
                    "beta_to_m_value",
                    FunctionTransformer(beta_to_m_values, validate=False),
                )
            )

        steps.extend(
            [
                ("scaler", StandardScaler()),
                (
                    "elasticnet",
                    ElasticNetCV(
                        l1_ratio=list(self.l1_ratio_grid),
                        n_alphas=self.n_alphas,
                        cv=self.cv,
                        n_jobs=-1,
                        max_iter=self.max_iter,
                        tol=self.tol,
                        random_state=42,
                    ),
                ),
            ]
        )

        self.pipeline_ = Pipeline(steps)
        self.pipeline_.fit(X_num, np.asarray(y).reshape(-1))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline_.predict(self.preprocess_df(X))


def get_baseline_elasticnet_presets() -> list[ElasticNetPreset]:
    return [
        ElasticNetPreset(
            name="baseline_v1",
            l1_ratio_grid=(0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0),
            n_alphas=25,
            cv=5,
            max_iter=2500,
            tol=1e-3,
            use_m_value=False,
            notes="Exact `model_1.py` baseline.",
        ),
        ElasticNetPreset(
            name="baseline_v2_more_alphas",
            l1_ratio_grid=(0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 1.0),
            n_alphas=40,
            cv=5,
            max_iter=6000,
            tol=5e-4,
            use_m_value=False,
            notes="Same ratios, denser alpha path, slightly stricter optimization.",
        ),
        ElasticNetPreset(
            name="baseline_v3_low_l1",
            l1_ratio_grid=(0.03, 0.05, 0.07, 0.1, 0.15, 0.2),
            n_alphas=35,
            cv=5,
            max_iter=6000,
            tol=1e-4,
            use_m_value=False,
            notes="Denser low-l1 search around the public 3.9 model behavior.",
        ),
        ElasticNetPreset(
            name="baseline_v4_low_l1_mvalue",
            l1_ratio_grid=(0.03, 0.05, 0.07, 0.1, 0.15, 0.2),
            n_alphas=35,
            cv=5,
            max_iter=6000,
            tol=1e-4,
            use_m_value=True,
            notes="Same low-l1 search with M-value transform.",
        ),
    ]


def build_elasticnet_from_preset(preset: ElasticNetPreset) -> ElasticNetCVRegressor:
    return ElasticNetCVRegressor(
        l1_ratio_grid=preset.l1_ratio_grid,
        n_alphas=preset.n_alphas,
        cv=preset.cv,
        max_iter=preset.max_iter,
        tol=preset.tol,
        use_m_value=preset.use_m_value,
        variance_threshold=preset.variance_threshold,
    )
