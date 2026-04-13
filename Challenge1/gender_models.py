from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from pipelines import build_best_ridge_feature_model, build_final_model


def _numeric_columns(X: pd.DataFrame) -> list[str]:
    return [column for column in X.columns if column != "gender"]


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    finite_scores = np.nan_to_num(scores, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    k = max(1, min(k, finite_scores.shape[0]))
    return np.argsort(finite_scores)[-k:]


def _fit_numeric_ridge(X: np.ndarray, y: np.ndarray, k: int, alpha: float) -> dict[str, object]:
    effective_k = min(k, X.shape[1])
    selector = SelectKBest(score_func=f_regression, k=effective_k)
    X_selected = selector.fit_transform(X, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)
    return {
        "selector": selector,
        "scaler": scaler,
        "model": model,
    }


def _predict_numeric_ridge(bundle: dict[str, object], X: np.ndarray) -> np.ndarray:
    selector = bundle["selector"]
    scaler = bundle["scaler"]
    model = bundle["model"]
    X_selected = selector.transform(X)
    X_scaled = scaler.transform(X_selected)
    return model.predict(X_scaled)


def _nan_safe_corr(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X_centered = X - X.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()
    numerator = X_centered.T @ y_centered
    denominator = np.sqrt((X_centered**2).sum(axis=0) * (y_centered**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denominator
    return np.nan_to_num(corr, nan=0.0)


class GenderSpecificRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        female_k: int = 3500,
        male_k: int = 1000,
        female_alpha: float = 0.01,
        male_alpha: float = 0.01,
    ):
        self.female_k = female_k
        self.male_k = male_k
        self.female_alpha = female_alpha
        self.male_alpha = male_alpha

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_array = np.asarray(y).reshape(-1)
        self.numeric_columns_ = _numeric_columns(X)
        X_numeric = X[self.numeric_columns_].to_numpy()
        genders = X["gender"].astype(str).to_numpy()

        self.models_ = {}
        for gender_value in ["f", "m"]:
            mask = genders == gender_value
            if not np.any(mask):
                continue
            k = self.female_k if gender_value == "f" else self.male_k
            alpha = self.female_alpha if gender_value == "f" else self.male_alpha
            self.models_[gender_value] = _fit_numeric_ridge(
                X_numeric[mask],
                y_array[mask],
                k=k,
                alpha=alpha,
            )

        fallback_k = max(self.female_k, self.male_k)
        fallback_alpha = min(self.female_alpha, self.male_alpha)
        self.fallback_model_ = _fit_numeric_ridge(
            X_numeric,
            y_array,
            k=fallback_k,
            alpha=fallback_alpha,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X[self.numeric_columns_].to_numpy()
        genders = X["gender"].astype(str).to_numpy()
        predictions = np.zeros(X.shape[0], dtype=float)

        for gender_value in ["f", "m"]:
            mask = genders == gender_value
            if not np.any(mask):
                continue
            model = self.models_.get(gender_value, self.fallback_model_)
            predictions[mask] = _predict_numeric_ridge(model, X_numeric[mask])

        unknown_mask = ~(np.isin(genders, ["f", "m"]))
        if np.any(unknown_mask):
            predictions[unknown_mask] = _predict_numeric_ridge(
                self.fallback_model_,
                X_numeric[unknown_mask],
            )

        return predictions


class GenderInteractionRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, main_k: int = 3500, interaction_k: int = 300, alpha: float = 0.01):
        self.main_k = main_k
        self.interaction_k = interaction_k
        self.alpha = alpha

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_array = np.asarray(y).reshape(-1)
        self.numeric_columns_ = _numeric_columns(X)
        X_numeric = X[self.numeric_columns_].to_numpy()
        male_indicator = (X["gender"].astype(str).to_numpy() == "m").astype(float)

        main_scores, _ = f_regression(X_numeric, y_array)
        self.main_indices_ = _top_k_indices(main_scores, self.main_k)
        X_main = X_numeric[:, self.main_indices_]

        male_mask = male_indicator == 1.0
        female_mask = ~male_mask.astype(bool)
        male_corr = _nan_safe_corr(X_main[male_mask], y_array[male_mask])
        female_corr = _nan_safe_corr(X_main[female_mask], y_array[female_mask])
        diff_scores = np.abs(male_corr - female_corr)
        self.interaction_indices_ = _top_k_indices(diff_scores, self.interaction_k)

        self.scaler_ = StandardScaler()
        X_main_scaled = self.scaler_.fit_transform(X_main)
        interaction_block = X_main_scaled[:, self.interaction_indices_] * male_indicator[:, None]
        design = np.hstack([X_main_scaled, male_indicator[:, None], interaction_block])

        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(design, y_array)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X[self.numeric_columns_].to_numpy()
        male_indicator = (X["gender"].astype(str).to_numpy() == "m").astype(float)
        X_main = X_numeric[:, self.main_indices_]
        X_main_scaled = self.scaler_.transform(X_main)
        interaction_block = X_main_scaled[:, self.interaction_indices_] * male_indicator[:, None]
        design = np.hstack([X_main_scaled, male_indicator[:, None], interaction_block])
        return self.model_.predict(design)


class ClusterAugmentedRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        main_k: int = 3500,
        alpha: float = 0.01,
        n_clusters: int = 3,
        pca_components: int = 20,
    ):
        self.main_k = main_k
        self.alpha = alpha
        self.n_clusters = n_clusters
        self.pca_components = pca_components

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_array = np.asarray(y).reshape(-1)
        self.numeric_columns_ = _numeric_columns(X)
        X_numeric = X[self.numeric_columns_].to_numpy()
        male_indicator = (X["gender"].astype(str).to_numpy() == "m").astype(float)

        main_scores, _ = f_regression(X_numeric, y_array)
        self.main_indices_ = _top_k_indices(main_scores, self.main_k)
        X_main = X_numeric[:, self.main_indices_]

        self.scaler_ = StandardScaler()
        X_main_scaled = self.scaler_.fit_transform(X_main)
        effective_components = min(
            self.pca_components,
            X_main_scaled.shape[0] - 1,
            X_main_scaled.shape[1],
        )
        self.pca_ = PCA(n_components=max(2, effective_components), random_state=42)
        X_pca = self.pca_.fit_transform(X_main_scaled)

        self.kmeans_ = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=42)
        cluster_labels = self.kmeans_.fit_predict(X_pca)
        cluster_one_hot = np.eye(self.n_clusters)[cluster_labels]
        design = np.hstack([X_main_scaled, male_indicator[:, None], cluster_one_hot])

        self.model_ = Ridge(alpha=self.alpha)
        self.model_.fit(design, y_array)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X[self.numeric_columns_].to_numpy()
        male_indicator = (X["gender"].astype(str).to_numpy() == "m").astype(float)
        X_main = X_numeric[:, self.main_indices_]
        X_main_scaled = self.scaler_.transform(X_main)
        X_pca = self.pca_.transform(X_main_scaled)
        cluster_labels = self.kmeans_.predict(X_pca)
        cluster_one_hot = np.eye(self.n_clusters)[cluster_labels]
        design = np.hstack([X_main_scaled, male_indicator[:, None], cluster_one_hot])
        return self.model_.predict(design)


class GenderBlendRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        female_inter_weight: float = 0.0,
        male_inter_weight: float = 0.6,
        interaction_main_k: int = 3500,
        interaction_k: int = 100,
        interaction_alpha: float = 0.01,
        base_model_variant: str = "plain_ridge",
    ):
        self.female_inter_weight = female_inter_weight
        self.male_inter_weight = male_inter_weight
        self.interaction_main_k = interaction_main_k
        self.interaction_k = interaction_k
        self.interaction_alpha = interaction_alpha
        self.base_model_variant = base_model_variant

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        target = np.asarray(y).reshape(-1)
        if self.base_model_variant == "plain_ridge":
            self.global_model_ = build_final_model(list(X.columns))
        elif self.base_model_variant == "bagged_ridge":
            self.global_model_ = build_best_ridge_feature_model(list(X.columns))
        else:
            raise ValueError(f"Unsupported base_model_variant: {self.base_model_variant}")
        self.interaction_model_ = GenderInteractionRidgeRegressor(
            main_k=self.interaction_main_k,
            interaction_k=self.interaction_k,
            alpha=self.interaction_alpha,
        )
        self.global_model_.fit(X, target)
        self.interaction_model_.fit(X, target)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        global_pred = np.asarray(self.global_model_.predict(X)).reshape(-1)
        interaction_pred = np.asarray(self.interaction_model_.predict(X)).reshape(-1)
        inter_weight = np.where(
            X["gender"].astype(str).to_numpy() == "f",
            self.female_inter_weight,
            self.male_inter_weight,
        )
        return (1.0 - inter_weight) * global_pred + inter_weight * interaction_pred


def build_best_gender_blend_model() -> GenderBlendRegressor:
    return GenderBlendRegressor(
        female_inter_weight=0.0,
        male_inter_weight=0.7,
        interaction_main_k=3500,
        interaction_k=100,
        interaction_alpha=0.01,
        base_model_variant="bagged_ridge",
    )
