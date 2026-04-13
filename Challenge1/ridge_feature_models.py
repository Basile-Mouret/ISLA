from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import f_regression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


def _numeric_columns(X: pd.DataFrame) -> list[str]:
    return [column for column in X.columns if column != "gender"]


def _male_indicator(X: pd.DataFrame) -> np.ndarray:
    return (X["gender"].astype(str).to_numpy() == "m").astype(float)


def _safe_abs_corr(X_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    X_centered = X_values - X_values.mean(axis=0, keepdims=True)
    y_centered = y_values - y_values.mean()
    numerator = X_centered.T @ y_centered
    denominator = np.sqrt((X_centered**2).sum(axis=0) * (y_centered**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denominator
    return np.abs(np.nan_to_num(corr, nan=0.0))


def _rank_to_unit(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(scores))
    if scores.shape[0] <= 1:
        return np.zeros_like(scores, dtype=float)
    return order.astype(float) / float(scores.shape[0] - 1)


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, scores.shape[0]))
    return np.argsort(scores)[-k:]


def _fit_ridge_bundle_with_target(
    X_numeric: np.ndarray,
    male_indicator: np.ndarray,
    selected_indices: np.ndarray,
    y_values: np.ndarray,
    alpha: float,
) -> dict[str, object]:
    X_selected = X_numeric[:, selected_indices]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    design = np.column_stack([X_scaled, male_indicator])
    model = Ridge(alpha=alpha)
    model.fit(design, y_values)
    return {"scaler": scaler, "model": model, "selected_indices": selected_indices}


def _predict_ridge_bundle(bundle: dict[str, object], X_numeric: np.ndarray, male_indicator: np.ndarray) -> np.ndarray:
    X_selected = X_numeric[:, bundle["selected_indices"]]
    X_scaled = bundle["scaler"].transform(X_selected)
    design = np.column_stack([X_scaled, male_indicator])
    return bundle["model"].predict(design)


def compute_feature_scores(
    X_numeric: np.ndarray,
    y_values: np.ndarray,
    genders: np.ndarray,
    score_method: str,
    gap_penalty: float,
) -> np.ndarray:
    if score_method == "f_score":
        scores, _ = f_regression(X_numeric, y_values)
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    if score_method == "abs_corr":
        return _safe_abs_corr(X_numeric, y_values)

    if score_method == "gender_stable":
        overall = _safe_abs_corr(X_numeric, y_values)
        male_mask = genders == "m"
        female_mask = genders == "f"
        male_scores = _safe_abs_corr(X_numeric[male_mask], y_values[male_mask])
        female_scores = _safe_abs_corr(X_numeric[female_mask], y_values[female_mask])
        shared = np.minimum(male_scores, female_scores)
        gap = np.abs(male_scores - female_scores)

        overall_rank = _rank_to_unit(overall)
        shared_rank = _rank_to_unit(shared)
        gap_rank = _rank_to_unit(gap)
        return overall_rank + shared_rank - gap_penalty * gap_rank

    raise ValueError(f"Unsupported score method: {score_method}")


class StableScoreRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        k: int = 3500,
        alpha: float = 0.01,
        score_method: str = "f_score",
        n_resamples: int = 15,
        sample_fraction: float = 0.8,
        gap_penalty: float = 0.5,
        random_state: int = 42,
    ):
        self.k = k
        self.alpha = alpha
        self.score_method = score_method
        self.n_resamples = n_resamples
        self.sample_fraction = sample_fraction
        self.gap_penalty = gap_penalty
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_values = np.asarray(y).reshape(-1)
        self.numeric_columns_ = _numeric_columns(X)
        X_numeric = X[self.numeric_columns_].to_numpy()
        genders = X["gender"].astype(str).to_numpy()
        male_indicator = _male_indicator(X)

        rng = np.random.default_rng(self.random_state)
        aggregated = np.zeros(X_numeric.shape[1], dtype=float)
        resample_size = max(2, math.ceil(self.sample_fraction * X_numeric.shape[0]))

        for _ in range(self.n_resamples):
            sample_idx = rng.choice(X_numeric.shape[0], size=resample_size, replace=False)
            scores = compute_feature_scores(
                X_numeric[sample_idx],
                y_values[sample_idx],
                genders[sample_idx],
                score_method=self.score_method,
                gap_penalty=self.gap_penalty,
            )
            aggregated += _rank_to_unit(scores)

        self.aggregated_scores_ = aggregated / float(self.n_resamples)
        self.selected_indices_ = _top_k_indices(self.aggregated_scores_, self.k)
        self.selected_features_ = [self.numeric_columns_[idx] for idx in self.selected_indices_]
        self.bundle_ = _fit_ridge_bundle_with_target(
            X_numeric,
            male_indicator,
            self.selected_indices_,
            y_values,
            alpha=self.alpha,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X[self.numeric_columns_].to_numpy()
        male_indicator = _male_indicator(X)
        return _predict_ridge_bundle(self.bundle_, X_numeric, male_indicator)


class BaggedScoreRidgeRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        k: int = 3500,
        alpha: float = 0.01,
        score_method: str = "f_score",
        n_estimators: int = 7,
        sample_fraction: float = 0.8,
        gap_penalty: float = 0.5,
        random_state: int = 42,
    ):
        self.k = k
        self.alpha = alpha
        self.score_method = score_method
        self.n_estimators = n_estimators
        self.sample_fraction = sample_fraction
        self.gap_penalty = gap_penalty
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray):
        y_values = np.asarray(y).reshape(-1)
        self.numeric_columns_ = _numeric_columns(X)
        X_numeric = X[self.numeric_columns_].to_numpy()
        genders = X["gender"].astype(str).to_numpy()
        male_indicator = _male_indicator(X)

        rng = np.random.default_rng(self.random_state)
        resample_size = max(2, math.ceil(self.sample_fraction * X_numeric.shape[0]))

        self.bundles_ = []
        self.selected_indices_list_ = []
        selection_frequency = np.zeros(X_numeric.shape[1], dtype=int)

        for _ in range(self.n_estimators):
            sample_idx = rng.choice(X_numeric.shape[0], size=resample_size, replace=False)
            scores = compute_feature_scores(
                X_numeric[sample_idx],
                y_values[sample_idx],
                genders[sample_idx],
                score_method=self.score_method,
                gap_penalty=self.gap_penalty,
            )
            selected_indices = _top_k_indices(scores, self.k)
            selection_frequency[selected_indices] += 1
            self.selected_indices_list_.append(selected_indices)
            self.bundles_.append(
                _fit_ridge_bundle_with_target(
                    X_numeric,
                    male_indicator,
                    selected_indices,
                    y_values,
                    alpha=self.alpha,
                )
            )

        self.selection_frequency_ = selection_frequency
        self.consensus_indices_ = _top_k_indices(selection_frequency.astype(float), self.k)
        self.consensus_features_ = [self.numeric_columns_[idx] for idx in self.consensus_indices_]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X_numeric = X[self.numeric_columns_].to_numpy()
        male_indicator = _male_indicator(X)
        predictions = [
            _predict_ridge_bundle(bundle, X_numeric, male_indicator) for bundle in self.bundles_
        ]
        return np.mean(np.vstack(predictions), axis=0)
