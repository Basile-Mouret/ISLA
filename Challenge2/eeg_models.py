import os

import numpy as np
import pandas as pd
from package_submission import package_submission_dir
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


SUBJECTS = ["A", "B", "C", "D", "E", "F"]


def bandpass_array(X, sample_rate_hz, low_cut_hz, high_cut_hz, order=4):
    sos = butter(
        order,
        [low_cut_hz, high_cut_hz],
        btype="bandpass",
        fs=sample_rate_hz,
        output="sos",
    )
    return sosfiltfilt(sos, X, axis=-1)


class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate_hz=256.0, low_cut_hz=8.0, high_cut_hz=30.0, order=4):
        self.sample_rate_hz = sample_rate_hz
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return bandpass_array(
            X,
            sample_rate_hz=self.sample_rate_hz,
            low_cut_hz=self.low_cut_hz,
            high_cut_hz=self.high_cut_hz,
            order=self.order,
        )


class LogVarianceTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        var = np.nan_to_num(np.var(X, axis=-1), nan=1e-10, posinf=1e-10, neginf=1e-10)
        return np.log(np.clip(var, 1e-10, None))


class WelchBandPowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate_hz=256.0, bands=((8.0, 12.0), (13.0, 30.0)), n_perseg=256):
        self.sample_rate_hz = sample_rate_hz
        self.bands = bands
        self.n_perseg = n_perseg

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        n_perseg = min(self.n_perseg, X.shape[-1])
        freqs, psd = welch(X, fs=self.sample_rate_hz, nperseg=n_perseg, axis=-1)
        band_features = []
        for low_cut_hz, high_cut_hz in self.bands:
            band_mask = (freqs >= low_cut_hz) & (freqs <= high_cut_hz)
            band_power = psd[..., band_mask].mean(axis=-1)
            band_features.append(np.log(np.clip(band_power, 1e-10, None)))
        return np.concatenate(band_features, axis=1)


class CSPFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=6, reg=1e-6):
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise ValueError("CSPFeatureTransformer supports binary classification only")

        covariances = {label: [] for label in classes}
        for trial, label in zip(X, y):
            covariances[label].append(self._normalized_covariance(trial))

        class_covariances = {
            label: np.mean(class_covariances, axis=0)
            for label, class_covariances in covariances.items()
        }
        composite_covariance = class_covariances[classes[0]] + class_covariances[classes[1]]

        eigenvalues, eigenvectors = eigh(class_covariances[classes[0]], composite_covariance)
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]

        selected_indices = []
        left_index = 0
        right_index = eigenvectors.shape[1] - 1
        while len(selected_indices) < self.n_components:
            selected_indices.append(left_index)
            left_index += 1
            if len(selected_indices) < self.n_components:
                selected_indices.append(right_index)
                right_index -= 1

        self.filters_ = eigenvectors[:, selected_indices].T
        return self

    def transform(self, X):
        projected = np.einsum("fc,nct->nft", self.filters_, X)
        variances = np.var(projected, axis=-1)
        normalized_variances = variances / np.clip(variances.sum(axis=1, keepdims=True), 1e-10, None)
        return np.log(np.clip(normalized_variances, 1e-10, None))

    def _normalized_covariance(self, trial):
        covariance = np.cov(trial, bias=True)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += self.reg * np.eye(covariance.shape[0])
        covariance /= np.clip(np.trace(covariance), 1e-10, None)
        return covariance


class FilterBankCSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sample_rate_hz=256.0,
        bands=((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0)),
        order=4,
        n_components=4,
        reg=1e-6,
    ):
        self.sample_rate_hz = sample_rate_hz
        self.bands = bands
        self.order = order
        self.n_components = n_components
        self.reg = reg

    def fit(self, X, y):
        self.band_models_ = []
        for low_cut_hz, high_cut_hz in self.bands:
            X_band = bandpass_array(
                X,
                sample_rate_hz=self.sample_rate_hz,
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                order=self.order,
            )
            csp = CSPFeatureTransformer(n_components=self.n_components, reg=self.reg)
            csp.fit(X_band, y)
            self.band_models_.append(((low_cut_hz, high_cut_hz), csp))
        return self

    def transform(self, X):
        features = []
        for (low_cut_hz, high_cut_hz), csp in self.band_models_:
            X_band = bandpass_array(
                X,
                sample_rate_hz=self.sample_rate_hz,
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                order=self.order,
            )
            features.append(csp.transform(X_band))
        return np.concatenate(features, axis=1)


class CovarianceVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, add_log_diag=True):
        self.add_log_diag = add_log_diag

    def fit(self, X, y=None):
        n_channels = X.shape[1]
        self.upper_triangle_indices_ = np.triu_indices(n_channels, k=1)
        return self

    def transform(self, X):
        feature_rows = []
        for trial in X:
            covariance = np.cov(trial, bias=True)
            covariance = 0.5 * (covariance + covariance.T)

            diagonal = np.clip(np.diag(covariance), 1e-10, None)
            scale = np.sqrt(diagonal)
            correlation = covariance / np.outer(scale, scale)
            row_features = [correlation[self.upper_triangle_indices_]]

            if self.add_log_diag:
                row_features.append(np.log(diagonal))

            feature_rows.append(np.concatenate(row_features))

        return np.asarray(feature_rows)


def build_model(model_name, sample_rate_hz=256.0):
    if model_name == "baseline_logvar_lda":
        return Pipeline([
            ("log_var", LogVarianceTransformer()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if model_name == "bandpass_8_30_logvar_lda":
        return Pipeline([
            (
                "bandpass",
                BandpassFilter(
                    sample_rate_hz=sample_rate_hz,
                    low_cut_hz=8.0,
                    high_cut_hz=30.0,
                    order=4,
                ),
            ),
            ("log_var", LogVarianceTransformer()),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if model_name == "bandpower_welch_lda":
        return Pipeline([
            (
                "bandpower",
                WelchBandPowerTransformer(
                    sample_rate_hz=sample_rate_hz,
                    bands=((8.0, 12.0), (13.0, 30.0)),
                    n_perseg=256,
                ),
            ),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if model_name == "csp_8_30_lda":
        return Pipeline([
            (
                "bandpass",
                BandpassFilter(
                    sample_rate_hz=sample_rate_hz,
                    low_cut_hz=8.0,
                    high_cut_hz=30.0,
                    order=4,
                ),
            ),
            ("csp", CSPFeatureTransformer(n_components=6, reg=1e-5)),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if model_name == "filterbank_csp_lda":
        return Pipeline([
            (
                "filterbank_csp",
                FilterBankCSPTransformer(
                    sample_rate_hz=sample_rate_hz,
                    bands=((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 30.0)),
                    order=4,
                    n_components=4,
                    reg=1e-5,
                ),
            ),
            ("select", SelectKBest(score_func=f_classif, k=8)),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ])

    if model_name == "covariance_svm":
        return Pipeline([
            (
                "bandpass",
                BandpassFilter(
                    sample_rate_hz=sample_rate_hz,
                    low_cut_hz=8.0,
                    high_cut_hz=30.0,
                    order=4,
                ),
            ),
            ("covariance", CovarianceVectorizer(add_log_diag=True)),
            ("scale", StandardScaler()),
            ("clf", LinearSVC(C=0.1, max_iter=10000)),
        ])

    raise ValueError(f"Unknown model: {model_name}")


def default_model_names():
    return [
        "baseline_logvar_lda",
        "bandpass_8_30_logvar_lda",
        "bandpower_welch_lda",
        "csp_8_30_lda",
        "filterbank_csp_lda",
        "covariance_svm",
    ]


def load_subject_data(data_dir, subject):
    X_train = np.load(os.path.join(data_dir, f"subject_{subject}_X_train.npy"))
    y_train = np.load(os.path.join(data_dir, f"subject_{subject}_y_train.npy"))
    X_test = np.load(os.path.join(data_dir, f"subject_{subject}_X_test.npy"))
    return X_train, y_train, X_test


def write_submission(model_name, predictions_by_subject, submissions_dir="submissions"):
    output_dir = os.path.join(submissions_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    for subject, predictions in predictions_by_subject.items():
        pd.DataFrame({"y_pred": predictions}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"),
            index=False,
        )

    return package_submission_dir(output_dir, os.path.join(submissions_dir, f"{model_name}.zip"))
