import numpy as np
import mne
from eeg_models import BandpassFilter
from mne.decoding import CSP
from pyriemann.classification import FgMDM, MDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


mne.set_log_level("ERROR")


class TemporalCropper(BaseEstimator, TransformerMixin):
    def __init__(self, start=0, stop=None):
        self.start = start
        self.stop = stop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[..., self.start:self.stop]


class FilterBankCSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sample_rate_hz=256.0,
        bands=((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)),
        filter_order=4,
        n_components=4,
        reg="oas",
        cov_est="epoch",
    ):
        self.sample_rate_hz = sample_rate_hz
        self.bands = bands
        self.filter_order = filter_order
        self.n_components = n_components
        self.reg = reg
        self.cov_est = cov_est

    def fit(self, X, y):
        self.band_models_ = []
        for low_cut_hz, high_cut_hz in self.bands:
            bandpass = BandpassFilter(
                sample_rate_hz=self.sample_rate_hz,
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                order=self.filter_order,
            )
            csp = CSP(
                n_components=self.n_components,
                reg=self.reg,
                log=True,
                cov_est=self.cov_est,
                norm_trace=False,
                component_order="mutual_info",
            )
            X_band = bandpass.transform(X)
            csp.fit(X_band, y)
            self.band_models_.append((bandpass, csp))
        return self

    def transform(self, X):
        features = []
        for bandpass, csp in self.band_models_:
            X_band = bandpass.transform(X)
            features.append(csp.transform(X_band))
        return np.concatenate(features, axis=1)


def default_window_specs(n_times):
    window_specs = [{"name": f"samples_0_{n_times}", "start": 0, "stop": n_times}]

    targeted_windows = [
        (0, min(1024, n_times)),
        (256, min(1280, n_times)),
        (512, min(1536, n_times)),
        (0, min(768, n_times)),
        (256, min(1024, n_times)),
        (512, min(1280, n_times)),
        (768, min(1536, n_times)),
    ]
    for start, stop in targeted_windows:
        if start >= stop or stop > n_times:
            continue
        window_specs.append({"name": f"samples_{start}_{stop}", "start": start, "stop": stop})

    unique_window_specs = []
    seen = set()
    for window_spec in window_specs:
        key = (window_spec["start"], window_spec["stop"])
        if key in seen:
            continue
        seen.add(key)
        unique_window_specs.append(window_spec)
    return unique_window_specs


def default_model_names():
    return [
        "riemann_ts_lr_8_30",
        "riemann_ts_lr_6_35",
        "riemann_fgmdm_8_30",
        "mne_csp_8_30_lda",
        "mne_fbcsp_lda",
    ]


def build_model(model_name, sample_rate_hz, start, stop):
    prefix = [("crop", TemporalCropper(start=start, stop=stop))]

    if model_name == "riemann_ts_lr_8_30":
        return Pipeline(
            prefix
            + [
                ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
                ("cov", Covariances(estimator="oas")),
                ("tangent", TangentSpace(metric="riemann")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=4000)),
            ]
        )

    if model_name == "riemann_ts_lr_6_35":
        return Pipeline(
            prefix
            + [
                ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=6.0, high_cut_hz=35.0, order=4)),
                ("cov", Covariances(estimator="oas")),
                ("tangent", TangentSpace(metric="riemann")),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=4000)),
            ]
        )

    if model_name == "riemann_fgmdm_8_30":
        return Pipeline(
            prefix
            + [
                ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
                ("cov", Covariances(estimator="oas")),
                ("clf", FgMDM(metric="riemann")),
            ]
        )

    if model_name == "riemann_mdm_8_30":
        return Pipeline(
            prefix
            + [
                ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
                ("cov", Covariances(estimator="oas")),
                ("clf", MDM(metric="riemann")),
            ]
        )

    if model_name == "mne_csp_8_30_lda":
        return Pipeline(
            prefix
            + [
                ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
                (
                    "csp",
                    CSP(
                        n_components=6,
                        reg="oas",
                        log=True,
                        cov_est="epoch",
                        norm_trace=False,
                        component_order="mutual_info",
                    ),
                ),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )

    if model_name == "mne_csp_6_35_lda":
        return Pipeline(
            prefix
            + [
                ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=6.0, high_cut_hz=35.0, order=4)),
                (
                    "csp",
                    CSP(
                        n_components=6,
                        reg="oas",
                        log=True,
                        cov_est="epoch",
                        norm_trace=False,
                        component_order="mutual_info",
                    ),
                ),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )

    if model_name == "mne_fbcsp_lda":
        return Pipeline(
            prefix
            + [
                (
                    "fbcsp",
                    FilterBankCSPTransformer(
                        sample_rate_hz=sample_rate_hz,
                        bands=((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)),
                        filter_order=4,
                        n_components=4,
                        reg="oas",
                        cov_est="epoch",
                    ),
                ),
                ("select", SelectKBest(score_func=f_classif, k=12)),
                ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
            ]
        )

    raise ValueError(f"Unknown model: {model_name}")
