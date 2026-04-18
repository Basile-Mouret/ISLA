import mne
import numpy as np
from mne.decoding import CSP
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from scipy.signal import butter, sosfiltfilt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


mne.set_log_level("ERROR")

MODEL_SPECS = [
    {"model": "mne_fbcsp_lda", "family": "fbcsp"},
    {"model": "fbcsp_broad_c6_k16_lda", "family": "fbcsp"},
    {"model": "fbcsp_dense_c6_k20_lda", "family": "fbcsp"},
    {"model": "mne_csp_8_30_lda", "family": "csp"},
    {"model": "riemann_ts_lr_6_35", "family": "riemann_ts"},
    {"model": "riemann_ts_lr_8_30", "family": "riemann_ts"},
    {"model": "riemann_fgmdm_8_30", "family": "riemann_fgmdm"},
    {"model": "riemann_fgmdm_8_30_lwf", "family": "riemann_fgmdm"},
]


class TemporalCropper(BaseEstimator, TransformerMixin):
    def __init__(self, start=0, stop=None):
        self.start = start
        self.stop = stop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[..., self.start:self.stop]


class BandpassFilter(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate_hz=256.0, low_cut_hz=8.0, high_cut_hz=30.0, order=4):
        self.sample_rate_hz = sample_rate_hz
        self.low_cut_hz = low_cut_hz
        self.high_cut_hz = high_cut_hz
        self.order = order

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        sos = butter(
            self.order,
            [self.low_cut_hz, self.high_cut_hz],
            btype="bandpass",
            fs=self.sample_rate_hz,
            output="sos",
        )
        return sosfiltfilt(sos, X, axis=-1)


class FilterBankCSPTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        sample_rate_hz=256.0,
        bands=((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)),
        filter_order=4,
        n_components=4,
        reg="oas",
    ):
        self.sample_rate_hz = sample_rate_hz
        self.bands = bands
        self.filter_order = filter_order
        self.n_components = n_components
        self.reg = reg

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
                cov_est="epoch",
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


def _build_fbcsp_model(sample_rate_hz, start, stop, bands, n_components, select_k):
    return Pipeline([
        ("crop", TemporalCropper(start=start, stop=stop)),
        (
            "fbcsp",
            FilterBankCSPTransformer(
                sample_rate_hz=sample_rate_hz,
                bands=bands,
                filter_order=4,
                n_components=n_components,
                reg="oas",
            ),
        ),
        ("select", SelectKBest(score_func=f_classif, k=select_k)),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])


def build_model(model_name, sample_rate_hz, start, stop):
    if model_name == "mne_fbcsp_lda":
        return _build_fbcsp_model(
            sample_rate_hz,
            start,
            stop,
            bands=((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)),
            n_components=4,
            select_k=12,
        )

    if model_name == "fbcsp_broad_c6_k16_lda":
        return _build_fbcsp_model(
            sample_rate_hz,
            start,
            stop,
            bands=((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)),
            n_components=6,
            select_k=16,
        )

    if model_name == "fbcsp_dense_c6_k20_lda":
        return _build_fbcsp_model(
            sample_rate_hz,
            start,
            stop,
            bands=((6.0, 10.0), (8.0, 12.0), (10.0, 14.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0), (28.0, 32.0)),
            n_components=6,
            select_k=20,
        )

    if model_name == "mne_csp_8_30_lda":
        return Pipeline([
            ("crop", TemporalCropper(start=start, stop=stop)),
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
        ])

    if model_name == "riemann_ts_lr_6_35":
        return Pipeline([
            ("crop", TemporalCropper(start=start, stop=stop)),
            ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=6.0, high_cut_hz=35.0, order=4)),
            ("cov", Covariances(estimator="oas")),
            ("tangent", TangentSpace(metric="riemann")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=4000)),
        ])

    if model_name == "riemann_ts_lr_8_30":
        return Pipeline([
            ("crop", TemporalCropper(start=start, stop=stop)),
            ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
            ("cov", Covariances(estimator="oas")),
            ("tangent", TangentSpace(metric="riemann")),
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=4000)),
        ])

    if model_name == "riemann_fgmdm_8_30":
        return Pipeline([
            ("crop", TemporalCropper(start=start, stop=stop)),
            ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
            ("cov", Covariances(estimator="oas")),
            ("clf", FgMDM(metric="riemann")),
        ])

    if model_name == "riemann_fgmdm_8_30_lwf":
        return Pipeline([
            ("crop", TemporalCropper(start=start, stop=stop)),
            ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
            ("cov", Covariances(estimator="lwf")),
            ("clf", FgMDM(metric="riemann")),
        ])

    raise ValueError(f"Unknown model: {model_name}")


def weighted_vote(prediction_rows, weights):
    labels = sorted({label for predictions in prediction_rows for label in predictions.tolist()})
    label_to_index = {label: index for index, label in enumerate(labels)}
    vote_matrix = np.zeros((prediction_rows[0].shape[0], len(labels)), dtype=float)

    for predictions, weight in zip(prediction_rows, weights):
        for index, label in enumerate(predictions):
            vote_matrix[index, label_to_index[label]] += weight

    return np.asarray([labels[index] for index in vote_matrix.argmax(axis=1)])


def predict_weighted_ensemble(rows, X_train, y_train, X_test, sample_rate_hz):
    prediction_rows = []
    weights = []
    for row in rows:
        model = build_model(
            model_name=row["model"],
            sample_rate_hz=sample_rate_hz,
            start=int(row["start"]),
            stop=int(row["stop"]),
        )
        model.fit(X_train, y_train)
        prediction_rows.append(model.predict(X_test))
        weights.append(float(row["weight"]))
    return weighted_vote(prediction_rows, weights)


def evaluate_candidate_task(
    subject,
    model_name,
    family,
    start,
    stop,
    sample_rate_hz,
    X_train,
    y_train,
    split_list,
    cache_path,
    inner_threads,
    logger,
    verbose,
    log_label,
):
    oof_pred = np.empty(y_train.shape[0], dtype="<U16")
    fold_scores = []

    with threadpool_limits(limits=inner_threads):
        for fold_index, (train_idx, valid_idx) in enumerate(split_list, start=1):
            if verbose >= 2:
                logger.info("%s FOLD %d/%d", log_label, fold_index, len(split_list))
            model = build_model(model_name=model_name, sample_rate_hz=sample_rate_hz, start=start, stop=stop)
            model.fit(X_train[train_idx], y_train[train_idx])
            pred = model.predict(X_train[valid_idx])
            oof_pred[valid_idx] = pred
            fold_score = accuracy_score(y_train[valid_idx], pred)
            fold_scores.append(fold_score)
            if verbose >= 2:
                logger.info("%s FOLD %d/%d SCORE %.4f", log_label, fold_index, len(split_list), fold_score)

    temp_path = f"{cache_path}.tmp.npy"
    np.save(temp_path, oof_pred)
    os.replace(temp_path, cache_path)

    return {
        "subject": subject,
        "model": model_name,
        "family": family,
        "start": int(start),
        "stop": int(stop),
        "mean_accuracy": float(np.mean(fold_scores)),
        "std_accuracy": float(np.std(fold_scores)),
    }
