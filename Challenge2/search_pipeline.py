import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import itertools
import logging
import os
import zipfile

import mne
import numpy as np
import pandas as pd
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
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


mne.set_log_level("ERROR")

SUBJECTS = ["A", "B", "C", "D", "E", "F"]
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


def load_subject_data(data_dir, subject):
    X_train = np.load(f"{data_dir}/subject_{subject}_X_train.npy")
    y_train = np.load(f"{data_dir}/subject_{subject}_y_train.npy")
    X_test = np.load(f"{data_dir}/subject_{subject}_X_test.npy")
    return X_train, y_train, X_test


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


def validate_prediction_csv(csv_path):
    with open(csv_path, newline="") as handle:
        rows = list(csv.reader(handle))

    if not rows:
        raise ValueError(f"{csv_path} is empty")
    if rows[0] != ["y_pred"]:
        raise ValueError(f"{csv_path} must contain exactly one header column named 'y_pred'")
    if len(rows) != 61:
        raise ValueError(f"{csv_path} must contain exactly 60 predictions")

    valid_labels = {"left_hand", "right_hand"}
    for row in rows[1:]:
        if len(row) != 1 or row[0] not in valid_labels:
            raise ValueError(f"{csv_path} contains an invalid prediction value: {row}")


def write_prediction_dir(output_dir, predictions_by_subject):
    os.makedirs(output_dir, exist_ok=True)
    for subject in SUBJECTS:
        pd.DataFrame({"y_pred": predictions_by_subject[subject]}).to_csv(
            os.path.join(output_dir, f"subject_{subject}_y_pred.csv"),
            index=False,
        )


def package_submission_dir(model_dir, output_path=None):
    model_dir = os.path.abspath(model_dir)
    model_name = os.path.basename(os.path.normpath(model_dir))
    output_path = os.path.abspath(output_path or os.path.join("submissions", f"{model_name}.zip"))

    csv_paths = []
    for subject in SUBJECTS:
        csv_path = os.path.join(model_dir, f"subject_{subject}_y_pred.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Missing required file: {csv_path}")
        validate_prediction_csv(csv_path)
        csv_paths.append(csv_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for csv_path in csv_paths:
            archive.write(csv_path, arcname=os.path.basename(csv_path))

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Heavy subject-wise model search with diversity-aware top-k voting selection."
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--submissions-dir", default="submissions")
    parser.add_argument("--sample-rate-hz", type=float, default=256.0)
    parser.add_argument("--cv-folds", type=int, default=4)
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Logging verbosity: 0=summary only, 1=per-candidate, 2=per-fold.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel candidate workers.")
    parser.add_argument(
        "--inner-threads",
        type=int,
        default=1,
        help="Number of BLAS/OpenMP threads per worker (set to 1 to avoid oversubscription).",
    )

    parser.add_argument("--start-min", type=int, default=192)
    parser.add_argument("--start-max", type=int, default=960)
    parser.add_argument("--start-step", type=int, default=64)
    parser.add_argument("--stop-min", type=int, default=1088)
    parser.add_argument("--stop-max", type=int, default=1537)
    parser.add_argument("--stop-step", type=int, default=64)
    parser.add_argument("--min-window-len", type=int, default=384)

    parser.add_argument("--top-per-model", type=int, default=8)
    parser.add_argument("--max-candidates-per-subject", type=int, default=48)

    parser.add_argument("--min-ensemble-size", type=int, default=2)
    parser.add_argument("--max-ensemble-size", type=int, default=3)
    parser.add_argument("--max-same-model", type=int, default=1)
    parser.add_argument("--max-same-family", type=int, default=2)
    parser.add_argument("--min-window-shift", type=int, default=96)
    parser.add_argument("--min-disagreement", type=float, default=0.04)
    parser.add_argument("--diversity-weight", type=float, default=0.03)
    parser.add_argument("--double-fault-weight", type=float, default=0.02)

    parser.add_argument("--estimate-only", action="store_true")

    parser.add_argument("--results-csv", default="submissions/heavy_search_results.csv")
    parser.add_argument("--selection-csv", default="submissions/heavy_search_selection.csv")
    parser.add_argument("--models-csv", default="submissions/heavy_subject_models.csv")
    parser.add_argument("--cache-dir", default="submissions/heavy_oof_cache")
    parser.add_argument("--output-model-name", default="heavy_top3_diverse")
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def setup_logging(log_file_path, verbose):
    logger = logging.getLogger("search_pipeline")
    logger.setLevel(logging.INFO if verbose > 0 else logging.WARNING)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def set_thread_env(inner_threads):
    value = str(int(inner_threads))
    os.environ["OMP_NUM_THREADS"] = value
    os.environ["OPENBLAS_NUM_THREADS"] = value
    os.environ["MKL_NUM_THREADS"] = value
    os.environ["BLIS_NUM_THREADS"] = value
    os.environ["VECLIB_MAXIMUM_THREADS"] = value
    os.environ["NUMEXPR_NUM_THREADS"] = value


def run_tag(args):
    payload = repr(
        (
            args.cv_folds,
            args.start_min,
            args.start_max,
            args.start_step,
            args.stop_min,
            args.stop_max,
            args.stop_step,
            args.min_window_len,
            args.top_per_model,
            args.max_candidates_per_subject,
            args.min_ensemble_size,
            args.max_ensemble_size,
            args.max_same_model,
            args.max_same_family,
            args.min_window_shift,
            args.min_disagreement,
            args.diversity_weight,
            args.double_fault_weight,
            args.sample_rate_hz,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def make_window_grid(n_times, args):
    starts = list(range(args.start_min, args.start_max + 1, args.start_step))
    stops = list(range(args.stop_min, args.stop_max + 1, args.stop_step))
    if n_times not in stops:
        stops.append(n_times)

    windows = []
    for start in starts:
        for stop in stops:
            if stop > n_times:
                continue
            if stop - start < args.min_window_len:
                continue
            if stop <= start:
                continue
            windows.append((start, stop))

    windows.append((0, n_times))
    return sorted(set(windows))


def candidate_key(subject, model_name, start, stop):
    return (subject, model_name, int(start), int(stop))


def oof_cache_path(cache_dir, subject, model_name, start, stop):
    key = f"{subject}__{model_name}__{int(start)}__{int(stop)}".replace("/", "_")
    return os.path.join(cache_dir, f"{key}.npy")


def load_results(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = ["subject", "model", "family", "start", "stop", "mean_accuracy", "std_accuracy"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"Missing column '{column}' in existing results file: {path}")
    df["start"] = df["start"].astype(int)
    df["stop"] = df["stop"].astype(int)
    return df.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="last")


def append_result_row(path, row):
    row_df = pd.DataFrame([row])
    row_df.to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def estimate_fit_counts(subjects, windows, cv_folds):
    candidates = len(subjects) * len(MODEL_SPECS) * len(windows)
    return candidates, candidates * cv_folds


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


def top_candidate_pool(subject_results, args):
    parts = []
    for model_name in subject_results["model"].unique():
        parts.append(
            subject_results[subject_results["model"] == model_name]
            .sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
            .head(args.top_per_model)
        )

    if not parts:
        return pd.DataFrame()

    pool = pd.concat(parts, ignore_index=True)
    pool = pool.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="first")
    return (
        pool.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
        .head(args.max_candidates_per_subject)
        .reset_index(drop=True)
    )


def disagreement(pred_a, pred_b):
    return float(np.mean(pred_a != pred_b))


def double_fault(pred_a, pred_b, y_true):
    return float(np.mean((pred_a != y_true) & (pred_b != y_true)))


def combo_valid(combo_rows, args):
    model_counts = {}
    family_counts = {}
    for row in combo_rows:
        model_counts[row.model] = model_counts.get(row.model, 0) + 1
        family_counts[row.family] = family_counts.get(row.family, 0) + 1
        if model_counts[row.model] > args.max_same_model:
            return False
        if family_counts[row.family] > args.max_same_family:
            return False

    for left, right in itertools.combinations(combo_rows, 2):
        if left.family != right.family:
            continue
        if abs(int(left.start) - int(right.start)) < args.min_window_shift and abs(int(left.stop) - int(right.stop)) < args.min_window_shift:
            return False
    return True


def combo_metrics(combo_rows, oof_lookup, y_true, args):
    oof_rows = [oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] for row in combo_rows]
    weights = [float(row.mean_accuracy) for row in combo_rows]
    ensemble_pred = weighted_vote(oof_rows, weights)
    ensemble_acc = accuracy_score(y_true, ensemble_pred)

    if len(combo_rows) == 1:
        avg_disagreement = 0.0
        avg_double_fault = 0.0
    else:
        disagreements = []
        double_faults = []
        for left_index in range(len(combo_rows)):
            for right_index in range(left_index + 1, len(combo_rows)):
                left_pred = oof_rows[left_index]
                right_pred = oof_rows[right_index]
                disagreements.append(disagreement(left_pred, right_pred))
                double_faults.append(double_fault(left_pred, right_pred, y_true))
        avg_disagreement = float(np.mean(disagreements))
        avg_double_fault = float(np.mean(double_faults))

    score = ensemble_acc + args.diversity_weight * avg_disagreement - args.double_fault_weight * avg_double_fault
    return {
        "ensemble_accuracy": float(ensemble_acc),
        "avg_disagreement": avg_disagreement,
        "avg_double_fault": avg_double_fault,
        "score": float(score),
    }


def select_diverse_combo(subject_pool, oof_lookup, y_true, args):
    rows = list(subject_pool.itertuples(index=False))
    best_combo = None
    best_metrics = None

    sizes = list(range(args.min_ensemble_size, args.max_ensemble_size + 1))
    if args.min_ensemble_size <= 1:
        sizes = [1] + sizes

    for size in sizes:
        for combo_idx in itertools.combinations(range(len(rows)), size):
            combo_rows = [rows[index] for index in combo_idx]
            if not combo_valid(combo_rows, args):
                continue

            metrics = combo_metrics(combo_rows, oof_lookup, y_true, args)
            if size > 1 and metrics["avg_disagreement"] < args.min_disagreement:
                continue

            if best_combo is None:
                best_combo = combo_rows
                best_metrics = metrics
                continue

            left = (
                metrics["score"],
                metrics["ensemble_accuracy"],
                metrics["avg_disagreement"],
                -metrics["avg_double_fault"],
            )
            right = (
                best_metrics["score"],
                best_metrics["ensemble_accuracy"],
                best_metrics["avg_disagreement"],
                -best_metrics["avg_double_fault"],
            )
            if left > right:
                best_combo = combo_rows
                best_metrics = metrics

    if best_combo is None:
        best_combo = [rows[0]]
        best_metrics = combo_metrics(best_combo, oof_lookup, y_true, args)

    return best_combo, best_metrics


def main():
    args = parse_args()
    os.makedirs(args.submissions_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    set_thread_env(args.inner_threads)

    tag = run_tag(args)
    log_file = args.log_file or os.path.join(args.submissions_dir, f"heavy_search_{tag}.log")
    logger = setup_logging(log_file, args.verbose)

    logger.info("Run tag: %s", tag)
    logger.info("Results CSV: %s", args.results_csv)
    logger.info("Selection CSV: %s", args.selection_csv)
    logger.info("Models CSV: %s", args.models_csv)
    logger.info("Output model name: %s", args.output_model_name)
    logger.info("Verbose level: %d", args.verbose)
    logger.info("Parallel workers (n_jobs): %d", args.n_jobs)
    logger.info("Inner threads per worker: %d", args.inner_threads)

    X_a, _, _ = load_subject_data(args.data_dir, "A")
    windows = make_window_grid(X_a.shape[-1], args)
    logger.info("Window grid size: %d", len(windows))

    total_candidates, total_cv_fits = estimate_fit_counts(SUBJECTS, windows, args.cv_folds)
    existing = load_results(args.results_csv)
    done_keys = set()
    if not existing.empty:
        done_keys = {
            candidate_key(row.subject, row.model, row.start, row.stop)
            for row in existing.itertuples(index=False)
            if os.path.exists(oof_cache_path(args.cache_dir, row.subject, row.model, row.start, row.stop))
        }
    remaining_configs = total_candidates - len(done_keys)

    logger.info("Total candidate configs: %d", total_candidates)
    logger.info("Total CV fits (cold run): %d", total_cv_fits)
    logger.info("Cached configs with OOF present: %d", len(done_keys))
    logger.info("Remaining configs: %d", remaining_configs)
    logger.info("Remaining CV fits: %d", remaining_configs * args.cv_folds)

    if args.estimate_only:
        return

    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)
    processed = 0

    for subject in SUBJECTS:
        X_train, y_train, _ = load_subject_data(args.data_dir, subject)
        split_list = list(splitter.split(X_train, y_train))
        logger.info("Searching subject %s (%d train samples)", subject, X_train.shape[0])

        pending = []
        for spec in MODEL_SPECS:
            for start, stop in windows:
                key = candidate_key(subject, spec["model"], start, stop)
                if key in done_keys:
                    logger.info("[cache skip] %s | %s | samples_%d_%d", subject, spec["model"], start, stop)
                    continue
                pending.append((
                    spec["model"],
                    spec["family"],
                    start,
                    stop,
                    key,
                    oof_cache_path(args.cache_dir, subject, spec["model"], start, stop),
                ))

        logger.info("Subject %s pending candidates: %d", subject, len(pending))
        if not pending:
            continue

        if args.n_jobs <= 1:
            for model_name, family, start, stop, key, cache_path in pending:
                processed += 1
                label = f"{subject} | {model_name} | samples_{start}_{stop}"
                logger.info("[search %d/%d] START %s", processed, remaining_configs, label)
                row = evaluate_candidate_task(
                    subject=subject,
                    model_name=model_name,
                    family=family,
                    start=start,
                    stop=stop,
                    sample_rate_hz=args.sample_rate_hz,
                    X_train=X_train,
                    y_train=y_train,
                    split_list=split_list,
                    cache_path=cache_path,
                    inner_threads=args.inner_threads,
                    logger=logger,
                    verbose=args.verbose,
                    log_label=f"[search {processed}/{remaining_configs}] {label}",
                )
                append_result_row(args.results_csv, row)
                done_keys.add(key)
                logger.info(
                    "[search %d/%d] DONE %s mean=%.4f std=%.4f",
                    processed,
                    remaining_configs,
                    label,
                    row["mean_accuracy"],
                    row["std_accuracy"],
                )
        else:
            with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
                future_map = {}
                for model_name, family, start, stop, key, cache_path in pending:
                    label = f"{subject} | {model_name} | samples_{start}_{stop}"
                    logger.info("[submit] %s", label)
                    future = executor.submit(
                        evaluate_candidate_task,
                        subject,
                        model_name,
                        family,
                        start,
                        stop,
                        args.sample_rate_hz,
                        X_train,
                        y_train,
                        split_list,
                        cache_path,
                        args.inner_threads,
                        logger,
                        args.verbose,
                        label,
                    )
                    future_map[future] = (model_name, start, stop, key)

                for future in as_completed(future_map):
                    model_name, start, stop, key = future_map[future]
                    processed += 1
                    label = f"{subject} | {model_name} | samples_{start}_{stop}"
                    row = future.result()
                    append_result_row(args.results_csv, row)
                    done_keys.add(key)
                    logger.info(
                        "[search %d/%d] DONE %s mean=%.4f std=%.4f",
                        processed,
                        remaining_configs,
                        label,
                        row["mean_accuracy"],
                        row["std_accuracy"],
                    )

    results = load_results(args.results_csv)
    selection_rows = []
    model_rows = []
    final_predictions = {}

    for subject in SUBJECTS:
        X_train, y_train, X_test = load_subject_data(args.data_dir, subject)
        subject_results = results[results["subject"] == subject]
        if subject_results.empty:
            raise ValueError(f"No search results found for subject {subject}")

        pool = top_candidate_pool(subject_results, args)
        logger.info("Subject %s pool size: %d", subject, len(pool))

        oof_lookup = {}
        for row in pool.itertuples(index=False):
            cache_path = oof_cache_path(args.cache_dir, row.subject, row.model, row.start, row.stop)
            if not os.path.exists(cache_path):
                raise FileNotFoundError(
                    f"Missing OOF cache for {row.subject}/{row.model}/{row.start}:{row.stop}: {cache_path}"
                )
            oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] = np.load(cache_path)

        selected_combo, metrics = select_diverse_combo(pool, oof_lookup, y_train, args)
        logger.info(
            "Subject %s selected combo size=%d acc=%.4f disagree=%.4f double_fault=%.4f score=%.4f",
            subject,
            len(selected_combo),
            metrics["ensemble_accuracy"],
            metrics["avg_disagreement"],
            metrics["avg_double_fault"],
            metrics["score"],
        )

        subject_rows = []
        for rank, row in enumerate(selected_combo, start=1):
            subject_rows.append(
                {
                    "subject": subject,
                    "rank": rank,
                    "model": row.model,
                    "family": row.family,
                    "start": int(row.start),
                    "stop": int(row.stop),
                    "weight": float(row.mean_accuracy),
                }
            )

        final_predictions[subject] = predict_weighted_ensemble(subject_rows, X_train, y_train, X_test, args.sample_rate_hz)
        model_rows.extend(subject_rows)
        selection_rows.append(
            {
                "subject": subject,
                "ensemble_size": len(selected_combo),
                "ensemble_oof_accuracy": metrics["ensemble_accuracy"],
                "avg_disagreement": metrics["avg_disagreement"],
                "avg_double_fault": metrics["avg_double_fault"],
                "selection_score": metrics["score"],
            }
        )

    selection_df = pd.DataFrame(selection_rows).sort_values("subject")
    selection_df.to_csv(args.selection_csv, index=False)
    pd.DataFrame(model_rows).sort_values(["subject", "rank"]).to_csv(args.models_csv, index=False)

    output_dir = os.path.join(args.submissions_dir, args.output_model_name)
    write_prediction_dir(output_dir, final_predictions)
    zip_path = package_submission_dir(output_dir, os.path.join(args.submissions_dir, f"{args.output_model_name}.zip"))
    logger.info("Generated submission zip: %s", zip_path)
    logger.info("Selection summary:\n%s", selection_df.to_string(index=False))


if __name__ == "__main__":
    main()
