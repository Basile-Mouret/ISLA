import argparse
import hashlib
import logging
import os

import numpy as np
import pandas as pd
from eeg_models import SUBJECTS, BandpassFilter, load_subject_data, write_submission
from mne.decoding import CSP
from mne_pyriemann_models import FilterBankCSPTransformer, TemporalCropper, build_model as build_base_model
from pyriemann.classification import FgMDM
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class FilterBankTangentSpaceTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sample_rate_hz=256.0, bands=(), filter_order=4, estimator="oas", metric="riemann"):
        self.sample_rate_hz = sample_rate_hz
        self.bands = bands
        self.filter_order = filter_order
        self.estimator = estimator
        self.metric = metric

    def fit(self, X, y=None):
        self.band_models_ = []
        for low_cut_hz, high_cut_hz in self.bands:
            bandpass = BandpassFilter(
                sample_rate_hz=self.sample_rate_hz,
                low_cut_hz=low_cut_hz,
                high_cut_hz=high_cut_hz,
                order=self.filter_order,
            )
            tangent = TangentSpace(metric=self.metric)
            X_band = bandpass.transform(X)
            X_cov = Covariances(estimator=self.estimator).transform(X_band)
            tangent.fit(X_cov)
            self.band_models_.append((bandpass, tangent))
        return self

    def transform(self, X):
        features = []
        for bandpass, tangent in self.band_models_:
            X_band = bandpass.transform(X)
            X_cov = Covariances(estimator=self.estimator).transform(X_band)
            features.append(tangent.transform(X_cov))
        return np.concatenate(features, axis=1)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a deeper subject-specific MNE/CSP and pyRiemann fine-tuning search around prior winning configs."
    )
    parser.add_argument("--data-dir", default="data", help="Directory containing challenge .npy files.")
    parser.add_argument(
        "--submissions-dir",
        default="submissions",
        help="Directory where reports and submission archives are written.",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=256.0,
        help="Sampling rate used for filtering. Override if the dataset uses a different rate.",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=SUBJECTS,
        choices=SUBJECTS,
        help="Subset of subjects to retune. Untuned subjects fall back to prior best configs.",
    )
    parser.add_argument(
        "--base-results-path",
        default="submissions/mne_pyriemann_cv_results.csv",
        help="CSV from the prior MNE/pyRiemann search used to seed windows and model families.",
    )
    parser.add_argument(
        "--base-best-path",
        default="submissions/mne_pyriemann_subject_best_configs.csv",
        help="CSV containing the prior best config per subject.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=4,
        help="Number of stratified folds for repeated subject-specific CV.",
    )
    parser.add_argument(
        "--cv-repeats",
        type=int,
        default=1,
        help="Number of repeated CV rounds. Default is 1 to avoid extra recomputation cost.",
    )
    parser.add_argument(
        "--anchor-top-configs",
        type=int,
        default=2,
        help="Number of top prior configs per tuned subject to use as window anchors.",
    )
    parser.add_argument(
        "--start-deltas",
        nargs="+",
        type=int,
        default=[-128, -64, 0, 64, 128],
        help="Offsets applied to anchor window starts during fine-grained search.",
    )
    parser.add_argument(
        "--stop-deltas",
        nargs="+",
        type=int,
        default=[-64, 0, 64],
        help="Offsets applied to anchor window stops during fine-grained search.",
    )
    parser.add_argument(
        "--top-vote-models",
        type=int,
        default=5,
        help="Top fine-tuned configs per tuned subject used in the weighted vote submission.",
    )
    parser.add_argument(
        "--spec-profile",
        choices=["full", "focused"],
        default="full",
        help="Model-spec search profile. Use 'focused' for a much smaller shortlist of the most promising new variants.",
    )
    return parser.parse_args()


def load_required_csv(csv_path, description):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {description}: {csv_path}")
    return pd.read_csv(csv_path)


def make_cv_splitter(cv_folds, cv_repeats):
    if cv_repeats <= 1:
        return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=42)


def make_run_tag(args):
    payload = repr(
        (
            tuple(args.subjects),
            args.sample_rate_hz,
            args.cv_folds,
            args.cv_repeats,
            args.anchor_top_configs,
            tuple(args.start_deltas),
            tuple(args.stop_deltas),
            args.spec_profile,
        )
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def checkpoint_path(args):
    return os.path.join(args.submissions_dir, f"mne_pyriemann_finetune_resume_{make_run_tag(args)}.csv")


def log_path(args):
    return os.path.join(args.submissions_dir, f"mne_pyriemann_finetune_resume_{make_run_tag(args)}.log")


def setup_logging(log_file_path):
    logger = logging.getLogger("mne_pyriemann_finetune")
    logger.setLevel(logging.INFO)
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


def result_key(subject, model, start, stop):
    return (subject, model, int(start), int(stop))


def append_checkpoint_row(csv_path, row):
    row_df = pd.DataFrame([row])
    write_header = not os.path.exists(csv_path)
    row_df.to_csv(csv_path, mode="a", header=write_header, index=False)


def dedupe_results(df):
    if df.empty:
        return df
    return (
        df.drop_duplicates(subset=["subject", "model", "start", "stop"], keep="last")
        .reset_index(drop=True)
    )


def candidate_specs(subject, model_specs, window_specs):
    return [
        {
            "subject": subject,
            "model_spec": model_spec,
            "window_spec": window_spec,
            "key": result_key(subject, model_spec["name"], window_spec["start"], window_spec["stop"]),
        }
        for window_spec in window_specs
        for model_spec in model_specs
    ]


def prepare_result_rows(df, source):
    prepared = df.copy()
    if "family" not in prepared.columns:
        prepared["family"] = prepared["model"].map(family_from_model_name)
    if "anchor_model" not in prepared.columns:
        prepared["anchor_model"] = prepared["model"]
    prepared["source"] = source
    return prepared


def family_from_model_name(model_name):
    if "fbcsp" in model_name:
        return "fbcsp"
    if "riemann" in model_name or "fgmdm" in model_name or "mdm" in model_name:
        return "riemann"
    if "csp" in model_name:
        return "csp"
    return "other"


def band_sets():
    return {
        "broad": ((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)),
        "dense": ((6.0, 10.0), (8.0, 12.0), (10.0, 14.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0), (28.0, 32.0)),
        "focus": ((8.0, 12.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0), (28.0, 32.0)),
    }


def dedupe_specs(model_specs):
    unique_specs = []
    seen = set()
    for spec in model_specs:
        if spec["name"] in seen:
            continue
        seen.add(spec["name"])
        unique_specs.append(spec)
    return unique_specs


def subject_model_specs(prior_model_names, profile="full"):
    bands = band_sets()
    families = {family_from_model_name(model_name) for model_name in prior_model_names}
    model_specs = []

    if "fbcsp" in families or not families:
        model_specs.extend([
            {
                "name": "fbcsp_broad_c6_k16_lda",
                "family": "fbcsp",
                "bands": bands["broad"],
                "n_components": 6,
                "select_k": 16,
                "classifier": "lda",
            },
            {
                "name": "fbcsp_dense_c4_k16_lda",
                "family": "fbcsp",
                "bands": bands["dense"],
                "n_components": 4,
                "select_k": 16,
                "classifier": "lda",
            },
            {
                "name": "fbcsp_dense_c6_k20_lda",
                "family": "fbcsp",
                "bands": bands["dense"],
                "n_components": 6,
                "select_k": 20,
                "classifier": "lda",
            },
            {
                "name": "fbcsp_focus_c6_k12_lr",
                "family": "fbcsp",
                "bands": bands["focus"],
                "n_components": 6,
                "select_k": 12,
                "classifier": "lr",
                "C": 1.0,
            },
        ])

    if "riemann" in families or not families:
        model_specs.extend([
            {
                "name": "riemann_ts_6_35_lwf_riemann",
                "family": "riemann_ts",
                "low_cut_hz": 6.0,
                "high_cut_hz": 35.0,
                "estimator": "lwf",
                "metric": "riemann",
                "C": 1.0,
            },
            {
                "name": "riemann_ts_8_30_oas_logeuclid",
                "family": "riemann_ts",
                "low_cut_hz": 8.0,
                "high_cut_hz": 30.0,
                "estimator": "oas",
                "metric": "logeuclid",
                "C": 1.0,
            },
            {
                "name": "riemann_fgmdm_8_30_lwf",
                "family": "riemann_fgmdm",
                "low_cut_hz": 8.0,
                "high_cut_hz": 30.0,
                "estimator": "lwf",
                "metric": "riemann",
            },
        ])
        model_specs.extend([
            {
                "name": "fbriemann_broad_oas_riemann",
                "family": "fbriemann_ts",
                "bands": bands["broad"],
                "estimator": "oas",
                "metric": "riemann",
                "C": 1.0,
            },
            {
                "name": "fbriemann_dense_oas_riemann",
                "family": "fbriemann_ts",
                "bands": bands["dense"],
                "estimator": "oas",
                "metric": "riemann",
                "C": 1.0,
            },
            {
                "name": "fbriemann_focus_lwf_riemann",
                "family": "fbriemann_ts",
                "bands": bands["focus"],
                "estimator": "lwf",
                "metric": "riemann",
                "C": 1.0,
            },
        ])

    if "csp" in families or not families:
        model_specs.extend([
            {
                "name": "csp_6_35_c6_lda",
                "family": "csp",
                "low_cut_hz": 6.0,
                "high_cut_hz": 35.0,
                "n_components": 6,
                "classifier": "lda",
            },
            {
                "name": "csp_8_30_c8_lda",
                "family": "csp",
                "low_cut_hz": 8.0,
                "high_cut_hz": 30.0,
                "n_components": 8,
                "classifier": "lda",
            },
        ])

    model_specs = dedupe_specs(model_specs)

    if profile == "focused":
        focused_names = {
            "fbcsp_broad_c6_k16_lda",
            "fbcsp_dense_c6_k20_lda",
            "riemann_ts_8_30_oas_logeuclid",
            "riemann_fgmdm_8_30_lwf",
            "csp_8_30_c8_lda",
        }
        model_specs = [spec for spec in model_specs if spec["name"] in focused_names]

    return model_specs


def build_finetune_model(model_spec, sample_rate_hz, start, stop):
    prefix = [("crop", TemporalCropper(start=start, stop=stop))]

    if model_spec["family"] == "fbcsp":
        steps = prefix + [
            (
                "fbcsp",
                FilterBankCSPTransformer(
                    sample_rate_hz=sample_rate_hz,
                    bands=model_spec["bands"],
                    filter_order=4,
                    n_components=model_spec["n_components"],
                    reg="oas",
                    cov_est="epoch",
                ),
            ),
            ("select", SelectKBest(score_func=f_classif, k=model_spec["select_k"])),
        ]
        if model_spec["classifier"] == "lda":
            steps.append(("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")))
        else:
            steps.extend([
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=model_spec.get("C", 1.0), max_iter=4000)),
            ])
        return Pipeline(steps)

    if model_spec["family"] == "csp":
        steps = prefix + [
            (
                "bandpass",
                BandpassFilter(
                    sample_rate_hz=sample_rate_hz,
                    low_cut_hz=model_spec["low_cut_hz"],
                    high_cut_hz=model_spec["high_cut_hz"],
                    order=4,
                ),
            ),
            (
                "csp",
                CSP(
                    n_components=model_spec["n_components"],
                    reg="oas",
                    log=True,
                    cov_est="epoch",
                    norm_trace=False,
                    component_order="mutual_info",
                ),
            ),
            ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
        ]
        return Pipeline(steps)

    if model_spec["family"] == "riemann_ts":
        return Pipeline(
            prefix
            + [
                (
                    "bandpass",
                    BandpassFilter(
                        sample_rate_hz=sample_rate_hz,
                        low_cut_hz=model_spec["low_cut_hz"],
                        high_cut_hz=model_spec["high_cut_hz"],
                        order=4,
                    ),
                ),
                ("cov", Covariances(estimator=model_spec["estimator"])),
                ("tangent", TangentSpace(metric=model_spec["metric"])),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=model_spec.get("C", 1.0), max_iter=4000)),
            ]
        )

    if model_spec["family"] == "riemann_fgmdm":
        return Pipeline(
            prefix
            + [
                (
                    "bandpass",
                    BandpassFilter(
                        sample_rate_hz=sample_rate_hz,
                        low_cut_hz=model_spec["low_cut_hz"],
                        high_cut_hz=model_spec["high_cut_hz"],
                        order=4,
                    ),
                ),
                ("cov", Covariances(estimator=model_spec["estimator"])),
                ("clf", FgMDM(metric=model_spec["metric"])),
            ]
        )

    if model_spec["family"] == "fbriemann_ts":
        return Pipeline(
            prefix
            + [
                (
                    "fb_tangent",
                    FilterBankTangentSpaceTransformer(
                        sample_rate_hz=sample_rate_hz,
                        bands=model_spec["bands"],
                        filter_order=4,
                        estimator=model_spec["estimator"],
                        metric=model_spec["metric"],
                    ),
                ),
                ("scale", StandardScaler()),
                ("clf", LogisticRegression(C=model_spec.get("C", 1.0), max_iter=4000)),
            ]
        )

    raise ValueError(f"Unsupported fine-tune family: {model_spec['family']}")


def fine_window_specs(anchor_rows, n_times, start_deltas, stop_deltas, min_length=384):
    window_specs = []
    for row in anchor_rows.itertuples(index=False):
        for start_delta in start_deltas:
            for stop_delta in stop_deltas:
                start = int(row.start) + start_delta
                stop = int(row.stop) + stop_delta
                start = max(0, start)
                stop = min(n_times, stop)
                if stop - start < min_length:
                    continue
                window_specs.append({
                    "name": f"samples_{start}_{stop}",
                    "start": start,
                    "stop": stop,
                    "anchor_model": row.model,
                })

    if not window_specs:
        window_specs.append({"name": f"samples_0_{n_times}", "start": 0, "stop": n_times, "anchor_model": "fallback"})

    unique_windows = []
    seen = set()
    for window_spec in window_specs:
        key = (window_spec["start"], window_spec["stop"])
        if key in seen:
            continue
        seen.add(key)
        unique_windows.append(window_spec)
    return unique_windows


def subject_anchor_rows(subject, base_results, base_best, top_n):
    if base_results is not None:
        subject_rows = base_results[base_results["subject"] == subject]
        if not subject_rows.empty:
            return (
                subject_rows.sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
                .head(top_n)
                .reset_index(drop=True)
            )
    subject_rows = base_best[base_best["subject"] == subject]
    if subject_rows.empty:
        raise ValueError(f"No prior config found for subject {subject}")
    return subject_rows.reset_index(drop=True)


def evaluate_subject(
    subject,
    X_train,
    y_train,
    sample_rate_hz,
    splitter,
    model_specs,
    window_specs,
    resume_path,
    completed_keys,
    logger,
    progress_state,
):
    rows = []
    skipped = 0
    subject_candidates = candidate_specs(subject, model_specs, window_specs)
    subject_total = len(subject_candidates)

    for subject_index, candidate in enumerate(subject_candidates, start=1):
        model_spec = candidate["model_spec"]
        window_spec = candidate["window_spec"]
        key = candidate["key"]

        progress_state["processed"] += 1
        global_prefix = f"[global {progress_state['processed']}/{progress_state['total']}]"
        subject_prefix = f"[subject {subject} {subject_index}/{subject_total}]"
        config_label = f"{model_spec['name']} @ {window_spec['name']}"

        if key in completed_keys:
            skipped += 1
            progress_state["skipped"] += 1
            logger.info("%s %s SKIP already computed %s", global_prefix, subject_prefix, config_label)
            continue

        logger.info("%s %s START %s", global_prefix, subject_prefix, config_label)

        fold_scores = []
        split_iterator = list(splitter.split(X_train, y_train))
        for fold_index, (train_indices, valid_indices) in enumerate(split_iterator, start=1):
            logger.info(
                "%s %s FOLD %d/%d %s",
                global_prefix,
                subject_prefix,
                fold_index,
                len(split_iterator),
                config_label,
            )
            model = build_finetune_model(
                model_spec=model_spec,
                sample_rate_hz=sample_rate_hz,
                start=window_spec["start"],
                stop=window_spec["stop"],
            )
            model.fit(X_train[train_indices], y_train[train_indices])
            predictions = model.predict(X_train[valid_indices])
            fold_score = accuracy_score(y_train[valid_indices], predictions)
            fold_scores.append(fold_score)
            logger.info(
                "%s %s FOLD %d/%d SCORE %.4f %s",
                global_prefix,
                subject_prefix,
                fold_index,
                len(split_iterator),
                fold_score,
                config_label,
            )

        row = {
            "subject": subject,
            "model": model_spec["name"],
            "family": model_spec["family"],
            "window": window_spec["name"],
            "start": window_spec["start"],
            "stop": window_spec["stop"],
            "anchor_model": window_spec["anchor_model"],
            "mean_accuracy": float(np.mean(fold_scores)),
            "std_accuracy": float(np.std(fold_scores)),
            "fold_scores": repr(fold_scores),
        }
        rows.append(row)
        append_checkpoint_row(resume_path, row)
        completed_keys.add(key)
        progress_state["completed"] += 1
        logger.info(
            "%s %s DONE %s mean=%.4f std=%.4f checkpoint=%s",
            global_prefix,
            subject_prefix,
            config_label,
            row["mean_accuracy"],
            row["std_accuracy"],
            os.path.basename(resume_path),
        )
    return pd.DataFrame(rows), skipped


def predict_from_finetune_row(row, data_dir, sample_rate_hz):
    X_train, y_train, X_test = load_subject_data(data_dir, row.subject)
    model_specs = subject_model_specs([row.model], profile="full")
    model_spec = next(spec for spec in model_specs if spec["name"] == row.model)
    model = build_finetune_model(
        model_spec=model_spec,
        sample_rate_hz=sample_rate_hz,
        start=int(row.start),
        stop=int(row.stop),
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def predict_from_base_row(row, data_dir, sample_rate_hz):
    X_train, y_train, X_test = load_subject_data(data_dir, row.subject)
    model = build_base_model(row.model, sample_rate_hz=sample_rate_hz, start=int(row.start), stop=int(row.stop))
    model.fit(X_train, y_train)
    return model.predict(X_test)


def weighted_vote(prediction_rows, weights):
    labels = sorted({label for predictions in prediction_rows for label in predictions.tolist()})
    label_to_index = {label: index for index, label in enumerate(labels)}
    vote_matrix = np.zeros((prediction_rows[0].shape[0], len(labels)), dtype=float)

    for predictions, weight in zip(prediction_rows, weights):
        for index, label in enumerate(predictions):
            vote_matrix[index, label_to_index[label]] += weight

    return np.asarray([labels[index] for index in vote_matrix.argmax(axis=1)])


def main():
    args = parse_args()
    os.makedirs(args.submissions_dir, exist_ok=True)
    resume_path = checkpoint_path(args)
    logger = setup_logging(log_path(args))
    aggregate_results_path = os.path.join(args.submissions_dir, "mne_pyriemann_finetune_cv_results.csv")

    tuned_subjects = list(dict.fromkeys(args.subjects))
    base_best = load_required_csv(args.base_best_path, "prior best-config CSV")
    base_results = None
    if os.path.exists(args.base_results_path):
        base_results = pd.read_csv(args.base_results_path)
    base_best_prepared = prepare_result_rows(base_best, "base")
    base_vote_prepared = prepare_result_rows(base_results, "base") if base_results is not None else base_best_prepared.copy()
    existing_aggregate_results = pd.read_csv(aggregate_results_path) if os.path.exists(aggregate_results_path) else pd.DataFrame()
    existing_aggregate_results = dedupe_results(existing_aggregate_results)
    resume_results = pd.read_csv(resume_path) if os.path.exists(resume_path) else pd.DataFrame()
    resume_results = dedupe_results(resume_results)
    completed_keys = set()
    if not resume_results.empty:
        completed_keys = {
            result_key(row.subject, row.model, row.start, row.stop)
            for row in resume_results.itertuples(index=False)
        }
        logger.info("Resuming from %s with %d completed configs.", resume_path, len(completed_keys))
    else:
        logger.info("Checkpoint file: %s", resume_path)
    logger.info("Live log file: %s", log_path(args))

    splitter = make_cv_splitter(args.cv_folds, args.cv_repeats)

    subject_plans = []
    total_candidates = 0
    total_remaining = 0
    for subject in tuned_subjects:
        X_train, y_train, _ = load_subject_data(args.data_dir, subject)
        anchor_rows = subject_anchor_rows(subject, base_results, base_best, args.anchor_top_configs)
        window_specs = fine_window_specs(
            anchor_rows=anchor_rows,
            n_times=X_train.shape[-1],
            start_deltas=args.start_deltas,
            stop_deltas=args.stop_deltas,
        )
        model_specs = subject_model_specs(anchor_rows["model"].tolist(), profile=args.spec_profile)
        candidates = candidate_specs(subject, model_specs, window_specs)
        remaining = sum(1 for candidate in candidates if candidate["key"] not in completed_keys)
        total_candidates += len(candidates)
        total_remaining += remaining
        subject_plans.append(
            {
                "subject": subject,
                "X_train": X_train,
                "y_train": y_train,
                "anchor_rows": anchor_rows,
                "window_specs": window_specs,
                "model_specs": model_specs,
                "candidate_total": len(candidates),
                "remaining": remaining,
            }
        )

    logger.info(
        "Planned configs: total=%d remaining=%d skipped_already=%d",
        total_candidates,
        total_remaining,
        total_candidates - total_remaining,
    )
    for plan in subject_plans:
        logger.info(
            "Subject %s plan: anchors=%s windows=%d model_specs=%d total=%d remaining=%d",
            plan["subject"],
            plan["anchor_rows"][["model", "window"]].to_dict("records"),
            len(plan["window_specs"]),
            len(plan["model_specs"]),
            plan["candidate_total"],
            plan["remaining"],
        )

    progress_state = {"processed": 0, "completed": 0, "skipped": 0, "total": total_candidates}

    for plan in subject_plans:
        logger.info(
            "Evaluating subject %s with %d windows, %d model specs, %d remaining configs.",
            plan["subject"],
            len(plan["window_specs"]),
            len(plan["model_specs"]),
            plan["remaining"],
        )
        _, skipped = evaluate_subject(
            subject=plan["subject"],
            X_train=plan["X_train"],
            y_train=plan["y_train"],
            sample_rate_hz=args.sample_rate_hz,
            splitter=splitter,
            model_specs=plan["model_specs"],
            window_specs=plan["window_specs"],
            resume_path=resume_path,
            completed_keys=completed_keys,
            logger=logger,
            progress_state=progress_state,
        )
        if skipped:
            logger.info("Skipped %d already-computed configs for subject %s.", skipped, plan["subject"])

    current_run_results_df = pd.read_csv(resume_path) if os.path.exists(resume_path) else pd.DataFrame()
    current_run_results_df = dedupe_results(current_run_results_df)
    preserved_results_df = existing_aggregate_results.copy()
    if not preserved_results_df.empty:
        preserved_results_df = preserved_results_df[~preserved_results_df["subject"].isin(tuned_subjects)]
    tuned_results_df = dedupe_results(pd.concat([preserved_results_df, current_run_results_df], ignore_index=True))
    tuned_results_path = aggregate_results_path
    tuned_results_df.to_csv(tuned_results_path, index=False)
    logger.info(
        "Search complete. processed=%d completed=%d skipped=%d saved_results=%s (preserved=%d current_run=%d total=%d)",
        progress_state["processed"],
        progress_state["completed"],
        progress_state["skipped"],
        tuned_results_path,
        len(preserved_results_df),
        len(current_run_results_df),
        len(tuned_results_df),
    )

    tuned_prepared_df = pd.DataFrame()
    tuned_best = pd.DataFrame()
    if not tuned_results_df.empty:
        tuned_prepared_df = prepare_result_rows(tuned_results_df, "fine_tuned")
        tuned_best = (
            tuned_prepared_df.sort_values(["subject", "mean_accuracy", "std_accuracy"], ascending=[True, False, True])
            .groupby("subject", as_index=False)
            .first()
        )

    final_best_rows = []
    for subject in SUBJECTS:
        candidates = [base_best_prepared[base_best_prepared["subject"] == subject]]
        if not tuned_best.empty and not tuned_best[tuned_best["subject"] == subject].empty:
            candidates.append(tuned_best[tuned_best["subject"] == subject])
        subject_best = (
            pd.concat(candidates, ignore_index=True)
            .sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
            .iloc[0]
        )
        final_best_rows.append(subject_best)

    final_best = pd.DataFrame(final_best_rows)
    final_best_path = os.path.join(args.submissions_dir, "mne_pyriemann_finetune_subject_best_configs.csv")
    final_best.to_csv(final_best_path, index=False)

    if not tuned_prepared_df.empty:
        tuned_summary = (
            tuned_prepared_df.groupby(["subject", "model"], as_index=False)
            .agg(mean_accuracy=("mean_accuracy", "max"), std_accuracy=("mean_accuracy", "std"))
            .sort_values(["subject", "mean_accuracy"], ascending=[True, False])
        )
        tuned_summary.to_csv(
            os.path.join(args.submissions_dir, "mne_pyriemann_finetune_model_summary.csv"),
            index=False,
        )

    best_predictions = {}
    for row in final_best.itertuples(index=False):
        if row.source == "fine_tuned":
            best_predictions[row.subject] = predict_from_finetune_row(row, args.data_dir, args.sample_rate_hz)
        else:
            best_predictions[row.subject] = predict_from_base_row(row, args.data_dir, args.sample_rate_hz)

    best_zip_path = write_submission(
        "mne_pyriemann_finetune_subject_best",
        best_predictions,
        submissions_dir=args.submissions_dir,
    )

    vote_rows_per_subject = []
    for subject in SUBJECTS:
        candidates = [base_vote_prepared[base_vote_prepared["subject"] == subject]]
        if not tuned_prepared_df.empty:
            subject_tuned_rows = tuned_prepared_df[tuned_prepared_df["subject"] == subject]
            if not subject_tuned_rows.empty:
                candidates.append(subject_tuned_rows)
        subject_vote_rows = (
            pd.concat(candidates, ignore_index=True)
            .sort_values(["mean_accuracy", "std_accuracy"], ascending=[False, True])
            .head(args.top_vote_models)
        )
        vote_rows_per_subject.append(subject_vote_rows)

    vote_rows = pd.concat(vote_rows_per_subject, ignore_index=True)
    vote_rows_path = os.path.join(args.submissions_dir, "mne_pyriemann_finetune_subject_top_vote_configs.csv")
    vote_rows.to_csv(vote_rows_path, index=False)

    vote_predictions = {}
    for subject in SUBJECTS:
        subject_vote_rows = vote_rows[vote_rows["subject"] == subject]
        prediction_rows = []
        weights = []
        for row in subject_vote_rows.itertuples(index=False):
            if getattr(row, "source", "fine_tuned") == "base":
                prediction_rows.append(predict_from_base_row(row, args.data_dir, args.sample_rate_hz))
            else:
                prediction_rows.append(predict_from_finetune_row(row, args.data_dir, args.sample_rate_hz))
            weights.append(max(row.mean_accuracy, 1e-6))
        vote_predictions[subject] = weighted_vote(prediction_rows, weights)

    vote_zip_path = write_submission(
        f"mne_pyriemann_finetune_subject_top{args.top_vote_models}_vote",
        vote_predictions,
        submissions_dir=args.submissions_dir,
    )

    logger.info("Generated submission: %s", best_zip_path)
    logger.info("Generated submission: %s", vote_zip_path)
    logger.info("Final best configs:\n%s", final_best[["subject", "model", "window", "mean_accuracy", "std_accuracy", "source"]].to_string(index=False))


if __name__ == "__main__":
    main()
