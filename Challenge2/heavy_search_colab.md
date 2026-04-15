---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3
    name: python3
---

<!-- #region id="title" -->
# Heavy MI-EEG Search (Colab)

This notebook runs a **heavy subject-wise search** for the motor imagery challenge with:
- dense temporal window search
- 4-fold CV
- FBCSP / CSP / Riemannian models
- diversity-aware top-k ensemble selection (disagreement + double-fault)
- cache + resume support
- final submission zip generation

<!-- #endregion -->

```python id="deps"
!pip -q install --upgrade pip
!pip -q install "decorator<5" "jedi>=0.18" numpy pandas scikit-learn scipy mne pyriemann
import decorator, jedi
print('decorator', decorator.__version__, 'jedi', jedi.__version__)
```

```python id="imports"
from pathlib import Path
import hashlib
import itertools
import logging
import os
import shutil
import subprocess
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

mne.set_log_level("ERROR")

BASE_DIR = Path("/content/mi_heavy_search")
DATA_DIR = BASE_DIR / "data"
SUBMISSIONS_DIR = BASE_DIR / "submissions"
CACHE_DIR = SUBMISSIONS_DIR / "heavy_oof_cache"

for path in [BASE_DIR, DATA_DIR, SUBMISSIONS_DIR, CACHE_DIR]:
    path.mkdir(parents=True, exist_ok=True)

SUBJECTS = ["A", "B", "C", "D", "E", "F"]
print("Working directory:", BASE_DIR)
```

```python id="download-data"
DATA_URL = "https://cloud.univ-grenoble-alpes.fr/public.php/dav/files/XABwo9ygAdqorRt/?accept=zip"
ZIP_PATH = BASE_DIR / "challenge_data.zip"
RAW_EXTRACT_DIR = BASE_DIR / "_raw_zip"

def expected_files():
    files = []
    for subject in SUBJECTS:
        files.extend([
            f"subject_{subject}_X_train.npy",
            f"subject_{subject}_y_train.npy",
            f"subject_{subject}_X_test.npy",
        ])
    return files

def data_ready():
    return all((DATA_DIR / name).exists() for name in expected_files())

if not data_ready():
    if not ZIP_PATH.exists():
        print("Downloading data zip...")
        subprocess.run(["bash", "-lc", f"curl -L '{DATA_URL}' -o '{ZIP_PATH}'"], check=True)
    else:
        print("Zip already downloaded:", ZIP_PATH)

    if RAW_EXTRACT_DIR.exists():
        shutil.rmtree(RAW_EXTRACT_DIR)
    RAW_EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    print("Extracting zip...")
    with zipfile.ZipFile(ZIP_PATH, "r") as archive:
        archive.extractall(RAW_EXTRACT_DIR)

    npy_files = list(RAW_EXTRACT_DIR.rglob("subject_*_*.npy"))
    if not npy_files:
        raise RuntimeError("No expected .npy files found after extraction.")

    for npy_path in npy_files:
        shutil.copy2(npy_path, DATA_DIR / npy_path.name)

    missing = [name for name in expected_files() if not (DATA_DIR / name).exists()]
    if missing:
        raise RuntimeError(f"Missing expected data files after extraction: {missing}")

print("Data ready in:", DATA_DIR)
print("Sample files:")
for name in sorted(expected_files())[:6]:
    print(" -", name)
```

```python id="models"
def load_subject_data(data_dir, subject):
    X_train = np.load(data_dir / f"subject_{subject}_X_train.npy")
    y_train = np.load(data_dir / f"subject_{subject}_y_train.npy")
    X_test = np.load(data_dir / f"subject_{subject}_X_test.npy")
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
        ("fbcsp", FilterBankCSPTransformer(sample_rate_hz=sample_rate_hz, bands=bands, filter_order=4, n_components=n_components, reg="oas")),
        ("select", SelectKBest(score_func=f_classif, k=select_k)),
        ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
    ])

def build_model(model_name, sample_rate_hz, start, stop):
    if model_name == "mne_fbcsp_lda":
        return _build_fbcsp_model(sample_rate_hz, start, stop, ((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)), 4, 12)
    if model_name == "fbcsp_broad_c6_k16_lda":
        return _build_fbcsp_model(sample_rate_hz, start, stop, ((4.0, 8.0), (8.0, 12.0), (12.0, 16.0), (16.0, 24.0), (24.0, 32.0), (32.0, 40.0)), 6, 16)
    if model_name == "fbcsp_dense_c6_k20_lda":
        return _build_fbcsp_model(sample_rate_hz, start, stop, ((6.0, 10.0), (8.0, 12.0), (10.0, 14.0), (12.0, 16.0), (16.0, 20.0), (20.0, 24.0), (24.0, 28.0), (28.0, 32.0)), 6, 20)
    if model_name == "mne_csp_8_30_lda":
        return Pipeline([
            ("crop", TemporalCropper(start=start, stop=stop)),
            ("bandpass", BandpassFilter(sample_rate_hz=sample_rate_hz, low_cut_hz=8.0, high_cut_hz=30.0, order=4)),
            ("csp", CSP(n_components=6, reg="oas", log=True, cov_est="epoch", norm_trace=False, component_order="mutual_info")),
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
```

```python id="search-helpers"
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

def make_window_grid(n_times, config):
    starts = list(range(config['start_min'], config['start_max'] + 1, config['start_step']))
    stops = list(range(config['stop_min'], config['stop_max'] + 1, config['stop_step']))
    if n_times not in stops:
        stops.append(n_times)

    windows = []
    for start in starts:
        for stop in stops:
            if stop > n_times or stop <= start:
                continue
            if stop - start < config['min_window_len']:
                continue
            windows.append((start, stop))

    windows.append((0, n_times))
    return sorted(set(windows))

def candidate_key(subject, model_name, start, stop):
    return (subject, model_name, int(start), int(stop))

def oof_cache_path(cache_dir, subject, model_name, start, stop):
    key = f"{subject}__{model_name}__{int(start)}__{int(stop)}".replace("/", "_")
    return cache_dir / f"{key}.npy"

def load_results(path):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = ["subject", "model", "family", "start", "stop", "mean_accuracy", "std_accuracy"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {path}")
    df['start'] = df['start'].astype(int)
    df['stop'] = df['stop'].astype(int)
    return df.drop_duplicates(subset=['subject','model','start','stop'], keep='last').reset_index(drop=True)

def append_result_row(path, row):
    path = Path(path)
    row_df = pd.DataFrame([row])
    row_df.to_csv(path, mode='a', header=not path.exists(), index=False)

def top_candidate_pool(subject_results, config):
    parts = []
    for model_name in subject_results['model'].unique():
        parts.append(
            subject_results[subject_results['model'] == model_name]
            .sort_values(['mean_accuracy', 'std_accuracy'], ascending=[False, True])
            .head(config['top_per_model'])
        )
    if not parts:
        return pd.DataFrame()
    pool = pd.concat(parts, ignore_index=True)
    pool = pool.drop_duplicates(subset=['subject','model','start','stop'], keep='first')
    pool = pool.sort_values(['mean_accuracy','std_accuracy'], ascending=[False, True])
    return pool.head(config['max_candidates_per_subject']).reset_index(drop=True)

def disagreement(pred_a, pred_b):
    return float(np.mean(pred_a != pred_b))

def double_fault(pred_a, pred_b, y_true):
    return float(np.mean((pred_a != y_true) & (pred_b != y_true)))

def combo_valid(combo_rows, config):
    model_counts = {}
    family_counts = {}
    for row in combo_rows:
        model_counts[row.model] = model_counts.get(row.model, 0) + 1
        family_counts[row.family] = family_counts.get(row.family, 0) + 1
        if model_counts[row.model] > config['max_same_model']:
            return False
        if family_counts[row.family] > config['max_same_family']:
            return False

    for left, right in itertools.combinations(combo_rows, 2):
        if left.family != right.family:
            continue
        if abs(int(left.start) - int(right.start)) < config['min_window_shift'] and abs(int(left.stop) - int(right.stop)) < config['min_window_shift']:
            return False
    return True

def combo_metrics(combo_rows, oof_lookup, y_true, config):
    oof_rows = [oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] for row in combo_rows]
    weights = [float(row.mean_accuracy) for row in combo_rows]
    ensemble_pred = weighted_vote(oof_rows, weights)
    ensemble_acc = accuracy_score(y_true, ensemble_pred)

    if len(combo_rows) == 1:
        avg_disagreement = 0.0
        avg_double_fault = 0.0
    else:
        dis = []
        dflt = []
        for i in range(len(combo_rows)):
            for j in range(i + 1, len(combo_rows)):
                a = oof_rows[i]
                b = oof_rows[j]
                dis.append(disagreement(a, b))
                dflt.append(double_fault(a, b, y_true))
        avg_disagreement = float(np.mean(dis))
        avg_double_fault = float(np.mean(dflt))

    score = ensemble_acc + config['diversity_weight'] * avg_disagreement - config['double_fault_weight'] * avg_double_fault
    return {
        'ensemble_accuracy': float(ensemble_acc),
        'avg_disagreement': avg_disagreement,
        'avg_double_fault': avg_double_fault,
        'score': float(score),
    }

def select_diverse_combo(subject_pool, oof_lookup, y_true, config):
    rows = list(subject_pool.itertuples(index=False))
    best_combo = None
    best_metrics = None

    sizes = list(range(config['min_ensemble_size'], config['max_ensemble_size'] + 1))
    if config['min_ensemble_size'] <= 1:
        sizes = [1] + sizes

    for size in sizes:
        for combo_idx in itertools.combinations(range(len(rows)), size):
            combo = [rows[i] for i in combo_idx]
            if not combo_valid(combo, config):
                continue

            metrics = combo_metrics(combo, oof_lookup, y_true, config)
            if size > 1 and metrics['avg_disagreement'] < config['min_disagreement']:
                continue

            if best_combo is None:
                best_combo = combo
                best_metrics = metrics
                continue

            left = (metrics['score'], metrics['ensemble_accuracy'], metrics['avg_disagreement'], -metrics['avg_double_fault'])
            right = (best_metrics['score'], best_metrics['ensemble_accuracy'], best_metrics['avg_disagreement'], -best_metrics['avg_double_fault'])
            if left > right:
                best_combo = combo
                best_metrics = metrics

    if best_combo is None:
        fallback = [rows[0]]
        best_combo = fallback
        best_metrics = combo_metrics(fallback, oof_lookup, y_true, config)

    return best_combo, best_metrics

def write_models_md(path, selected_rows):
    lines = []
    lines.append('# Heavy Search Subject Models')
    lines.append('')
    lines.append('Top-k weighted vote selected with diversity constraints (disagreement + double-fault aware selection).')
    lines.append('')
    lines.append('| subject | rank | model | start | stop | weight | note |')
    lines.append('| --- | --- | --- | --- | --- | --- | --- |')
    for subject in SUBJECTS:
        rows = [r for r in selected_rows if r['subject'] == subject]
        rows = sorted(rows, key=lambda x: x['rank'])
        for row in rows:
            lines.append(f"| {row['subject']} | {row['rank']} | {row['model']} | {row['start']} | {row['stop']} | {row['weight']:.12f} | heavy-diverse |")
    Path(path).write_text('\n'.join(lines) + '\n', encoding='utf-8')

def make_submission_zip(pred_dir, zip_path):
    pred_dir = Path(pred_dir)
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for subject in SUBJECTS:
            csv_path = pred_dir / f"subject_{subject}_y_pred.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing prediction file: {csv_path}")
            zf.write(csv_path, arcname=csv_path.name)

def estimate_fit_counts(config):
    windows = make_window_grid(1537, config)
    candidates = len(SUBJECTS) * len(MODEL_SPECS) * len(windows)
    return {'windows': len(windows), 'candidates': candidates, 'cv_fits': candidates * config['cv_folds']}

def run_heavy_search(config):
    submissions_dir = Path(config['submissions_dir'])
    cache_dir = Path(config['cache_dir'])
    results_csv = Path(config['results_csv'])
    selection_csv = Path(config['selection_csv'])
    models_csv = Path(config['models_csv'])
    models_md = Path(config['models_md'])

    submissions_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    tag_payload = repr(tuple(sorted(config.items())))
    run_tag = hashlib.sha1(tag_payload.encode('utf-8')).hexdigest()[:12]
    log_file = Path(config.get('log_file') or (submissions_dir / f"heavy_search_{run_tag}.log"))

    logger = logging.getLogger(f"heavy_search_{run_tag}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info('Run tag: %s', run_tag)
    logger.info('Log file: %s', log_file)

    X_a, _, _ = load_subject_data(Path(config['data_dir']), 'A')
    windows = make_window_grid(X_a.shape[-1], config)

    total_candidates = len(SUBJECTS) * len(MODEL_SPECS) * len(windows)
    total_cv_fits = total_candidates * config['cv_folds']

    existing = load_results(results_csv)
    done_keys = set()
    if not existing.empty:
        done_keys = {
            candidate_key(row.subject, row.model, row.start, row.stop)
            for row in existing.itertuples(index=False)
            if oof_cache_path(cache_dir, row.subject, row.model, row.start, row.stop).exists()
        }

    remaining_configs = total_candidates - len(done_keys)
    remaining_cv_fits = remaining_configs * config['cv_folds']

    logger.info('Window grid size: %d', len(windows))
    logger.info('Total candidate configs: %d', total_candidates)
    logger.info('Total CV fits (cold run): %d', total_cv_fits)
    logger.info('Cached configs with OOF present: %d', len(done_keys))
    logger.info('Remaining configs: %d', remaining_configs)
    logger.info('Remaining CV fits: %d', remaining_cv_fits)

    splitter = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=42)
    processed = 0

    for subject in SUBJECTS:
        X_train, y_train, _ = load_subject_data(Path(config['data_dir']), subject)
        split_list = list(splitter.split(X_train, y_train))
        logger.info('Searching subject %s (%d train samples)', subject, X_train.shape[0])

        for spec in MODEL_SPECS:
            model_name = spec['model']
            family = spec['family']
            for start, stop in windows:
                key = candidate_key(subject, model_name, start, stop)
                cache_path = oof_cache_path(cache_dir, subject, model_name, start, stop)
                if key in done_keys:
                    logger.info('[cache skip] %s | %s | samples_%d_%d', subject, model_name, start, stop)
                    continue

                processed += 1
                label = f"{subject} | {model_name} | samples_{start}_{stop}"
                logger.info('[search %d/%d] START %s', processed, max(remaining_configs, 1), label)

                oof_pred = np.empty(y_train.shape[0], dtype='<U16')
                fold_scores = []
                for fold_idx, (train_idx, valid_idx) in enumerate(split_list, start=1):
                    logger.info('[search %d/%d] FOLD %d/%d %s', processed, max(remaining_configs, 1), fold_idx, len(split_list), label)
                    model = build_model(model_name=model_name, sample_rate_hz=config['sample_rate_hz'], start=start, stop=stop)
                    model.fit(X_train[train_idx], y_train[train_idx])
                    pred = model.predict(X_train[valid_idx])
                    oof_pred[valid_idx] = pred
                    sc = accuracy_score(y_train[valid_idx], pred)
                    fold_scores.append(sc)
                    logger.info('[search %d/%d] FOLD %d/%d SCORE %.4f %s', processed, max(remaining_configs, 1), fold_idx, len(split_list), sc, label)

                np.save(cache_path, oof_pred)
                row = {
                    'subject': subject,
                    'model': model_name,
                    'family': family,
                    'start': int(start),
                    'stop': int(stop),
                    'mean_accuracy': float(np.mean(fold_scores)),
                    'std_accuracy': float(np.std(fold_scores)),
                }
                append_result_row(results_csv, row)
                done_keys.add(key)
                logger.info('[search %d/%d] DONE %s mean=%.4f std=%.4f', processed, max(remaining_configs, 1), label, row['mean_accuracy'], row['std_accuracy'])

    results = load_results(results_csv)
    selection_rows = []
    model_rows = []
    final_predictions = {}

    for subject in SUBJECTS:
        X_train, y_train, X_test = load_subject_data(Path(config['data_dir']), subject)
        subject_results = results[results['subject'] == subject]
        pool = top_candidate_pool(subject_results, config)
        logger.info('Subject %s pool size: %d', subject, len(pool))

        oof_lookup = {}
        for row in pool.itertuples(index=False):
            cache_path = oof_cache_path(cache_dir, row.subject, row.model, row.start, row.stop)
            oof_lookup[candidate_key(row.subject, row.model, row.start, row.stop)] = np.load(cache_path)

        selected_combo, metrics = select_diverse_combo(pool, oof_lookup, y_train, config)
        logger.info('Subject %s selected combo size=%d acc=%.4f disagree=%.4f double_fault=%.4f score=%.4f', subject, len(selected_combo), metrics['ensemble_accuracy'], metrics['avg_disagreement'], metrics['avg_double_fault'], metrics['score'])

        pred_rows = []
        weights = []
        for rank, row in enumerate(selected_combo, start=1):
            model = build_model(model_name=row.model, sample_rate_hz=config['sample_rate_hz'], start=int(row.start), stop=int(row.stop))
            model.fit(X_train, y_train)
            pred_rows.append(model.predict(X_test))
            weights.append(float(row.mean_accuracy))
            model_rows.append({'subject': subject, 'rank': rank, 'model': row.model, 'start': int(row.start), 'stop': int(row.stop), 'weight': float(row.mean_accuracy), 'family': row.family})

        final_predictions[subject] = weighted_vote(pred_rows, weights)
        selection_rows.append({'subject': subject, 'ensemble_size': len(selected_combo), 'ensemble_oof_accuracy': metrics['ensemble_accuracy'], 'avg_disagreement': metrics['avg_disagreement'], 'avg_double_fault': metrics['avg_double_fault'], 'selection_score': metrics['score']})

    selection_df = pd.DataFrame(selection_rows).sort_values('subject')
    selection_df.to_csv(selection_csv, index=False)

    models_df = pd.DataFrame(model_rows)
    models_df.to_csv(models_csv, index=False)
    write_models_md(models_md, model_rows)

    output_dir = submissions_dir / config['output_model_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    for subject in SUBJECTS:
        pd.DataFrame({'y_pred': final_predictions[subject]}).to_csv(output_dir / f"subject_{subject}_y_pred.csv", index=False)

    zip_path = submissions_dir / f"{config['output_model_name']}.zip"
    make_submission_zip(output_dir, zip_path)

    logger.info('Generated submission zip: %s', zip_path)
    logger.info('Selection summary:\n%s', selection_df.to_string(index=False))

    return {'zip_path': zip_path, 'selection_df': selection_df, 'models_df': models_df, 'log_file': log_file, 'results_csv': results_csv, 'models_md': models_md}
```

```python id="config"
SEARCH_CONFIG = {
    'data_dir': DATA_DIR,
    'submissions_dir': SUBMISSIONS_DIR,
    'cache_dir': CACHE_DIR,
    'results_csv': SUBMISSIONS_DIR / 'heavy_search_results.csv',
    'selection_csv': SUBMISSIONS_DIR / 'heavy_search_selection.csv',
    'models_csv': SUBMISSIONS_DIR / 'heavy_subject_models.csv',
    'models_md': SUBMISSIONS_DIR / 'heavy_subject_models.md',
    'output_model_name': 'heavy_top3_diverse_colab',

    'sample_rate_hz': 256.0,
    'cv_folds': 4,

    'start_min': 192,
    'start_max': 960,
    'start_step': 64,
    'stop_min': 1088,
    'stop_max': 1537,
    'stop_step': 64,
    'min_window_len': 384,

    'top_per_model': 8,
    'max_candidates_per_subject': 48,

    'min_ensemble_size': 2,
    'max_ensemble_size': 3,
    'max_same_model': 1,
    'max_same_family': 2,
    'min_window_shift': 96,
    'min_disagreement': 0.04,
    'diversity_weight': 0.03,
    'double_fault_weight': 0.02,
}

estimate = estimate_fit_counts(SEARCH_CONFIG)
print('Window count:', estimate['windows'])
print('Candidate configs:', estimate['candidates'])
print('CV fits (cold run):', estimate['cv_fits'])
```

```python id="run-search"
artifacts = run_heavy_search(SEARCH_CONFIG)
print('Submission zip:', artifacts['zip_path'])
display(artifacts['selection_df'])
display(artifacts['models_df'].sort_values(['subject', 'rank']))
```

```python id="download-zip"
# Optional: download submission zip from Colab
try:
    from google.colab import files
    files.download(str(artifacts['zip_path']))
except Exception as exc:
    print('Not running in Colab or automatic download failed:', exc)
    print('Zip path:', artifacts['zip_path'])
```
