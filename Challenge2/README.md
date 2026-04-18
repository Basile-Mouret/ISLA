## Overview

The final approach is a subject-specific classical EEG pipeline for binary motor imagery classification (`left_hand` vs `right_hand`).

Each subject is handled independently. The search script evaluates a fixed set of model families over many temporal crops, keeps the strongest candidates per family, and then chooses a final 2- or 3-model ensemble with explicit diversity constraints. The final frozen submission is the `top3_diverse` ensemble found by that search.

The two runnable scripts are:

- `search_pipeline.py`: full heavy search with caching, CV, diversity-aware selection, and submission packaging.
- `train_top3_diverse.py`: trains only the exact frozen final ensemble and packages a submission.

Shared code used by both scripts lives in the `utils/` package:

- `utils/models.py`
- `utils/file_manager.py`
- `utils/selection.py`
- `utils/helpers.py`

## Data Flow

For each subject, each candidate model follows the same high-level pattern:

1. Temporal crop: keep samples `start:stop`.
2. Band filtering or filter-bank decomposition.
3. Feature extraction.
4. A simple linear or geometry-based classifier.
5. Subject-level ensemble voting across the selected top models.

Let one trial be `X \in R^{C \times T}`, where `C` is the number of EEG channels and `T` is the number of time samples.

## Model Families

### 1. CSP + LDA

Script name: `mne_csp_8_30_lda`

Pipeline:

1. Crop the trial to `X[:, start:stop]`.
2. Apply a zero-phase 4th-order Butterworth band-pass filter from 8 to 30 Hz.
3. Estimate class-wise covariance structure and fit Common Spatial Patterns (CSP) with:
   - `n_components=6`
   - shrinkage regularization `reg="oas"`
   - epoch-wise covariance estimation
   - log-variance features
4. Train shrinkage LDA (`solver="lsqr", shrinkage="auto"`).

For binary classes with class covariance matrices `Sigma_1` and `Sigma_2`, CSP finds spatial filters `w` that maximize the generalized Rayleigh quotient

```math
\max_w \frac{w^\top \Sigma_1 w}{w^\top \Sigma_2 w}.
```

This yields filters that emphasize variance for one class while suppressing the other. After projection, CSP features are log-variances:

```math
f_j(X) = \log\left(\mathrm{var}(w_j^\top X)\right).
```

LDA then finds a linear separator in that feature space.

### 2. Filter-Bank CSP + LDA

Script names:

- `mne_fbcsp_lda`
- `fbcsp_broad_c6_k16_lda`
- `fbcsp_dense_c6_k20_lda`

These are all FBCSP variants. The idea is to run CSP separately on several frequency bands, concatenate the resulting log-variance features, optionally select the most discriminative ones, and classify with LDA.

For band `b`, let `X_b` be the band-passed trial. CSP produces features

```math
f_{b,j}(X) = \log\left(\mathrm{var}(w_{b,j}^\top X_b)\right).
```

All bandwise features are concatenated into one vector, then `SelectKBest(f_classif)` keeps the top `k` ANOVA-ranked features before LDA.

The exact variants are:

#### `mne_fbcsp_lda`

- Bands: `(4,8)`, `(8,12)`, `(12,16)`, `(16,24)`, `(24,32)`, `(32,40)` Hz
- CSP components per band: `4`
- Selected features: `12`
- Classifier: shrinkage LDA

#### `fbcsp_broad_c6_k16_lda`

- Same 6 broad bands as above
- CSP components per band: `6`
- Selected features: `16`
- Classifier: shrinkage LDA

#### `fbcsp_dense_c6_k20_lda`

- Overlapping denser bands: `(6,10)`, `(8,12)`, `(10,14)`, `(12,16)`, `(16,20)`, `(20,24)`, `(24,28)`, `(28,32)` Hz
- CSP components per band: `6`
- Selected features: `20`
- Classifier: shrinkage LDA

These models worked well because motor imagery information is concentrated in mu and beta rhythms, but the best discriminative sub-band differs by subject.

### 3. Riemannian Tangent Space + Logistic Regression

Script names:

- `riemann_ts_lr_6_35`
- `riemann_ts_lr_8_30`

Pipeline:

1. Crop the trial.
2. Band-pass filter to either 6-35 Hz or 8-30 Hz.
3. Estimate a regularized covariance matrix for each trial using OAS shrinkage.
4. Map covariance matrices from the SPD manifold to the tangent space at a reference mean covariance.
5. Standardize tangent features.
6. Fit logistic regression with `C=1.0`, `max_iter=4000`.

If `C_i \in S_{++}^C` is the trial covariance and `G` is the reference covariance mean, the affine-invariant Riemannian tangent-space mapping is locally

```math
T_i = \log\left(G^{-1/2} C_i G^{-1/2}\right),
```

followed by vectorization of the symmetric matrix. This turns the non-Euclidean covariance geometry into a locally Euclidean feature representation where a linear classifier can be used.

The covariance regularization is shrinkage of the sample covariance `S` toward a scaled identity:

```math
\hat{\Sigma} = (1-\lambda) S + \lambda \frac{\mathrm{tr}(S)}{C} I.
```

The code uses `estimator="oas"` for these tangent-space models.

### 4. Riemannian FgMDM

Script names:

- `riemann_fgmdm_8_30`
- `riemann_fgmdm_8_30_lwf`

Pipeline:

1. Crop the trial.
2. Band-pass filter to 8-30 Hz.
3. Estimate covariance with either:
   - OAS shrinkage for `riemann_fgmdm_8_30`
   - Ledoit-Wolf shrinkage for `riemann_fgmdm_8_30_lwf`
4. Apply FgMDM (`pyriemann.classification.FgMDM`) with the Riemannian metric.

`FgMDM` is a geodesic-filtered version of minimum-distance-to-mean classification. In pyRiemann it is effectively an `FGDA + MDM` pipeline:

1. Project SPD covariance matrices to tangent space.
2. Apply Fisher Geodesic Discriminant Analysis (FGDA), which uses LDA in tangent space to keep the most discriminative directions.
3. Project the filtered points back to the SPD manifold.
4. Compute class centroids on the manifold.
5. Predict by nearest centroid under the Riemannian distance.

The MDM decision rule is

```math
\hat{k} = \arg\min_k d_R(C, M_k),
```

where `M_k` is the class centroid and the affine-invariant Riemannian distance is

```math
d_R(C_1, C_2) = \left\| \log\left(C_1^{-1/2} C_2 C_1^{-1/2}\right) \right\|_F.
```

The `lwf` variant changes only the covariance estimator from OAS to Ledoit-Wolf shrinkage.

## Search Procedure

### Candidate Space

The heavy search is entirely subject-specific. Subject `A` is searched using only subject `A` data, and so on.

The script evaluates every combination of:

- 8 model definitions
- a dense grid of temporal windows
- 4-fold stratified cross-validation

The temporal crop grid is defined in samples at 256 Hz:

- `start` in `192, 256, 320, ..., 960`
- `stop` in `1088, 1152, 1216, ..., 1537`
- minimum crop length `384` samples
- plus one explicit full-trial window `(0, n_times)`

At 256 Hz, this means the search focuses on the motor-imagery portion of the trial while still allowing substantial variation in onset and duration.

### Cross-Validation

Every candidate is scored with:

- `StratifiedKFold(n_splits=4, shuffle=True, random_state=42)`

For each fold:

1. Fit the full candidate pipeline on the training folds.
2. Predict the validation fold.
3. Store the fold accuracy.
4. Store the out-of-fold predictions for every training sample.

The search stores:

- `mean_accuracy`
- `std_accuracy`
- the full out-of-fold prediction vector for each candidate

The cached out-of-fold predictions are critical because the final ensemble selection is based on how candidates behave on the same held-out samples, not only on their individual mean scores.

### Caching and Resume

The search is designed to resume without recomputing finished candidates.

- Per-candidate CV summaries are appended to `heavy_search_results.csv`.
- Per-candidate out-of-fold predictions are saved as `.npy` files in `heavy_oof_cache/`.
- If both the CSV row and OOF cache exist, the candidate is skipped on the next run.

This is why reruns can continue from partial work instead of restarting the full search.

### Candidate Pool Before Ensemble Selection

After the full per-subject search:

1. For each model definition, keep the top `8` windows sorted by descending `mean_accuracy` and ascending `std_accuracy`.
2. Merge those per-model lists.
3. Drop duplicates.
4. Keep the best `48` total candidates for that subject.

This trims the ensemble search to a manageable but still diverse pool.

## Final Ensemble Selection

### Allowed Ensemble Size

The final selection searches all valid combinations of size 2 or 3.

The defaults used in the heavy run were:

- `min_ensemble_size=2`
- `max_ensemble_size=3`

In practice, the selected ensemble for every subject ended up having size 3.

### Constraints

For a candidate combination to be valid:

1. A model definition cannot appear more than once.
   - `max_same_model=1`
2. A model family cannot appear more than twice.
   - `max_same_family=2`
3. If two candidates come from the same family, they cannot be almost the same time window.
   - the combination is rejected when both `|start_a - start_b| < 96` and `|stop_a - stop_b| < 96`
4. Multi-model ensembles must have enough behavioral diversity.
   - `avg_disagreement >= 0.04`

### Diversity Metrics

For two prediction vectors `p_a` and `p_b`, disagreement is

```math
\mathrm{disagreement}(p_a, p_b) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[p_{a,i} \neq p_{b,i}].
```

Double fault is

```math
\mathrm{doublefault}(p_a, p_b, y) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[p_{a,i} \neq y_i \land p_{b,i} \neq y_i].
```

For an ensemble, the script averages these pairwise values across all model pairs.

### Weighted Voting

Given predictions `\hat{y}^{(m)}` from each selected model `m`, the final trial prediction is a weighted majority vote:

```math
\hat{y}_i = \arg\max_{c \in \mathcal{C}} \sum_m w_m \mathbf{1}[\hat{y}^{(m)}_i = c].
```

The weight `w_m` is that model's cross-validated `mean_accuracy`.

### Selection Score

The ensemble selection objective used in the search is

```math
\text{score} = \text{ensemble\_accuracy} + 0.03 \cdot \text{avg\_disagreement} - 0.02 \cdot \text{avg\_double\_fault}.
```

So the search prefers:

- high out-of-fold ensemble accuracy first,
- some disagreement between models,
- low shared failure rate.

Tie-breaking is effectively by the tuple:

```text
(selection score, ensemble accuracy, avg disagreement, -avg double fault)
```

## Frozen Final `top3_diverse` Ensemble

The final model is the exact ensemble below, with the exact search-derived vote weights.

| Subject | Rank | Model | Window (`start:stop`) | Weight |
| --- | --- | --- | --- | --- |
| A | 1 | `fbcsp_broad_c6_k16_lda` | `320:1344` | `0.8785714285714286` |
| A | 2 | `mne_fbcsp_lda` | `384:1088` | `0.8357142857142857` |
| A | 3 | `riemann_fgmdm_8_30_lwf` | `768:1152` | `0.65` |
| B | 1 | `riemann_ts_lr_8_30` | `832:1344` | `0.7928571428571429` |
| B | 2 | `mne_csp_8_30_lda` | `512:1536` | `0.7785714285714287` |
| B | 3 | `fbcsp_broad_c6_k16_lda` | `192:1408` | `0.6785714285714286` |
| C | 1 | `mne_fbcsp_lda` | `256:1280` | `0.9571428571428572` |
| C | 2 | `riemann_ts_lr_6_35` | `832:1536` | `0.9285714285714284` |
| C | 3 | `fbcsp_dense_c6_k20_lda` | `704:1536` | `0.9071428571428573` |
| D | 1 | `fbcsp_dense_c6_k20_lda` | `640:1280` | `0.9428571428571428` |
| D | 2 | `mne_csp_8_30_lda` | `320:1152` | `0.9428571428571428` |
| D | 3 | `fbcsp_broad_c6_k16_lda` | `512:1408` | `0.9142857142857144` |
| E | 1 | `riemann_ts_lr_6_35` | `640:1408` | `0.8785714285714287` |
| E | 2 | `fbcsp_broad_c6_k16_lda` | `320:1216` | `0.8571428571428571` |
| E | 3 | `mne_fbcsp_lda` | `512:1472` | `0.8428571428571429` |
| F | 1 | `mne_fbcsp_lda` | `512:1472` | `0.8571428571428572` |
| F | 2 | `fbcsp_broad_c6_k16_lda` | `256:1537` | `0.7857142857142857` |
| F | 3 | `riemann_ts_lr_6_35` | `640:1216` | `0.75` |

This is exactly what `train_top3_diverse.py` reproduces.

## Why This Worked

The final search converged to a simple pattern:

- some subjects were strongly FBCSP-dominant,
- some benefited from a mix of CSP/FBCSP and Riemannian geometry,
- the best time window was clearly subject-dependent,
- the best single model was often not the best ensemble member,
- diversity mattered, but only mildly; too much overlap hurt, but the best ensemble still relied on strong individual models.

In practice, the search found that weighted and equal-weight top-3 voting behaved very similarly on the public leaderboard, so most of the gain came from choosing the right three models and windows rather than from fine weight tuning.

## Sources

1. H. Ramoser, J. Muller-Gerking, G. Pfurtscheller, "Optimal spatial filtering of single trial EEG during imagined hand movement," IEEE Transactions on Rehabilitation Engineering, 2000.
2. K. K. Ang, Z. Y. Chin, C. Wang, C. Guan, H. Zhang, "Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b," Frontiers in Neuroscience, 2012.
3. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Riemannian geometry applied to BCI classification," LVA/ICA, 2010.
4. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Multiclass Brain-Computer Interface Classification by Riemannian Geometry," IEEE Transactions on Biomedical Engineering, 2012.
5. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Classification of covariance matrices using a Riemannian-based kernel for BCI applications," Neurocomputing, 2013.
6. Y. Chen, A. Wiesel, Y. C. Eldar, A. O. Hero, "Shrinkage algorithms for MMSE covariance estimation," IEEE Transactions on Signal Processing, 2010. This is the OAS shrinkage estimator used in several pipelines.
7. O. Ledoit, M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," Journal of Multivariate Analysis, 2004. This is the shrinkage idea behind the `lwf` covariance variant.
