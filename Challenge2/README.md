## Overview

This repo solves the binary motor imagery task (`left_hand` vs `right_hand`) with a subject-specific classical EEG pipeline. Each subject is searched independently. The heavy search scores a fixed set of model families over many time windows, keeps the strongest candidates, and selects a small diverse ensemble. The frozen final submission is the `top3_diverse` ensemble found by that search.

The two entrypoints are `search_pipeline.py` for the full heavy search and `train_top3_diverse.py` for the frozen final ensemble. Shared code lives in `utils/models.py`, `utils/file_manager.py`, `utils/selection.py`, and `utils/helpers.py`.

## Data Flow

For each subject, a candidate model crops the trial to `start:stop`, applies band filtering or filter-bank decomposition, extracts features, fits a simple classifier, and then contributes to subject-level ensemble voting. A trial is denoted `X \in R^{C \times T}`, where `C` is the number of EEG channels and `T` is the number of time samples.

## Model Families

### CSP + LDA

The `mne_csp_8_30_lda` pipeline crops the trial, applies a zero-phase 4th-order Butterworth band-pass filter from 8 to 30 Hz, fits Common Spatial Patterns with `n_components=6`, `reg="oas"`, epoch-wise covariance estimation, and log-variance features, then trains shrinkage LDA with `solver="lsqr", shrinkage="auto"`.

For binary classes with covariance matrices `Sigma_1` and `Sigma_2`, CSP finds spatial filters `w` that maximize

$$
\max_w \frac{w^\top \Sigma_1 w}{w^\top \Sigma_2 w}.
$$

After projection, the CSP features are

$$
f_j(X) = \log\left(\mathrm{var}(w_j^\top X)\right).
$$

LDA then finds a linear separator in that feature space.

### Filter-Bank CSP + LDA

The FBCSP variants are `mne_fbcsp_lda`, `fbcsp_broad_c6_k16_lda`, and `fbcsp_dense_c6_k20_lda`. They run CSP separately on several frequency bands, concatenate the log-variance features, optionally keep the best ones with `SelectKBest(f_classif)`, and classify with LDA.

For band `b`, let `X_b` be the band-passed trial. The CSP features are

$$
f_{b,j}(X) = \log\left(\mathrm{var}(w_{b,j}^\top X_b)\right).
$$

`mne_fbcsp_lda` uses broad bands from 4 to 40 Hz, 4 CSP components per band, and 12 selected features. `fbcsp_broad_c6_k16_lda` uses the same bands, 6 CSP components, and 16 selected features. `fbcsp_dense_c6_k20_lda` uses denser overlapping bands from 6 to 32 Hz, 6 CSP components, and 20 selected features.

These models work well because motor imagery information is concentrated in mu and beta rhythms, but the best sub-band depends on the subject.

### Riemannian Tangent Space + Logistic Regression

The tangent-space variants are `riemann_ts_lr_6_35` and `riemann_ts_lr_8_30`. Each pipeline crops the trial, band-pass filters it to either 6-35 Hz or 8-30 Hz, estimates covariance with OAS shrinkage, maps the covariance matrix to tangent space at a reference mean covariance, standardizes the tangent features, and fits logistic regression with `C=1.0` and `max_iter=4000`.

If `C_i \in S_{++}^C` is the trial covariance and `G` is the reference mean covariance, the local tangent-space map is

$$
T_i = \log\left(G^{-1/2} C_i G^{-1/2}\right).
$$

The symmetric matrix is then vectorized and passed to a linear classifier.

The covariance shrinkage has the form

$$
\hat{\Sigma} = (1-\lambda) S + \lambda \frac{\mathrm{tr}(S)}{C} I.
$$

### Riemannian FgMDM

The FgMDM variants are `riemann_fgmdm_8_30` and `riemann_fgmdm_8_30_lwf`. Each pipeline crops the trial, band-pass filters it to 8-30 Hz, estimates covariance with either OAS or Ledoit-Wolf shrinkage, and applies `pyriemann.classification.FgMDM` with the Riemannian metric.

In pyRiemann, `FgMDM` is effectively an `FGDA + MDM` pipeline: project SPD covariance matrices to tangent space, keep the most discriminative directions with Fisher Geodesic Discriminant Analysis, project back to the SPD manifold, compute class centroids, and classify by nearest centroid under the Riemannian distance.

The MDM decision rule is

$$
\hat{k} = \arg\min_k d_R(C, M_k),
$$

where `M_k` is the class centroid and the affine-invariant Riemannian distance is

$$
d_R(C_1, C_2) = \left\| \log\left(C_1^{-1/2} C_2 C_1^{-1/2}\right) \right\|_F.
$$

## Search Procedure

### Candidate Space

The heavy search is fully subject-specific. For each subject, it evaluates 8 model definitions over a dense grid of temporal windows with 4-fold stratified cross-validation.

At 256 Hz, starts range from `192` to `960` in steps of `64`, stops range from `1088` to `1537` in steps of `64`, the minimum crop length is `384` samples, and the full window `(0, n_times)` is also included. This targets the motor imagery segment while still allowing substantial variation in onset and duration.

### Cross-Validation

Each candidate is scored with `StratifiedKFold(n_splits=4, shuffle=True, random_state=42)`. On each fold, the model is fit on the training split, evaluated on the validation split, and its out-of-fold predictions are stored. The search keeps `mean_accuracy`, `std_accuracy`, and the full out-of-fold prediction vector for each candidate.

The out-of-fold predictions matter because the final ensemble is selected from joint behavior on the same held-out samples, not just from individual mean accuracy.

### Caching and Resume

The search is resumable. Per-candidate CV summaries are appended to `heavy_search_results.csv`, per-candidate out-of-fold predictions are saved in `heavy_oof_cache/`, and a candidate is skipped on the next run if both artifacts already exist.

### Candidate Pool Before Ensemble Selection

After the full search for one subject, the script keeps the top 8 windows for each model definition, sorted by descending `mean_accuracy` and ascending `std_accuracy`, merges those lists, drops duplicates, and keeps the best 48 candidates. This keeps the ensemble search manageable without collapsing diversity.

## Final Ensemble Selection

### Ensemble Size and Constraints

The final selection searches all valid combinations of size 2 or 3. In the heavy run, the defaults were `min_ensemble_size=2` and `max_ensemble_size=3`, and every selected subject ensemble ended up with size 3.

A combination is valid only if the same model definition appears at most once, the same family appears at most twice, two same-family candidates are not almost identical in time window, and the ensemble has enough behavioral diversity. In code, the same-family window check rejects pairs with both `|start_a - start_b| < 96` and `|stop_a - stop_b| < 96`, and the diversity threshold requires `avg_disagreement >= 0.04` for multi-model ensembles.

### Diversity Metrics

For two prediction vectors `p_a` and `p_b`, disagreement is

$$
\mathrm{disagreement}(p_a, p_b) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[p_{a,i} \neq p_{b,i}].
$$

Double fault is

$$
\mathrm{doublefault}(p_a, p_b, y) = \frac{1}{n} \sum_{i=1}^{n} \mathbf{1}[p_{a,i} \neq y_i \land p_{b,i} \neq y_i].
$$

For an ensemble, these values are averaged over all model pairs.

### Weighted Voting and Selection Score

Given predictions `\hat{y}^{(m)}` from each selected model `m`, the final prediction is a weighted majority vote:

$$
\hat{y}_i = \arg\max_{c \in \mathcal{C}} \sum_m w_m \mathbf{1}[\hat{y}^{(m)}_i = c].
$$

The weight `w_m` is that model's cross-validated `mean_accuracy`.

The ensemble selection score is

$$
\text{score} = \text{ensemble\_accuracy} + 0.03 \cdot \text{avg\_disagreement} - 0.02 \cdot \text{avg\_double\_fault}.
$$

This prefers high out-of-fold ensemble accuracy, some disagreement, and a low shared failure rate. Tie-breaking follows `(selection score, ensemble accuracy, avg disagreement, -avg double fault)`.

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

The final pattern was simple. Some subjects were strongly FBCSP-dominant. Some benefited from mixing CSP or FBCSP with Riemannian models. The best time window clearly depended on the subject. The best single model was often not the best ensemble member. Diversity helped, but only mildly. Most of the gain came from choosing the right three models and windows, not from fine-tuning the vote weights.

## Sources

1. H. Ramoser, J. Muller-Gerking, G. Pfurtscheller, "Optimal spatial filtering of single trial EEG during imagined hand movement," IEEE Transactions on Rehabilitation Engineering, 2000.
2. K. K. Ang, Z. Y. Chin, C. Wang, C. Guan, H. Zhang, "Filter Bank Common Spatial Pattern Algorithm on BCI Competition IV Datasets 2a and 2b," Frontiers in Neuroscience, 2012.
3. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Riemannian geometry applied to BCI classification," LVA/ICA, 2010.
4. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Multiclass Brain-Computer Interface Classification by Riemannian Geometry," IEEE Transactions on Biomedical Engineering, 2012.
5. A. Barachant, S. Bonnet, M. Congedo, C. Jutten, "Classification of covariance matrices using a Riemannian-based kernel for BCI applications," Neurocomputing, 2013.
6. Y. Chen, A. Wiesel, Y. C. Eldar, A. O. Hero, "Shrinkage algorithms for MMSE covariance estimation," IEEE Transactions on Signal Processing, 2010.
7. O. Ledoit, M. Wolf, "A well-conditioned estimator for large-dimensional covariance matrices," Journal of Multivariate Analysis, 2004.
