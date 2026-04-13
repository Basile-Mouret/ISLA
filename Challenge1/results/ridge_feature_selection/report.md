# Ridge Feature Selection Analysis

- Search CV: `1 x 3` repeated stratified folds on age bins
- Final comparison CV: `2 x 5` repeated stratified folds on age bins
- Goal: improve Ridge primarily through better CpG feature selection rather than changing the core linear model

## Plot Files

- `feature_score_diagnostics.png`
- `overlap_and_frequency.png`
- `final_model_scores.png`

## Best Search Candidate Per Family

| Family | Model | Search RMSE | Search Std | Params |
| --- | --- | --- | --- | --- |
| bagged_fscore | bagged_fscore_k3500 | 4.7400 | 0.0725 | {"alpha": 0.01, "k": 3500, "n_estimators": 7, "random_state": 42, "sample_fraction": 0.8, "score_method": "f_score"} |
| bagged_gender | bagged_gender_k3500_gp0.25 | 4.7327 | 0.0824 | {"alpha": 0.01, "gap_penalty": 0.25, "k": 3500, "n_estimators": 7, "random_state": 42, "sample_fraction": 0.8, "score_method": "gender_stable"} |
| baseline | global_ridge_best | 4.7711 | 0.0233 | {} |
| stable_fscore | stable_fscore_k3000 | 4.6858 | 0.0484 | {"alpha": 0.01, "k": 3000, "n_resamples": 15, "random_state": 42, "sample_fraction": 0.8, "score_method": "f_score"} |
| stable_gender | stable_gender_k3500_gp0.25 | 4.7417 | 0.0904 | {"alpha": 0.01, "gap_penalty": 0.25, "k": 3500, "n_resamples": 15, "random_state": 42, "sample_fraction": 0.8, "score_method": "gender_stable"} |

## Final Repeated-CV Comparison

| Model | Final CV RMSE | Final CV Std |
| --- | --- | --- |
| bagged_fscore_k3500 | 4.4479 | 0.2019 |
| bagged_gender_k3500_gp0.25 | 4.4523 | 0.1861 |
| stable_fscore_k3000 | 4.4532 | 0.2259 |
| global_ridge_best | 4.4577 | 0.2611 |
| stable_gender_k3500_gp0.25 | 4.5050 | 0.1832 |

## Feature Overlap With Baseline

| Model | Selected CpGs | Overlap | Jaccard |
| --- | --- | --- | --- |
| stable_fscore_k3000 | 3000 | 3000 | 0.8571 |
| stable_gender_k3500_gp0.25 | 3500 | 3159 | 0.8224 |
| bagged_fscore_k3500 | 3500 | 3371 | 0.9289 |
| bagged_gender_k3500_gp0.25 | 3500 | 3107 | 0.7981 |

## Interpretation

- The current best model from this ridge-only feature-selection analysis is `bagged_fscore_k3500`
- If the best stable or bagged selector beats the baseline, that means selection variance matters and more robust CpG ranking is helping
- If the gain is small, then the baseline `f_regression` selector was already close to optimal and future gains likely require model combination rather than another single selector tweak
