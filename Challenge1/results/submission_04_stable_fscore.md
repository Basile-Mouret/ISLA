# Submission 04 - Stable F-Score Ridge

- Submission file: `submissions/submission_04_stable_fscore/y_pred.csv`
- Model source: `StableScoreRidgeRegressor(...)` generated directly from `ridge_feature_models.py`
- Local CV source: `results/candidate_followup.md`
- Local CV setting: `2 x 5` repeated stratified folds on age bins

## Chosen Configuration

- Model class: `StableScoreRidgeRegressor`
- Feature score: `f_score`
- Selected CpGs: `2800`
- Ridge alpha: `0.01`
- Number of resamples for stable ranking: `15`
- Resample fraction: `0.7`

## Why This Model

- It is the conservative fallback candidate after bagged Ridge
- It uses a smaller, more stable feature subset and lower local CV variance than the plain Ridge baseline

## Output Validation

- File name: `y_pred.csv`
- Column count: `1`
- Column name: `age`
- Row count: `200`
- Values: numeric, no missing values
