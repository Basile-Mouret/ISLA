# Submission 03 - Bagged Ridge

- Submission file: `submissions/submission_03_bagged_ridge/y_pred.csv`
- Model source: `build_best_ridge_feature_model()` in `pipelines.py`
- Local CV source: `results/ridge_feature_refine/report.md`
- Local CV setting: `2 x 5` repeated stratified folds on age bins

## Chosen Configuration

- Model class: `BaggedScoreRidgeRegressor`
- Feature score: `f_score`
- Selected CpGs per ensemble member: `3200`
- Ridge alpha: `0.01`
- Ensemble size: `7`
- Resample fraction: `0.7`

## Why This Model

- It is the strongest pure Ridge-only candidate after the first two public submissions
- It improves locally over the original plain Ridge baseline while avoiding the fragile subgroup interaction logic

## Output Validation

- File name: `y_pred.csv`
- Column count: `1`
- Column name: `age`
- Row count: `200`
- Values: numeric, no missing values
