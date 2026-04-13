# Submission 02 - Bagged Gender Blend

- Submission file: `submissions/submission_02_bagged_gender_blend/y_pred.csv`
- Model source: `build_best_gender_blend_model()` in `gender_models.py`
- Local CV source: `results/gender_cluster_analysis/bagged_blend_report.md`
- Local CV setting: `2 x 5` repeated stratified folds on age bins

## Chosen Configuration

- Base model: bagged Ridge feature selector
- Base Ridge settings:
  - `k = 3200`
  - `alpha = 0.01`
  - `n_estimators = 7`
  - `sample_fraction = 0.7`
- Interaction model settings:
  - `interaction_main_k = 3500`
  - `interaction_k = 100`
  - `interaction_alpha = 0.01`
- Blend weights:
  - female interaction weight `0.0`
  - male interaction weight `0.7`

## Why This Model

- It is the best local model found so far
- It improves on the plain Ridge submission baseline from `4.4577` to `4.4007` repeated-CV RMSE
- The gain comes from combining a stronger Ridge base with a male-specific interaction correction

## Output Validation

- File name: `y_pred.csv`
- Column count: `1`
- Column name: `age`
- Row count: `200`
- Values: numeric, no missing values
