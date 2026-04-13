# Submission Candidate - Bagged Ridge Feature Selection

- Builder: `build_best_ridge_feature_model()` in `pipelines.py`
- Source sweep: `results/ridge_feature_refine/report.md`
- This is the best pure-Ridge feature-selection model found so far

## Configuration

- Model class: `BaggedScoreRidgeRegressor`
- Feature score: `f_score`
- Number of selected CpGs per ensemble member: `3200`
- Ridge alpha: `0.01`
- Ensemble size: `7`
- Resample fraction for feature scoring: `0.7`

## Local Performance

- Final repeated-CV RMSE: `4.4224`
- Final repeated-CV std: `0.2133`

## Position Relative To Other Models

- Better than the plain global Ridge baseline at `4.4577`
- Slightly worse than the current best gender-aware blend at `4.4145`
- Good fallback candidate when a pure Ridge-only submission is preferred
