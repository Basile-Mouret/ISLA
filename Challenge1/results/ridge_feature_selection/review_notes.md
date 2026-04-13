# Ridge Feature Selection Review Notes

## Self-Review Summary

I reviewed the custom Ridge feature-selection code in:

- `ridge_feature_models.py`
- `ridge_feature_selection_analysis.py`
- `pipelines.py`

and the implementation looks methodologically sound for the intended experiments.

## What Was Checked

### Cross-validation leakage

- Feature scoring happens inside each estimator's `fit`
- During CV, each estimator is cloned and fit only on the fold's training split
- No feature ranks or selected CpGs are computed using validation-fold labels
- Therefore the custom selectors are fold-local and do not leak validation information

### Estimator behavior

- Custom estimators inherit from `BaseEstimator` and `RegressorMixin`
- They expose all tunable settings in `__init__`, so sklearn cloning works correctly
- `predict` uses only fitted attributes produced during `fit`

### Comparison validity

- The quick search uses a cheaper `1 x 3` CV only for screening families
- The final family comparison is done under the stronger `2 x 5` repeated CV setup
- The repeated-CV ranking is therefore the meaningful comparison table

### Interpretation caveats

- The observed gains are real but still modest
- The bagged feature-selection Ridge improves on the plain global Ridge baseline, but not enough to explain the full public leaderboard gap by itself
- Because the bagged model trains multiple subsets on the same fold training data, it is best viewed as variance reduction through feature-subset diversity, not as a fundamentally different model class

## Current Conclusion

- The code is usable for continued Ridge feature-selection experiments
- The best Ridge-only direction so far is bagged `f_score` selection around `k=3500`
- The next sensible step is to refine that neighborhood rather than return to gender-penalized selectors, which underperformed in the Ridge-only setting
