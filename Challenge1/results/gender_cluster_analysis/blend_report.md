# Gender-Aware Blend Follow-Up

- Source analysis: `results/gender_cluster_analysis/report.md`
- Search results: `results/gender_cluster_analysis/blend_search.json`
- Base models:
  - `global_ridge_best`
  - `gender_interaction_ik100_a0.01`

## Why This Follow-Up Was Tested

- The interaction model was slightly worse overall than the global Ridge model
- However, it improved male RMSE while hurting female RMSE only modestly
- That suggested the two models might be complementary rather than strictly competing

## Search Strategy

- First, search on `1 x 5` CV for:
  - simple global/interactions blends
  - gender-conditional blends
  - gender-specific bias correction of the global Ridge model
- Then evaluate the best candidate from each family on `2 x 5` repeated CV

## Best Search Candidates

- Best simple blend: interaction weight `0.1`
- Best gender-conditional blend:
  - female interaction weight `0.0`
  - male interaction weight `0.6`
- Best bias correction: full estimated gender bias subtraction from the global Ridge model

## Final 2x5 CV Results

| Model | CV RMSE | CV Std | Notes |
| --- | ---: | ---: | --- |
| gender_blend_best | 4.4145 | 0.1300 | female uses pure global Ridge, male uses 40% global + 60% interaction |
| bias_corrected_ridge_best | 4.4167 | 0.1091 | subtract training-fold gender residual mean |
| simple_blend_best | 4.4169 | 0.1130 | 90% global Ridge + 10% interaction for everyone |
| global_ridge_best | 4.4577 | 0.2611 | current submission Ridge baseline |
| gender_interaction_best | 4.4608 | 0.2904 | interaction model alone |

## Interpretation

- The strongest improvement comes from using the interaction model only for male predictions
- Female predictions are best left as the original global Ridge model
- This means gender-specific slope adjustments appear useful for the male subgroup, but not worth applying uniformly
- The gain is meaningful locally: about `0.043` RMSE better than the current best Ridge baseline

## Practical Recommendation

- The current most promising local model is a `GenderBlendRegressor` with:
  - `female_inter_weight = 0.0`
  - `male_inter_weight = 0.6`
  - `interaction_main_k = 3500`
  - `interaction_k = 100`
  - `interaction_alpha = 0.01`

- This model is now available in `gender_models.py` as `GenderBlendRegressor`
