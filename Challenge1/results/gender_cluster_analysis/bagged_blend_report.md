# Gender Blend With Bagged Ridge Base

- Search results: `results/gender_cluster_analysis/bagged_blend_search.json`
- Base Ridge candidates compared:
  - `plain_ridge_base`
  - `bagged_ridge_base`
- Interaction branch: `GenderInteractionRidgeRegressor(main_k=3500, interaction_k=100, alpha=0.01)`

## Why This Follow-Up Was Needed

- The Ridge-only feature-selection sweep found a better pure Ridge model than the original global Ridge baseline
- The natural next question was whether the gender-aware blend should use that stronger Ridge model as its base
- During the re-check, the older plain-blend number was re-evaluated with the estimator implementation and found to be `4.4490`, not the earlier optimistic `4.4145`

## Search Strategy

- Coarse search on `1 x 5` CV for a gender-conditional blend using the bagged Ridge base
- Female interaction weight grid: `0.0, 0.1, 0.2`
- Male interaction weight grid: `0.3, 0.5, 0.6, 0.7, 0.9`
- Final ranking on `2 x 5` repeated CV

## Best Search Setting

- Female interaction weight: `0.0`
- Male interaction weight: `0.7`
- Search CV RMSE: `4.3748`

## Final 2x5 CV Results

| Model | CV RMSE | CV Std | Notes |
| --- | ---: | ---: | --- |
| bagged_ridge_gender_blend_best_search | 4.4007 | 0.2573 | female uses pure bagged Ridge, male uses 30% bagged Ridge + 70% interaction |
| bagged_ridge_gender_blend_prev_weights | 4.4009 | 0.2484 | same as above but with male interaction weight 0.6 |
| bagged_ridge_base | 4.4224 | 0.2133 | best pure Ridge-only selector so far |
| plain_ridge_gender_blend | 4.4490 | 0.2763 | original gender blend re-checked with direct estimator evaluation |
| plain_ridge_base | 4.4577 | 0.2611 | original submission Ridge baseline |
| gender_interaction_only | 4.4608 | 0.2904 | interaction model alone |

## Interpretation

- Yes, plugging the stronger Ridge model into the blend helps
- The new bagged-Ridge gender blend is the best local model seen so far
- Most of the gain comes from improving the shared base model; the interaction branch still acts like a male-specific correction
- The exact male interaction weight is not very sensitive around `0.6-0.7`

## Practical Recommendation

- Current best local candidate:
  - `GenderBlendRegressor(base_model_variant="bagged_ridge", female_inter_weight=0.0, male_inter_weight=0.7, interaction_main_k=3500, interaction_k=100, interaction_alpha=0.01)`
- If we want a second submission candidate after the plain Ridge upload, this is the strongest option right now
