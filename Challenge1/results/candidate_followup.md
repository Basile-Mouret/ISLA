# Candidate Follow-Up After First Two Public Submissions

- Source metrics: `results/candidate_followup.json`
- Goal: prioritize more robust follow-up candidates after the gender-aware blend underperformed on public data

## Final 2x5 CV Ranking

| Model | Local CV RMSE | Local CV Std |
| --- | ---: | ---: |
| bagged_gender_blend_male_0.7 | 4.4007 | 0.2573 |
| bagged_bias_correction | 4.4224 | 0.2133 |
| bagged_ridge_base | 4.4224 | 0.2133 |
| plain_bagged_blend_w0.4 | 4.4269 | 0.2323 |
| stable_fscore_k2800_r15_sf0.7 | 4.4406 | 0.2007 |
| plain_ridge_base | 4.4577 | 0.2611 |

## Recommended Next Public Candidates

1. `bagged_ridge_base`
   - Best pure Ridge-only model so far
   - Improves locally over the public-tested plain Ridge baseline
   - Avoids the subgroup-specific interaction term that appears fragile on the public subset

2. `stable_fscore_k2800_r15_sf0.7`
   - Slightly worse local RMSE than bagged Ridge
   - More conservative feature subset and lower repeated-CV variance
   - Good robustness candidate if bagged Ridge still disappoints publicly

3. `plain_bagged_blend_w0.4`
   - A simple non-gender blend of plain Ridge and bagged Ridge
   - Useful if we want a small ensemble without sex-specific correction logic

## What Not To Prioritize Next

- Another interaction-heavy gender blend submission right away
- More hard gender-split models
- More cluster-driven Ridge variants

## Practical Interpretation

- Public results are currently telling us that extra complexity is not paying off unless it is extremely robust
- The next best move is to test stronger Ridge feature selection without subgroup corrections
