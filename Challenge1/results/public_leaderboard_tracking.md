# Public Leaderboard Tracking

## Submitted Models

| Submission | File | Local CV RMSE | Public RMSE | Notes |
| --- | --- | ---: | ---: | --- |
| submission_01_ridge | `y_pred.csv` | 4.4577 | 4.10 | plain Ridge baseline |
| submission_02_bagged_gender_blend | `submissions/submission_02_bagged_gender_blend/y_pred.csv` | 4.4007 | 4.19 | bagged Ridge plus male interaction blend |

## Interpretation

- The public leaderboard is noisy, but these two submissions still provide useful signal
- The plain Ridge baseline beat the more complex gender-aware blend on public data
- That suggests the interaction correction is helping locally but not transferring cleanly to the public subset
- The safest conclusion is that public performance currently prefers robust Ridge structure over subgroup-specific corrections

## What This Changes

- The next submission should prioritize the best pure Ridge-only candidate rather than another interaction-heavy blend
- The strongest next public candidate is the refined bagged Ridge selector
- The second-best fallback is a stable Ridge selector with a smaller, more conservative feature set
