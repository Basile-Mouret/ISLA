# Refined Ridge Feature Selection Sweep

- Search CV: `1 x 3` repeated stratified folds on age bins
- Final comparison CV: `2 x 5` repeated stratified folds on age bins
- Focus: refine the strongest Ridge-only selector family found earlier (`bagged_fscore`) and compare it against a tightened `stable_fscore` sweep

## Best Search Candidate Per Family

| Family | Model | Search RMSE | Search Std | Params |
| --- | --- | --- | --- | --- |
| bagged_fscore_refine | bagged_fscore_k3200_e7_sf0.7 | 4.6742 | 0.1016 | {"alpha": 0.01, "k": 3200, "n_estimators": 7, "random_state": 42, "sample_fraction": 0.7, "score_method": "f_score"} |
| baseline | global_ridge_best | 4.7711 | 0.0233 | {} |
| stable_fscore_refine | stable_fscore_k2800_r15_sf0.7 | 4.6647 | 0.1005 | {"alpha": 0.01, "k": 2800, "n_resamples": 15, "random_state": 42, "sample_fraction": 0.7, "score_method": "f_score"} |

## Final Repeated-CV Comparison

| Model | Final CV RMSE | Final CV Std |
| --- | --- | --- |
| bagged_fscore_k3200_e7_sf0.7 | 4.4224 | 0.2133 |
| stable_fscore_k2800_r15_sf0.7 | 4.4406 | 0.2007 |
| global_ridge_best | 4.4577 | 0.2611 |

## Interpretation

- The best refined Ridge-only model from this sweep is `bagged_fscore_k3200_e7_sf0.7`
- If this beats the earlier `bagged_fscore_k3500`, the bagging hyperparameters still matter and the Ridge-only path has a little more room
- If it ties the earlier result, Ridge feature selection is likely close to saturated and future gains should come from model combination rather than more selector tuning
