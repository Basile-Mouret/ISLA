# Finalists Experiments

- Generated at: `2026-03-23T12:44:46.372280+00:00`
- Training rows: `489`
- Training columns: `10001`
- Target mean age: `51.671`
- CV strategy: `1 x 3` repeated stratified folds on age bins
- Run status: `completed`
- Completed models: `1/1`

## Ranked Results

| Rank | Model | CV RMSE | CV Std | Mean Fit Time (s) | Wall Time (s) | CV Fits | Best Params | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | elasticnet_select_raw_refined_lite | 4.5368 | 0.0827 | 75.04 | 1166.81 | 81 | `model__alpha`=0.01, `model__l1_ratio`=0.2, `select__k`=5000 | Smaller refined ElasticNet sweep on raw beta values. |

## Top Candidate Details

### elasticnet_select_raw_refined_lite

- Best CV RMSE: `4.5368`
- Fold-to-fold std: `0.0827`
- Mean fit time: `75.04` seconds
- Total wall time: `1166.81` seconds
- CV fits: `81`
- Grid candidates: `27`
- Refit time: `56.02` seconds
- Best params: `model__alpha`=0.01, `model__l1_ratio`=0.2, `select__k`=5000
- Notes: Smaller refined ElasticNet sweep on raw beta values.

Top grid candidates:

| RMSE | Std | Params |
| ---: | ---: | --- |
| 4.5368 | 0.0827 | `model__alpha`=0.01, `model__l1_ratio`=0.2, `select__k`=5000 |
| 4.5419 | 0.0815 | `model__alpha`=0.03, `model__l1_ratio`=0.2, `select__k`=5000 |
| 4.5490 | 0.0913 | `model__alpha`=0.02, `model__l1_ratio`=0.2, `select__k`=5000 |
