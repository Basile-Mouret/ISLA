# Benchmark Experiments

- Generated at: `2026-03-23T11:51:12.375607+00:00`
- Training rows: `489`
- Training columns: `10001`
- Target mean age: `51.671`
- CV strategy: `2 x 5` repeated stratified folds on age bins
- Run status: `completed`
- Completed models: `6/6`

## Ranked Results

| Rank | Model | CV RMSE | CV Std | Mean Fit Time (s) | Wall Time (s) | CV Fits | Best Params | Notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | ridge_select_raw | 4.4731 | 0.2134 | 0.72 | 23.31 | 200 | `model__alpha`=0.1, `select__k`=3000 | Univariate filter plus Ridge on raw beta values. |
| 2 | ridge_select_mvalue | 4.4749 | 0.2094 | 1.17 | 35.01 | 200 | `model__alpha`=0.1, `select__k`=3000 | Univariate filter plus Ridge on M-values. |
| 3 | pls_raw | 4.9772 | 0.3643 | 2.69 | 23.64 | 60 | `model__n_components`=15 | Supervised latent factors with PLS. |
| 4 | ridge_all_raw | 4.9809 | 0.3636 | 0.39 | 5.42 | 50 | `model__alpha`=0.1 | All features, scaled beta values, Ridge. |
| 5 | pca_ridge_raw | 5.1628 | 0.3457 | 3.61 | 58.87 | 200 | `model__alpha`=0.1, `reduce__n_components`=150 | Dimensionality reduction with PCA followed by Ridge. |
| 6 | dummy_mean | 11.8225 | 0.4005 | 0.10 | 1.36 | 10 | - | Sanity-check baseline. |

## Top Candidate Details

### ridge_select_raw

- Best CV RMSE: `4.4731`
- Fold-to-fold std: `0.2134`
- Mean fit time: `0.72` seconds
- Total wall time: `23.31` seconds
- CV fits: `200`
- Grid candidates: `20`
- Refit time: `0.30` seconds
- Best params: `model__alpha`=0.1, `select__k`=3000
- Notes: Univariate filter plus Ridge on raw beta values.

Top grid candidates:

| RMSE | Std | Params |
| ---: | ---: | --- |
| 4.4731 | 0.2134 | `model__alpha`=0.1, `select__k`=3000 |
| 4.4732 | 0.2133 | `model__alpha`=1, `select__k`=3000 |
| 4.4748 | 0.2124 | `model__alpha`=10, `select__k`=3000 |

### ridge_select_mvalue

- Best CV RMSE: `4.4749`
- Fold-to-fold std: `0.2094`
- Mean fit time: `1.17` seconds
- Total wall time: `35.01` seconds
- CV fits: `200`
- Grid candidates: `20`
- Refit time: `0.37` seconds
- Best params: `model__alpha`=0.1, `select__k`=3000
- Notes: Univariate filter plus Ridge on M-values.

Top grid candidates:

| RMSE | Std | Params |
| ---: | ---: | --- |
| 4.4749 | 0.2094 | `model__alpha`=0.1, `select__k`=3000 |
| 4.4749 | 0.2094 | `model__alpha`=1, `select__k`=3000 |
| 4.4753 | 0.2088 | `model__alpha`=10, `select__k`=3000 |

### pls_raw

- Best CV RMSE: `4.9772`
- Fold-to-fold std: `0.3643`
- Mean fit time: `2.69` seconds
- Total wall time: `23.64` seconds
- CV fits: `60`
- Grid candidates: `6`
- Refit time: `0.91` seconds
- Best params: `model__n_components`=15
- Notes: Supervised latent factors with PLS.

Top grid candidates:

| RMSE | Std | Params |
| ---: | ---: | --- |
| 4.9772 | 0.3643 | `model__n_components`=15 |
| 4.9809 | 0.3636 | `model__n_components`=40 |
| 4.9809 | 0.3636 | `model__n_components`=25 |

### ridge_all_raw

- Best CV RMSE: `4.9809`
- Fold-to-fold std: `0.3636`
- Mean fit time: `0.39` seconds
- Total wall time: `5.42` seconds
- CV fits: `50`
- Grid candidates: `5`
- Refit time: `0.32` seconds
- Best params: `model__alpha`=0.1
- Notes: All features, scaled beta values, Ridge.

Top grid candidates:

| RMSE | Std | Params |
| ---: | ---: | --- |
| 4.9809 | 0.3636 | `model__alpha`=0.1 |
| 4.9811 | 0.3636 | `model__alpha`=1 |
| 4.9826 | 0.3634 | `model__alpha`=10 |

### pca_ridge_raw

- Best CV RMSE: `5.1628`
- Fold-to-fold std: `0.3457`
- Mean fit time: `3.61` seconds
- Total wall time: `58.87` seconds
- CV fits: `200`
- Grid candidates: `20`
- Refit time: `2.84` seconds
- Best params: `model__alpha`=0.1, `reduce__n_components`=150
- Notes: Dimensionality reduction with PCA followed by Ridge.

Top grid candidates:

| RMSE | Std | Params |
| ---: | ---: | --- |
| 5.1628 | 0.3457 | `model__alpha`=0.1, `reduce__n_components`=150 |
| 5.1630 | 0.3456 | `model__alpha`=1, `reduce__n_components`=150 |
| 5.1643 | 0.3454 | `model__alpha`=10, `reduce__n_components`=150 |
