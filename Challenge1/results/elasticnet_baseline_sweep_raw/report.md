# ElasticNet Baseline Sweep

- Validation strategy: `12` repeated stratified shuffle splits
- Validation size per split: `100` rows
- Focus: small set of baseline-like ElasticNetCV presets derived from `model_1.py`

## Ranked Results

| Preset | Mean RMSE | Std | Median | Best | Worst | Mean alpha | Chosen l1 ratios |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline_v1 | 4.6604 | 0.4394 | 4.7771 | 3.9325 | 5.2078 | 0.1198 | 0.1:5, 0.5:7 |

## Presets Evaluated

- `baseline_v1`: exact `model_1.py`
- `baseline_v2_more_alphas`: same l1 ratios with denser alpha path and stricter optimization
- `baseline_v3_low_l1`: concentrates search near low `l1_ratio` values, since the fitted baseline tends to choose `0.1`
- `baseline_v4_low_l1_mvalue`: same low-l1 search with M-values
