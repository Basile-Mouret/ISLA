# Current Analysis

- Benchmark source: `results/latest_benchmark.md`
- Refined Ridge source: `results/latest_finalists.md`
- Manual note from interrupted run: `elasticnet_select_raw` reached CV RMSE `4.4450` with CV std `0.2381`
- CV setup: `2 x 5` repeated stratified folds on age bins

## Updated Scoreboard

- `ridge_select_raw_refined` improved filtered Ridge from `4.4731` to `4.4577`
- The best refined Ridge parameters are `select__k=3500` and `model__alpha=0.01`
- The interrupted `elasticnet_select_raw` benchmark run at `4.4450` still remains the best local result seen so far
- `ridge_select_mvalue` remains essentially tied with raw Ridge at the benchmark stage, so raw beta values still look like the right default branch

## What The Refined Ridge Run Tells Us

- Moving from `k=3000` to `k=3500` helped, so the benchmark grid really was a bit too narrow on feature count
- The best `alpha` shifted down from `0.1` to `0.01`, confirming that weaker Ridge regularization is better in this region
- The top five refined Ridge candidates are all at `k=3500`, while `alpha` varies from `0.01` through `1.0` with almost no score change
- That means the local optimum is broad in `alpha` but sharp enough in `k` to matter
- The improvement is real but modest: about `0.015` RMSE better than the benchmark Ridge winner

## Interpretation

- The strongest signal so far is still linear and filter-based
- Supervised feature selection is clearly important; using all features or unsupervised PCA remains worse
- The model does not appear very sensitive to the exact Ridge penalty once `k` is near the right range
- This suggests future gains are more likely to come from better feature subsets, better blends, or better model families than from endlessly fine-tuning Ridge `alpha`

## Local CV Versus Public Leaderboard

- Local CV is still the best honest estimate before submission, but it is not identical to the public leaderboard
- Because each CV fold trains on about `80%` of the training data, CV often slightly overestimates the final error of the refit-on-all-data submission model
- On the other hand, the public leaderboard uses only `100` hidden samples, so it is noisy and can be either better or worse than local CV by chance
- The most realistic expectation is that public RMSE should be in the same broad range as local CV, perhaps somewhat lower or higher by a few tenths, but not dramatically lower without a much stronger model
- So a local `4.46` model might submit at something like low-to-mid `4`s or maybe high `3`s if we are lucky, but it should not be expected to turn into a `1.x` leaderboard model on refit alone

## Recommended Next Steps

1. Run `elasticnet_select_raw_refined`, because ElasticNet is still the best local family seen so far
2. After that, compare the best refined Ridge and best refined ElasticNet and test a simple average blend
3. If the blend helps locally, use it for the first submission candidate
4. Add one gender-aware branch after the linear finalists, such as sex-specific models or interaction-expanded linear models
5. Keep tracking both local CV and public leaderboard outcomes in markdown after each submission

## Priority Order

1. Refined `elasticnet_select_raw`
2. Blend of best Ridge and best ElasticNet
3. Optional gender-aware linear variant
