# Submission 01 - Ridge

- Submission file: `y_pred.csv`
- Model family: `ridge_select_raw_refined`
- Local CV source: `results/latest_finalists.md`
- Local CV setting: `2 x 5` repeated stratified folds on age bins

## Chosen Configuration

- Preprocessing: one-hot encode `gender`, standardize CpG values
- Feature selection: `SelectKBest(f_regression, k=3500)`
- Regressor: `Ridge(alpha=0.01)`
- Methylation scale: raw beta values

## Why This Model

- It is the best Ridge model validated so far
- It improved over the benchmark Ridge run from `4.4731` to `4.4577` RMSE
- It is much faster and simpler than ElasticNet while remaining locally competitive

## Output Validation

- File name: `y_pred.csv`
- Column count: `1`
- Column name: `age`
- Row count: `200`
- Values: numeric, no missing values

## Notes

- This is the first submission candidate, chosen for robustness and speed rather than absolute local best score
- If the public leaderboard is close, the next likely candidate should be the best ElasticNet or a Ridge/ElasticNet blend
