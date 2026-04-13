# Gender and Cluster Analysis

- Training rows: `489`
- Numeric methylation features: `10000`
- Search CV: `1 x 5` repeated stratified folds on age bins
- Final comparison CV: `2 x 5` repeated stratified folds on age bins

## Plot Files

- `age_by_gender.png`
- `correlation_diagnostics.png`
- `pca_overview.png`
- `cluster_diagnostics.png`
- `model_comparison.png`

## Gender Summary

- Female samples: `348`
- Male samples: `141`
- Mean female age: `50.713`
- Mean male age: `54.035`

## Correlation Findings

- Among the top 200 age-correlated CpGs overall, `200` keep the same correlation sign in both genders
- That suggests the main age signal is largely shared, but the strength of some CpGs differs by gender

Top overall correlated features:

| Feature | Overall Corr | Male Corr | Female Corr | Gap |
| --- | --- | --- | --- | --- |
| cg00329615 | -0.6210 | -0.5493 | -0.6345 | 0.0851 |
| cg22736354 | 0.6062 | 0.5619 | 0.6181 | -0.0563 |
| cg18826637 | -0.5832 | -0.5186 | -0.5941 | 0.0755 |
| cg10804656 | 0.5744 | 0.5387 | 0.5959 | -0.0572 |
| cg05991454 | 0.5741 | 0.5057 | 0.5956 | -0.0899 |
| cg17436656 | -0.5400 | -0.4123 | -0.5750 | 0.1627 |
| cg12534424 | 0.5336 | 0.4567 | 0.5809 | -0.1242 |
| cg25584930 | 0.5301 | 0.4995 | 0.5419 | -0.0424 |
| cg16477091 | 0.5137 | 0.4788 | 0.5324 | -0.0536 |
| cg10806820 | 0.5120 | 0.4943 | 0.5119 | -0.0176 |

Top gender-gap features:

| Feature | Overall Corr | Male Corr | Female Corr | Gap |
| --- | --- | --- | --- | --- |
| cg15067806 | 0.1212 | -0.1349 | 0.2245 | -0.3594 |
| cg14381550 | 0.1088 | -0.1093 | 0.2314 | -0.3406 |
| cg05006211 | -0.0966 | -0.3423 | -0.0221 | -0.3202 |
| cg27148800 | 0.2692 | 0.0454 | 0.3622 | -0.3168 |
| cg25999148 | 0.0247 | -0.2333 | 0.0694 | -0.3027 |
| cg10940997 | 0.0489 | -0.1667 | 0.1298 | -0.2965 |
| cg06664254 | 0.2797 | 0.0705 | 0.3644 | -0.2939 |
| cg19695507 | 0.1832 | -0.0480 | 0.2428 | -0.2908 |
| cg01993865 | 0.0066 | -0.1975 | 0.0916 | -0.2890 |
| cg18860310 | 0.1182 | -0.1007 | 0.1879 | -0.2887 |

## Cluster Findings

- Best unsupervised cluster count by silhouette on the PCA representation: `2`
- The cluster summary table below helps check whether clusters mainly reflect age structure, gender structure, or both

| Cluster | Samples | Mean Age | Female Fraction |
| --- | --- | --- | --- |
| 0 | 161 | 57.4845 | 0.7267 |
| 1 | 328 | 48.8171 | 0.7043 |

## Candidate Search Results

These are the top quick-screen candidates from each gender-aware family.

| Family | Candidate | Search RMSE | Search Std | Params |
| --- | --- | --- | --- | --- |
| gender_split_ridge | gender_split_fk3500_mk500_a0.1 | 4.7992 | 0.2732 | {"female_alpha": 0.1, "female_k": 3500, "male_alpha": 0.1, "male_k": 500} |
| gender_split_ridge | gender_split_fk3500_mk500_a0.01 | 4.7992 | 0.2731 | {"female_alpha": 0.01, "female_k": 3500, "male_alpha": 0.01, "male_k": 500} |
| gender_split_ridge | gender_split_fk3000_mk500_a0.1 | 4.8184 | 0.2338 | {"female_alpha": 0.1, "female_k": 3000, "male_alpha": 0.1, "male_k": 500} |
| gender_interaction_ridge | gender_interaction_ik100_a0.01 | 4.4278 | 0.1476 | {"alpha": 0.01, "interaction_k": 100, "main_k": 3500} |
| gender_interaction_ridge | gender_interaction_ik100_a0.1 | 4.4278 | 0.1476 | {"alpha": 0.1, "interaction_k": 100, "main_k": 3500} |
| gender_interaction_ridge | gender_interaction_ik100_a1.0 | 4.4280 | 0.1475 | {"alpha": 1.0, "interaction_k": 100, "main_k": 3500} |
| cluster_augmented_ridge | cluster_augmented_k2_a0.01 | 4.4159 | 0.1100 | {"alpha": 0.01, "main_k": 3500, "n_clusters": 2, "pca_components": 20} |
| cluster_augmented_ridge | cluster_augmented_k2_a0.1 | 4.4159 | 0.1100 | {"alpha": 0.1, "main_k": 3500, "n_clusters": 2, "pca_components": 20} |
| cluster_augmented_ridge | cluster_augmented_k3_a0.01 | 4.4168 | 0.1097 | {"alpha": 0.01, "main_k": 3500, "n_clusters": 3, "pca_components": 20} |

## Final 2x5 CV Comparison

| Model | Final CV RMSE | Final CV Std |
| --- | --- | --- |
| cluster_augmented_k2_a0.01 | 4.4574 | 0.2617 |
| global_ridge_best | 4.4577 | 0.2611 |
| gender_interaction_ik100_a0.01 | 4.4608 | 0.2904 |
| gender_split_fk3500_mk500_a0.1 | 4.8706 | 0.2823 |

## RMSE by Gender

| Model | Overall RMSE | Female RMSE | Male RMSE | Female Bias | Male Bias |
| --- | --- | --- | --- | --- | --- |
| cluster_augmented_k2_a0.01 | 4.3962 | 4.3291 | 4.5575 | -0.0359 | 0.4018 |
| global_ridge_best | 4.3965 | 4.3295 | 4.5578 | -0.0365 | 0.4029 |
| gender_interaction_ik100_a0.01 | 4.4015 | 4.3529 | 4.5193 | -0.0276 | 0.3775 |
| gender_split_fk3500_mk500_a0.1 | 4.7490 | 4.6216 | 5.0497 | 0.0684 | -0.0542 |

## Interpretation

- The shared linear age signal is strong across genders, but some CpGs change slope strength by sex
- If a gender-aware model beats the global Ridge baseline, the likely mechanism is not a completely different feature set by sex, but modestly different weighting or sex-specific residual correction
- The best model from this analysis is `cluster_augmented_k2_a0.01` under the final repeated-CV comparison
- If the improvement over the global Ridge baseline is tiny, that means gender structure is real but not large enough to materially move leaderboard performance on its own
