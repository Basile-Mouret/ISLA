# Model Improvement Plan

## Current Best Honest Local Result

- Best repeated-CV result so far: `bagged_ridge_gender_blend_best_search` at RMSE `4.4007`
- Source: `results/gender_cluster_analysis/bagged_blend_report.md`
- Previous best global Ridge baseline: `4.4577`

## What We Learned

- Hard male/female model splits are not useful; they overfit and perform worse
- Gender is still useful as a small correction on top of the shared age signal
- Pure unsupervised clustering is not enough on its own
- The best improvements now come from **combining** strong linear models rather than replacing them with a radically different single model
- The feature-selection stage matters more than the exact Ridge penalty in the current best region
- A stronger base Ridge model improves the gender-aware blend further, which confirms the blend is additive rather than a fragile one-off

## Highest-Priority Next Directions

1. **OOF stacking / ensembling**
   - Base learners: best global Ridge, best ElasticNet, gender-aware blend, cluster-augmented Ridge
   - Meta-learner: small Ridge on out-of-fold predictions
   - Rationale: the best improvement found so far came from combining complementary models

2. **Proper ElasticNet re-run under the final CV protocol**
   - Use a coarse screen on `1 x 3` CV, then confirm the top few candidates on `2 x 5`
   - Focus around `k=3000..6000`, `alpha=0.005..0.03`, `l1_ratio=0.05..0.25`
   - Rationale: the earlier interrupted ElasticNet run was stronger than the Ridge baseline, but later lite runs were not directly comparable because CV changed

3. **Bagged / stability-selected Ridge**
   - Repeat feature ranking over bootstrap or fold resamples
   - Fit multiple filtered Ridge models and average predictions
   - Rationale: current solutions are sensitive to the chosen feature subset; averaging can reduce variance

4. **Residual modeling**
   - Fit the best linear baseline first
   - Fit a second small model on residuals using selected interaction features or low-dimensional latent factors
   - Rationale: the gender-interaction model appears useful mainly as a correction model

## Lower-Priority Directions

- Hard gender split models
- Pure PCA-based models
- Naive kernelization without careful structure

## Quick Experimental Guardrails

- Compare serious models only under the same final CV protocol before ranking them
- Save out-of-fold predictions for every serious candidate, because blending is likely the strongest path now
- Keep markdown records for every submission candidate and every leaderboard score
