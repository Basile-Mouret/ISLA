# Cross-Validation for This Challenge

## Why We Need It

In the challenge we know `(X_train, y_train)` but we do **not** know `y_test` for `X_test`.
That means we cannot directly compute the real test RMSE before submission.

So we build a **proxy** for hidden-test performance by repeatedly splitting the training set into:

- a temporary training subset
- a temporary validation subset

and measuring how well a model trained on the temporary training subset predicts the held-out validation subset.

This procedure is called **cross-validation** (CV).


## Notation

Let the training data be

```text
D = {(x_1, y_1), ..., (x_n, y_n)}
```

where:

- `x_i in R^p` is the feature vector for sample `i`
- `y_i in R` is the age for sample `i`
- here `n = 489`
- here `p = 10001`

We want to learn a prediction function

```text
f_hat : R^p -> R
```

from the training data.

For a regression model, the true quantity we care about is the **generalization risk** under squared loss:

```text
R(f_hat) = E[(Y - f_hat(X))^2]
```

If we want RMSE instead of MSE, we take the square root:

```text
RMSE(f_hat) = sqrt(R(f_hat))
```

The problem is that the expectation above is over the unknown test distribution, so we cannot compute it exactly.


## Simple Holdout Validation

The simplest idea is to split the data once into:

- train set `D_train`
- validation set `D_val`

Train on `D_train`, predict on `D_val`, then compute

```text
RMSE_holdout = sqrt((1 / |D_val|) * sum_{i in D_val} (y_i - y_hat_i)^2)
```

This is easy, but it has two weaknesses:

1. it wastes data, because only part of the dataset is used to fit the model
2. it is noisy, because the answer depends heavily on one particular split


## K-Fold Cross-Validation

In **K-fold CV**, we partition the index set `{1, ..., n}` into `K` disjoint subsets:

```text
I_1, I_2, ..., I_K
```

called **folds**, with approximately equal size.

For each fold `k`:

- training indices are all samples not in `I_k`
- validation indices are the samples in `I_k`

So we fit

```text
f_hat^(-k)
```

using all data except fold `k`, and evaluate on fold `k`.

The fold-level RMSE is

```text
RMSE_k = sqrt((1 / |I_k|) * sum_{i in I_k} (y_i - f_hat^(-k)(x_i))^2)
```

Then the usual reported CV score is the arithmetic mean of the fold RMSEs:

```text
CV_RMSE = (1 / K) * sum_{k=1}^K RMSE_k
```

Important subtlety:

- this is **not exactly the same** as computing one single RMSE from all out-of-fold residuals and then taking one square root
- the two are usually close, but mathematically they differ because the square root is nonlinear

The alternative pooled out-of-fold RMSE would be

```text
OOF_RMSE = sqrt((1 / n) * sum_{k=1}^K sum_{i in I_k} (y_i - f_hat^(-k)(x_i))^2)
```

Our current runner uses sklearn's fold-based scoring and averages fold scores.


## Repeated K-Fold Cross-Validation

One K-fold split can still be noisy, especially when `n` is small.

So we repeat the whole K-fold procedure `R` times with different splits.

Let `RMSE_{r,k}` be the RMSE from repetition `r` and fold `k`.

Then the repeated-CV estimate is

```text
Repeated_CV_RMSE = (1 / (R*K)) * sum_{r=1}^R sum_{k=1}^K RMSE_{r,k}
```

and we also compute the standard deviation across the `R*K` fold scores.

In our benchmark runs:

- `K = 5`
- `R = 2`
- so each candidate model is evaluated on `10` validation folds total


## What Stratification Means Here

In classification, stratification usually preserves class proportions.

For regression there are no classes, so we create **age bins**. In our code this is done by cutting the age values into quantile-based bins and stratifying on those bins.

That means each fold gets roughly similar proportions of:

- younger individuals
- middle-aged individuals
- older individuals

Mathematically, if `b_i` is the bin label for sample `i`, stratification tries to make

```text
P(b = j | i in I_k)
```

similar across folds `k`.

This reduces validation noise due to accidental age imbalance.


## What Our Runner Does Exactly

Our experiment runner currently uses:

1. age-bin stratification
2. repeated 5-fold CV
3. RMSE as the score
4. all preprocessing **inside** each fold

That last point is very important.

For a pipeline like:

- encode `gender`
- standardize methylation values
- select top `k` features with `f_regression`
- fit Ridge or ElasticNet

the feature selection and scaling are re-fit inside each training fold.

This avoids **data leakage**.


## Data Leakage: Why Fold-Local Preprocessing Matters

Suppose we selected the top `k` CpGs using the full dataset before CV. Then the validation fold would influence which features are chosen.

Formally, the selected feature set `S` would become a function of all labels:

```text
S = S(D)
```

instead of only the training part of fold `k`:

```text
S_k = S(D \ I_k)
```

That leaks target information from the validation fold into training.

The resulting CV estimate is optimistically biased, often substantially so in high-dimensional problems like this one.


## Why Cross-Validation Is Especially Important Here

This challenge has:

- `n = 489` samples
- `p = 10001` columns

So we are in a **high-dimensional** regime: `p >> n`.

In such settings:

- overfitting is easy
- hyperparameter choices matter a lot
- feature selection can look artificially good if leakage occurs
- one lucky train/validation split can be very misleading

Repeated CV is therefore much more trustworthy than a single holdout split.


## Bias and Variance of the CV Estimate

Cross-validation is still an **estimator**, and every estimator has bias and variance.

### Bias

In fold `k`, the model is trained on only about

```text
n_train = n * (K - 1) / K
```

samples.

For `K = 5` and `n = 489`, that is about

```text
489 * 4 / 5 = 391.2
```

training samples per fold.

But your final submission model is trained on all `489` training samples.

So CV often slightly **overestimates** the final model error, because the fold models are trained on less data than the final refit.

This pushes CV RMSE a bit upward.

### Variance

The exact folds matter. Different splits give slightly different answers.

Repeated CV reduces this variance by averaging over more fold configurations.

That is why we record both:

- mean CV RMSE
- fold-to-fold standard deviation


## Hyperparameter Tuning With CV

Suppose a model family has hyperparameters `lambda`.

For example, in filtered Ridge:

- `k` = number of retained CpG probes
- `alpha` = L2 regularization strength

For each candidate `lambda`, CV produces an estimated risk:

```text
CV(lambda)
```

We then choose

```text
lambda_hat = argmin_lambda CV(lambda)
```

This is model selection by empirical risk minimization over the hyperparameter grid.

In practice, if the best value sits on the edge of the grid, that often means the search range should be extended.


## Why Tuned CV Scores Can Still Be Optimistic

There is an extra subtlety: once you search many hyperparameter combinations and keep the minimum CV score, that minimum itself is a random variable selected for looking good.

So the reported best score

```text
min_lambda CV(lambda)
```

is usually a bit optimistic as an estimate of the performance of the **model-selection procedure**.

This is called **selection bias** or **winner's curse**.

The gold-standard fix is **nested CV**:

- outer CV estimates generalization
- inner CV tunes hyperparameters

But nested CV is much more expensive.

For practical competition work, repeated CV on a moderate grid is usually a reasonable compromise.


## Why The Public Leaderboard Can Differ From Local CV

The challenge public score is computed on the first `100` hidden test samples, not on our training folds.

So there are at least four reasons the public score can differ from local CV:

1. **Sampling noise**
   - the public set has only `100` points
   - RMSE on `100` points can fluctuate a lot

2. **Distribution shift**
   - the public subset may not match the training distribution perfectly

3. **Refitting on all training data**
   - submissions train on all `489` known labels, which can improve performance versus CV fold models trained on about `391` samples each

4. **Hyperparameter selection noise**
   - a model that looks best locally may not be best on the public subset just by chance


## Should Public RMSE Be Lower or Higher Than Local CV?

There is no deterministic rule.

### Why public RMSE can be lower

- the final model uses all `489` training samples instead of only `~391` per fold
- the public subset may happen to be easier than average

### Why public RMSE can be higher

- the public subset may be harder
- local tuning may overfit the CV procedure
- the public set is small and therefore noisy

### Practical expectation

Usually the public score is in the **same rough range** as repeated CV, not radically different.

So if local repeated CV is around `4.45`, I would not expect a public score of `1.2` unless:

- our local evaluation is badly mis-specified, or
- the leading submissions are using substantially stronger models or domain tricks we have not yet implemented


## Why Leaderboard Scores Can Be Much Better Than Ours

If top submissions are around `1.x` while we are around `4.4`, possibilities include:

1. they found a much stronger model family or feature-engineering trick
2. they are using stacking or blending effectively
3. the public subset is unusually easy and the private score may be less extreme
4. they used domain-specific epigenetic clock structure that we have not exploited yet

So the gap is not impossible, but we should treat it as evidence that there is probably still substantial modeling headroom.


## Challenge-Specific Numbers

For our current benchmark setup:

- `n = 489`
- `K = 5`
- `R = 2`
- each candidate gets `10` validation fold scores
- each fold trains on about `391` samples and validates on about `98` samples

For example, a model grid with `20` hyperparameter settings requires:

```text
20 * 10 = 200
```

fits.

That is exactly why some models become slow: each candidate is not one fit, but ten fits under repeated CV.


## Interpreting Our Current Results

When `ridge_select_raw` gets around `4.47`, that means:

- across repeated held-out validation folds
- training on roughly `80%` of the data each time
- the typical age prediction error is about `4.47` years in RMSE units

It does **not** mean the hidden challenge test RMSE will be exactly `4.47`.
It means `4.47` is our best current estimate of the model's out-of-sample performance.


## Bottom Line

- Cross-validation is our local stand-in for hidden-test evaluation.
- It is mathematically an estimator of out-of-sample error, not the hidden score itself.
- Repeated stratified K-fold CV is a good choice here because the dataset is small and high-dimensional.
- Public leaderboard scores can be a bit lower or higher than local CV, but usually not by several RMSE points without a genuinely better model.
- The right workflow is: improve local CV honestly, submit only strong candidates, and compare local estimates against leaderboard feedback without overfitting to it.
