# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
#
# # Lab 1: Social_Network_Ads (Logistic Regression)
#
# We follow the slides starting at `Lab 1` and build the requested 2D classifier using `Age` and `EstimatedSalary` to predict `Purchased`.
#
# \[
# \Pr(Y=1\mid x)=\sigma(eta_0+eta_1\,	ext{Age}+eta_2\,	ext{EstimatedSalary}),
# \qquad
# \sigma(z)=
# rac{1}{1+e^{-z}}.
# \]
#

# %%

from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update({
    "figure.figsize": (8, 5.5),
    "axes.spines.top": False,
    "axes.spines.right": False,
})

DATA_PATH = Path("Dataset1.csv")
point_palette = {"No purchase": "#4C78A8", "Purchase": "#E45756"}
probability_cmap = LinearSegmentedColormap.from_list(
    "purchase_probability",
    ["#EEF4FF", "#F9F6F0", "#FFD9CC"],
)
salary_formatter = FuncFormatter(lambda value, _: f"${value/1000:.0f}k")


def plot_feature_space(ax, frame, model=None, title=""):
    plot_frame = frame.copy()
    plot_frame["PurchasedLabel"] = plot_frame["Purchased"].map({0: "No purchase", 1: "Purchase"})

    if model is not None:
        age = np.linspace(plot_frame["Age"].min() - 2, plot_frame["Age"].max() + 2, 300)
        salary = np.linspace(
            plot_frame["EstimatedSalary"].min() - 5000,
            plot_frame["EstimatedSalary"].max() + 5000,
            300,
        )
        age_grid, salary_grid = np.meshgrid(age, salary)
        grid = pd.DataFrame(
            {"Age": age_grid.ravel(), "EstimatedSalary": salary_grid.ravel()}
        )
        proba_grid = model.predict_proba(grid)[:, 1].reshape(age_grid.shape)
        ax.contourf(
            age_grid,
            salary_grid,
            proba_grid,
            levels=np.linspace(0, 1, 11),
            cmap=probability_cmap,
            alpha=0.55,
        )
        ax.contour(
            age_grid,
            salary_grid,
            proba_grid,
            levels=[0.5],
            colors="#1f1f1f",
            linewidths=2,
            linestyles="--",
        )

    sns.scatterplot(
        data=plot_frame,
        x="Age",
        y="EstimatedSalary",
        hue="PurchasedLabel",
        hue_order=["No purchase", "Purchase"],
        palette=point_palette,
        s=70,
        alpha=0.9,
        linewidth=0.6,
        edgecolor="white",
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Estimated salary")
    ax.yaxis.set_major_formatter(salary_formatter)
    ax.legend(title="Purchased", frameon=True)


def plot_confusion(ax, cm, title):
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=sns.light_palette("#4C78A8", as_cmap=True),
        cbar=False,
        square=True,
        linewidths=1,
        linecolor="white",
        ax=ax,
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("Actual class")
    ax.set_xticklabels([0, 1])
    ax.set_yticklabels([0, 1], rotation=0)



# %% [markdown]
#
# ## Step 1 - Load & Inspect the Dataset
#

# %%

df = pd.read_csv(DATA_PATH)
features = ["Age", "EstimatedSalary"]
target = "Purchased"

X = df[features]
y = df[target]

overview = pd.DataFrame(
    {
        "Value": [
            df.shape[0],
            df.shape[1],
            len(features),
            int(y.value_counts().loc[0]),
            int(y.value_counts().loc[1]),
            round(y.mean(), 4),
            int(df.isna().sum().sum()),
        ]
    },
    index=[
        "Observations",
        "Variables in CSV",
        "Predictors used",
        "Class 0 count",
        "Class 1 count",
        "Purchase rate",
        "Missing values",
    ],
)

structure = pd.DataFrame({"dtype": df.dtypes.astype(str)}).T
missing = pd.DataFrame(df.isna().sum()).T
missing.index = ["Missing values"]

display(df.head())
display(structure)
display(overview)
display(missing)


# %% [markdown]
#
# **Short answers**
#
# - The CSV contains **400 observations** and **5 variables**; the model uses the 2 requested predictors: `Age` and `EstimatedSalary`.
# - The target is **not balanced**: $\hat P(Y=1)=143/400=0.3575$ and $\hat P(Y=0)=257/400=0.6425$.
# - There are **no missing values** in any column.
#

# %% [markdown]
#
# ## Step 2 - Visualize the Classes (Separability?)
#

# %%

fig, ax = plt.subplots(figsize=(8.4, 6))
plot_feature_space(ax, df[features + [target]], title="Feature space: Age vs EstimatedSalary")
plt.tight_layout()
plt.show()


# %% [markdown]
#
# **Short answers**
#
# - The classes are **not perfectly linearly separable**: the purchase class concentrates in the older / higher-salary region, but the two groups still overlap.
# - A linear decision boundary has the form $eta_0 + eta_1\,	ext{Age} + eta_2\,	ext{EstimatedSalary} = 0$, so it is a **straight line** in the feature plane. With positive fitted coefficients, that line has a **negative slope**.
#

# %% [markdown]
#
# ## Step 3 - Fit Logistic Regression (Train/Test)
#

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

clf = LogisticRegression(max_iter=5000)
clf.fit(X_train, y_train)

beta0 = clf.intercept_[0]
beta_age, beta_salary = clf.coef_[0]

split_sizes = pd.DataFrame({"Set": ["Train", "Test"], "Size": [len(X_train), len(X_test)]})
coef_table = pd.DataFrame(
    {
        "Coefficient": [beta_age, beta_salary],
        "Sign": [
            "positive" if beta_age > 0 else "negative",
            "positive" if beta_salary > 0 else "negative",
        ],
        "Interpretation step": ["+1 year", "+$10k salary"],
        "Odds multiplier for step": [np.exp(beta_age), np.exp(beta_salary * 10000)],
    },
    index=["Age", "EstimatedSalary"],
)

display(split_sizes)
display(coef_table.round({"Coefficient": 6, "Odds multiplier for step": 3}))

train_frame = X_train.copy()
train_frame["Purchased"] = y_train.values

fig, ax = plt.subplots(figsize=(8.4, 6))
plot_feature_space(ax, train_frame, model=clf, title="Training data and fitted logistic boundary")
plt.tight_layout()
plt.show()


# %% [markdown]
#
# **Short answers**
#
# - Both fitted coefficients are **positive**, so increasing either predictor increases the **log-odds** of purchase.
# - Therefore, **both predictors increase the odds of purchase**. Interpreted on practical scales, the fitted model gives about a **$1.249	imes$** change in odds for **+1 year of age** and about a **$1.458	imes$** change in odds for **+$10k of salary**.
# - The raw coefficient magnitudes should not be compared directly because the predictors are measured in different units.
#

# %% [markdown]
#
# ## Step 4 - Predictions and Model Evaluation
#

# %%

y_proba = clf.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)

metrics = pd.DataFrame(
    {
        "Metric": ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"],
        "Value": [
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred),
            roc_auc_score(y_test, y_proba),
        ],
    }
)

display(metrics.round(3))

cm = confusion_matrix(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, y_proba)
auc = roc_auc_score(y_test, y_proba)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
plot_confusion(axes[0], cm, r"Confusion matrix ($\tau=0.5$)")

axes[1].plot(fpr, tpr, color="#E45756", linewidth=2.5, label=f"AUC = {auc:.3f}")
axes[1].plot([0, 1], [0, 1], linestyle="--", color="#9AA0A6", linewidth=1.5)
axes[1].set_title("ROC curve", fontsize=13, fontweight="bold", pad=10)
axes[1].set_xlabel("False positive rate")
axes[1].set_ylabel("True positive rate")
axes[1].legend(frameon=True)
axes[1].set_aspect("equal", adjustable="box")

plt.tight_layout()
plt.show()


# %% [markdown]
#
# **Short answers**
#
# - Predicted probabilities are the fitted values $\hat p(x)=\Pr(Y=1\mid X=x)$, while predicted classes apply the rule $\hat y = \mathbf{1}\{\hat p(x) \ge 0.5\}$.
# - On the test set, the model reaches **84% accuracy**.
# - Accuracy alone is **not sufficient** because it depends on the threshold and hides the false-positive / false-negative trade-off. The confusion matrix, precision, recall, F1-score, and ROC AUC give a fuller picture.
#

# %% [markdown]
#
# ## Step 5 - Effect of the Classification Threshold
#

# %%

thresholds = [0.3, 0.5, 0.7]
rows = []
cms = {}

for tau in thresholds:
    pred_tau = (y_proba >= tau).astype(int)
    cm_tau = confusion_matrix(y_test, pred_tau)
    tn, fp, fn, tp = cm_tau.ravel()
    cms[tau] = cm_tau
    rows.append(
        {
            "Threshold": tau,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp,
            "Accuracy": accuracy_score(y_test, pred_tau),
            "Precision": precision_score(y_test, pred_tau, zero_division=0),
            "Recall": recall_score(y_test, pred_tau),
        }
    )

threshold_df = pd.DataFrame(rows)
display(threshold_df.round(3))

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4))
for ax, tau in zip(axes, thresholds):
    plot_confusion(ax, cms[tau], fr"$\tau = {tau}$")
plt.tight_layout()
plt.show()


# %% [markdown]
#
# **Short answers**
#
# - Lowering the threshold to $	au=0.3$ labels more users as purchasers: **false positives rise** from **4** to **16**, while **false negatives fall** from **12** to **6**. So recall improves, but precision drops.
# - Raising the threshold to $	au=0.7$ does the opposite: **false positives fall** from **4** to **2**, while **false negatives rise** from **12** to **20**. So precision improves slightly, but recall drops sharply.
# - In short: **lower thresholds favor recall**, whereas **higher thresholds favor precision**.
#

# %% [markdown]
#
# ## Step 6 - Regularization
#
# The slides continue with a regularization extension. For a fair L1/L2 comparison, the predictors are standardized first because the penalty is applied directly to coefficient size.
#

# %%

C_values = [10, 1, 0.1, 0.01]
records = []

for penalty in ["l2", "l1"]:
    for C in C_values:
        kwargs = {"penalty": penalty, "C": C, "max_iter": 5000}
        if penalty == "l1":
            kwargs["solver"] = "liblinear"

        regularized_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(**kwargs)),
            ]
        )
        regularized_model.fit(X_train, y_train)

        beta_age_scaled, beta_salary_scaled = regularized_model.named_steps["logit"].coef_[0]
        records.append(
            {
                "Penalty": penalty.upper(),
                "C": C,
                "Age coef (scaled)": beta_age_scaled,
                "Salary coef (scaled)": beta_salary_scaled,
                "Test accuracy": accuracy_score(y_test, regularized_model.predict(X_test)),
            }
        )

reg_df = pd.DataFrame(records)
display(reg_df.round(3))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, penalty in zip(axes, ["L2", "L1"]):
    subset = reg_df[reg_df["Penalty"] == penalty].sort_values("C", ascending=False)
    ax.plot(subset["C"], subset["Age coef (scaled)"], marker="o", linewidth=2, color="#4C78A8", label="Age")
    ax.plot(subset["C"], subset["Salary coef (scaled)"], marker="o", linewidth=2, color="#E45756", label="EstimatedSalary")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_title(f"{penalty} regularization", fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel("C (decreases left to right)")
    ax.grid(True, which="both", axis="x", alpha=0.25)
axes[0].set_ylabel("Standardized coefficient")
axes[1].legend(frameon=True)

plt.tight_layout()
plt.show()


# %% [markdown]
#
# **Short answers**
#
# - If **interpretability** is the priority, choose **L1 regularization** because it can set some coefficients exactly to zero and produce a sparse model.
# - If **stability** is the priority, choose **L2 regularization** because it shrinks coefficients smoothly instead of making abrupt variable-selection decisions.
# - As $C$ decreases, regularization becomes stronger and the coefficients shrink toward zero. Under **L1**, that shrinkage is more aggressive and can drive coefficients to **exactly zero**.
# - In this split, test accuracy stays close to **0.84** under mild regularization, reaches **0.86** for **L1 with $C=0.1$**, and drops sharply for very strong regularization (**0.64** at **L1 with $C=0.01$**), which is a clear sign of underfitting.
#

# %% [markdown]
#
# ## Takeaway
#
# With only two predictors, logistic regression gives an interpretable linear boundary, a strong ranking score (**ROC AUC $pprox 0.91$**), and a clear precision-recall trade-off controlled by the classification threshold.
#
