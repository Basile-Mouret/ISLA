from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_regression
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from gender_models import (
    ClusterAugmentedRidgeRegressor,
    GenderInteractionRidgeRegressor,
    GenderSpecificRidgeRegressor,
)
from pipelines import build_final_model, ensure_target, make_cv_splits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--output-dir",
        default="results/gender_cluster_analysis",
        help="Directory for plots and markdown report.",
    )
    return parser.parse_args()


def resolve_training_paths(data_dir: Path) -> tuple[Path, Path]:
    direct_X = data_dir / "X_train.csv"
    direct_y = data_dir / "y_train.csv"
    nested_X = data_dir / "train" / "X_train.csv"
    nested_y = data_dir / "train" / "y_train.csv"

    if direct_X.exists() and direct_y.exists():
        return direct_X, direct_y
    if nested_X.exists() and nested_y.exists():
        return nested_X, nested_y
    raise FileNotFoundError(f"Could not find training data under {data_dir}")


def load_training_data(data_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    X_path, y_path = resolve_training_paths(data_dir)
    X = pd.read_csv(X_path)
    y = ensure_target(pd.read_csv(y_path))
    return X, y


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def log(message: str) -> None:
    print(message, flush=True)


def save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def numeric_columns(X: pd.DataFrame) -> list[str]:
    return [column for column in X.columns if column != "gender"]


def safe_corr_matrix(X_values: np.ndarray, y_values: np.ndarray) -> np.ndarray:
    X_centered = X_values - X_values.mean(axis=0, keepdims=True)
    y_centered = y_values - y_values.mean()
    numerator = X_centered.T @ y_centered
    denominator = np.sqrt((X_centered**2).sum(axis=0) * (y_centered**2).sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = numerator / denominator
    return np.nan_to_num(corr, nan=0.0)


def compute_gender_correlation_frame(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    feature_columns = numeric_columns(X)
    X_numeric = X[feature_columns].to_numpy()
    y_values = y.to_numpy()
    male_mask = X["gender"].eq("m").to_numpy()
    female_mask = X["gender"].eq("f").to_numpy()

    overall_corr = safe_corr_matrix(X_numeric, y_values)
    male_corr = safe_corr_matrix(X_numeric[male_mask], y_values[male_mask])
    female_corr = safe_corr_matrix(X_numeric[female_mask], y_values[female_mask])

    correlation_frame = pd.DataFrame(
        {
            "feature": feature_columns,
            "overall_corr": overall_corr,
            "male_corr": male_corr,
            "female_corr": female_corr,
        }
    )
    correlation_frame["abs_overall_corr"] = correlation_frame["overall_corr"].abs()
    correlation_frame["abs_male_corr"] = correlation_frame["male_corr"].abs()
    correlation_frame["abs_female_corr"] = correlation_frame["female_corr"].abs()
    correlation_frame["corr_gap"] = correlation_frame["male_corr"] - correlation_frame["female_corr"]
    correlation_frame["abs_corr_gap"] = correlation_frame["corr_gap"].abs()
    return correlation_frame


def plot_age_by_gender(X: pd.DataFrame, y: pd.Series, output_dir: Path) -> dict[str, float]:
    ages_by_gender = {gender: y[X["gender"] == gender].to_numpy() for gender in ["f", "m"]}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.arange(15, 76, 3)
    axes[0].hist(ages_by_gender["f"], bins=bins, alpha=0.65, label="female", color="#d62728")
    axes[0].hist(ages_by_gender["m"], bins=bins, alpha=0.65, label="male", color="#1f77b4")
    axes[0].set_title("Age Distribution by Gender")
    axes[0].set_xlabel("Age")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].boxplot(
        [ages_by_gender["f"], ages_by_gender["m"]],
        tick_labels=["female", "male"],
        patch_artist=True,
        boxprops={"facecolor": "#dddddd"},
    )
    axes[1].set_title("Age Spread by Gender")
    axes[1].set_ylabel("Age")
    save_figure(output_dir / "age_by_gender.png")

    return {
        "female_mean_age": float(ages_by_gender["f"].mean()),
        "male_mean_age": float(ages_by_gender["m"].mean()),
        "female_count": int(ages_by_gender["f"].shape[0]),
        "male_count": int(ages_by_gender["m"].shape[0]),
    }


def plot_correlation_diagnostics(correlation_frame: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(
        correlation_frame["female_corr"],
        correlation_frame["male_corr"],
        s=10,
        alpha=0.18,
        color="#4c78a8",
        linewidths=0,
    )
    lim = np.max(np.abs(correlation_frame[["female_corr", "male_corr"]].to_numpy()))
    axes[0].plot([-lim, lim], [-lim, lim], linestyle="--", color="black", linewidth=1)
    axes[0].set_title("Feature Correlation: Female vs Male")
    axes[0].set_xlabel("Female Pearson correlation with age")
    axes[0].set_ylabel("Male Pearson correlation with age")

    top_gap = correlation_frame.nlargest(20, "abs_corr_gap").sort_values("corr_gap")
    axes[1].barh(top_gap["feature"], top_gap["corr_gap"], color="#f28e2b")
    axes[1].set_title("Top 20 Gender Correlation Gaps")
    axes[1].set_xlabel("Male corr - Female corr")
    save_figure(output_dir / "correlation_diagnostics.png")

    top_overall = correlation_frame.nlargest(15, "abs_overall_corr")[
        ["feature", "overall_corr", "male_corr", "female_corr", "corr_gap"]
    ]
    top_gap_table = top_gap[["feature", "overall_corr", "male_corr", "female_corr", "corr_gap"]]
    return {
        "top_overall": top_overall.to_dict(orient="records"),
        "top_gender_gap": top_gap_table.to_dict(orient="records"),
        "same_sign_top_200": int(
            (
                np.sign(correlation_frame.nlargest(200, "abs_overall_corr")["male_corr"])
                == np.sign(correlation_frame.nlargest(200, "abs_overall_corr")["female_corr"])
            ).sum()
        ),
    }


def compute_pca_and_clusters(
    X: pd.DataFrame,
    y: pd.Series,
    correlation_frame: pd.DataFrame,
    output_dir: Path,
) -> tuple[dict[str, object], list[str]]:
    top_features = correlation_frame.nlargest(500, "abs_overall_corr")["feature"].tolist()
    X_top = X[top_features].to_numpy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_top)

    pca_2d = PCA(n_components=2, random_state=42)
    embedding_2d = pca_2d.fit_transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    gender_colors = np.where(X["gender"].eq("m"), "#1f77b4", "#d62728")
    axes[0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=gender_colors, s=24, alpha=0.8)
    axes[0].set_title("PCA of Top 500 Correlated CpGs")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")

    scatter = axes[1].scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=y.to_numpy(),
        cmap="viridis",
        s=24,
        alpha=0.8,
    )
    axes[1].set_title("Same PCA, Colored by Age")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    fig.colorbar(scatter, ax=axes[1], label="Age")
    save_figure(output_dir / "pca_overview.png")

    pca_cluster = PCA(n_components=min(10, X_scaled.shape[0] - 1, X_scaled.shape[1]), random_state=42)
    cluster_embedding = pca_cluster.fit_transform(X_scaled)
    cluster_metrics = []
    for k in range(2, 7):
        labels = KMeans(n_clusters=k, n_init="auto", random_state=42).fit_predict(cluster_embedding)
        cluster_metrics.append(
            {
                "n_clusters": k,
                "silhouette": float(silhouette_score(cluster_embedding, labels)),
                "inertia": float(KMeans(n_clusters=k, n_init="auto", random_state=42).fit(cluster_embedding).inertia_),
            }
        )

    cluster_metrics_frame = pd.DataFrame(cluster_metrics)
    best_k = int(cluster_metrics_frame.sort_values("silhouette", ascending=False).iloc[0]["n_clusters"])
    best_kmeans = KMeans(n_clusters=best_k, n_init="auto", random_state=42)
    best_labels = best_kmeans.fit_predict(cluster_embedding)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(cluster_metrics_frame["n_clusters"], cluster_metrics_frame["silhouette"], marker="o")
    axes[0].set_title("Cluster Silhouette by K")
    axes[0].set_xlabel("Number of clusters")
    axes[0].set_ylabel("Silhouette score")

    for gender_value, marker in [("f", "o"), ("m", "^")]:
        mask = X["gender"].eq(gender_value).to_numpy()
        axes[1].scatter(
            embedding_2d[mask, 0],
            embedding_2d[mask, 1],
            c=best_labels[mask],
            cmap="tab10",
            s=30,
            alpha=0.8,
            marker=marker,
            label=f"{gender_value} samples",
        )
    axes[1].set_title(f"PCA with KMeans Clusters (k={best_k})")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].legend()
    save_figure(output_dir / "cluster_diagnostics.png")

    cluster_summary = pd.DataFrame(
        {
            "cluster": best_labels,
            "gender": X["gender"].to_numpy(),
            "age": y.to_numpy(),
        }
    )
    cluster_rollup = (
        cluster_summary.groupby("cluster")
        .agg(
            sample_count=("age", "size"),
            mean_age=("age", "mean"),
            female_fraction=("gender", lambda values: float(np.mean(values == "f"))),
        )
        .reset_index()
    )

    return {
        "best_cluster_count": best_k,
        "cluster_metrics": cluster_metrics,
        "cluster_summary": cluster_rollup.to_dict(orient="records"),
        "explained_variance_ratio_2d": pca_2d.explained_variance_ratio_.tolist(),
    }, top_features


def evaluate_estimator_cv(
    estimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
    keep_oof: bool = False,
) -> dict[str, object]:
    fold_rmses = []
    oof_sum = np.zeros(X.shape[0], dtype=float) if keep_oof else None
    oof_count = np.zeros(X.shape[0], dtype=int) if keep_oof else None

    for fold_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        estimator_fold = clone(estimator)
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_valid = X.iloc[valid_idx].reset_index(drop=True)
        y_valid = y.iloc[valid_idx].to_numpy()

        estimator_fold.fit(X_train, y_train)
        predictions = np.asarray(estimator_fold.predict(X_valid)).reshape(-1)
        fold_rmses.append(rmse(y_valid, predictions))

        if keep_oof:
            oof_sum[valid_idx] += predictions
            oof_count[valid_idx] += 1

    result = {
        "rmse": float(np.mean(fold_rmses)),
        "rmse_std": float(np.std(fold_rmses)),
        "fold_rmses": fold_rmses,
    }
    if keep_oof:
        result["oof_predictions"] = oof_sum / np.maximum(oof_count, 1)
    return result


def search_family(
    family_name: str,
    candidates: list[tuple[str, dict[str, object], object]],
    X: pd.DataFrame,
    y: pd.Series,
    cv_splits,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    log(f"Searching {family_name} over {len(candidates)} candidates...")
    results = []
    for candidate_index, (label, params, estimator) in enumerate(candidates, start=1):
        log(f"  [{candidate_index}/{len(candidates)}] {label}")
        evaluation = evaluate_estimator_cv(estimator, X, y, cv_splits, keep_oof=False)
        results.append(
            {
                "family": family_name,
                "label": label,
                "params": params,
                "rmse": evaluation["rmse"],
                "rmse_std": evaluation["rmse_std"],
            }
        )
    ranked = sorted(results, key=lambda row: (row["rmse"], row["rmse_std"]))
    return ranked[0], ranked


def build_gender_split_candidates() -> list[tuple[str, dict[str, object], object]]:
    candidates = []
    for female_k in [3000, 3500, 5000]:
        for male_k in [500, 1000, 1500]:
            for alpha in [0.01, 0.1]:
                params = {
                    "female_k": female_k,
                    "male_k": male_k,
                    "female_alpha": alpha,
                    "male_alpha": alpha,
                }
                label = f"gender_split_fk{female_k}_mk{male_k}_a{alpha}"
                candidates.append((label, params, GenderSpecificRidgeRegressor(**params)))
    return candidates


def build_interaction_candidates() -> list[tuple[str, dict[str, object], object]]:
    candidates = []
    for interaction_k in [100, 300, 500]:
        for alpha in [0.01, 0.1, 1.0]:
            params = {
                "main_k": 3500,
                "interaction_k": interaction_k,
                "alpha": alpha,
            }
            label = f"gender_interaction_ik{interaction_k}_a{alpha}"
            candidates.append((label, params, GenderInteractionRidgeRegressor(**params)))
    return candidates


def build_cluster_candidates() -> list[tuple[str, dict[str, object], object]]:
    candidates = []
    for n_clusters in [2, 3, 4]:
        for alpha in [0.01, 0.1]:
            params = {
                "main_k": 3500,
                "alpha": alpha,
                "n_clusters": n_clusters,
                "pca_components": 20,
            }
            label = f"cluster_augmented_k{n_clusters}_a{alpha}"
            candidates.append((label, params, ClusterAugmentedRidgeRegressor(**params)))
    return candidates


def summarize_oof_predictions(
    model_name: str,
    y: pd.Series,
    predictions: np.ndarray,
    genders: pd.Series,
) -> dict[str, float | str]:
    residuals = predictions - y.to_numpy()
    result = {
        "model": model_name,
        "overall_rmse": rmse(y.to_numpy(), predictions),
        "female_rmse": rmse(y[genders == "f"].to_numpy(), predictions[genders == "f"]),
        "male_rmse": rmse(y[genders == "m"].to_numpy(), predictions[genders == "m"]),
        "female_bias": float(residuals[genders == "f"].mean()),
        "male_bias": float(residuals[genders == "m"].mean()),
    }
    return result


def plot_model_comparison(
    model_summaries: list[dict[str, object]],
    residual_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    model_names = [row["model"] for row in model_summaries]
    x = np.arange(len(model_names))
    width = 0.25
    axes[0].bar(x - width, [row["overall_rmse"] for row in model_summaries], width=width, label="overall")
    axes[0].bar(x, [row["female_rmse"] for row in model_summaries], width=width, label="female")
    axes[0].bar(x + width, [row["male_rmse"] for row in model_summaries], width=width, label="male")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names, rotation=20, ha="right")
    axes[0].set_ylabel("RMSE")
    axes[0].set_title("OOF RMSE by Model and Gender")
    axes[0].legend()

    plot_groups = [
        residual_frame.loc[
            (residual_frame["model"] == model_name) & (residual_frame["gender"] == gender_value),
            "residual",
        ].to_numpy()
        for model_name in model_names
        for gender_value in ["f", "m"]
    ]
    labels = [f"{model_name}\n{gender_value}" for model_name in model_names for gender_value in ["f", "m"]]
    axes[1].boxplot(
        plot_groups,
        tick_labels=labels,
        patch_artist=True,
        boxprops={"facecolor": "#dddddd"},
    )
    axes[1].axhline(0.0, linestyle="--", color="black", linewidth=1)
    axes[1].set_ylabel("Prediction residual (pred - age)")
    axes[1].set_title("Residual Distribution by Gender")
    save_figure(output_dir / "model_comparison.png")


def format_float(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict[str, object]], columns: list[tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, separator] + body)


def write_report(
    output_dir: Path,
    age_summary: dict[str, float],
    correlation_summary: dict[str, object],
    cluster_summary: dict[str, object],
    search_results: dict[str, list[dict[str, object]]],
    final_results: list[dict[str, object]],
    model_summaries: list[dict[str, object]],
    best_model_name: str,
) -> None:
    search_rows = []
    for family_name, rows in search_results.items():
        for row in rows[:3]:
            search_rows.append(
                {
                    "family": family_name,
                    "candidate": row["label"],
                    "rmse": row["rmse"],
                    "rmse_std": row["rmse_std"],
                    "params": json.dumps(row["params"], sort_keys=True),
                }
            )

    final_rows = []
    for row in final_results:
        final_rows.append(
            {
                "model": row["model"],
                "rmse": row["rmse"],
                "rmse_std": row["rmse_std"],
            }
        )

    gender_rows = []
    for row in model_summaries:
        gender_rows.append(
            {
                "model": row["model"],
                "overall_rmse": row["overall_rmse"],
                "female_rmse": row["female_rmse"],
                "male_rmse": row["male_rmse"],
                "female_bias": row["female_bias"],
                "male_bias": row["male_bias"],
            }
        )

    report = f"""# Gender and Cluster Analysis

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

- Female samples: `{age_summary['female_count']}`
- Male samples: `{age_summary['male_count']}`
- Mean female age: `{format_float(age_summary['female_mean_age'], 3)}`
- Mean male age: `{format_float(age_summary['male_mean_age'], 3)}`

## Correlation Findings

- Among the top 200 age-correlated CpGs overall, `{correlation_summary['same_sign_top_200']}` keep the same correlation sign in both genders
- That suggests the main age signal is largely shared, but the strength of some CpGs differs by gender

Top overall correlated features:

{markdown_table(correlation_summary['top_overall'][:10], [('feature', 'Feature'), ('overall_corr', 'Overall Corr'), ('male_corr', 'Male Corr'), ('female_corr', 'Female Corr'), ('corr_gap', 'Gap')])}

Top gender-gap features:

{markdown_table(correlation_summary['top_gender_gap'][:10], [('feature', 'Feature'), ('overall_corr', 'Overall Corr'), ('male_corr', 'Male Corr'), ('female_corr', 'Female Corr'), ('corr_gap', 'Gap')])}

## Cluster Findings

- Best unsupervised cluster count by silhouette on the PCA representation: `{cluster_summary['best_cluster_count']}`
- The cluster summary table below helps check whether clusters mainly reflect age structure, gender structure, or both

{markdown_table(cluster_summary['cluster_summary'], [('cluster', 'Cluster'), ('sample_count', 'Samples'), ('mean_age', 'Mean Age'), ('female_fraction', 'Female Fraction')])}

## Candidate Search Results

These are the top quick-screen candidates from each gender-aware family.

{markdown_table(search_rows, [('family', 'Family'), ('candidate', 'Candidate'), ('rmse', 'Search RMSE'), ('rmse_std', 'Search Std'), ('params', 'Params')])}

## Final 2x5 CV Comparison

{markdown_table(final_rows, [('model', 'Model'), ('rmse', 'Final CV RMSE'), ('rmse_std', 'Final CV Std')])}

## RMSE by Gender

{markdown_table(gender_rows, [('model', 'Model'), ('overall_rmse', 'Overall RMSE'), ('female_rmse', 'Female RMSE'), ('male_rmse', 'Male RMSE'), ('female_bias', 'Female Bias'), ('male_bias', 'Male Bias')])}

## Interpretation

- The shared linear age signal is strong across genders, but some CpGs change slope strength by sex
- If a gender-aware model beats the global Ridge baseline, the likely mechanism is not a completely different feature set by sex, but modestly different weighting or sex-specific residual correction
- The best model from this analysis is `{best_model_name}` under the final repeated-CV comparison
- If the improvement over the global Ridge baseline is tiny, that means gender structure is real but not large enough to materially move leaderboard performance on its own
"""

    (output_dir / "report.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_training_data(Path(args.data_dir))
    genders = X["gender"]

    age_summary = plot_age_by_gender(X, y, output_dir)
    correlation_frame = compute_gender_correlation_frame(X, y)
    correlation_frame.to_csv(output_dir / "correlations.csv", index=False)
    correlation_summary = plot_correlation_diagnostics(correlation_frame, output_dir)
    cluster_summary, _ = compute_pca_and_clusters(X, y, correlation_frame, output_dir)

    quick_cv = make_cv_splits(y, n_splits=5, n_repeats=1)
    final_cv = make_cv_splits(y, n_splits=5, n_repeats=2)

    search_results = {}

    gender_split_best, gender_split_ranked = search_family(
        "gender_split_ridge",
        build_gender_split_candidates(),
        X,
        y,
        quick_cv,
    )
    search_results["gender_split_ridge"] = gender_split_ranked

    interaction_best, interaction_ranked = search_family(
        "gender_interaction_ridge",
        build_interaction_candidates(),
        X,
        y,
        quick_cv,
    )
    search_results["gender_interaction_ridge"] = interaction_ranked

    cluster_best, cluster_ranked = search_family(
        "cluster_augmented_ridge",
        build_cluster_candidates(),
        X,
        y,
        quick_cv,
    )
    search_results["cluster_augmented_ridge"] = cluster_ranked

    final_model_specs = [
        ("global_ridge_best", build_final_model(list(X.columns))),
        (
            gender_split_best["label"],
            GenderSpecificRidgeRegressor(**gender_split_best["params"]),
        ),
        (
            interaction_best["label"],
            GenderInteractionRidgeRegressor(**interaction_best["params"]),
        ),
        (
            cluster_best["label"],
            ClusterAugmentedRidgeRegressor(**cluster_best["params"]),
        ),
    ]

    final_results = []
    model_summaries = []
    residual_rows = []
    best_model_name = None
    best_rmse = float("inf")

    for model_name, estimator in final_model_specs:
        log(f"Final evaluation: {model_name}")
        evaluation = evaluate_estimator_cv(estimator, X, y, final_cv, keep_oof=True)
        final_results.append(
            {
                "model": model_name,
                "rmse": evaluation["rmse"],
                "rmse_std": evaluation["rmse_std"],
            }
        )
        summary = summarize_oof_predictions(model_name, y, evaluation["oof_predictions"], genders)
        model_summaries.append(summary)

        residuals = evaluation["oof_predictions"] - y.to_numpy()
        for gender_value in ["f", "m"]:
            mask = genders.eq(gender_value).to_numpy()
            for value in residuals[mask]:
                residual_rows.append(
                    {"model": model_name, "gender": gender_value, "residual": float(value)}
                )

        if evaluation["rmse"] < best_rmse:
            best_rmse = evaluation["rmse"]
            best_model_name = model_name

    final_results = sorted(final_results, key=lambda row: (row["rmse"], row["rmse_std"]))
    model_summaries = sorted(model_summaries, key=lambda row: row["overall_rmse"])
    residual_frame = pd.DataFrame(residual_rows)
    plot_model_comparison(model_summaries, residual_frame, output_dir)

    write_report(
        output_dir=output_dir,
        age_summary=age_summary,
        correlation_summary=correlation_summary,
        cluster_summary=cluster_summary,
        search_results=search_results,
        final_results=final_results,
        model_summaries=model_summaries,
        best_model_name=best_model_name,
    )

    summary_payload = {
        "age_summary": age_summary,
        "cluster_summary": cluster_summary,
        "search_results": search_results,
        "final_results": final_results,
        "model_summaries": model_summaries,
        "best_model_name": best_model_name,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    log(f"Wrote gender and cluster analysis to {output_dir}")


if __name__ == "__main__":
    main()
