"""Dimensionality Reduction and Clustering verification helper module.

Enforces pre-scaling verification, PCA variance conservation, K-Means partition
completeness, and visual artifact generation across output directories.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def standardize_features(
    data: pd.DataFrame,
    feature_columns: Sequence[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize feature matrix and assert zero mean and unit variance."""
    if data.empty:
        raise ValueError("Cannot standardize an empty dataset.")

    missing_cols = [c for c in feature_columns if c not in data.columns]
    if missing_cols:
        raise KeyError(f"Feature columns not found: {missing_cols}")

    feature_matrix = data[list(feature_columns)]
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(feature_matrix)

    scaled_dataframe = pd.DataFrame(
        scaled_values,
        columns=list(feature_columns),
        index=data.index,
    )

    # Invariant assertions
    mean_discrepancy = scaled_dataframe.mean().abs().max()
    std_discrepancy = (scaled_dataframe.std(ddof=0) - 1.0).abs().max()

    assert mean_discrepancy < 1e-6, (
        f"Standardization invariant violated: max mean offset is {mean_discrepancy:.2e}"
    )
    assert std_discrepancy < 1e-6, (
        f"Standardization invariant violated: max std offset is {std_discrepancy:.2e}"
    )

    print(
        f"[Scaling] Successfully standardized {len(feature_columns)} features "
        f"across {len(data):,} rows."
    )
    return scaled_dataframe, scaler


def compute_pca_decomposition(
    scaled_data: pd.DataFrame,
    variance_threshold: float = 0.80,
) -> tuple[PCA, pd.DataFrame, int]:
    """Decompose features via PCA, assert variance conservation, and identify cutoff."""
    if scaled_data.empty:
        raise ValueError("Cannot perform PCA on an empty dataset.")

    pca_model = PCA()
    projected_values = pca_model.fit_transform(scaled_data)

    variance_ratios = pca_model.explained_variance_ratio_
    cumulative_variance = variance_ratios.cumsum()

    # Mathematical Invariant: sum of explained variance ratios must equal 1.0
    total_variance_sum = variance_ratios.sum()
    assert abs(total_variance_sum - 1.0) < 1e-6, (
        f"Variance conservation violated: total variance sum is {total_variance_sum:.6f}"
    )

    cutoff_index = int((cumulative_variance >= variance_threshold).argmax() + 1)

    delimiter = "=" * 60
    print(f"\n{delimiter}\nPCA EXPLAINED VARIANCE PROFILE\n{delimiter}")
    for idx, (individual, cumulative) in enumerate(
        zip(variance_ratios, cumulative_variance),
        start=1,
    ):
        annotation = (
            f" <-- [{variance_threshold:.0%} Cutoff Reached]"
            if idx == cutoff_index
            else ""
        )
        print(
            f"  PC{idx:02d}: Individual = {individual:6.2%} | "
            f"Cumulative = {cumulative:6.2%}{annotation}"
        )
    print(f"{delimiter}\n")

    component_columns = [
        f"PC{idx}" for idx in range(1, scaled_data.shape[1] + 1)
    ]
    projected_dataframe = pd.DataFrame(
        projected_values,
        columns=component_columns,
        index=scaled_data.index,
    )

    return pca_model, projected_dataframe, cutoff_index


def fit_kmeans_partition(
    data: pd.DataFrame,
    cluster_count: int = 3,
    random_state: int = 42,
) -> tuple[KMeans, pd.Series, float]:
    """Fit K-Means clustering, verify partition invariants, and report silhouette."""
    if cluster_count <= 1:
        raise ValueError("cluster_count must be at least 2.")

    kmeans_model = KMeans(
        n_clusters=cluster_count,
        random_state=random_state,
        n_init=10,
    )
    cluster_array = kmeans_model.fit_predict(data)
    cluster_labels = pd.Series(
        cluster_array,
        name="cluster",
        index=data.index,
    )

    # Invariant assertions: non-empty clusters and complete partition
    assert len(cluster_labels) == len(data), "Cluster partition length mismatch."
    assert cluster_labels.nunique() == cluster_count, (
        f"Degenerate cluster detected: expected {cluster_count} unique clusters, "
        f"got {cluster_labels.nunique()}."
    )
    assert (cluster_labels.value_counts() > 0).all(), (
        "Empty cluster detected in partition."
    )

    silhouette_metric = float(silhouette_score(data, cluster_labels))

    print(
        f"[Clustering] Fitted K={cluster_count} clusters. "
        f"Average Silhouette Score: {silhouette_metric:.4f}"
    )
    print("Cluster Membership Summary:")
    for cluster_id, count in cluster_labels.value_counts().sort_index().items():
        proportion = (count / len(data)) * 100
        print(f"  - Cluster {cluster_id}: {count:,} samples ({proportion:.2f}%)")
    print()

    return kmeans_model, cluster_labels, silhouette_metric


def save_clustering_visuals(
    pca_model: PCA,
    projected_data: pd.DataFrame,
    cluster_labels: pd.Series,
    output_directory: str = "output",
    filename_prefix: str = "clustering",
) -> pathlib.Path:
    """Generate dual-panel Scree plot and 2D PCA cluster projection figure."""
    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    variance_ratios = pca_model.explained_variance_ratio_
    cumulative_variance = variance_ratios.cumsum()
    components = list(range(1, len(variance_ratios) + 1))

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Scree plot with cumulative step line
    axes[0].bar(
        components,
        variance_ratios,
        alpha=0.65,
        color="#2b5c8f",
        label="Individual Variance",
    )
    axes[0].step(
        components,
        cumulative_variance,
        where="mid",
        color="#d95f02",
        linewidth=2.2,
        label="Cumulative Variance",
    )
    axes[0].axhline(
        0.80,
        color="#7570b3",
        linestyle="--",
        linewidth=1.5,
        label="80% Cutoff",
    )
    axes[0].set_title("PCA Scree Plot", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Principal Component", fontsize=10)
    axes[0].set_ylabel("Explained Variance Ratio", fontsize=10)
    axes[0].set_xticks(components)
    axes[0].legend(loc="center right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: 2D projection scatter colored by cluster
    sns.scatterplot(
        x=projected_data["PC1"],
        y=projected_data["PC2"],
        hue=cluster_labels.astype(str),
        palette="viridis",
        ax=axes[1],
        alpha=0.85,
        s=45,
    )
    axes[1].set_title(
        "2D PCA Projection by Cluster",
        fontsize=11,
        fontweight="bold",
    )
    axes[1].set_xlabel(
        f"PC1 ({variance_ratios[0]:.1%} variance)",
        fontsize=10,
    )
    axes[1].set_ylabel(
        f"PC2 ({variance_ratios[1]:.1%} variance)",
        fontsize=10,
    )
    axes[1].legend(title="Cluster", loc="best")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    artifact_path = output_path / f"{filename_prefix}_dashboard.png"
    figure.savefig(artifact_path, dpi=300)
    plt.close(figure)

    print(f"[Visual Saved] Artifact written to: {artifact_path.resolve()}")
    return artifact_path


if __name__ == "__main__":
    # Self-test demonstration with synthetic multi-feature clustered data
    print("Initiating Dimensionality Reduction & Clustering Verification Self-Test...")

    features_array, true_labels = make_blobs(
        n_samples=400,
        n_features=6,
        centers=3,
        cluster_std=1.2,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(1, 7)]
    dataset = pd.DataFrame(features_array, columns=feature_names)

    # Stage 1: Standardize and verify invariants
    scaled_df, fitted_scaler = standardize_features(dataset, feature_names)

    # Stage 2: PCA decomposition and variance conservation check
    fitted_pca, projected_df, cutoff = compute_pca_decomposition(
        scaled_df,
        variance_threshold=0.80,
    )

    # Stage 3: K-Means clustering and partition completeness check
    kmeans_engine, assigned_clusters, silhouette = fit_kmeans_partition(
        projected_df[["PC1", "PC2"]],
        cluster_count=3,
    )

    # Stage 4: Visual artifact generation
    saved_artifact = save_clustering_visuals(
        pca_model=fitted_pca,
        projected_data=projected_df,
        cluster_labels=assigned_clusters,
        output_directory="output",
        filename_prefix="test_dim_reduction",
    )

    print(
        f"Self-test successfully verified. Output image written: {saved_artifact}"
    )
