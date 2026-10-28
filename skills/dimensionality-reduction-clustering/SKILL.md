---
name: dimensionality-reduction-clustering
description: >-
  Performs Principal Component Analysis (PCA), Scree plot validation, K-Means clustering,
  and silhouette verification in Python or R with strict scaling checks, variance conservation
  assertions, and saved visual artifacts.
---

# Dimensionality Reduction & Clustering Verification Skill (Polyglot)

Use this skill whenever reducing feature spaces (e.g., PCA, t-SNE) or identifying natural groupings (e.g., K-Means, hierarchical clustering). This workflow enforces standardization pre-checks, variance conservation invariants, cluster partition completeness, and visual proof across **Python** and **R**.

---

## Universal Workflow

```
[ Stage 1: Feature Standardization & Scaling Verification ]
                           │
                           ▼
[ Stage 2: Principal Component Analysis & Scree Invariant Check ]
                           │
                           ▼
[ Stage 3: K-Means Clustering & Partition Integrity Assertions ]
                           │
                           ▼
[ Stage 4: Dual Verification (Terminal Tables & Saved Visuals) ]
                           │
                           ▼
[ Stage 5: Execution & Zero-Error Check ]
```

---

## Reference Implementation Scripts

- **Python Module**: [`scripts/dim_reduction_cluster.py`](./scripts/dim_reduction_cluster.py)
- **R Module**: [`scripts/dim_reduction_cluster.R`](./scripts/dim_reduction_cluster.R)

---

## Mathematical Invariants & Rules of Thumb

1. **Standardization Precondition**:
   Before PCA or distance-based clustering, all continuous features must be standardized to zero mean and unit variance:
   $$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
   *Invariant Assertion*: $\text{mean}(\tilde{x}_j) \approx 0.0$ and $\text{std}(\tilde{x}_j) \approx 1.0$ for all columns $j$.

2. **Variance Conservation**:
   The individual explained variance ratios $r_k$ across all $p$ components must sum to 1.0:
   $$\sum_{k=1}^p r_k = 1.0 \pm 10^{-6}$$
   Identify the cutoff index $K$ where cumulative explained variance exceeds the threshold $\tau$ (typically 80% or 90%).

3. **Cluster Partition Invariants**:
   - Every cluster must be non-empty: $|C_k| > 0$ for all $k \in \{1, \dots, K\}$.
   - No data points lost: $\sum_{k=1}^K |C_k| = n$.
   - Clusters are disjoint: $C_j \cap C_k = \emptyset$.

---

## Language Implementation Patterns

### Pattern A: Python (`scikit-learn`, `pandas`, `matplotlib`, `seaborn`)

#### 1. Standardization & Invariant Check
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_and_verify(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, StandardScaler]:
    """Standardize features and verify zero mean and unit variance."""
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(df[feature_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=feature_cols)
    
    # Invariant assertions
    assert (scaled_df.mean().abs() < 1e-6).all(), "Standardization failed: means are not zero"
    assert ((scaled_df.std(ddof=0) - 1.0).abs() < 1e-6).all(), "Standardization failed: std is not 1.0"
    print(f"[Scaling] Standardized {len(feature_cols)} features across {len(df):,} rows.")
    return scaled_df, scaler
```

#### 2. PCA & Scree Invariant Verification
```python
from sklearn.decomposition import PCA

def run_pca_with_invariants(scaled_df: pd.DataFrame, variance_threshold: float = 0.80) -> tuple[PCA, pd.DataFrame]:
    """Fit PCA, verify variance conservation, and identify optimal component cutoff."""
    pca = PCA()
    projected_features = pca.fit_transform(scaled_df)
    
    ratios = pca.explained_variance_ratio_
    cumulative = ratios.cumsum()
    
    # Mathematical Invariant
    assert abs(ratios.sum() - 1.0) < 1e-6, "Variance conservation invariant violated!"
    
    cutoff_components = int((cumulative >= variance_threshold).argmax() + 1)
    
    print("\n" + "=" * 50)
    print("PCA EXPLAINED VARIANCE TABLE")
    print("=" * 50)
    for i, (r, c) in enumerate(zip(ratios, cumulative), start=1):
        marker = " <-- [80% Threshold Met]" if i == cutoff_components else ""
        print(f"PC{i:02d}: Individual = {r:6.2%} | Cumulative = {c:6.2%}{marker}")
    print("=" * 50 + "\n")
    
    projected_df = pd.DataFrame(
        projected_features,
        columns=[f"PC{i}" for i in range(1, scaled_df.shape[1] + 1)]
    )
    return pca, projected_df
```

#### 3. K-Means Clustering & Partition Verification
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def run_kmeans_clustering(data: pd.DataFrame, n_clusters: int = 3, random_state: int = 42) -> tuple[KMeans, pd.Series]:
    """Fit K-Means, verify partition completeness, and calculate silhouette."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = pd.Series(kmeans.fit_predict(data), name="cluster")
    
    # Invariants
    assert len(cluster_labels) == len(data), "Cluster partition length mismatch!"
    assert cluster_labels.nunique() == n_clusters, "Degenerate cluster detected!"
    assert (cluster_labels.value_counts() > 0).all(), "Empty cluster detected!"
    
    score = silhouette_score(data, cluster_labels)
    print(f"\n[Clustering] K={n_clusters} fitted. Average Silhouette Score: {score:.4f}")
    print("Cluster Sizes:")
    for cluster_id, count in cluster_labels.value_counts().sort_index().items():
        print(f"  - Cluster {cluster_id}: {count:,} ({count / len(data):.1%})")
    
    return kmeans, cluster_labels
```

#### 4. Save Visual Artifacts to `output/`
```python
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

def save_dim_reduction_plots(
    pca_model: PCA,
    projected_df: pd.DataFrame,
    cluster_labels: pd.Series,
    output_dir: str = "output"
) -> pathlib.Path:
    """Save Scree plot and 2D PCA cluster projection to output folder."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree Plot
    ratios = pca_model.explained_variance_ratio_
    cumulative = ratios.cumsum()
    components = range(1, len(ratios) + 1)
    
    axes[0].bar(components, ratios, alpha=0.6, color="#2b5c8f", label="Individual Variance")
    axes[0].step(components, cumulative, where="mid", color="#d95f02", linewidth=2, label="Cumulative")
    axes[0].axhline(0.80, color="gray", linestyle="--", label="80% Threshold")
    axes[0].set_title("PCA Scree Plot", fontweight="bold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")
    axes[0].legend(loc="best")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # 2D PCA Cluster Scatter
    sns.scatterplot(
        x=projected_df["PC1"],
        y=projected_df["PC2"],
        hue=cluster_labels.astype(str),
        palette="viridis",
        ax=axes[1],
        alpha=0.8,
        s=40
    )
    axes[1].set_title("2D PCA Projection colored by Cluster", fontweight="bold")
    axes[1].set_xlabel(f"PC1 ({ratios[0]:.1%} variance)")
    axes[1].set_ylabel(f"PC2 ({ratios[1]:.1%} variance)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    plot_file = path / "pca_clustering_dashboard.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Artifact written to: {plot_file.resolve()}")
    return plot_file
```

---

### Pattern B: R (`stats`, `ggplot2`, `cluster`)

#### 1. Standardization & Invariant Check
```r
standardize_and_verify <- function(data, feature_cols) {
  raw_mat <- as.matrix(data[, feature_cols])
  scaled_mat <- scale(raw_mat)
  
  col_means <- colMeans(scaled_mat)
  col_sds <- apply(scaled_mat, 2, sd)
  
  stopifnot(all(abs(col_means) < 1e-6))
  stopifnot(all(abs(col_sds - 1.0) < 1e-6))
  
  cat(sprintf("[Scaling] Standardized %d features across %s rows.\n",
              length(feature_cols), format(nrow(data), big.mark = ",")))
  return(as.data.frame(scaled_mat))
}
```

#### 2. PCA & Scree Invariant Verification
```r
run_pca_with_invariants <- function(scaled_df, variance_threshold = 0.80) {
  pca_res <- prcomp(scaled_df, center = FALSE, scale. = FALSE)
  eigenvalues <- pca_res$sdev^2
  ratios <- eigenvalues / sum(eigenvalues)
  cumulative <- cumsum(ratios)
  
  stopifnot(abs(sum(ratios) - 1.0) < 1e-6)
  cutoff_idx <- which(cumulative >= variance_threshold)[1]
  
  cat("\n==================================================\n")
  cat("PCA EXPLAINED VARIANCE TABLE (R)\n")
  cat("==================================================\n")
  for (i in seq_along(ratios)) {
    marker <- if (i == cutoff_idx) " <-- [80% Threshold Met]" else ""
    cat(sprintf("PC%02d: Individual = %5.1f%% | Cumulative = %5.1f%%%s\n",
                i, ratios[i] * 100, cumulative[i] * 100, marker))
  }
  cat("==================================================\n\n")
  
  return(list(pca = pca_res, ratios = ratios, cumulative = cumulative))
}
```

#### 3. K-Means Clustering & Partition Verification
```r
run_kmeans_clustering <- function(data, n_clusters = 3, seed = 42) {
  set.seed(seed)
  km_res <- kmeans(data, centers = n_clusters, nstart = 10)
  
  # Invariants
  stopifnot(length(km_res$cluster) == nrow(data))
  stopifnot(length(unique(km_res$cluster)) == n_clusters)
  stopifnot(all(km_res$size > 0))
  
  cat(sprintf("\n[Clustering] K=%d fitted. Total within-cluster SS: %.2f\n", n_clusters, km_res$tot.withinss))
  cat("Cluster Sizes:\n")
  for (k in seq_len(n_clusters)) {
    cat(sprintf("  - Cluster %d: %s (%.1f%%)\n",
                k, format(km_res$size[k], big.mark = ","),
                (km_res$size[k] / nrow(data)) * 100))
  }
  return(km_res)
}
```

#### 4. Save Visual Artifacts to `output/`
```r
library(ggplot2)

save_dim_reduction_plots <- function(pca_list, cluster_labels, output_dir = "output") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  plot_path <- file.path(output_dir, "pca_clustering_dashboard.png")
  
  # 2D projection data
  proj_df <- as.data.frame(pca_list$pca$x[, 1:2])
  proj_df$Cluster <- as.factor(cluster_labels)
  
  ratios <- pca_list$ratios
  p <- ggplot(proj_df, aes(x = PC1, y = PC2, color = Cluster)) +
    geom_point(alpha = 0.8, size = 2) +
    theme_minimal() +
    labs(
      title = "2D PCA Projection colored by Cluster",
      x = sprintf("PC1 (%.1f%% variance)", ratios[1] * 100),
      y = sprintf("PC2 (%.1f%% variance)", ratios[2] * 100)
    )
  
  ggsave(plot_path, plot = p, width = 8, height = 5, dpi = 300)
  cat(sprintf("[Visual Saved] Artifact written to: %s\n", normalizePath(plot_path)))
  return(plot_path)
}
```

---

## Universal Verification Checklist

Before reporting completion on dimensionality reduction or clustering tasks:
1. **Standardization Check**: Confirm pre-scaling passed zero mean and unit variance tests.
2. **Variance Conservation**: Verify $\sum r_k = 1.0 \pm 10^{-6}$ and identify the 80% cumulative threshold component count.
3. **Partition Integrity**: Confirm zero empty clusters and complete partition coverage.
4. **Visual Artifact Verification**: Confirm `output/pca_clustering_dashboard.png` exists with $>0$ bytes.
5. **Execution**: Confirm test scripts execute with zero runtime errors.
