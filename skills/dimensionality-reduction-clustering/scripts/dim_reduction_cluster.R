# ==============================================================================
# Dimensionality Reduction & Clustering Verification Module (R)
#
# Enforces pre-scaling verification, PCA variance conservation, K-Means partition
# completeness, and visual artifact generation across output directories.
# ==============================================================================

standardize_features <- function(data, feature_columns) {
  if (nrow(data) == 0) {
    stop("Cannot standardize an empty dataset.")
  }

  missing_cols <- setdiff(feature_columns, names(data))
  if (length(missing_cols) > 0) {
    stop(sprintf("Feature columns not found: %s", paste(missing_cols, collapse = ", ")))
  }

  feature_matrix <- as.matrix(data[, feature_columns])
  scaled_matrix <- scale(feature_matrix)

  col_means <- colMeans(scaled_matrix)
  col_sds <- apply(scaled_matrix, 2, sd)

  # Invariant assertions: mean = 0, std = 1
  stopifnot(all(abs(col_means) < 1e-6))
  stopifnot(all(abs(col_sds - 1.0) < 1e-6))

  cat(sprintf("[Scaling] Successfully standardized %d features across %s rows.\n",
              length(feature_columns), format(nrow(data), big.mark = ",")))
  return(as.data.frame(scaled_matrix))
}

compute_pca_decomposition <- function(scaled_data, variance_threshold = 0.80) {
  if (nrow(scaled_data) == 0) {
    stop("Cannot perform PCA on an empty dataset.")
  }

  pca_res <- prcomp(scaled_data, center = FALSE, scale. = FALSE)
  eigenvalues <- pca_res$sdev^2
  variance_ratios <- eigenvalues / sum(eigenvalues)
  cumulative_variance <- cumsum(variance_ratios)

  # Mathematical Invariant: sum of variance ratios must equal 1.0
  stopifnot(abs(sum(variance_ratios) - 1.0) < 1e-6)

  cutoff_index <- which(cumulative_variance >= variance_threshold)[1]

  banner <- strrep("=", 60)
  cat(sprintf("\n%s\nPCA EXPLAINED VARIANCE PROFILE (R)\n%s\n", banner, banner))
  for (i in seq_along(variance_ratios)) {
    marker <- if (i == cutoff_index) sprintf(" <-- [%.0f%% Cutoff Reached]", variance_threshold * 100) else ""
    cat(sprintf("  PC%02d: Individual = %6.2f%% | Cumulative = %6.2f%%%s\n",
                i, variance_ratios[i] * 100, cumulative_variance[i] * 100, marker))
  }
  cat(banner, "\n\n")

  projected_df <- as.data.frame(pca_res$x)
  return(list(
    pca = pca_res,
    ratios = variance_ratios,
    cumulative = cumulative_variance,
    projected = projected_df,
    cutoff = cutoff_index
  ))
}

fit_kmeans_partition <- function(data, cluster_count = 3, seed = 42) {
  if (cluster_count <= 1) {
    stop("cluster_count must be at least 2.")
  }

  set.seed(seed)
  km_res <- kmeans(data, centers = cluster_count, nstart = 10)

  # Invariant assertions: non-empty clusters and exact partition length
  stopifnot(length(km_res$cluster) == nrow(data))
  stopifnot(length(unique(km_res$cluster)) == cluster_count)
  stopifnot(all(km_res$size > 0))

  cat(sprintf("[Clustering] Fitted K=%d clusters. Total within-cluster SS: %.2f\n",
              cluster_count, km_res$tot.withinss))
  cat("Cluster Membership Summary:\n")
  for (k in seq_len(cluster_count)) {
    pct <- (km_res$size[k] / nrow(data)) * 100
    cat(sprintf("  - Cluster %d: %s samples (%.2f%%)\n",
                k, format(km_res$size[k], big.mark = ","), pct))
  }
  cat("\n")

  return(km_res)
}

save_clustering_visuals <- function(pca_info, cluster_labels, output_directory = "output", filename_prefix = "clustering") {
  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  plot_path <- file.path(output_directory, sprintf("%s_dashboard.png", filename_prefix))

  # If ggplot2 is available, use it; otherwise fall back to base R png
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    proj_df <- pca_info$projected[, 1:2]
    proj_df$Cluster <- as.factor(cluster_labels)
    ratios <- pca_info$ratios

    p <- ggplot2::ggplot(proj_df, ggplot2::aes(x = PC1, y = PC2, color = Cluster)) +
      ggplot2::geom_point(alpha = 0.8, size = 2) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = "2D PCA Projection colored by Cluster",
        x = sprintf("PC1 (%.1f%% variance)", ratios[1] * 100),
        y = sprintf("PC2 (%.1f%% variance)", ratios[2] * 100)
      )

    ggplot2::ggsave(plot_path, plot = p, width = 8, height = 5, dpi = 300)
  } else {
    png(filename = plot_path, width = 900, height = 450, res = 100)
    par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

    # Scree Plot
    ratios <- pca_info$ratios
    barplot(ratios, names.arg = sprintf("PC%d", seq_along(ratios)),
            col = "#2b5c8f", main = "PCA Scree Plot", ylab = "Variance Ratio")

    # 2D Scatter
    plot(pca_info$projected$PC1, pca_info$projected$PC2, col = cluster_labels, pch = 19,
         xlab = sprintf("PC1 (%.1f%%)", ratios[1] * 100),
         ylab = sprintf("PC2 (%.1f%%)", ratios[2] * 100),
         main = "2D PCA Clusters")
    dev.off()
  }

  cat(sprintf("[Visual Saved] Artifact written to: %s\n", normalizePath(plot_path, winslash = "/")))
  invisible(plot_path)
}

# ------------------------------------------------------------------------------
# Self-Test Demonstration
# ------------------------------------------------------------------------------
if (!interactive()) {
  cat("Initiating Dimensionality Reduction & Clustering Verification Self-Test (R)...\n")
  set.seed(42)
  n <- 400

  # Synthetic 3-cluster 6-feature data
  c1 <- matrix(rnorm(n * 2, mean = 0, sd = 1), ncol = 6)
  c2 <- matrix(rnorm(n * 2, mean = 4, sd = 1), ncol = 6)
  synthetic_matrix <- rbind(c1, c2)
  colnames(synthetic_matrix) <- sprintf("feature_%d", 1:6)
  df <- as.data.frame(synthetic_matrix)

  scaled_df <- standardize_features(df, colnames(df))
  pca_res <- compute_pca_decomposition(scaled_df, variance_threshold = 0.80)
  km <- fit_kmeans_partition(pca_res$projected[, 1:2], cluster_count = 2)

  saved <- save_clustering_visuals(pca_res, km$cluster, output_directory = "output", filename_prefix = "r_dim_reduction")
  cat(sprintf("R self-test complete. Output generated at: %s\n", saved))
}
