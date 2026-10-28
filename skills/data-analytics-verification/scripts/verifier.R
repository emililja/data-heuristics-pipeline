# ==============================================================================
# Verification and EDA Helper Module for Step-by-Step Data Analysis (R)
#
# Provides standardized terminal inspection printouts, transition logging,
# ASCII console histograms, mathematical invariant assertions, and adaptive
# visual artifact generation (ggplot2 or base R PNG devices).
# ==============================================================================

inspect_dataset <- function(data, dataset_name = "Dataset") {
  if (nrow(data) == 0) {
    cat(sprintf("\n[WARNING] Dataset '%s' is empty.\n", dataset_name))
    return(invisible(NULL))
  }

  banner <- strrep("=", 60)
  cat(sprintf("\n%s\nBASELINE PROFILE: %s\n%s\n", banner, dataset_name, banner))
  cat(sprintf("Dimensions: %s rows x %s columns\n", format(nrow(data), big.mark = ","), ncol(data)))
  cat(sprintf("Duplicate Rows: %s\n", format(sum(duplicated(data)), big.mark = ",")))

  null_counts <- colSums(is.na(data))
  cols_with_nulls <- null_counts[null_counts > 0]
  if (length(cols_with_nulls) == 0) {
    cat("Missing Values: None\n")
  } else {
    cat("\nMissing Values per Column:\n")
    for (col_name in names(cols_with_nulls)) {
      count <- cols_with_nulls[[col_name]]
      pct <- (count / nrow(data)) * 100
      cat(sprintf("  - %s: %s (%.2f%%)\n", col_name, format(count, big.mark = ","), pct))
    }
  }

  cat("\nColumn Data Types:\n")
  for (col_name in names(data)) {
    cat(sprintf("  - %s: %s\n", col_name, class(data[[col_name]])[1]))
  }

  cat("\nSummary Statistics:\n")
  print(summary(data))
  cat(banner, "\n\n")
  invisible(NULL)
}

log_transition <- function(before_data, after_data, operation_name) {
  before_count <- nrow(before_data)
  after_count <- nrow(after_data)
  dropped_count <- before_count - after_count
  pct_reduced <- if (before_count > 0) (dropped_count / before_count) * 100 else 0

  cat(sprintf(
    "[Transition: %s] %s -> %s rows (%s delta, %.2f%% reduced)\n",
    operation_name,
    format(before_count, big.mark = ","),
    format(after_count, big.mark = ","),
    format(dropped_count, big.mark = ","),
    pct_reduced
  ))
  invisible(NULL)
}

print_distribution_table <- function(values, label = "Metric") {
  clean_values <- values[!is.na(values)]
  if (length(clean_values) == 0) {
    cat(sprintf("[Distribution] Series '%s' has no valid values.\n", label))
    return(invisible(NULL))
  }

  quantiles <- quantile(clean_values, probs = c(0.05, 0.25, 0.50, 0.75, 0.95))
  delimiter <- strrep("-", 45)

  cat(sprintf("\n%s\nDistribution Quantiles: %s\n%s\n", delimiter, label, delimiter))
  cat(sprintf(
    "Count: %s | Mean: %.4f | Std: %.4f\n",
    format(length(clean_values), big.mark = ","),
    mean(clean_values),
    sd(clean_values)
  ))
  cat(sprintf("  p05 (5th percentile)  : %.4f\n", quantiles[["5%"]]))
  cat(sprintf("  p25 (1st quartile)    : %.4f\n", quantiles[["25%"]]))
  cat(sprintf("  p50 (median)          : %.4f\n", quantiles[["50%"]]))
  cat(sprintf("  p75 (3rd quartile)    : %.4f\n", quantiles[["75%"]]))
  cat(sprintf("  p95 (95th percentile) : %.4f\n", quantiles[["95%"]]))
  cat(delimiter, "\n\n")
  invisible(NULL)
}

print_ascii_histogram <- function(values, label = "Metric", bin_count = 10, max_bar_width = 30) {
  clean_values <- values[!is.na(values)]
  if (length(clean_values) == 0) {
    cat(sprintf("[Histogram] Series '%s' has no valid numeric values.\n", label))
    return(invisible(NULL))
  }

  h <- hist(clean_values, breaks = bin_count, plot = FALSE)
  counts <- h$counts
  breaks <- h$breaks
  max_count <- max(counts, na.rm = TRUE)
  if (max_count == 0) max_count <- 1

  cat(sprintf("--- ASCII Histogram: %s ---\n", label))
  for (i in seq_along(counts)) {
    lower <- breaks[i]
    upper <- breaks[i + 1]
    freq <- counts[i]
    bar_len <- as.integer((freq / max_count) * max_bar_width)
    bar_str <- strrep("#", bar_len)
    cat(sprintf("[%8.2f - %8.2f] | %-*s (%s)\n", lower, upper, max_bar_width, bar_str, format(freq, big.mark = ",")))
  }
  cat(strrep("-", 55), "\n\n")
  invisible(NULL)
}

save_eda_figure <- function(data, column_name, output_directory = "output", filename_prefix = "eda") {
  if (!column_name %in% names(data)) {
    stop(sprintf("Column '%s' not found in dataset.", column_name))
  }

  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  plot_path <- file.path(output_directory, sprintf("%s_%s_distribution.png", filename_prefix, column_name))

  # Use ggplot2 if installed; otherwise fall back to base R graphics
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p <- ggplot2::ggplot(data, ggplot2::aes(x = .data[[column_name]])) +
      ggplot2::geom_histogram(
        ggplot2::aes(y = ggplot2::after_stat(density)),
        bins = 30,
        fill = "#2b5c8f",
        color = "white",
        alpha = 0.8
      ) +
      ggplot2::geom_density(color = "#e66101", linewidth = 1) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = sprintf("Distribution of %s", column_name),
        x = column_name,
        y = "Density"
      )
    ggplot2::ggsave(plot_path, plot = p, width = 8, height = 4.5, dpi = 300)
  } else {
    png(filename = plot_path, width = 800, height = 400, res = 100)
    par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))
    clean_vals <- data[[column_name]][!is.na(data[[column_name]])]
    hist(clean_vals, main = sprintf("Histogram: %s", column_name), xlab = column_name, col = "#2b5c8f", border = "white")
    boxplot(clean_vals, main = sprintf("Boxplot: %s", column_name), ylab = column_name, col = "#4e9da4")
    dev.off()
  }

  cat(sprintf("[Visual Saved] Artifact written to: %s\n", normalizePath(plot_path, winslash = "/")))
  invisible(plot_path)
}

assert_invariants <- function(data, non_negative_columns = NULL, unique_columns = NULL, probability_columns = NULL) {
  if (!is.null(non_negative_columns)) {
    for (col_name in non_negative_columns) {
      if (col_name %in% names(data)) {
        neg_count <- sum(data[[col_name]] < 0, na.rm = TRUE)
        if (neg_count > 0) {
          stop(sprintf("Invariant violated: column '%s' has %d negative values.", col_name, neg_count))
        }
      }
    }
  }

  if (!is.null(unique_columns)) {
    for (col_name in unique_columns) {
      if (col_name %in% names(data)) {
        if (any(duplicated(data[[col_name]]))) {
          stop(sprintf("Invariant violated: column '%s' contains duplicate values.", col_name))
        }
      }
    }
  }

  if (!is.null(probability_columns)) {
    for (col_name in probability_columns) {
      if (col_name %in% names(data)) {
        probs <- data[[col_name]][!is.na(data[[col_name]])]
        if (any(probs < 0.0 | probs > 1.0)) {
          stop(sprintf("Invariant violated: column '%s' contains probabilities outside [0, 1].", col_name))
        }
      }
    }
  }
  invisible(TRUE)
}

# ------------------------------------------------------------------------------
# Demonstration / Self-Test (when run as a standalone script)
# ------------------------------------------------------------------------------
if (!interactive()) {
  set.seed(42)
  sample_size <- 500

  raw_records <- data.frame(
    transaction_id = sprintf("TX_%05d", seq_len(sample_size)),
    amount = rexp(sample_size, rate = 1 / 50.0),
    fee_rate = runif(sample_size, min = 0.01, max = 0.05),
    status = sample(c("COMPLETED", "PENDING", "FAILED"), size = sample_size, replace = TRUE, prob = c(0.85, 0.10, 0.05)),
    stringsAsFactors = FALSE
  )

  # Step 1: Baseline inspection
  inspect_dataset(raw_records, dataset_name = "Synthetic Transactions (R)")

  # Step 2: Incremental transformation with transition logging
  filtered_records <- subset(raw_records, status == "COMPLETED")
  filtered_records$net_amount <- filtered_records$amount * (1.0 - filtered_records$fee_rate)
  rownames(filtered_records) <- NULL
  log_transition(raw_records, filtered_records, "Filter COMPLETED status")

  # Step 3: Assert mathematical invariants
  assert_invariants(
    filtered_records,
    non_negative_columns = c("amount", "net_amount"),
    unique_columns = c("transaction_id"),
    probability_columns = c("fee_rate")
  )

  # Step 4: Terminal quantile and ASCII distribution printouts
  print_distribution_table(filtered_records$net_amount, label = "Net Transaction Amount")
  print_ascii_histogram(filtered_records$net_amount, label = "Net Transaction Amount")

  # Step 5: Save visual artifact
  saved_file <- save_eda_figure(
    filtered_records,
    column_name = "net_amount",
    output_directory = "output",
    filename_prefix = "r_test_run"
  )
  cat(sprintf("R self-test complete. Output generated at: %s\n", saved_file))
}
