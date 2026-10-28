---
name: data-analytics-verification
description: >-
  Performs exploratory data analysis (EDA), data cleaning, tabular transformations,
  and statistical checks in Python or R using a strict step-by-step verification
  workflow with terminal summary printouts and saved visual artifacts.
---

# Step-by-Step Data Analytics & Verification Skill (Polyglot)

Use this skill whenever executing exploratory data analysis (EDA), data cleaning, tabular feature engineering, or statistical validation. This workflow provides end-to-end data integrity checks, terminal verification tables, and visual proof across **Python** and **R**.

---

## Universal Verification Flow

```
[ Step 1: Ingestion & Baseline Profile ]
                  │
                  ▼
[ Step 2: Incremental Transformation with Invariant Asserts ]
                  │
                  ▼
[ Step 3: Terminal Summary & ASCII Distribution Printout ]
                  │
                  ▼
[ Step 4: Persistence of Visual Artifacts in output/ ]
                  │
                  ▼
[ Step 5: Terminal Execution & Zero-Error Check ]
```

---

## Reference Implementation Scripts

- **Python Module**: [`scripts/verifier.py`](./scripts/verifier.py) (includes Pandas profiling, transition tracking, ASCII histograms, adaptive Matplotlib/Altair graphics, and invariant assertions).
- **R Module**: [`scripts/verifier.R`](./scripts/verifier.R) (includes Tidyverse profiling, pipe transition logging, base R / ggplot2 visuals, and `stopifnot()` checks).

---

## Language Implementation Patterns

### Pattern A: Python (`pandas`, `numpy`, `matplotlib` / `altair`)

#### 1. Baseline Ingestion & Profiling
```python
import pandas as pd

def inspect_dataset(data: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """Print structural dimensions, null percentages, and five-number summary."""
    print(f"\n{'=' * 60}\nBASELINE PROFILE: {dataset_name}\n{'=' * 60}")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print(f"Duplicates: {data.duplicated().sum():,} duplicate rows")
    
    null_counts = data.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if columns_with_nulls.empty:
        print("Missing Values: None")
    else:
        print("\nMissing Values per Column:")
        for col, count in columns_with_nulls.items():
            pct = (count / len(data)) * 100
            print(f"  - {col}: {count:,} ({pct:.2f}%)")
            
    print("\nNumerical Summary:")
    print(data.describe().T[["count", "mean", "std", "min", "50%", "max"]])
    print("=" * 60 + "\n")
```

#### 2. Incremental Transformation & Invariant Assertions
```python
def clean_records(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Filter records using readable method chains and assert domain invariants."""
    before_count = len(raw_df)
    
    cleaned_df = (
        raw_df
        .dropna(subset=["transaction_id", "amount"])
        .drop_duplicates(subset=["transaction_id"])
        .query("amount >= 0")
        .reset_index(drop=True)
    )
    
    delta = before_count - len(cleaned_df)
    print(f"[Transition] Cleaned records: {before_count:,} -> {len(cleaned_df):,} rows ({delta:,} dropped)")
    
    # Mathematical and logical assertions
    assert len(cleaned_df) > 0, "Cleaned dataset must not be empty"
    assert (cleaned_df["amount"] >= 0).all(), "Amounts must be non-negative"
    assert cleaned_df["transaction_id"].is_unique, "Transaction IDs must be unique"
    return cleaned_df
```

#### 3. Terminal Quantiles & ASCII Histogram
```python
import numpy as np

def print_terminal_metrics(series: pd.Series, label: str) -> None:
    """Print quantile table and ASCII histogram to console."""
    clean = series.dropna()
    quantiles = clean.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    print(f"\n--- Quantiles: {label} ---")
    print(f"Mean: {clean.mean():.4f} | Std: {clean.std():.4f}")
    for q, val in quantiles.items():
        print(f"  p{int(q*100):02d}: {val:.4f}")
        
    counts, edges = np.histogram(clean, bins=8)
    max_count = max(counts) if max(counts) > 0 else 1
    print(f"\n--- ASCII Distribution: {label} ---")
    for i in range(len(counts)):
        bar = "#" * int((counts[i] / max_count) * 25)
        print(f"[{edges[i]:7.2f} - {edges[i+1]:7.2f}] | {bar:<25} ({counts[i]:,})")
    print("-" * 45 + "\n")
```

#### 4. Saved Visual Artifacts in `output/`
```python
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns

def save_distribution_plot(data: pd.DataFrame, column: str, output_dir: str = "output") -> pathlib.Path:
    """Save dual histogram and boxplot visualization to output directory."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=data[column].dropna(), kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title(f"Distribution: {column}", fontweight="bold")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    sns.boxplot(x=data[column].dropna(), ax=axes[1], color="#4e9da4")
    axes[1].set_title(f"Boxplot: {column}", fontweight="bold")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    plot_file = path / f"{column}_distribution.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Artifact written to: {plot_file.resolve()}")
    return plot_file
```

---

### Pattern B: R (`tidyverse`, `ggplot2`)

#### 1. Baseline Ingestion & Profiling
```r
inspect_dataset <- function(data, dataset_name = "Dataset") {
  cat(sprintf("\n%s\nBASELINE PROFILE: %s\n%s\n", strrep("=", 60), dataset_name, strrep("=", 60)))
  cat(sprintf("Dimensions: %s rows x %s columns\n", format(nrow(data), big.mark = ","), ncol(data)))
  cat(sprintf("Duplicate Rows: %s\n", format(sum(duplicated(data)), big.mark = ",")))
  
  null_counts <- colSums(is.na(data))
  cols_with_nulls <- null_counts[null_counts > 0]
  if (length(cols_with_nulls) == 0) {
    cat("Missing Values: None\n")
  } else {
    cat("\nMissing Values per Column:\n")
    for (col_name in names(cols_with_nulls)) {
      pct <- (cols_with_nulls[[col_name]] / nrow(data)) * 100
      cat(sprintf("  - %s: %s (%.2f%%)\n", col_name, format(cols_with_nulls[[col_name]], big.mark = ","), pct))
    }
  }
  
  cat("\nSummary Statistics:\n")
  print(summary(data))
  cat(strrep("=", 60), "\n\n")
}
```

#### 2. Incremental Transformation & Invariant Assertions
```r
clean_records <- function(raw_df) {
  before_count <- nrow(raw_df)
  
  cleaned_df <- raw_df |>
    dplyr::filter(!is.na(transaction_id), !is.na(amount), amount >= 0) |>
    dplyr::distinct(transaction_id, .keep_all = TRUE)
  
  after_count <- nrow(cleaned_df)
  delta <- before_count - after_count
  cat(sprintf("[Transition] Cleaned records: %s -> %s rows (%s dropped)\n",
              format(before_count, big.mark = ","),
              format(after_count, big.mark = ","),
              format(delta, big.mark = ",")))
  
  # Assertions
  stopifnot(after_count > 0)
  stopifnot(all(cleaned_df$amount >= 0))
  stopifnot(!any(duplicated(cleaned_df$transaction_id)))
  
  return(cleaned_df)
}
```

#### 3. Terminal Quantiles & Summary
```r
print_terminal_metrics <- function(values, label = "Metric") {
  clean_values <- values[!is.na(values)]
  quantiles <- quantile(clean_values, probs = c(0.05, 0.25, 0.50, 0.75, 0.95))
  
  cat(sprintf("\n--- Quantiles: %s ---\n", label))
  cat(sprintf("Mean: %.4f | Std: %.4f\n", mean(clean_values), sd(clean_values)))
  for (q_name in names(quantiles)) {
    cat(sprintf("  %s: %.4f\n", q_name, quantiles[[q_name]]))
  }
  cat(strrep("-", 40), "\n\n")
}
```

#### 4. Saved Visual Artifacts in `output/`
```r
library(ggplot2)

save_distribution_plot <- function(data, column_name, output_dir = "output") {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  plot_path <- file.path(output_dir, sprintf("%s_distribution.png", column_name))
  
  p <- ggplot(data, aes(x = .data[[column_name]])) +
    geom_histogram(aes(y = after_stat(density)), bins = 30, fill = "#2b5c8f", color = "white", alpha = 0.8) +
    geom_density(color = "#e66101", linewidth = 1) +
    theme_minimal() +
    labs(title = sprintf("Distribution of %s", column_name), x = column_name, y = "Density")
  
  ggsave(plot_path, plot = p, width = 8, height = 4.5, dpi = 300)
  cat(sprintf("[Visual Saved] Artifact written to: %s\n", normalizePath(plot_path)))
  return(plot_path)
}
```

---

## Universal Verification Checklist

Before marking any analytical sub-goal or task complete:
1. **Execution**: Run the script directly in terminal (`python ...` or `Rscript ...`) and confirm return code `0`.
2. **Terminal Double-Check**: Verify console output contains baseline profiles, before/after transition counts, and distribution quantiles.
3. **Artifact Integrity**: Check that visualization files exist in `output/` and are non-empty ($> 0$ bytes).
4. **Logical Invariants**: Confirm zero runtime assertion failures (valid probabilities, non-negative counts, unique keys).
