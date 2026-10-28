# Data Analytics & Verification Plugin

An Antigravity agent plugin designed to enforce a **verification-first, step-by-step exploratory data analysis (EDA) and transformation workflow**. 

It guarantees empirical data integrity across both **Python** and **R** pipelines through baseline profiling, observable transition metrics, mathematical invariant assertions, terminal ASCII double-checks, and saved visualization artifacts.

---

## Architecture & Plugin Layout

```text
data-heuristics-pipeline/
├── plugin.json                                        # Plugin manifest
├── README.md                                          # Documentation & user guide
├── rules/
│   └── AGENTS.md                                      # Always-on operating rules and constraints
├── agents/
│   └── data-scientist.md                              # Dedicated analytical auditor subagent
└── skills/
    ├── data-analytics-verification/                   # Skill: Tabular EDA & clean transformations
    │   ├── SKILL.md                                   # On-demand procedural guide (Python & R)
    │   └── scripts/
    │       ├── verifier.py                            # Standalone Python verification helper
    │       └── verifier.R                             # Standalone R verification helper
    └── dimensionality-reduction-clustering/           # Skill: PCA & K-Means clustering verification
        ├── SKILL.md                                   # On-demand procedural guide (Python & R)
        └── scripts/
            ├── dim_reduction_cluster.py               # Standalone Python module (PCA + K-Means)
            └── dim_reduction_cluster.R                # Standalone R module (PCA + K-Means)
```

---

## The 5-Step Verification Protocol

Every analytical task and data transformation follows this non-negotiable operational cycle:

```
[ Step 1: Ingestion & Baseline Profile ]
                  │
                  ▼
[ Step 2: Incremental Transformation with Invariant Asserts ]
                  │
                  ▼
[ Step 3: Terminal Quantiles & ASCII Distribution Printout ]
                  │
                  ▼
[ Step 4: Persistence of Visual Artifacts in output/ ]
                  │
                  ▼
[ Step 5: Terminal Execution & Zero-Error Check ]
```

### 1. Ingestion & Baseline Profile
Never transform data blind. Always inspect and print:
- Dimensions (`df.shape` in Python / `dim(df)` in R)
- Duplicate counts (`df.duplicated().sum()` / `sum(duplicated(df))`)
- Missing value counts and percentages per column
- Five-number distribution summaries (`describe()` / `summary()`)

### 2. Incremental Transformation & Invariant Assertions
Break pipelines into traceable microsteps. Format logic in parenthesized vertical method chains (`pandas`) or vertical pipes (`|>`, `%>%` in R).
- Log row counts before and after every filter, join, or drop.
- Assert domain invariants (`assert` / `stopifnot()`): non-negative costs, valid probability sums ($\sum p_i = 1$), unique primary keys, non-empty results.

### 3. Terminal Quantile & ASCII Distribution Printouts
Print immediate, readable proof to console `stdout` without requiring the user to open external viewers:
- 5th, 25th, 50th (median), 75th, and 95th percentiles.
- Pure ASCII distribution bar charts generated directly from sample frequencies.

### 4. Persistence of Visual Artifacts in `output/`
Every data script saves concrete visual evidence to an `output/` directory:
- **Python**: Dual-panel PNGs (Histogram + KDE and Boxplot) via Matplotlib/Seaborn, or interactive HTML charts via Altair.
- **R**: High-DPI PNGs via `ggplot2::ggsave()` or base R graphics devices.

### 5. Terminal Execution & Zero-Error Check
Verify scripts by running them synchronously via the terminal (`python ...` or `Rscript ...`) to ensure exit code `0` with zero unhandled exceptions before concluding any turn.

---

## Quickstart & Usage

### Running the Python Verification Module

You can execute the built-in self-test directly:

```bash
python skills/data-analytics-verification/scripts/verifier.py
```

Or import the helper functions into your project:

```python
from skills.data_analytics_verification.scripts.verifier import (
    inspect_dataset,
    log_transition,
    print_distribution_table,
    print_ascii_histogram,
    save_eda_figure,
    assert_invariants,
)

# 1. Inspect
inspect_dataset(df, dataset_name="Customer Orders")

# 2. Transform & Log
cleaned_df = (
    df
    .dropna(subset=["order_id", "total_price"])
    .query("total_price >= 0")
    .reset_index(drop=True)
)
log_transition(df, cleaned_df, "Filter valid orders")

# 3. Assert Invariants
assert_invariants(
    cleaned_df,
    non_negative_columns=["total_price"],
    unique_columns=["order_id"],
)

# 4. Terminal Double-Check
print_distribution_table(cleaned_df["total_price"], label="Order Total")
print_ascii_histogram(cleaned_df["total_price"], label="Order Total")

# 5. Save Visual
save_eda_figure(cleaned_df, column_name="total_price", output_directory="output")
```

### Running the R Verification Module

Execute the R self-test:

```bash
Rscript skills/data-analytics-verification/scripts/verifier.R
```

Or source the functions in an R session:

```r
source("skills/data-analytics-verification/scripts/verifier.R")

# 1. Inspect
inspect_dataset(df, dataset_name = "Customer Orders")

# 2. Transform & Log
cleaned_df <- df |>
  dplyr::filter(!is.na(order_id), !is.na(total_price), total_price >= 0) |>
  dplyr::distinct(order_id, .keep_all = TRUE)
log_transition(df, cleaned_df, "Filter valid orders")

# 3. Assert Invariants
assert_invariants(
  cleaned_df,
  non_negative_columns = c("total_price"),
  unique_columns = c("order_id")
)

# 4. Terminal Double-Check
print_distribution_table(cleaned_df$total_price, label = "Order Total")
print_ascii_histogram(cleaned_df$total_price, label = "Order Total")

# 5. Save Visual
save_eda_figure(cleaned_df, column_name = "total_price", output_directory = "output")
```

### Running the Dimensionality Reduction & Clustering Module

#### Python
```bash
python skills/dimensionality-reduction-clustering/scripts/dim_reduction_cluster.py
```
Or import into your pipeline:
```python
from skills.dimensionality_reduction_clustering.scripts.dim_reduction_cluster import (
    standardize_features,
    compute_pca_decomposition,
    fit_kmeans_partition,
    save_clustering_visuals,
)

# 1. Standardize features and assert mean=0, std=1
scaled_df, scaler = standardize_features(df, feature_columns=["f1", "f2", "f3", "f4"])

# 2. PCA decomposition with variance conservation check
pca_model, projected_df, cutoff = compute_pca_decomposition(scaled_df, variance_threshold=0.80)

# 3. K-Means clustering with partition completeness check
kmeans_model, cluster_labels, silhouette = fit_kmeans_partition(projected_df[["PC1", "PC2"]], cluster_count=3)

# 4. Save visual dashboard (Scree plot + 2D PCA cluster projection)
save_clustering_visuals(pca_model, projected_df, cluster_labels, output_directory="output")
```

#### R
```bash
Rscript skills/dimensionality-reduction-clustering/scripts/dim_reduction_cluster.R
```

---

## Plugin Deployment & Registration

To make this plugin active across Antigravity:

### Option A: Workspace Customization (Project-Level)
Copy or link this plugin directory under your workspace root:
```text
your-project/
└── .agents/
    └── plugins/
        └── data-analytics-verification/  <-- (contents of this repository)
```

### Option B: Machine-Wide Global Plugin
Copy this repository into your global Antigravity plugins directory:
```bash
# Windows
Copy-Item -Recurse . "$HOME\.gemini\config\plugins\data-analytics-verification"

# Linux / macOS
cp -r . ~/.gemini/config/plugins/data-analytics-verification
```

Once installed, Antigravity automatically detects the plugin, injects [`rules/AGENTS.md`](./rules/AGENTS.md), activates the `data-scientist` subagent, and equips the agent with [`data-analytics-verification`](./skills/data-analytics-verification/SKILL.md).
