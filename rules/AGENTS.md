# Step-by-Step Data Analytics & Verification Protocol (Polyglot: Python & R)

When executing exploratory data analysis (EDA), data transformations, or mathematical/heuristic modeling, adhere strictly to this verification-first operational protocol. This protocol applies equally across **Python** and **R** workflows.

---

## 1. Core Operating Cycle: Orient, Inspect, Transform, Verify

Every data task must proceed through explicit, observable phases. Never execute monolithic, batch, or unverified transformations.

### Phase 1: Baseline Ingestion & Profiling (Inspect First)
Before applying any filtering, cleaning, feature engineering, or modeling:
- **Inspect Structural Dimensions:** Print raw dimensions (`df.shape` in Python; `dim(df)` or `glimpse(df)` in R).
- **Inspect Data Integrity:** Check and print missing value counts per column (`df.isna().sum()` in Python; `colSums(is.na(df))` in R) and duplicate record counts (`df.duplicated().sum()` in Python; `sum(duplicated(df))` in R).
- **Profile Distributions:** Print numerical five-number summaries and categorical frequencies (`df.describe()` / `value_counts()` in Python; `summary(df)` / `table()` in R).
- **Surface Anomalies:** Explicitly call out unexpected nulls, zero variances, skewness, or boundary outliers to the user before proceeding.

### Phase 2: Incremental Transformation & Intermediate Prints
Transformations must be executed in traceable, modular microsteps:
- **Before-and-After Logging:** Print record counts and summary metrics immediately before and after any filtering, join, deduplication, or imputation (e.g., `Rows before filter: 1,200 -> after: 1,142 (-58)`).
- **Explicit Invariant Assertions:** Assert domain constraints and mathematical rules using runtime checks (`assert` in Python; `stopifnot()` or `assertthat` in R).
- **Readable Pipeline Formatting:** Use parenthesized multi-line method chains in Python or vertical pipes (`|>` or `%>%`) in R:
  ```python
  # Python pattern
  cleaned_df = (
      raw_df
      .dropna(subset=["id", "timestamp"])
      .query("metric >= 0")
      .reset_index(drop=True)
  )
  ```
  ```r
  # R pattern
  cleaned_df <- raw_df |>
    dplyr::filter(!is.na(id), !is.na(timestamp), metric >= 0) |>
    dplyr::distinct(id, .keep_all = TRUE)
  ```

### Phase 3: Dual Verification (Terminal Summary & Saved Visualizations)
Every data processing or analysis script must produce two forms of concrete evidence:
1. **Terminal Double-Check:** Print a formatted text summary (ASCII distribution quantiles, five-number tables) directly to console `stdout`.
2. **Saved Visual Artifacts:**
   - Always persist visualizations (distributions, correlation matrices, residual plots, line trends) to an `output/` directory.
   - Ensure the directory exists before saving (`pathlib.Path("output").mkdir(parents=True, exist_ok=True)` in Python; `dir.create("output", showWarnings = FALSE)` in R).
   - Python: Save high-resolution PNGs via Matplotlib/Seaborn or interactive HTML via Altair.
   - R: Save high-resolution PNGs via `ggplot2::ggsave()` or base R devices.

---

## 2. Mathematical & Logical Accuracy Rules of Thumb

Maintain absolute fidelity to mathematical, statistical, and operational invariants:
- **Normalization & Probabilities:** Verify that probability distributions sum to $1.0 \pm 10^{-6}$ and normalized vectors stay within expected ranges $[0, 1]$ or $[-1, 1]$.
- **Data Leakage & Train/Test Splits:** Fit scalers, imputers, and encoders strictly on training splits. Never calculate global transformation statistics across combined sets.
- **Aggregation Soundness:** Never average averages or percentages without appropriate weighting. Always output group record counts alongside aggregated means/medians.
- **Heuristics & Optimization Invariants:** When evaluating heuristics or algorithms, log cost/energy progress across iterations, verify constraint compliance at each move/transition, and assert monotonic improvements where required.

---

## 3. Script Execution Verification

Before considering any task complete:
- Execute the script directly in the terminal (e.g., `python script.py` or `Rscript script.R`) to guarantee zero runtime errors or unhandled warnings.
- Confirm the generated visualization file exists in `output/` and has non-zero size.
- Inspect the terminal output printout to double-check that analytical results match domain expectations.
