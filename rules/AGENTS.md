# Step-by-Step Data Analytics, Modeling & Verification Protocol (Polyglot: Python & R)

When executing exploratory data analysis (EDA), data transformations, statistical modeling, exact optimization (LP/MILP), or modular metaheuristics, adhere strictly to this verification-first operational protocol. This protocol applies equally across **Python** and **R** workflows.

---

## 1. Core Operating Cycle: Orient, Inspect, Transform/Model, Verify

Every data and modeling task must proceed through explicit, observable phases. Never execute monolithic, batch, or unverified transformations.

### Phase 1: Baseline Ingestion & Profiling (Inspect First)
Before applying any filtering, feature engineering, modeling, or optimization:
- **Inspect Structural Dimensions:** Print raw dimensions (`df.shape` in Python; `dim(df)` or `glimpse(df)` in R).
- **Inspect Data Integrity:** Check and print missing value counts and percentages per column (`df.isna().sum()` in Python; `colSums(is.na(df))` in R) and duplicate record counts (`df.duplicated().sum()` in Python; `sum(duplicated(df))` in R).
- **Profile Distributions:** Print numerical five-number summaries and categorical frequencies (`df.describe()` / `value_counts()` in Python; `summary(df)` / `table()` in R).

### Phase 2: Incremental Transformations & Observable Intermediate States
Transformations and modeling pipelines must be executed in traceable, modular microsteps:
- **Before-and-After Logging:** Print record counts and summary metrics immediately before and after any filtering, join, deduplication, imputation, or train/test split (e.g., `Rows before filter: 1,200 -> after: 1,142 (-58)`).
- **Explicit Invariant Assertions:** Assert domain constraints and mathematical rules using runtime checks (`assert` in Python; `stopifnot()` in R).
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
Every data processing, machine learning, or optimization script must produce two forms of concrete evidence:
1. **Terminal Double-Check:** Print a formatted text summary (ASCII distribution quantiles, before/after deltas, baseline comparison tables, slack margin tables) directly to console `stdout`.
2. **Saved Visual Artifacts:**
   - Always persist visualizations (distributions, correlation heatmaps, ROC/PR curves, slack margins, convergence/diversity dashboards) to an `output/` directory.
   - Ensure the directory exists before saving (`pathlib.Path("output").mkdir(parents=True, exist_ok=True)` in Python; `dir.create("output", showWarnings = FALSE)` in R).
   - Python: Save high-resolution PNGs via Matplotlib/Seaborn or interactive HTML via Altair.
   - R: Save high-resolution PNGs via `ggplot2::ggsave()` or base R devices.

---

## 2. Empirical Findings & Anomaly Communication Mandate

Never merely dump tables or terminal outputs without quantitative synthesis. Actively surface data anomalies and translate raw numbers into empirical observations:

1. **Active Anomaly Surfacing**:
   - **Sentinel Values & Hidden Nulls**: Explicitly detect and communicate suspicious values masquerading as legitimate data (e.g., `-999`, `9999`, `1970-01-01`, or an unnatural spike of exact `0.0` entries).
   - **Distribution Irregularities**: Call out extreme skewness, bimodal distributions, zero-variance columns, or boundary clipping (truncation artifacts).
   - **High-Leverage Outliers**: Distinguish between legitimate rare extremes and data entry errors before applying cleaning or scaling.
2. **Modeling & Optimization Diagnostic Insights**:
   - **Baseline Comparison**: For supervised ML, report the exact performance lift over trivial baselines (e.g., *"Model macro F1 is 0.84 vs. dummy majority-class baseline of 0.39, representing a +0.45 lift"*).
   - **Bottleneck Identification**: In LP/MILP, identify binding constraints ($s_i \approx 0$) as active capacity or economic bottlenecks limiting the objective.
   - **Search Stagnation & Diversity Loss**: In metaheuristics (GA/SA/VNS), actively flag when population diversity collapses or stagnation index exceeds patience ($I_{\text{stag}} \ge N_{\text{patience}}$), diagnosing premature local optimum entrapment.
3. **Actionable Recommendations**:
   - Conclude every analysis with 2–3 concise sentences outlining what the data implies and presenting clear options before proceeding to downstream modeling.

---

## 3. Comprehensive Mathematical & Operational Invariants

Maintain absolute fidelity to mathematical, statistical, and operational rules across all five domains:

### Domain 1: Tabular Transformations & EDA
- **Probabilities & Normalization**: Verify probability distributions sum to $1.0 \pm 10^{-6}$ and normalized vectors stay within $[0, 1]$ or $[-1, 1]$.
- **Aggregation Soundness**: Never average averages or percentages without appropriate weighting. Always output group record counts alongside aggregated means/medians.

### Domain 2: Dimensionality Reduction & Clustering
- **Standardization Precondition**: Confirm continuous features are standardized to $\text{mean} \approx 0.0$ and $\text{std} \approx 1.0$ before PCA or distance-based clustering.
- **Variance Conservation**: Assert that the sum of PCA explained variance ratios equals $1.0 \pm 10^{-6}$ ($\sum_{k=1}^p r_k = 1.0$).
- **Cluster Partition Integrity**: Assert zero empty clusters ($|C_k| > 0$) and complete dataset partition coverage.

### Domain 3: Supervised Machine Learning
- **Zero Data Leakage**: Transformers (scalers, encoders, imputers) must fit **strictly on the training split**. Never calculate global transformation statistics across combined sets.
- **Mandatory Baseline Superiority**: The candidate model must empirically outperform a trivial dummy baseline ($\text{F1}_{\text{model}} > \text{F1}_{\text{dummy}}$ for classification; $\text{RMSE}_{\text{model}} < \text{RMSE}_{\text{dummy}}$ for regression).
- **Probability Calibration**: Predicted multi-class probabilities must satisfy $\sum_{c=1}^C \hat{P}_c = 1.0 \pm 10^{-6}$.

### Domain 4: Exact Mathematical Programming (LP / MILP)
- **Constraint Slack Feasibility**: For inequality constraints $Ax \le b$, assert slack $s_i = b_i - (Ax)_i \ge -\epsilon$ ($\epsilon = 10^{-6}$). Negative slacks indicate infeasibility.
- **Integrality Invariance**: For integer or binary variables, assert $|x_j - \text{round}(x_j)| \le 10^{-6}$ (zero fractional relaxation leakage).
- **Solver Termination Status**: Assert solver exit status is `OPTIMAL` / `CONVERGED`.

### Domain 5: Modular Metaheuristics & Search Diagnostics
- **Monotonic Best-Found Trajectory**: Enforce $f_{\text{best}}(t+1) \le f_{\text{best}}(t) + 10^{-9}$ across all iterations/generations.
- **Cooling Validity**: For Simulated Annealing, enforce temperature decay $0 < T_{t+1} < T_t$ and valid Metropolis transition bounds ($P \in [0, 1]$).
- **Spatial Diversity & Stagnation**: Continuously monitor population dispersion ($\text{Div} = \frac{1}{Nd}\sum |s_{ij} - \bar{s}_j|$) to actively prevent premature convergence.
- **Hard Feasibility**: The returned best solution $s^*$ must satisfy all domain constraints ($g_i(s^*) \le 0$).

---

## 4. First-Principles Generalization & Theoretical Research Protocol

When requested to explain mathematical concepts or implement analytical algorithms, statistical tests, or heuristics not covered by Skills 1–5 (e.g., survival analysis, time series forecasting, Bayesian optimization, Ant Colony Optimization, or causal inference), adhere strictly to this 4-step generalization meta-protocol:

1. **Step A: Canonical Mathematical Formulation**:
   - Explicitly lay out the formulation in LaTeX/plain text before code touches disk: indices, decision parameters, state space, loss/objective function, boundary constraints, and theoretical assumptions.
   - For conceptual questions: provide formal mathematical intuition, geometric interpretations, and known failure modes (e.g., condition number explosion, curse of dimensionality).
2. **Step B: Invariant Deduction (Definition of Failure)**:
   - Deduce at least two deterministic or probabilistic invariants governing mathematical validity:
     - Examples: Probability bounds ($0 \le P \le 1, \sum P = 1$), energy monotonicity, non-negativity, stationarity, or conservation laws.
     - Implement explicit runtime assertions (`assert` in Python, `stopifnot()` in R) testing these invariants.
3. **Step C: Trivial Baseline Benchmarking**:
   - Determine the simplest naive benchmark (e.g., predicting previous step $y_{t-1}$ for time series, random permutation for TSP, or sample mean).
   - Assert and report empirical lift over this baseline.
4. **Step D: Dual Verification Deliverables**:
   - Produce a structured terminal summary table comparing candidate performance to the trivial baseline.
   - Save a high-resolution 2-panel diagnostic figure to an `output/` directory (e.g., Forecast vs. Actual + Residuals, or Objective vs. Pheromone Entropy).

---

## 5. Script Execution & Environment Verification

Before considering any task or microstep complete:
- **Execution & Status**: Execute the script directly in the terminal to guarantee return code `0` with zero unhandled exceptions.
  - Windows PowerShell activation: `(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\.venv\Scripts\Activate.ps1")`
  - Direct binary execution (preferred for CI/CD): `& ".\.venv\Scripts\python.exe" script.py` or `Rscript script.R`
- **Artifact Verification**: Confirm the generated visualization file exists in `output/` and has non-zero size.
- **Terminal Review**: Inspect the terminal output printout to double-check that analytical results match domain expectations.
