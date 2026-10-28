---
name: mathematical-research-formulation
description: >-
  Researches, formulates, and validates mathematical concepts, theoretical proofs,
  and novel analytics or heuristic algorithms that extend beyond the pre-built skills,
  enforcing first-principles verification across Python and R.
---

# Mathematical Research & First-Principles Formulation Skill (Polyglot)

Use this skill whenever:
1. The user asks deep conceptual, statistical, or mathematical questions (e.g., derivations, matrix decompositions, KKT duality, asymptotic convergence proofs, or loss function properties).
2. The user requests an analytical method, statistical model, or heuristic algorithm not covered by other skills in the plugin (e.g., Time-Series ARIMA/ETS, Survival Analysis, Bayesian Optimization / Gaussian Processes, Ant Colony Optimization, Particle Swarm Optimization, or SHAP explainability).

This workflow guarantees that theoretical rigor, testable invariant assertions, naive baseline comparisons, and diagnostic visual proof are maintained for any novel technique.

---

## Universal Research & Formulation Workflow

```
[ Step 1: Formal Mathematical Formulation & Theoretical Assumptions ]
                                  │
                                  ▼
[ Step 2: Invariant Deduction (Defining Mathematical Failure) ]
                                  │
                                  ▼
[ Step 3: Trivial Baseline Selection & Superiority Assertion ]
                                  │
                                  ▼
[ Step 4: Dual Verification (Terminal Table & Saved Visuals) ]
                                  │
                                  ▼
[ Step 5: Empirical Synthesis & Anomaly Reporting ]
```

---

## 1. Mathematical Research & Conceptual Inquiries

When explaining mathematical concepts, theories, or algorithms:
- **Canonical Notation**: Always define indices, variables, and parameters explicitly using standard LaTeX notation before referencing code.
- **Geometric Intuition & Derivations**: Connect algebraic equations to geometric representations (e.g., orthogonal projections in PCA, hyperplanes and polyhedra in LP, or energy landscapes in metaheuristics).
- **Theoretical Assumptions & Failure Modes**: Detail critical assumptions (e.g., i.i.d. observations, stationarity, homoscedasticity, convexity) and explicitly describe how the method degrades when assumptions are violated (e.g., ill-conditioned Hessian, collinearity, or local optimum traps).

---

## 2. The 4-Step First-Principles Algorithm Synthesis

When implementing a technique outside the standard skills, adhere to this four-step synthesis cycle:

### Step A: Canonical Mathematical Formulation
State the explicit optimization, estimation, or prediction objective:
$$\min_{\theta \in \Theta} \quad \mathcal{L}(\theta; \mathcal{D}) \quad \text{subject to} \quad g_i(\theta) \le 0, \quad \forall i \in \{1, \dots, m\}$$
Define parameter dimensions, search spaces, and domain boundaries.

### Step B: Invariant Deduction (Defining Failure)
Deduce at least two testable invariants that define mathematical correctness:
- **Probabilistic Invariants**: Probabilities bounded in $[0, 1]$ and summing to $1.0 \pm 10^{-6}$.
- **Monotonic / Lyapunov Invariants**: Best-found cost $f(t+1) \le f(t)$, or loss decreasing along gradient descent under line search.
- **Conservation & Boundaries**: Variance conservation, non-negative distances, or non-negative constraint slacks.
- **Stationarity & Residuals**: Residual errors $e_t$ must have near-zero mean and bounded variance.

### Step C: Trivial Baseline Selection
Every novel technique must prove empirical value over a trivial zero-knowledge heuristic:
| Problem Domain | Recommended Trivial Baseline | Validation Criterion |
| :--- | :--- | :--- |
| **Time Series Forecasting** | Naive Lag-1 Persistence ($\hat{y}_{t} = y_{t-1}$) | $\text{RMSE}_{\text{model}} < \text{RMSE}_{\text{persistence}}$ |
| **Survival Analysis** | Kaplan-Meier population median survival | Concordance Index $C > 0.50$ |
| **Combinatorial Search (ACO/PSO)**| Uniform random sampling / random walk | $f_{\text{best}}(\text{heuristic}) < f_{\text{best}}(\text{random})$ |
| **Anomaly Detection** | Simple 3-sigma / IQR boundary filter | Precision / Recall lift over random guessing |

### Step D: Dual Verification Deliverables
1. **Terminal Double-Check Table**: Print parameter estimates, baseline comparisons, and invariant test results directly to stdout.
2. **Saved Visual Dashboard**: Persist a high-resolution 2-panel figure to `output/` (e.g., Model vs. Baseline Forecast + Residual ACF plot).

---

## 3. Polyglot Reference Implementation Template

### Scenario: First-Principles Time Series Forecasting with Naive Persistence Baseline

#### Python Pattern (`numpy`, `pandas`, `matplotlib`)

```python
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def verify_novel_time_series_model(
    series: pd.Series,
    forecast_horizon: int = 10,
    output_dir: str = "output",
    prefix: str = "time_series",
) -> tuple[pd.Series, pd.Series, float, float]:
    """Demonstrates first-principles formulation, persistence baseline, and diagnostic visuals."""
    # Step 1: Formulation & Split
    train_series = series.iloc[:-forecast_horizon]
    test_series = series.iloc[-forecast_horizon:]
    
    # Step 2: Trivial Baseline (Lag-1 Persistence: y_hat_t = y_{t-1})
    last_train_value = train_series.iloc[-1]
    baseline_forecast = pd.Series(last_train_value, index=test_series.index)
    baseline_rmse = float(np.sqrt(np.mean((test_series - baseline_forecast) ** 2)))
    
    # Step 3: Candidate Model (e.g., Holt-Winters / Linear Trend Forecast)
    time_indices = np.arange(len(train_series))
    trend_slope, trend_intercept = np.polyfit(time_indices, train_series.values, deg=1)
    future_indices = np.arange(len(train_series), len(train_series) + forecast_horizon)
    model_forecast = pd.Series(trend_slope * future_indices + trend_intercept, index=test_series.index)
    model_rmse = float(np.sqrt(np.mean((test_series - model_forecast) ** 2)))
    
    # Step 4: Mathematical Invariant Checks
    residuals = test_series - model_forecast
    assert not np.isnan(model_rmse), "Forecast RMSE is NaN!"
    assert (np.abs(residuals.mean()) < baseline_rmse), "Severe residual bias detected!"
    
    print("\n" + "=" * 60)
    print("TIME SERIES FIRST-PRINCIPLES BENCHMARK")
    print("=" * 60)
    print(f"Naive Persistence Baseline RMSE : {baseline_rmse:.4f}")
    print(f"Candidate Model Trend RMSE      : {model_rmse:.4f} "
          f"({'BEATS BASELINE' if model_rmse < baseline_rmse else 'FAILED BASELINE'})")
    print(f"Forecast Residual Mean (Bias)   : {residuals.mean():.4f}")
    print("=" * 60 + "\n")
    
    # Step 5: Save Visual Dashboard
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # Panel 1: Forecast vs Actual
    axes[0].plot(train_series.index[-30:], train_series.iloc[-30:], label="Historical Train", color="black")
    axes[0].plot(test_series.index, test_series, label="Actual Ground Truth", color="#2b5c8f", lw=2)
    axes[0].plot(model_forecast.index, model_forecast, label=f"Model Forecast (RMSE: {model_rmse:.2f})", color="#d95f02", lw=2)
    axes[0].plot(baseline_forecast.index, baseline_forecast, label=f"Naive Persistence (RMSE: {baseline_rmse:.2f})", color="gray", linestyle="--")
    axes[0].set_title("Forecast vs. Naive Baseline", fontweight="bold")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Panel 2: Residual Distribution
    axes[1].hist(residuals, bins=10, color="#7570b3", edgecolor="black", alpha=0.8)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_title("Forecast Residual Distribution", fontweight="bold")
    axes[1].set_xlabel("Residual Error")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    fig.tight_layout()
    plot_file = path / f"{prefix}_diagnostics.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Time-series diagnostics written to: {plot_file.resolve()}")
    
    return model_forecast, baseline_forecast, model_rmse, baseline_rmse
```

---

## 4. Universal Verification Checklist

Before finalizing any novel research or beyond-scope algorithmic implementation:
1. **Formulation**: Confirm explicit equations for decision variables, loss, and domain constraints are articulated in text.
2. **Invariants**: Assert at least two mathematical failure rules (`assert` in Python, `stopifnot()` in R).
3. **Trivial Baseline**: Empirically evaluate against the simplest naive benchmark and report quantitative lift.
4. **Execution & Status**: Execute synchronously via the terminal to guarantee exit code `0`.
5. **Saved Visual Dashboard**: Confirm a 2-panel diagnostic figure is written to `output/` with size $>0$ bytes.
