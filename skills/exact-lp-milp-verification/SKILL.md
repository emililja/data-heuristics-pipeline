---
name: exact-lp-milp-verification
description: >-
  Solves and verifies Linear Programming (LP) and Mixed-Integer Linear Programming (MILP)
  models in Python or R with strict constraint feasibility audits, slack margin analysis,
  integrality checks, and visual sensitivity dashboards.
---

# Exact LP / MILP Verification Skill (Polyglot)

Use this skill whenever formulating, solving, or verifying Linear Programs (LP) and Mixed-Integer Linear Programs (MILP). This workflow guarantees mathematical feasibility, checks constraint slack margins ($s_i = b_i - A_i x \ge 0$), asserts integer restrictions, and generates visual sensitivity artifacts across **Python** and **R**.

---

## Universal Verification Flow

```
[ Stage 1: Formulation & Matrix Specification ]
                        │
                        ▼
[ Stage 2: Solver Execution & Status Validation ]
                        │
                        ▼
[ Stage 3: Constraint Slack & Feasibility Audit ]
                        │
                        ▼
[ Stage 4: Integrality Invariant Verification ]
                        │
                        ▼
[ Stage 5: Dual Verification (Terminal Table & Saved Visuals) ]
```

---

## Reference Implementation Scripts

- **Python Module**: [`scripts/lp_milp_solver.py`](./scripts/lp_milp_solver.py)
- **R Module**: [`scripts/lp_milp_solver.R`](./scripts/lp_milp_solver.R)

---

## Mathematical Invariants & Rules of Thumb

### 1. General Mathematical Formulation
Given decision variable vector $x = [x_1, \dots, x_n]^T \in \mathbb{R}^n$:
$$\min_{x} \quad z = c^T x$$
$$\text{subject to} \quad A_{\text{ub}} x \le b_{\text{ub}}, \quad A_{\text{eq}} x = b_{\text{eq}}, \quad l_j \le x_j \le u_j$$
$$\text{with integer restrictions} \quad x_j \in \mathbb{Z}, \quad \forall j \in \mathcal{I} \subseteq \{1, \dots, n\}$$
*(For maximization problems, invert the cost vector: $\max c^T x \iff \min -c^T x$).*

### 2. Constraint Slack & Feasibility Invariant
For every inequality constraint $i \in \{1, \dots, m_{\text{ub}}\}$, calculate slack:
$$s_i = b_{\text{ub}, i} - (A_{\text{ub}} x)_i$$
- **Feasibility Assertion**: $s_i \ge -\epsilon$ ($\epsilon = 10^{-6}$). Negative slack indicates constraint violation.
- **Binding / Active Constraint**: $s_i \approx 0$ (within $10^{-5}$). This constraint actively restricts the objective.
- **Non-Binding Constraint**: $s_i > 10^{-5}$. Surplus capacity is available.

### 3. Integrality Invariant
For every variable flagged as integer or binary ($j \in \mathcal{I}$):
$$|x_j - \text{round}(x_j)| \le 10^{-6}$$
Assert that no fractional relaxation values leak into integer solutions.

---

## Language Implementation Patterns

### Pattern A: Python (`scipy.optimize.milp`, `scipy.optimize.linprog`, `matplotlib`, `pandas`)

#### 1. MILP Formulation and Invariant Solving
```python
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp

def solve_milp_with_audit(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    integrality: np.ndarray,
    variable_names: list[str],
    constraint_names: list[str],
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Solve MILP via HiGHS, assert feasibility and integrality, and compute slacks."""
    num_vars = len(c)
    lb = np.zeros(num_vars) if lower_bounds is None else lower_bounds
    ub = np.full(num_vars, np.inf) if upper_bounds is None else upper_bounds
    
    constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    res = milp(c=c, integrality=integrality, constraints=constraints, bounds=(lb, ub))
    
    # Invariant 1: Solver status must be optimal
    assert res.success, f"MILP Solver failed: {res.status} ({res.message})"
    
    x_opt = res.x
    opt_val = res.fun
    
    # Invariant 2: Integrality check
    int_indices = np.where(integrality == 1)[0]
    for idx in int_indices:
        val = x_opt[idx]
        assert abs(val - round(val)) < 1e-6, f"Integrality violated on {variable_names[idx]}: {val}"
        
    # Invariant 3: Constraint slack audit
    lhs_values = A_ub @ x_opt
    slacks = b_ub - lhs_values
    
    # Assert non-negative slacks
    assert (slacks >= -1e-6).all(), "Feasibility violated: negative slacks detected!"
    
    audit_df = pd.DataFrame({
        "constraint": constraint_names,
        "lhs": lhs_values,
        "operator": "<=",
        "rhs": b_ub,
        "slack": slacks,
        "status": np.where(np.abs(slacks) < 1e-5, "ACTIVE (BINDING)", "FEASIBLE (SLACK)")
    })
    
    print("\n" + "=" * 65)
    print("MILP OPTIMAL SOLUTION & CONSTRAINT SLACK AUDIT")
    print("=" * 65)
    print(f"Optimal Objective Value: {opt_val:.4f}\n")
    print("Optimal Decision Variables:")
    for name, val in zip(variable_names, x_opt):
        print(f"  - {name}: {val:.4f}")
    print("\nConstraint Feasibility Audit Table:")
    print(audit_df.to_string(index=False))
    print("=" * 65 + "\n")
    
    return x_opt, opt_val, audit_df
```

#### 2. Save Constraint Slack Visualization in `output/`
```python
import pathlib
import matplotlib.pyplot as plt

def save_slack_plot(audit_df: pd.DataFrame, output_dir: str = "output", prefix: str = "milp") -> pathlib.Path:
    """Save horizontal bar chart of constraint slacks, highlighting binding constraints."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#d95f02" if s == "ACTIVE (BINDING)" else "#2b5c8f" for s in audit_df["status"]]
    
    y_pos = range(len(audit_df))
    ax.barh(y_pos, audit_df["slack"], color=colors, alpha=0.8, edgecolor="black")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(audit_df["constraint"])
    ax.invert_yaxis()
    ax.set_xlabel("Slack Margin (Capacity Remaining)")
    ax.set_title("Constraint Slack Audit (Orange = Active/Binding, Blue = Feasible Slack)", fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")
    
    fig.tight_layout()
    plot_file = path / f"{prefix}_slack_margins.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Constraint slack artifact: {plot_file.resolve()}")
    return plot_file
```

---

### Pattern B: R (`lpSolve`, `stats`)

#### 1. LP / MILP Formulation and Invariant Audit
```r
solve_milp_r <- function(obj_coeffs, A_mat, dir_vec, rhs_vec, int_vars, var_names, constr_names) {
  # Requires lpSolve package or base matrix audit
  if (requireNamespace("lpSolve", quietly = TRUE)) {
    res <- lpSolve::lp(
      direction = "min",
      objective.in = obj_coeffs,
      const.mat = A_mat,
      const.dir = dir_vec,
      const.rhs = rhs_vec,
      int.vec = int_vars
    )
    stopifnot(res$status == 0)
    x_opt <- res$solution
    opt_val <- res$objval
  } else {
    stop("lpSolve package required for R MILP execution.")
  }
  
  # Slack audit
  lhs_values <- as.vector(A_mat %*% x_opt)
  slacks <- rhs_vec - lhs_values
  
  stopifnot(all(slacks >= -1e-6))
  
  cat("\n==================================================\n")
  cat("MILP OPTIMAL SOLUTION (R)\n")
  cat("==================================================\n")
  cat(sprintf("Optimal Objective Value: %.4f\n\n", opt_val))
  for (i in seq_along(x_opt)) {
    cat(sprintf("  - %s: %.4f\n", var_names[i], x_opt[i]))
  }
  cat("==================================================\n\n")
  return(list(solution = x_opt, objval = opt_val, slacks = slacks))
}
```

---

## Universal Verification Checklist

Before marking any LP/MILP task complete:
1. **Mathematical Formulation**: State objective sense, variable indices, and constraints explicitly.
2. **Solver Termination**: Verify exit status is `0` (`OPTIMAL`).
3. **Slack Non-Negativity**: Verify all inequality constraint slacks satisfy $s_i \ge -\epsilon$ (zero violations).
4. **Integrality Invariants**: Confirm zero fractional leakage on all integer/binary decision variables.
5. **Visual Slack Margin**: Confirm `output/milp_slack_margins.png` is generated and saved.
