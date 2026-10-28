"""Exact Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) module.

Enforces solver optimality verification, constraint feasibility audits, slack margin
analysis, integrality checks, and visual sensitivity artifact generation.
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, milp


def audit_linear_constraints(
    A_matrix: np.ndarray,
    b_vector: np.ndarray,
    solution: np.ndarray,
    constraint_names: Sequence[str],
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Audit inequality constraints A @ x <= b and calculate exact slack margins."""
    lhs_values = A_matrix @ solution
    slack_margins = b_vector - lhs_values

    # Feasibility Invariant: slack margins must be non-negative (within tolerance)
    negative_violations = np.where(slack_margins < -tolerance)[0]
    if len(negative_violations) > 0:
        violating_names = [constraint_names[i] for i in negative_violations]
        raise ValueError(
            f"Constraint feasibility violated on constraints: {violating_names}. "
            f"Minimum slack: {slack_margins.min():.2e}"
        )

    statuses = [
        "ACTIVE (BINDING)" if abs(slack) < 1e-5 else "FEASIBLE (SLACK)"
        for slack in slack_margins
    ]

    return pd.DataFrame(
        {
            "constraint": list(constraint_names),
            "lhs": lhs_values,
            "operator": "<=",
            "rhs": b_vector,
            "slack": slack_margins,
            "status": statuses,
        }
    )


def solve_and_verify_milp(
    c: np.ndarray,
    A_ub: np.ndarray,
    b_ub: np.ndarray,
    integrality: np.ndarray | None = None,
    lower_bounds: np.ndarray | None = None,
    upper_bounds: np.ndarray | None = None,
    variable_names: Sequence[str] | None = None,
    constraint_names: Sequence[str] | None = None,
    problem_name: str = "Optimization Problem",
) -> tuple[np.ndarray, float, pd.DataFrame]:
    """Solve LP/MILP problem via HiGHS, assert feasibility and integrality, and print audit."""
    num_vars = len(c)
    var_labels = (
        [f"x_{i}" for i in range(1, num_vars + 1)]
        if variable_names is None
        else list(variable_names)
    )
    constr_labels = (
        [f"c_{j}" for j in range(1, len(b_ub) + 1)]
        if constraint_names is None
        else list(constraint_names)
    )

    lb = np.zeros(num_vars) if lower_bounds is None else lower_bounds
    ub = np.full(num_vars, np.inf) if upper_bounds is None else upper_bounds
    int_mask = np.zeros(num_vars) if integrality is None else integrality

    linear_constraints = LinearConstraint(A_ub, -np.inf, b_ub)
    result = milp(
        c=c,
        integrality=int_mask,
        constraints=linear_constraints,
        bounds=(lb, ub),
    )

    # Invariant 1: Solver status must be optimal
    assert result.success, (
        f"MILP solver did not converge: status {result.status} ({result.message})"
    )

    optimal_x = result.x
    optimal_objective = float(result.fun)

    # Invariant 2: Integrality assertion on discrete variables
    integer_indices = np.where(int_mask == 1)[0]
    for idx in integer_indices:
        val = optimal_x[idx]
        discrepancy = abs(val - round(val))
        assert discrepancy < 1e-6, (
            f"Integrality invariant violated on variable '{var_labels[idx]}': "
            f"value {val:.6f} has fractional offset {discrepancy:.2e}"
        )

    # Invariant 3: Constraint feasibility and slack audit
    audit_table = audit_linear_constraints(
        A_matrix=A_ub,
        b_vector=b_ub,
        solution=optimal_x,
        constraint_names=constr_labels,
    )

    delimiter = "=" * 65
    print(f"\n{delimiter}\n{problem_name.upper()}: OPTIMAL SOLUTION & SLACK AUDIT\n{delimiter}")
    print(f"Optimal Objective Value: {optimal_objective:.4f}")
    print(f"Solver Message: {result.message}\n")

    print("Optimal Decision Variables:")
    for name, val in zip(var_labels, optimal_x):
        print(f"  - {name}: {val:.4f}")

    print("\nConstraint Feasibility & Slack Margins:")
    print(audit_table.to_string(index=False))
    print(f"{delimiter}\n")

    return optimal_x, optimal_objective, audit_table


def save_slack_margin_figure(
    audit_dataframe: pd.DataFrame,
    output_directory: str = "output",
    filename_prefix: str = "milp",
) -> pathlib.Path:
    """Save horizontal bar chart of constraint slacks, color-coding binding bottlenecks."""
    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, ax = plt.subplots(figsize=(10, 5))

    bar_colors = [
        "#d95f02" if status == "ACTIVE (BINDING)" else "#2b5c8f"
        for status in audit_dataframe["status"]
    ]

    y_positions = list(range(len(audit_dataframe)))
    ax.barh(
        y_positions,
        audit_dataframe["slack"],
        color=bar_colors,
        alpha=0.85,
        edgecolor="black",
        height=0.6,
    )
    ax.set_yticks(y_positions)
    ax.set_yticklabels(audit_dataframe["constraint"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Remaining Slack / Capacity Margin", fontsize=10)
    ax.set_title(
        f"Constraint Slack Audit ({filename_prefix.upper()})\n"
        "(Orange = Active Bottleneck / Binding, Blue = Feasible Capacity Margin)",
        fontsize=11,
        fontweight="bold",
    )
    ax.grid(True, linestyle="--", alpha=0.5, axis="x")

    figure.tight_layout()
    artifact_file = output_path / f"{filename_prefix}_slack_margins.png"
    figure.savefig(artifact_file, dpi=300)
    plt.close(figure)

    print(f"[Visual Saved] Slack margin artifact written to: {artifact_file.resolve()}")
    return artifact_file


if __name__ == "__main__":
    print("Initiating Exact LP / MILP Verification Self-Test...")

    # Problem 1: Continuous Production Planning LP
    # Maximize Profit = 40*x1 + 30*x2  --> Minimize Cost = -40*x1 - 30*x2
    # Subject to:
    #   Labor: 2*x1 + x2 <= 100
    #   Material: x1 + x2 <= 80
    #   Machine Limit: x1 <= 40
    #   x1, x2 >= 0
    c_lp = np.array([-40.0, -30.0])
    A_lp = np.array(
        [
            [2.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
        ]
    )
    b_lp = np.array([100.0, 80.0, 40.0])
    lp_vars = ["Product_A", "Product_B"]
    lp_constrs = ["Labor_Hours", "Raw_Material", "Machine_Limit"]

    x_lp, z_lp, audit_lp = solve_and_verify_milp(
        c=c_lp,
        A_ub=A_lp,
        b_ub=b_lp,
        integrality=np.array([0, 0]),
        variable_names=lp_vars,
        constraint_names=lp_constrs,
        problem_name="Production Planning LP",
    )
    save_slack_margin_figure(
        audit_lp, output_directory="output", filename_prefix="production_lp"
    )

    # Problem 2: Mixed-Integer Binary Knapsack / Project Selection MILP
    # Maximize Value = 15*y1 + 25*y2 + 35*y3 + 45*y4 + 55*y5 --> Minimize -Value
    # Subject to:
    #   Weight Budget: 2*y1 + 3*y2 + 5*y3 + 7*y4 + 9*y5 <= 16
    #   Mutually Exclusive: y1 + y2 <= 1
    #   All y_j in {0, 1}
    c_milp = np.array([-15.0, -25.0, -35.0, -45.0, -55.0])
    A_milp = np.array(
        [
            [2.0, 3.0, 5.0, 7.0, 9.0],
            [1.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    b_milp = np.array([16.0, 1.0])
    milp_vars = [f"Project_{i}" for i in range(1, 6)]
    milp_constrs = ["Budget_Limit", "Mutually_Exclusive_Projects"]

    x_milp, z_milp, audit_milp = solve_and_verify_milp(
        c=c_milp,
        A_ub=A_milp,
        b_ub=b_milp,
        integrality=np.ones(5),  # All 5 are integer/binary
        lower_bounds=np.zeros(5),
        upper_bounds=np.ones(5),
        variable_names=milp_vars,
        constraint_names=milp_constrs,
        problem_name="Project Selection Binary MILP",
    )
    save_slack_margin_figure(
        audit_milp, output_directory="output", filename_prefix="project_milp"
    )

    print("Exact LP / MILP verification self-test successfully verified for both LP and MILP.")
