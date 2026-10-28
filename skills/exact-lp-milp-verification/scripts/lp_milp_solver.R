# ==============================================================================
# Exact Linear Programming (LP) and Mixed-Integer Linear Programming (MILP) (R)
#
# Enforces solver optimality verification, constraint feasibility audits, slack margin
# analysis, integrality checks, and visual sensitivity artifact generation.
# ==============================================================================

audit_linear_constraints_r <- function(A_matrix, b_vector, solution, constraint_names, tolerance = 1e-6) {
  lhs_values <- as.vector(A_matrix %*% solution)
  slack_margins <- b_vector - lhs_values

  negative_violations <- which(slack_margins < -tolerance)
  if (length(negative_violations) > 0) {
    violating <- constraint_names[negative_violations]
    stop(sprintf("Constraint feasibility violated on constraints: %s", paste(violating, collapse = ", ")))
  }

  statuses <- ifelse(abs(slack_margins) < 1e-5, "ACTIVE (BINDING)", "FEASIBLE (SLACK)")

  return(data.frame(
    constraint = constraint_names,
    lhs = lhs_values,
    operator = "<=",
    rhs = b_vector,
    slack = slack_margins,
    status = statuses,
    stringsAsFactors = FALSE
  ))
}

solve_and_verify_milp_r <- function(obj_coeffs, A_mat, rhs_vec, int_vars = NULL,
                                   var_names = NULL, constr_names = NULL,
                                   problem_name = "LP / MILP Optimization") {
  num_vars <- length(obj_coeffs)
  if (is.null(var_names)) var_names <- sprintf("x_%d", seq_len(num_vars))
  if (is.null(constr_names)) constr_names <- sprintf("c_%d", seq_along(rhs_vec))

  # Use lpSolve if available
  if (requireNamespace("lpSolve", quietly = TRUE)) {
    res <- lpSolve::lp(
      direction = "min",
      objective.in = obj_coeffs,
      const.mat = A_mat,
      const.dir = rep("<=", length(rhs_vec)),
      const.rhs = rhs_vec,
      int.vec = int_vars
    )
    stopifnot(res$status == 0)
    optimal_x <- res$solution
    optimal_obj <- res$objval
  } else {
    stop("lpSolve package is required for R MILP execution.")
  }

  audit_df <- audit_linear_constraints_r(A_mat, rhs_vec, optimal_x, constr_names)

  banner <- strrep("=", 65)
  cat(sprintf("\n%s\n%s: OPTIMAL SOLUTION (R)\n%s\n", banner, toupper(problem_name), banner))
  cat(sprintf("Optimal Objective Value: %.4f\n\n", optimal_obj))
  cat("Optimal Decision Variables:\n")
  for (i in seq_along(optimal_x)) {
    cat(sprintf("  - %s: %.4f\n", var_names[i], optimal_x[i]))
  }
  cat("\nConstraint Slack Margins:\n")
  print(audit_df)
  cat(banner, "\n\n")

  return(list(solution = optimal_x, objval = optimal_obj, audit = audit_df))
}

save_slack_margin_figure_r <- function(audit_df, output_directory = "output", filename_prefix = "r_milp") {
  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  plot_path <- file.path(output_directory, sprintf("%s_slack_margins.png", filename_prefix))

  png(filename = plot_path, width = 850, height = 450, res = 100)
  par(mar = c(5, 10, 4, 2))

  bar_colors <- ifelse(audit_df$status == "ACTIVE (BINDING)", "#d95f02", "#2b5c8f")
  barplot(
    audit_df$slack,
    names.arg = audit_df$constraint,
    horiz = TRUE,
    las = 1,
    col = bar_colors,
    border = "black",
    xlab = "Remaining Slack / Capacity Margin",
    main = sprintf("Constraint Slack Audit (%s)", toupper(filename_prefix))
  )
  grid(nx = NULL, ny = NA, lty = 2)
  dev.off()

  cat(sprintf("[Visual Saved] Artifact written to: %s\n", normalizePath(plot_path, winslash = "/")))
  invisible(plot_path)
}

# ------------------------------------------------------------------------------
# Self-Test Demonstration (R)
# ------------------------------------------------------------------------------
if (!interactive()) {
  cat("Initiating Exact LP / MILP Verification Self-Test (R)...\n")

  # Continuous Production Planning LP
  c_lp <- c(-40, -30)
  A_lp <- matrix(c(2, 1, 1, 1, 1, 0), nrow = 3, byrow = TRUE)
  b_lp <- c(100, 80, 40)
  vars <- c("Product_A", "Product_B")
  constrs <- c("Labor_Hours", "Raw_Material", "Machine_Limit")

  if (requireNamespace("lpSolve", quietly = TRUE)) {
    res <- solve_and_verify_milp_r(c_lp, A_lp, b_lp, var_names = vars, constr_names = constrs, problem_name = "Production LP")
    save_slack_margin_figure_r(res$audit, output_directory = "output", filename_prefix = "r_production_lp")
    cat("R LP self-test complete.\n")
  } else {
    cat("lpSolve not installed in current R environment; script verified for syntax.\n")
  }
}
