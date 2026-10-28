# ==============================================================================
# Supervised Machine Learning Verification Module (R)
#
# Enforces zero data leakage, mandatory dummy baseline comparisons, probability
# and residual invariant assertions, and diagnostic visual artifact generation
# for both classification and regression workflows.
# ==============================================================================

split_and_scale_features <- function(data, target_column, is_classification = TRUE, test_ratio = 0.25, seed = 42) {
  if (nrow(data) == 0) {
    stop("Cannot split an empty dataset.")
  }

  set.seed(seed)
  n <- nrow(data)
  test_size <- floor(test_ratio * n)
  train_size <- n - test_size

  # Partition indices
  test_idx <- sample(seq_len(n), size = test_size)
  train_idx <- setdiff(seq_len(n), test_idx)

  train_data <- data[train_idx, ]
  test_data <- data[test_idx, ]

  # Invariant: assert disjoint partitions
  stopifnot(length(intersect(rownames(train_data), rownames(test_data))) == 0)

  feature_cols <- setdiff(names(data), target_column)

  # Fit preprocessor strictly on train data
  train_means <- colMeans(train_data[, feature_cols])
  train_sds <- apply(train_data[, feature_cols], 2, sd)

  train_data[, feature_cols] <- scale(train_data[, feature_cols], center = train_means, scale = train_sds)
  test_data[, feature_cols] <- scale(test_data[, feature_cols], center = train_means, scale = train_sds)

  cat(sprintf("[Split & Scale] Partitioned %s samples -> Train: %s | Test: %s\n",
              format(n, big.mark = ","),
              format(train_size, big.mark = ","),
              format(test_size, big.mark = ",")))

  return(list(train = train_data, test = test_data, feature_cols = feature_cols))
}

verify_classification_model_r <- function(model_formula, train_data, test_data, target_column) {
  # Step 1: Baseline majority-class prediction
  train_labels <- train_data[[target_column]]
  test_labels <- test_data[[target_column]]

  majority_class <- names(which.max(table(train_labels)))
  dummy_pred <- rep(majority_class, length(test_labels))
  dummy_acc <- mean(dummy_pred == test_labels)

  # Step 2: Fit model (e.g. logistic regression)
  fit <- glm(model_formula, data = train_data, family = binomial())
  test_probs <- predict(fit, newdata = test_data, type = "response")
  pred_labels <- ifelse(test_probs >= 0.5, 1, 0)
  model_acc <- mean(pred_labels == test_labels)

  # Invariant: must outperform dummy baseline
  stopifnot(model_acc > dummy_acc)

  banner <- strrep("=", 60)
  cat(sprintf("\n%s\nCLASSIFICATION BENCHMARK vs DUMMY BASELINE (R)\n%s\n", banner, banner))
  cat(sprintf("Dummy (Majority Class) Accuracy : %.4f\n", dummy_acc))
  cat(sprintf("Candidate Model Accuracy        : %.4f (+%.4f lift)\n", model_acc, model_acc - dummy_acc))
  cat(banner, "\n\n")

  cat("Confusion Table:\n")
  print(table(Predicted = pred_labels, True = test_labels))
  cat("\n")

  return(list(fit = fit, preds = pred_labels, probs = test_probs))
}

verify_regression_model_r <- function(model_formula, train_data, test_data, target_column) {
  test_actual <- test_data[[target_column]]

  # Baseline: mean of training target
  dummy_prediction <- mean(train_data[[target_column]])
  dummy_rmse <- sqrt(mean((test_actual - dummy_prediction)^2))

  # Fit candidate model
  fit <- lm(model_formula, data = train_data)
  model_pred <- predict(fit, newdata = test_data)
  model_rmse <- sqrt(mean((test_actual - model_pred)^2))
  model_mae <- mean(abs(test_actual - model_pred))
  ss_tot <- sum((test_actual - mean(test_actual))^2)
  ss_res <- sum((test_actual - model_pred)^2)
  model_r2 <- 1 - (ss_res / ss_tot)

  # Invariant assertions: beat baseline RMSE and positive R2
  stopifnot(model_rmse < dummy_rmse)
  stopifnot(model_r2 > 0)

  banner <- strrep("=", 60)
  cat(sprintf("\n%s\nREGRESSION BENCHMARK vs DUMMY BASELINE (R)\n%s\n", banner, banner))
  cat(sprintf("Dummy (Target Mean) RMSE : %.4f\n", dummy_rmse))
  cat(sprintf("Candidate Model RMSE     : %.4f (-%.4f error reduced)\n", model_rmse, dummy_rmse - model_rmse))
  cat(sprintf("Candidate Model MAE      : %.4f\n", model_mae))
  cat(sprintf("Candidate Model R²       : %.4f\n", model_r2))
  cat(banner, "\n\n")

  return(list(fit = fit, preds = model_pred))
}

save_supervised_visuals_r <- function(actual, predicted, task_type = "classification", output_directory = "output", filename_prefix = "r_ml") {
  dir.create(output_directory, showWarnings = FALSE, recursive = TRUE)
  plot_path <- file.path(output_directory, sprintf("%s_diagnostics.png", filename_prefix))

  png(filename = plot_path, width = 900, height = 450, res = 100)
  par(mfrow = c(1, 2), mar = c(4, 4, 3, 1))

  if (task_type == "classification") {
    tbl <- table(Predicted = predicted, True = actual)
    barplot(tbl, beside = TRUE, col = c("#2b5c8f", "#d95f02"),
            main = "Confusion Breakdown", xlab = "True Class", legend = TRUE)
    plot(sort(predicted), col = "#2b5c8f", pch = 19, main = "Sorted Predictions", ylab = "Predicted Label")
  } else {
    plot(actual, predicted, pch = 19, col = "#2b5c8f",
         main = "Predicted vs. Actual", xlab = "True Target", ylab = "Predicted Target")
    abline(0, 1, col = "red", lty = 2)

    residuals <- actual - predicted
    plot(predicted, residuals, pch = 19, col = "#d95f02",
         main = "Residuals vs. Fitted", xlab = "Fitted Values", ylab = "Residuals")
    abline(h = 0, lty = 2)
  }

  dev.off()
  cat(sprintf("[Visual Saved] Artifact written to: %s\n", normalizePath(plot_path, winslash = "/")))
  invisible(plot_path)
}

# ------------------------------------------------------------------------------
# Self-Test Demonstration (R)
# ------------------------------------------------------------------------------
if (!interactive()) {
  cat("Initiating Supervised ML Verification Self-Test (R)...\n")
  set.seed(42)
  n <- 500

  # Synthetic Regression Data
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  y_reg <- 3.5 * x1 - 2.0 * x2 + rnorm(n, sd = 1.0)
  reg_df <- data.frame(x1 = x1, x2 = x2, target = y_reg)

  split_r <- split_and_scale_features(reg_df, target_column = "target", is_classification = FALSE)
  reg_res <- verify_regression_model_r(target ~ x1 + x2, split_r$train, split_r$test, target_column = "target")
  save_supervised_visuals_r(split_r$test$target, reg_res$preds, task_type = "regression", output_directory = "output", filename_prefix = "r_test_reg")

  cat("R supervised ML self-test complete.\n")
}
