---
name: supervised-ml-verification
description: >-
  Enforces strict validation, train/test leakage prevention, dummy baseline comparison,
  and diagnostic visualization for classification and regression tasks in Python or R.
---

# Supervised ML Verification Skill (Polyglot)

Use this skill whenever training, evaluating, or validating supervised machine learning models (both **Classification** and **Regression**). This workflow enforces zero-leakage data partitioning, mandatory trivial baseline comparisons, probability/residual invariant assertions, and diagnostic visual proof across **Python** and **R**.

---

## Universal Verification Flow

```
[ Stage 1: Split Isolation & Zero-Leakage Preprocessing ]
                           │
                           ▼
[ Stage 2: Trivial Baseline Benchmarking (Dummy Models) ]
                           │
                           ▼
[ Stage 3: Candidate Model Training & Invariant Assertions ]
                           │
                           ▼
[ Stage 4: Dual Verification (Terminal Tables & Saved Visuals) ]
                           │
                           ▼
[ Stage 5: Execution & Zero-Error Check ]
```

---

## Reference Implementation Scripts

- **Python Module**: [`scripts/supervised_ml.py`](./scripts/supervised_ml.py)
- **R Module**: [`scripts/supervised_ml.R`](./scripts/supervised_ml.R)

---

## Mathematical Invariants & Rules of Thumb

### 1. Zero Data Leakage Invariant
All transformations $\mathcal{T}$ (scaling, encoding, imputing) must fit **exclusively** on training data:
$$\hat{\theta}_{\text{preproc}} = \text{fit}(X_{\text{train}}), \quad X_{\text{train}}^{\text{scaled}} = \mathcal{T}(X_{\text{train}}; \hat{\theta}), \quad X_{\text{test}}^{\text{scaled}} = \mathcal{T}(X_{\text{test}}; \hat{\theta})$$
*Assert*: Global dataset statistics must never be computed before partitioning.

### 2. Mandatory Baseline Superiority
A model is only valid if it empirically outperforms a zero-knowledge trivial benchmark:
- **Classification**: $\text{F1}_{\text{macro}}(f_{\text{model}}, D_{\text{test}}) > \text{F1}_{\text{macro}}(f_{\text{dummy}}, D_{\text{test}})$ where dummy predicts the majority class.
- **Regression**: $\text{RMSE}(f_{\text{model}}, D_{\text{test}}) < \text{RMSE}(\bar{y}_{\text{train}}, D_{\text{test}})$ where dummy predicts the training target mean $\bar{y}_{\text{train}}$.

### 3. Probability & Residual Invariants
- **Classification Probabilities**: For all samples $i$ and classes $C$:
  $$\sum_{c=1}^C \hat{P}(y_i = c \mid x_i) = 1.0 \pm 10^{-6}, \quad 0 \le \hat{P} \le 1$$
- **Regression Residuals**: Residuals $e_i = y_i - \hat{y}_i$ must have near-zero mean $\frac{1}{n}\sum e_i \approx 0$ and no severe heteroscedastic fan-out.

---

## Language Implementation Patterns

### Pattern A: Python (`scikit-learn`, `pandas`, `matplotlib`, `seaborn`)

#### 1. Split Isolation & Scaling (No Leakage)
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def split_and_scale(
    X: pd.DataFrame,
    y: pd.Series,
    is_classification: bool = True,
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Partition data and fit scaler strictly on the training partition."""
    stratify = y if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    
    # Assert zero overlap
    assert len(X_train.index.intersection(X_test.index)) == 0, "Data leakage detected: indices overlap!"
    
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns, index=X_test.index)
    
    print(f"[Split] Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
```

#### 2. Classification Pipeline & Baseline Benchmark
```python
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score

def verify_classifier(model, X_train, y_train, X_test, y_test):
    """Train model, assert superiority over majority-class dummy, and verify probabilities."""
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    dummy_f1 = f1_score(y_test, dummy.predict(X_test), average="macro")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_f1 = f1_score(y_test, y_pred, average="macro")
    
    # Assert baseline superiority
    assert model_f1 > dummy_f1, f"Model failed to beat dummy baseline (Model F1: {model_f1:.4f} vs Dummy: {dummy_f1:.4f})"
    
    # Verify probability conservation
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)
        row_sums = probabilities.sum(axis=1)
        assert (abs(row_sums - 1.0) < 1e-6).all(), "Probability conservation violated!"
        
    print("\n" + "=" * 55)
    print("CLASSIFICATION BENCHMARK vs DUMMY BASELINE")
    print("=" * 55)
    print(f"Dummy (Majority Class) F1 Macro : {dummy_f1:.4f}")
    print(f"Candidate Model F1 Macro        : {model_f1:.4f} (+{(model_f1 - dummy_f1):.4f})")
    print(f"Candidate Model Accuracy        : {accuracy_score(y_test, y_pred):.4f}")
    print("=" * 55 + "\n")
    print(classification_report(y_test, y_pred))
    return y_pred
```

#### 3. Regression Pipeline & Baseline Benchmark
```python
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

def verify_regressor(model, X_train, y_train, X_test, y_test):
    """Train model, assert superiority over mean dummy, and verify residuals."""
    dummy = DummyRegressor(strategy="mean")
    dummy.fit(X_train, y_train)
    dummy_rmse = root_mean_squared_error(y_test, dummy.predict(X_test))
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    model_rmse = root_mean_squared_error(y_test, y_pred)
    model_r2 = r2_score(y_test, y_pred)
    
    # Assert baseline superiority
    assert model_rmse < dummy_rmse, f"Model failed to beat mean baseline (Model RMSE: {model_rmse:.4f} vs Dummy: {dummy_rmse:.4f})"
    assert model_r2 > 0.0, f"Model R² is non-positive: {model_r2:.4f}"
    
    print("\n" + "=" * 55)
    print("REGRESSION BENCHMARK vs DUMMY BASELINE")
    print("=" * 55)
    print(f"Dummy (Target Mean) RMSE : {dummy_rmse:.4f}")
    print(f"Candidate Model RMSE     : {model_rmse:.4f} (reduced by {dummy_rmse - model_rmse:.4f})")
    print(f"Candidate Model MAE      : {mean_absolute_error(y_test, y_pred):.4f}")
    print(f"Candidate Model R²       : {model_r2:.4f}")
    print("=" * 55 + "\n")
    return y_pred
```

#### 4. Save Visual Diagnostics in `output/`
```python
import pathlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_classification_plots(y_test, y_pred, y_prob, output_dir="output", prefix="clf"):
    """Save confusion matrix and ROC curve."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title("Confusion Matrix", fontweight="bold")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color="#2b5c8f", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
    axes[1].set_title("ROC Curve", fontweight="bold")
    axes[1].legend(loc="lower right")
    
    fig.tight_layout()
    plot_file = path / f"{prefix}_diagnostics.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Classification diagnostics: {plot_file.resolve()}")
    return plot_file

def save_regression_plots(y_test, y_pred, output_dir="output", prefix="reg"):
    """Save predicted vs actual and residual plots."""
    path = pathlib.Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    residuals = y_test - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="#2b5c8f")
    axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    axes[0].set_title("Predicted vs. Actual", fontweight="bold")
    
    axes[1].scatter(y_pred, residuals, alpha=0.6, color="#d95f02")
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set_title("Residuals vs. Fitted", fontweight="bold")
    
    fig.tight_layout()
    plot_file = path / f"{prefix}_diagnostics.png"
    fig.savefig(plot_file, dpi=300)
    plt.close(fig)
    print(f"[Visual Saved] Regression diagnostics: {plot_file.resolve()}")
    return plot_file
```

---

### Pattern B: R (`stats`, `ggplot2`)

#### 1. Split Isolation & Scaling
```r
split_and_scale <- function(data, target_col, is_classification = TRUE, split_ratio = 0.75, seed = 42) {
  set.seed(seed)
  n <- nrow(data)
  train_idx <- sample(seq_len(n), size = floor(split_ratio * n))
  
  train_df <- data[train_idx, ]
  test_df <- data[-train_idx, ]
  
  # Assert disjoint partitions
  stopifnot(length(intersect(rownames(train_df), rownames(test_df))) == 0)
  
  feature_cols <- setdiff(names(data), target_col)
  scaler_means <- colMeans(train_df[, feature_cols])
  scaler_sds <- apply(train_df[, feature_cols], 2, sd)
  
  train_df[, feature_cols] <- scale(train_df[, feature_cols], center = scaler_means, scale = scaler_sds)
  test_df[, feature_cols] <- scale(test_df[, feature_cols], center = scaler_means, scale = scaler_sds)
  
  cat(sprintf("[Split] Train samples: %s | Test samples: %s\n",
              format(nrow(train_df), big.mark = ","), format(nrow(test_df), big.mark = ",")))
  return(list(train = train_df, test = test_df))
}
```

#### 2. Classification & Regression Benchmarking
```r
verify_regressor_r <- function(model, train_df, test_df, target_col) {
  y_test <- test_df[[target_col]]
  
  # Baseline: mean of training target
  dummy_pred <- mean(train_df[[target_col]])
  dummy_rmse <- sqrt(mean((y_test - dummy_pred)^2))
  
  y_pred <- predict(model, newdata = test_df)
  model_rmse <- sqrt(mean((y_test - y_pred)^2))
  
  # Assert superiority
  stopifnot(model_rmse < dummy_rmse)
  
  cat("\n==================================================\n")
  cat("REGRESSION BENCHMARK vs DUMMY BASELINE (R)\n")
  cat("==================================================\n")
  cat(sprintf("Dummy (Target Mean) RMSE : %.4f\n", dummy_rmse))
  cat(sprintf("Candidate Model RMSE     : %.4f (reduced by %.4f)\n", model_rmse, dummy_rmse - model_rmse))
  cat("==================================================\n\n")
  return(y_pred)
}
```

---

## Universal Verification Checklist

Before reporting completion on any Supervised ML modeling task:
1. **Zero Data Leakage**: Confirm preprocessing scalers/encoders were fit solely on the training partition.
2. **Mandatory Baseline Check**: Confirm the model outperformed the dummy baseline (F1 for classification, RMSE for regression).
3. **Mathematical Invariants**: Verify predicted probabilities sum to $1.0 \pm 10^{-6}$ and regression residuals exhibit near-zero mean.
4. **Visual Diagnostic Persistence**: Confirm diagnostic plots (`confusion_matrix`, `roc_curve`, `residuals_plot`) exist in `output/` with $>0$ bytes.
5. **Execution**: Verify scripts complete with exit code `0`.
