"""Supervised Machine Learning verification helper module.

Enforces zero data leakage, mandatory dummy baseline comparisons, probability
and residual invariant assertions, and diagnostic visual artifact generation
for both classification and regression workflows.
"""

from __future__ import annotations

import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification, make_regression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_curve,
    root_mean_squared_error,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def split_and_scale_features(
    features: pd.DataFrame,
    target: pd.Series,
    is_classification: bool = True,
    test_ratio: float = 0.25,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """Partition data and fit preprocessor strictly on the training partition."""
    if features.empty:
        raise ValueError("Cannot split an empty feature matrix.")

    if not (0.05 <= test_ratio <= 0.50):
        raise ValueError("test_ratio must be between 0.05 and 0.50.")

    stratification = target if is_classification else None
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=stratification,
    )

    # Invariant: assert zero index overlap between train and test
    overlap = X_train.index.intersection(X_test.index)
    assert len(overlap) == 0, f"Data leakage detected: {len(overlap)} overlapping indices!"

    scaler = StandardScaler()
    scaled_train_values = scaler.fit_transform(X_train)
    scaled_test_values = scaler.transform(X_test)

    scaled_train_df = pd.DataFrame(
        scaled_train_values,
        columns=features.columns,
        index=X_train.index,
    )
    scaled_test_df = pd.DataFrame(
        scaled_test_values,
        columns=features.columns,
        index=X_test.index,
    )

    print(
        f"[Split & Scale] Partitioned {len(features):,} samples -> "
        f"Train: {len(X_train):,} ({1 - test_ratio:.0%}) | "
        f"Test: {len(X_test):,} ({test_ratio:.0%})"
    )
    return scaled_train_df, scaled_test_df, y_train, y_test, scaler


def verify_classification_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Train classification model, assert superiority over dummy baseline, and verify probabilities."""
    # Step 1: Fit majority-class dummy benchmark
    dummy_model = DummyClassifier(strategy="most_frequent")
    dummy_model.fit(X_train, y_train)
    dummy_predictions = dummy_model.predict(X_test)
    dummy_f1 = f1_score(y_test, dummy_predictions, average="macro", zero_division=0)

    # Step 2: Fit candidate model and evaluate
    model.fit(X_train, y_train)
    predicted_labels = model.predict(X_test)
    model_f1 = f1_score(y_test, predicted_labels, average="macro", zero_division=0)
    model_accuracy = accuracy_score(y_test, predicted_labels)

    # Invariant: candidate model must strictly outperform dummy baseline
    assert model_f1 > dummy_f1, (
        f"Baseline failure: model macro F1 ({model_f1:.4f}) does not outperform "
        f"dummy baseline ({dummy_f1:.4f})."
    )

    # Step 3: Verify probability conservation
    predicted_probabilities: np.ndarray | None = None
    if hasattr(model, "predict_proba"):
        predicted_probabilities = model.predict_proba(X_test)
        probability_sums = predicted_probabilities.sum(axis=1)
        max_deviation = np.abs(probability_sums - 1.0).max()
        assert max_deviation < 1e-6, (
            f"Probability invariant violated: max row sum deviation is {max_deviation:.2e}"
        )

    delimiter = "=" * 60
    print(f"\n{delimiter}\nCLASSIFICATION BENCHMARK vs DUMMY BASELINE\n{delimiter}")
    print(f"Dummy (Majority Class) Macro F1 : {dummy_f1:.4f}")
    print(
        f"Candidate Model Macro F1        : {model_f1:.4f} "
        f"(+{(model_f1 - dummy_f1):.4f} lift)"
    )
    print(f"Candidate Model Accuracy        : {model_accuracy:.4f}")
    print(f"{delimiter}\n")
    print(classification_report(y_test, predicted_labels, zero_division=0))

    return predicted_labels, predicted_probabilities


def verify_regression_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> np.ndarray:
    """Train regression model, assert superiority over mean dummy, and verify residuals."""
    # Step 1: Fit target-mean dummy benchmark
    dummy_model = DummyRegressor(strategy="mean")
    dummy_model.fit(X_train, y_train)
    dummy_predictions = dummy_model.predict(X_test)
    dummy_rmse = root_mean_squared_error(y_test, dummy_predictions)

    # Step 2: Fit candidate model and evaluate
    model.fit(X_train, y_train)
    predicted_values = model.predict(X_test)
    model_rmse = root_mean_squared_error(y_test, predicted_values)
    model_mae = mean_absolute_error(y_test, predicted_values)
    model_r2 = r2_score(y_test, predicted_values)

    # Invariants: must beat baseline RMSE and have positive R2
    assert model_rmse < dummy_rmse, (
        f"Baseline failure: model RMSE ({model_rmse:.4f}) is not lower than "
        f"mean dummy baseline RMSE ({dummy_rmse:.4f})."
    )
    assert model_r2 > 0.0, f"Quality failure: model R² ({model_r2:.4f}) is non-positive."

    residuals = y_test - predicted_values
    residual_mean = float(residuals.mean())
    assert abs(residual_mean) < (dummy_rmse * 0.10), (
        f"Residual invariant warning: residual mean is biased ({residual_mean:.4f})."
    )

    delimiter = "=" * 60
    print(f"\n{delimiter}\nREGRESSION BENCHMARK vs DUMMY BASELINE\n{delimiter}")
    print(f"Dummy (Target Mean) RMSE : {dummy_rmse:.4f}")
    print(
        f"Candidate Model RMSE     : {model_rmse:.4f} "
        f"(-{(dummy_rmse - model_rmse):.4f} error reduced)"
    )
    print(f"Candidate Model MAE      : {model_mae:.4f}")
    print(f"Candidate Model R²       : {model_r2:.4f}")
    print(f"Residual Mean (Bias)     : {residual_mean:.4f}")
    print(f"{delimiter}\n")

    return predicted_values


def save_classification_plots(
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None,
    output_directory: str = "output",
    filename_prefix: str = "classification",
) -> pathlib.Path:
    """Save confusion matrix and ROC curve diagnostics."""
    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], cbar=False)
    axes[0].set_title("Confusion Matrix", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Label")
    axes[0].set_ylabel("True Label")

    # Subplot 2: ROC Curve
    if y_prob is not None and len(np.unique(y_test)) == 2:
        positive_prob = y_prob[:, 1]
        fpr, tpr, _ = roc_curve(y_test, positive_prob)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, color="#2b5c8f", lw=2.2, label=f"ROC (AUC = {roc_auc:.3f})")
        axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random Chance")
        axes[1].set_title("ROC Curve", fontsize=11, fontweight="bold")
        axes[1].set_xlabel("False Positive Rate")
        axes[1].set_ylabel("True Positive Rate")
        axes[1].legend(loc="lower right")
        axes[1].grid(True, linestyle="--", alpha=0.5)
    else:
        axes[1].text(0.5, 0.5, "ROC not applicable (Multi-class / No Proba)", ha="center")

    figure.tight_layout()
    artifact_file = output_path / f"{filename_prefix}_diagnostics.png"
    figure.savefig(artifact_file, dpi=300)
    plt.close(figure)

    print(f"[Visual Saved] Classification diagnostics: {artifact_file.resolve()}")
    return artifact_file


def save_regression_plots(
    y_test: pd.Series,
    y_pred: np.ndarray,
    output_directory: str = "output",
    filename_prefix: str = "regression",
) -> pathlib.Path:
    """Save predicted vs. actual and residual diagnostic plots."""
    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    residuals = y_test.values - y_pred
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Subplot 1: Predicted vs. Actual
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="#2b5c8f", edgecolors="none")
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Ideal")
    axes[0].set_title("Predicted vs. Actual", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("True Target")
    axes[0].set_ylabel("Predicted Target")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Residuals vs. Fitted
    axes[1].scatter(y_pred, residuals, alpha=0.6, color="#d95f02", edgecolors="none")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs. Fitted", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Fitted Values")
    axes[1].set_ylabel("Residuals")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    figure.tight_layout()
    artifact_file = output_path / f"{filename_prefix}_diagnostics.png"
    figure.savefig(artifact_file, dpi=300)
    plt.close(figure)

    print(f"[Visual Saved] Regression diagnostics: {artifact_file.resolve()}")
    return artifact_file


if __name__ == "__main__":
    print("Initiating Supervised ML Verification Self-Test...")

    # Test Task 1: Classification Verification
    print("\n--- Running Classification Self-Test ---")
    clf_features, clf_labels = make_classification(
        n_samples=600,
        n_features=8,
        n_informative=5,
        n_classes=2,
        weights=[0.65, 0.35],
        random_state=42,
    )
    clf_X = pd.DataFrame(clf_features, columns=[f"feat_{i}" for i in range(1, 9)])
    clf_y = pd.Series(clf_labels, name="label")

    X_train_c, X_test_c, y_train_c, y_test_c, _ = split_and_scale_features(
        clf_X, clf_y, is_classification=True, test_ratio=0.25
    )

    clf_model = LogisticRegression(random_state=42)
    y_pred_c, y_prob_c = verify_classification_model(
        clf_model, X_train_c, y_train_c, X_test_c, y_test_c
    )
    save_classification_plots(
        y_test_c, y_pred_c, y_prob_c, output_directory="output", filename_prefix="test_clf"
    )

    # Test Task 2: Regression Verification
    print("\n--- Running Regression Self-Test ---")
    reg_features, reg_targets = make_regression(
        n_samples=600,
        n_features=8,
        n_informative=6,
        noise=12.0,
        random_state=42,
    )
    reg_X = pd.DataFrame(reg_features, columns=[f"reg_{i}" for i in range(1, 9)])
    reg_y = pd.Series(reg_targets, name="target")

    X_train_r, X_test_r, y_train_r, y_test_r, _ = split_and_scale_features(
        reg_X, reg_y, is_classification=False, test_ratio=0.25
    )

    reg_model = Ridge(random_state=42)
    y_pred_r = verify_regression_model(
        reg_model, X_train_r, y_train_r, X_test_r, y_test_r
    )
    save_regression_plots(
        y_test_r, y_pred_r, output_directory="output", filename_prefix="test_reg"
    )

    print("\nSupervised ML self-test successfully verified for both Classification and Regression.")
