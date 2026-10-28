"""Verification and EDA helper module for step-by-step data analysis.

Provides standardized terminal inspection printouts, transition logging,
ASCII console histograms, mathematical invariant assertions, and adaptive
visual artifact generation (Matplotlib/Seaborn PNGs and Altair HTML).
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np
import pandas as pd

# Optional visualization backends
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import altair as alt

    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False


def inspect_dataset(data: pd.DataFrame, dataset_name: str = "Dataset") -> None:
    """Print structural dimensions, null counts, and summary statistics to stdout."""
    if data.empty:
        print(f"\n[WARNING] Dataset '{dataset_name}' is empty.")
        return

    banner = "=" * 60
    print(f"\n{banner}\nBASELINE PROFILE: {dataset_name}\n{banner}")
    print(f"Shape: {data.shape[0]:,} rows x {data.shape[1]:,} columns")
    print(f"Duplicate Rows: {data.duplicated().sum():,}")

    null_counts = data.isna().sum()
    columns_with_nulls = null_counts[null_counts > 0]
    if columns_with_nulls.empty:
        print("Missing Values: None")
    else:
        print("\nMissing Values per Column:")
        for column_name, null_count in columns_with_nulls.items():
            percentage = (null_count / len(data)) * 100
            print(f"  - {column_name}: {null_count:,} ({percentage:.2f}%)")

    print("\nColumn Data Types:")
    for column_name, dtype in data.dtypes.items():
        print(f"  - {column_name}: {dtype}")

    numerical_columns = data.select_dtypes(include=[np.number])
    if not numerical_columns.empty:
        print("\nNumerical Five-Number Summary:")
        summary_table = numerical_columns.describe().T[
            ["count", "mean", "std", "min", "50%", "max"]
        ]
        print(summary_table.to_string())

    print(f"{banner}\n")


def log_transition(
    before_data: pd.DataFrame,
    after_data: pd.DataFrame,
    operation_name: str,
) -> None:
    """Print record counts and delta immediately after a transformation."""
    before_count = len(before_data)
    after_count = len(after_data)
    dropped_count = before_count - after_count
    percentage_change = (
        (dropped_count / before_count) * 100 if before_count > 0 else 0.0
    )

    print(
        f"[Transition: {operation_name}] "
        f"{before_count:,} -> {after_count:,} rows "
        f"({dropped_count:,} delta, {percentage_change:.2f}% reduced)"
    )


def print_distribution_table(series: pd.Series, label: str) -> None:
    """Print an ASCII quantile distribution table to terminal stdout."""
    clean_series = series.dropna()
    if clean_series.empty:
        print(f"[Distribution] Series '{label}' contains no valid values.")
        return

    quantiles = clean_series.quantile([0.05, 0.25, 0.50, 0.75, 0.95])
    delimiter = "-" * 45

    print(f"\n{delimiter}")
    print(f"Distribution Quantiles: {label}")
    print(f"{delimiter}")
    print(
        f"Count: {clean_series.count():,} | "
        f"Mean: {clean_series.mean():.4f} | "
        f"Std: {clean_series.std():.4f}"
    )
    print(f"  p05 (5th percentile)  : {quantiles.loc[0.05]:.4f}")
    print(f"  p25 (1st quartile)    : {quantiles.loc[0.25]:.4f}")
    print(f"  p50 (median)          : {quantiles.loc[0.50]:.4f}")
    print(f"  p75 (3rd quartile)    : {quantiles.loc[0.75]:.4f}")
    print(f"  p95 (95th percentile) : {quantiles.loc[0.95]:.4f}")
    print(f"{delimiter}\n")


def print_ascii_histogram(
    series: pd.Series,
    label: str,
    bin_count: int = 10,
    max_bar_width: int = 30,
) -> None:
    """Print a terminal ASCII distribution histogram using pure NumPy."""
    clean_series = series.dropna()
    if clean_series.empty:
        print(f"[Histogram] Series '{label}' contains no valid numeric values.")
        return

    counts, bin_edges = np.histogram(clean_series, bins=bin_count)
    max_count = max(counts) if max(counts) > 0 else 1

    print(f"--- ASCII Histogram: {label} ---")
    for index in range(bin_count):
        lower_bound = bin_edges[index]
        upper_bound = bin_edges[index + 1]
        frequency = counts[index]
        bar_length = int((frequency / max_count) * max_bar_width)
        bar_representation = "#" * bar_length
        print(
            f"[{lower_bound:8.2f} - {upper_bound:8.2f}] | "
            f"{bar_representation:<{max_bar_width}} ({frequency:,})"
        )
    print("-" * 55 + "\n")


def save_eda_figure(
    data: pd.DataFrame,
    column_name: str,
    output_directory: str = "output",
    filename_prefix: str = "eda",
) -> pathlib.Path:
    """Generate and persist distribution visual artifact (PNG via Matplotlib or HTML via Altair)."""
    if column_name not in data.columns:
        raise KeyError(f"Column '{column_name}' not found in dataframe.")

    output_path = pathlib.Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)
    target_series = data[column_name].dropna()

    if HAS_MATPLOTLIB:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Histogram + KDE
        sns.histplot(
            data=target_series,
            kde=True,
            ax=axes[0],
            color="#2b5c8f",
            bins=30,
        )
        axes[0].set_title(
            f"Histogram & KDE: {column_name}",
            fontsize=11,
            fontweight="bold",
        )
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Boxplot
        sns.boxplot(
            x=target_series,
            ax=axes[1],
            color="#4e9da4",
        )
        axes[1].set_title(
            f"Boxplot: {column_name}",
            fontsize=11,
            fontweight="bold",
        )
        axes[1].grid(True, linestyle="--", alpha=0.5)

        figure.tight_layout()
        plot_file = output_path / f"{filename_prefix}_{column_name}_distribution.png"
        figure.savefig(plot_file, dpi=300)
        plt.close(figure)
        print(f"[Visual Saved] PNG artifact written to: {plot_file.resolve()}")
        return plot_file

    if HAS_ALTAIR:
        chart = (
            alt.Chart(data)
            .mark_bar()
            .encode(
                x=alt.X(f"{column_name}:Q", bin=alt.Bin(maxbins=30), title=column_name),
                y=alt.Y("count()", title="Frequency"),
            )
            .properties(
                title=f"Distribution of {column_name}",
                width=600,
                height=350,
            )
        )
        plot_file = output_path / f"{filename_prefix}_{column_name}_distribution.html"
        chart.save(str(plot_file))
        print(f"[Visual Saved] Altair HTML artifact written to: {plot_file.resolve()}")
        return plot_file

    raise RuntimeError("No visualization library available (install matplotlib or altair).")


def assert_invariants(
    data: pd.DataFrame,
    non_negative_columns: Sequence[str] | None = None,
    unique_columns: Sequence[str] | None = None,
    probability_columns: Sequence[str] | None = None,
) -> None:
    """Assert mathematical and logical domain constraints across dataset columns."""
    if non_negative_columns:
        for column_name in non_negative_columns:
            if column_name in data.columns:
                negative_count = (data[column_name] < 0).sum()
                assert negative_count == 0, (
                    f"Invariant violated: column '{column_name}' has "
                    f"{negative_count} negative entries."
                )

    if unique_columns:
        for column_name in unique_columns:
            if column_name in data.columns:
                assert data[column_name].is_unique, (
                    f"Invariant violated: column '{column_name}' contains duplicate values."
                )

    if probability_columns:
        for column_name in probability_columns:
            if column_name in data.columns:
                probabilities = data[column_name].dropna()
                in_bounds = (probabilities >= 0.0) & (probabilities <= 1.0)
                assert in_bounds.all(), (
                    f"Invariant violated: probability column '{column_name}' "
                    f"contains values outside [0, 1]."
                )


if __name__ == "__main__":
    # Self-test demonstration with synthetic tabular data
    np.random.seed(42)
    sample_size = 500

    raw_records = pd.DataFrame(
        {
            "transaction_id": [f"TX_{idx:05d}" for idx in range(sample_size)],
            "amount": np.random.exponential(scale=50.0, size=sample_size),
            "fee_rate": np.random.uniform(0.01, 0.05, size=sample_size),
            "status": np.random.choice(
                ["COMPLETED", "PENDING", "FAILED"],
                size=sample_size,
                p=[0.85, 0.10, 0.05],
            ),
        }
    )

    # Step 1: Baseline inspection
    inspect_dataset(raw_records, dataset_name="Synthetic Transactions")

    # Step 2: Incremental transformation with transition logging
    filtered_records = (
        raw_records
        .query("status == 'COMPLETED'")
        .assign(net_amount=lambda df: df["amount"] * (1.0 - df["fee_rate"]))
        .reset_index(drop=True)
    )
    log_transition(raw_records, filtered_records, "Filter COMPLETED status")

    # Step 3: Assert mathematical invariants
    assert_invariants(
        filtered_records,
        non_negative_columns=["amount", "net_amount"],
        unique_columns=["transaction_id"],
        probability_columns=["fee_rate"],
    )

    # Step 4: Terminal quantile and ASCII distribution printouts
    print_distribution_table(filtered_records["net_amount"], label="Net Transaction Amount")
    print_ascii_histogram(filtered_records["net_amount"], label="Net Transaction Amount")

    # Step 5: Save visual artifact
    saved_path = save_eda_figure(
        filtered_records,
        column_name="net_amount",
        output_directory="output",
        filename_prefix="test_run",
    )
    print(f"Self-test verification complete. Artifact generated: {saved_path}")
