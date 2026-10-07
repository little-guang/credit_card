"""Explore the cleaned credit-card-default dataset.

Run with ``uv run python explore_data.py``.  The script prints useful summaries
and writes tables and histogram figures under ``data/reports``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "clean" / "credit_card_clean.csv"
DEFAULT_OUTPUT = ROOT / "data" / "reports"
TARGET = "default_payment_next_month"
ID_COLUMN = "ID"
PAYMENT_STATUS_COLUMNS = {"PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"}


def print_section(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def save_value_counts(data: pd.DataFrame, columns: list[str], output_dir: Path) -> None:
    """Print and save counts for low-cardinality categorical features."""
    for column in columns:
        counts = (
            data.groupby(column, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(column)
        )
        print(f"\n{column}")
        print(counts.to_string(index=False))
        counts.to_csv(output_dir / f"value_counts_{column}.csv", index=False, encoding="utf-8-sig")


def save_histograms(data: pd.DataFrame, columns: list[str], output_dir: Path, show: bool) -> None:
    """Create a compact histogram grid for continuous numerical features."""
    sns.set_theme(style="whitegrid")
    columns_per_row = 3
    rows = (len(columns) + columns_per_row - 1) // columns_per_row
    figure, axes = plt.subplots(rows, columns_per_row, figsize=(18, 4.5 * rows))

    for axis, column in zip(axes.flat, columns):
        sns.histplot(data=data, x=column, bins=30, ax=axis)
        axis.set_title(column)

    for axis in axes.flat[len(columns):]:
        axis.set_visible(False)

    figure.suptitle("Numerical feature distributions", y=1.01, fontsize=16)
    figure.tight_layout()
    figure.savefig(output_dir / "numeric_distributions.png", dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)


def explore(input_file: Path, output_dir: Path, show: bool = False) -> None:
    if not input_file.exists():
        raise FileNotFoundError(f"Clean data file not found: {input_file}")

    data = pd.read_csv(input_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    print_section("Dataset overview")
    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns)}")
    print("\nData types:")
    print(data.dtypes.to_string())
    print("\nMissing values:")
    print(data.isna().sum().to_string())
    print(f"\nDuplicate rows: {data.duplicated().sum():,}")

    categorical_columns = [
        column for column in data.columns
        if column in {"SEX", "EDUCATION", "MARRIAGE", TARGET} | PAYMENT_STATUS_COLUMNS
    ]
    numeric_columns = [
        column for column in data.select_dtypes(include="number").columns
        if column not in categorical_columns + [ID_COLUMN]
    ]

    print_section("Categorical feature counts")
    save_value_counts(data, categorical_columns, output_dir)

    print_section("Numerical feature summary")
    numeric_summary = data[numeric_columns].describe().T
    print(numeric_summary.to_string())
    numeric_summary.to_csv(output_dir / "numeric_summary.csv", encoding="utf-8-sig")

    save_histograms(data, numeric_columns, output_dir, show)
    print(f"\nReports saved to: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the cleaned credit-card dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV path")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Directory for tables and charts")
    parser.add_argument("--show", action="store_true", help="Also display the histogram window")
    args = parser.parse_args()
    explore(args.input, args.output_dir, args.show)


if __name__ == "__main__":
    main()
