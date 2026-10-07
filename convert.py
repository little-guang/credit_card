"""Clean the UCI credit-card-default dataset and write it as a CSV file."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW_FILE = ROOT / "data" / "raw" / "default of credit card clients.xls"
CLEAN_FILE = ROOT / "data" / "clean" / "credit_card_clean.csv"
COLUMNS = [
    "ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
    "default_payment_next_month",
]


def clean_data() -> Path:
    """Validate the source workbook and save a UTF-8 CSV for modelling."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"Raw data file not found: {RAW_FILE}")

    data = pd.read_excel(RAW_FILE, header=1)
    data.columns = [str(column).strip().replace(".", "_").replace(" ", "_") for column in data.columns]

    if data.columns.tolist() != COLUMNS:
        raise ValueError(f"Unexpected Excel columns: {data.columns.tolist()}")
    if data["ID"].duplicated().any():
        raise ValueError("The ID column contains duplicate values.")
    if data.isna().any().any():
        missing_columns = data.columns[data.isna().any()].tolist()
        raise ValueError(f"Missing values found in: {missing_columns}")

    CLEAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(CLEAN_FILE, index=False, encoding="utf-8-sig")
    print(f"Cleaned {len(data):,} rows -> {CLEAN_FILE}")
    return CLEAN_FILE


if __name__ == "__main__":
    clean_data()
