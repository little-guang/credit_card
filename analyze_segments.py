"""Summarize demographic and financial patterns for the interactive report."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
INPUT_FILE = ROOT / "data" / "clean" / "credit_card_clean.csv"
OUTPUT_FILE = ROOT / "docs" / "segment-analysis.json"
TARGET = "default_payment_next_month"

GROUP_LABELS = {
    "sex": {
        1: "男性",
        2: "女性",
    },
    "education": {
        1: "研究所",
        2: "大學",
        3: "高中",
        4: "其他",
        0: "未分類",
        5: "未分類",
        6: "未分類",
    },
    "marriage": {
        1: "已婚",
        2: "未婚",
        3: "其他",
        0: "未分類",
    },
}

GROUP_DESCRIPTIONS = {
    "age_group": "年齡",
    "sex": "性別",
    "education": "學歷",
    "marriage": "婚姻狀況",
}

MEASURE_DESCRIPTIONS = {
    "median_credit_limit": "信用額度中位數",
    "median_latest_bill": "最近一期帳單中位數",
    "median_avg_bill": "近六期平均帳單中位數",
    "median_latest_payment": "最近一期還款中位數",
    "median_avg_payment": "近六期平均還款中位數",
    "default_rate": "下一期違約比例",
}

CORRELATION_COLUMNS = {
    "AGE": "年齡",
    "LIMIT_BAL": "信用額度",
    "BILL_AMT1": "最近一期帳單",
    "avg_bill_6m": "近六期平均帳單",
    "PAY_AMT1": "最近一期還款",
    "avg_payment_6m": "近六期平均還款",
    "delayed_months": "近六期延遲次數",
    TARGET: "下一期違約",
}


def _json_number(value: object) -> float | None:
    if pd.isna(value) or not np.isfinite(float(value)):
        return None
    return round(float(value), 4)


def _prepare_data(data: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable recent/average financial measures and group labels."""
    data = data.rename(
        columns={
            "SEX": "sex",
            "EDUCATION": "education",
            "MARRIAGE": "marriage",
        }
    ).copy()
    bill_columns = [f"BILL_AMT{i}" for i in range(1, 7)]
    payment_columns = [f"PAY_AMT{i}" for i in range(1, 7)]
    status_columns = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

    data["avg_bill_6m"] = data[bill_columns].mean(axis=1)
    data["avg_payment_6m"] = data[payment_columns].mean(axis=1)
    data["delayed_months"] = (data[status_columns] > 0).sum(axis=1)
    data["age_group"] = pd.cut(
        data["AGE"],
        bins=[20, 30, 40, 50, 60, 80],
        labels=["21–29歲", "30–39歲", "40–49歲", "50–59歲", "60歲以上"],
        right=False,
    ).astype("string")

    for column, labels in GROUP_LABELS.items():
        data[column] = data[column].map(labels).fillna("未分類")

    return data


def _summarize_groups(data: pd.DataFrame) -> list[dict[str, object]]:
    group_columns = ["age_group", "sex", "education", "marriage"]
    group_rows: list[dict[str, object]] = []

    for dimension in group_columns:
        for label, group in data.groupby(dimension, observed=True, sort=True):
            group_rows.append(
                {
                    "dimension": dimension,
                    "group": str(label),
                    "count": int(len(group)),
                    "median_credit_limit": _json_number(group["LIMIT_BAL"].median()),
                    "median_latest_bill": _json_number(group["BILL_AMT1"].median()),
                    "median_avg_bill": _json_number(group["avg_bill_6m"].median()),
                    "median_latest_payment": _json_number(group["PAY_AMT1"].median()),
                    "median_avg_payment": _json_number(group["avg_payment_6m"].median()),
                    "default_rate": _json_number(group[TARGET].mean() * 100),
                }
            )
    return group_rows


def _correlations(data: pd.DataFrame) -> list[dict[str, object]]:
    matrix = data[list(CORRELATION_COLUMNS)].corr(method="spearman")
    rows: list[dict[str, object]] = []
    for index, (left, left_label) in enumerate(CORRELATION_COLUMNS.items()):
        for right, right_label in list(CORRELATION_COLUMNS.items())[index + 1 :]:
            rows.append(
                {
                    "x": left_label,
                    "y": right_label,
                    "rho": _json_number(matrix.loc[left, right]),
                }
            )
    return rows


def analyze(input_file: Path = INPUT_FILE, output_file: Path = OUTPUT_FILE) -> Path:
    if not input_file.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {input_file}")

    raw = pd.read_csv(input_file)
    data = _prepare_data(raw)
    payload = {
        "rows": int(len(data)),
        "default_rate": _json_number(data[TARGET].mean() * 100),
        "dimensions": GROUP_DESCRIPTIONS,
        "measures": MEASURE_DESCRIPTIONS,
        "groups": _summarize_groups(data),
        "correlations": _correlations(data),
        "notes": [
            "帳單及還款金額為資料中的歷史金額欄位；「近六期平均」是六個月份的簡單平均。",
            "信用額度、帳單、還款使用中位數呈現典型客戶，較不易被極少數大額數值拉高。",
            "分組差異及 Spearman 相關只代表這批歷史資料中的一起變化，不表示因果關係。",
            "學歷代碼 0、5、6 及婚姻代碼 0 合併為未分類。",
        ],
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Segment analysis saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    analyze()
