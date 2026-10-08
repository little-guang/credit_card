"""Summarize demographic and financial patterns for the interactive report."""

from __future__ import annotations

import json
import math
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


def _wilson_interval(events: int, total: int) -> tuple[float | None, float | None]:
    """Return a 95% Wilson confidence interval for a proportion, in percent."""
    if total == 0:
        return None, None

    z = 1.96
    proportion = events / total
    denominator = 1 + z**2 / total
    center = (proportion + z**2 / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return (center - margin) * 100, (center + margin) * 100


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
            default_count = int(group[TARGET].sum())
            ci_low, ci_high = _wilson_interval(default_count, len(group))
            group_rows.append(
                {
                    "dimension": dimension,
                    "group": str(label),
                    "count": int(len(group)),
                    "default_count": default_count,
                    "median_credit_limit": _json_number(group["LIMIT_BAL"].median()),
                    "median_latest_bill": _json_number(group["BILL_AMT1"].median()),
                    "median_avg_bill": _json_number(group["avg_bill_6m"].median()),
                    "median_latest_payment": _json_number(group["PAY_AMT1"].median()),
                    "median_avg_payment": _json_number(group["avg_payment_6m"].median()),
                    "default_rate": _json_number(group[TARGET].mean() * 100),
                    "default_rate_ci_low": _json_number(ci_low),
                    "default_rate_ci_high": _json_number(ci_high),
                }
            )
    return group_rows


def _summarize_risk_factor(
    data: pd.DataFrame,
    factor_id: str,
    title: str,
    description: str,
    groups: list[tuple[str, pd.Series]],
    reference_group: str,
) -> dict[str, object]:
    """Summarize observed default rates and contrasts against a reference group."""
    reference_label = next(
        (label for label, _ in groups if label == reference_group),
        None,
    )
    if reference_label is None:
        raise ValueError(f"Reference group {reference_group!r} is missing for {factor_id}.")
    reference = data.loc[dict(groups)[reference_label], TARGET]
    reference_rate = float(reference.mean()) if len(reference) else None

    rows: list[dict[str, object]] = []
    for label, mask in groups:
        values = data.loc[mask, TARGET]
        total = len(values)
        defaults = int(values.sum())
        rate = float(values.mean() * 100) if total else None
        ci_low, ci_high = _wilson_interval(defaults, total)
        difference = (
            rate - reference_rate * 100
            if rate is not None and reference_rate is not None
            else None
        )
        relative_rate = (
            rate / (reference_rate * 100)
            if rate is not None and reference_rate
            else None
        )
        rows.append(
            {
                "group": label,
                "count": total,
                "default_count": defaults,
                "default_rate": _json_number(rate),
                "default_rate_ci_low": _json_number(ci_low),
                "default_rate_ci_high": _json_number(ci_high),
                "difference_vs_reference_pp": _json_number(difference),
                "relative_rate_vs_reference": _json_number(relative_rate),
            }
        )

    return {
        "id": factor_id,
        "title": title,
        "description": description,
        "reference_group": reference_group,
        "eligible_count": int(sum(len(data.loc[mask]) for _, mask in groups)),
        "groups": rows,
    }


def _risk_factors(data: pd.DataFrame) -> list[dict[str, object]]:
    delay_labels = ["0 次", "1 次", "2 次", "3 次以上"]
    delay_masks = [
        data["delayed_months"] == 0,
        data["delayed_months"] == 1,
        data["delayed_months"] == 2,
        data["delayed_months"] >= 3,
    ]

    latest_labels = ["無正值延遲紀錄", "1 個月", "2 個月", "3 個月以上"]
    latest_masks = [
        data["PAY_0"] <= 0,
        data["PAY_0"] == 1,
        data["PAY_0"] == 2,
        data["PAY_0"] >= 3,
    ]

    utilization = data["BILL_AMT1"] / data["LIMIT_BAL"].replace(0, np.nan)
    utilization_labels = [
        "低於 30%",
        "30%–59%",
        "60%–89%",
        "90% 以上",
    ]
    utilization_masks = [
        (utilization >= 0) & (utilization < 0.3),
        (utilization >= 0.3) & (utilization < 0.6),
        (utilization >= 0.6) & (utilization < 0.9),
        utilization >= 0.9,
    ]

    bill_total = data[[f"BILL_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    payment_total = data[[f"PAY_AMT{i}" for i in range(1, 7)]].sum(axis=1)
    coverage = payment_total / bill_total.where(bill_total > 0)
    coverage_labels = ["低於 25%", "25%–49%", "50%–99%", "100% 以上"]
    coverage_masks = [
        (coverage >= 0) & (coverage < 0.25),
        (coverage >= 0.25) & (coverage < 0.5),
        (coverage >= 0.5) & (coverage < 1),
        coverage >= 1,
    ]

    return [
        _summarize_risk_factor(
            data,
            "delay_frequency",
            "近六期曾延遲的期數",
            "PAY_0、PAY_2 至 PAY_6 中大於 0 的期數；重複延遲可呈現比單一期狀態更持續的訊號。",
            list(zip(delay_labels, delay_masks, strict=True)),
            "0 次",
        ),
        _summarize_risk_factor(
            data,
            "latest_delay",
            "最近一期付款狀態",
            "PAY_0 大於 0 代表有正值延遲紀錄；非正值合併為參照組，不等於每筆都代表準時付款。",
            list(zip(latest_labels, latest_masks, strict=True)),
            "無正值延遲紀錄",
        ),
        _summarize_risk_factor(
            data,
            "credit_utilization",
            "最近一期帳單／信用額度",
            "以最近一期帳單除以正信用額度估算；僅納入非負使用率，負帳單或無法計算者不列入。",
            list(zip(utilization_labels, utilization_masks, strict=True)),
            "低於 30%",
        ),
        _summarize_risk_factor(
            data,
            "payment_coverage",
            "近六期還款金額／帳單金額",
            "六期還款總額除以六期帳單總額；只納入帳單總額為正的紀錄，且不同月份可能有時間落差，僅作探索性指標。",
            list(zip(coverage_labels, coverage_masks, strict=True)),
            "100% 以上",
        ),
    ]


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
        "risk_factors": _risk_factors(data),
        "correlations": _correlations(data),
        "notes": [
            "帳單及還款金額為資料中的歷史金額欄位；「近六期平均」是六個月份的簡單平均。",
            "信用額度、帳單、還款使用中位數呈現典型客戶，較不易被極少數大額數值拉高。",
            "違約率的區間為 95% Wilson 信賴區間；各組未經其他變數調整，分組差異與相關不表示因果關係。",
            "此資料只涵蓋六期還款狀態、額度、帳單及還款金額；分析結果不可當成個人授信判斷或因果證據。",
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
