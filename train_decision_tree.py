
from __future__ import annotations

# ============================================================
# 匯入套件
# ============================================================

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split


# ============================================================
# 基本設定
# ============================================================

RANDOM_STATE = 27
TEST_SIZE = 0.20
CV_FOLDS = 5

BASE_DIR = Path(__file__).resolve().parent

CLEAN_FILE = (
    BASE_DIR
    / "data"
    / "clean"
    / "credit_card_clean.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "decision_tree"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# 中文字型
# ============================================================

def setup_chinese_font():

    font_candidates = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "SimHei",
    ]

    available_fonts = {
        font.name
        for font in fm.fontManager.ttflist
    }

    for font_name in font_candidates:

        if font_name in available_fonts:

            plt.rcParams["font.sans-serif"] = [
                font_name
            ]

            plt.rcParams["axes.unicode_minus"] = False

            print(
                f"✓ Matplotlib 中文字型：{font_name}"
            )

            return

    plt.rcParams["axes.unicode_minus"] = False

    print(
        "⚠ 找不到指定中文字型"
    )

# ============================================================
# 欄位名稱
# ============================================================

COLUMN_MAP = {
    "ID": "ID",
    "LIMIT_BAL": "信用額度",
    "SEX": "性別",
    "EDUCATION": "教育程度",
    "MARRIAGE": "婚姻狀況",
    "AGE": "年齡",

    "default_payment_next_month": "default_payment_next_month",
    "Y": "default_payment_next_month",
}


# ============================================================
# 欄位名稱整理
# ============================================================

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:

    rename_dict = {}

    for column in df.columns:

        clean_name = str(column).strip()

        if clean_name in COLUMN_MAP:
            rename_dict[column] = COLUMN_MAP[clean_name]

    return df.rename(
        columns=rename_dict
    )


# ============================================================
# 特徵工程
# ============================================================

def create_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    bill_columns = [
        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",
    ]

    pay_columns = [
        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",
    ]

    pay_status_columns = [
        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",
    ]

    # --------------------------------------------------------
    # 信用額度使用率
    # --------------------------------------------------------

    df["信用額度使用率"] = (
        df["BILL_AMT1"]
        / df["信用額度"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 六個月還款帳單比
    # --------------------------------------------------------

    total_bill = df[bill_columns].sum(axis=1)
    total_pay = df[pay_columns].sum(axis=1)

    df["六個月還款帳單比"] = (
        total_pay
        / total_bill.replace(0, np.nan)
    )

    # --------------------------------------------------------
    # PAY 特徵
    # --------------------------------------------------------

    df["PAY平均延遲"] = (
        df[pay_status_columns].mean(axis=1)
    )

    df["PAY最大延遲"] = (
        df[pay_status_columns].max(axis=1)
    )

    df["PAY延遲次數"] = (
        (df[pay_status_columns] >= 1)
        .sum(axis=1)
    )

    df["PAY嚴重延遲次數"] = (
        (df[pay_status_columns] >= 3)
        .sum(axis=1)
    )

    # --------------------------------------------------------
    # 最近平均
    # --------------------------------------------------------

    df["最近帳單平均"] = (
        df[
            [
                "BILL_AMT1",
                "BILL_AMT2",
                "BILL_AMT3",
            ]
        ].mean(axis=1)
    )

    df["最近還款平均"] = (
        df[
            [
                "PAY_AMT1",
                "PAY_AMT2",
                "PAY_AMT3",
            ]
        ].mean(axis=1)
    )

    # --------------------------------------------------------
    # 變化量
    # --------------------------------------------------------

    df["帳單變化量"] = (
        df["BILL_AMT1"]
        - df["BILL_AMT6"]
    )

    df["還款變化量"] = (
        df["PAY_AMT1"]
        - df["PAY_AMT6"]
    )

    # --------------------------------------------------------
    # 變化率
    # --------------------------------------------------------

    df["帳單變化率"] = (
        (
            df["BILL_AMT1"]
            - df["BILL_AMT6"]
        )
        / df["BILL_AMT6"].replace(0, np.nan)
    )

    df["還款變化率"] = (
        (
            df["PAY_AMT1"]
            - df["PAY_AMT6"]
        )
        / df["PAY_AMT6"].replace(0, np.nan)
    )

    # --------------------------------------------------------
    # 波動
    # --------------------------------------------------------

    df["帳單波動"] = (
        df[bill_columns].std(axis=1)
    )

    df["還款波動"] = (
        df[pay_columns].std(axis=1)
    )

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return df


# ============================================================
# 主程式
# ============================================================

def main():

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_chinese_font()
    print("=" * 70)
    print("信用卡違約 Decision Tree")
    print("=" * 70)

    # ========================================================
    # 1. 讀取資料
    # ========================================================

    print("\n[1/6] 讀取資料")

    df = pd.read_csv(
        CLEAN_FILE
    )

    df = df.loc[
        :,
        ~df.columns.astype(str).str.startswith("Unnamed")
    ]

    df = rename_columns(df)

    print(f"資料筆數：{len(df):,}")
    print(f"欄位數：{len(df.columns):,}")

    # ========================================================
    # 2. 建立特徵
    # ========================================================

    print("\n[2/6] 建立特徵")

    df = create_features(df)

    target = "default_payment_next_month"

    features = [
        "信用額度",
        "性別",
        "教育程度",
        "婚姻狀況",
        "年齡",

        "PAY_0",
        "PAY_2",
        "PAY_3",
        "PAY_4",
        "PAY_5",
        "PAY_6",

        "BILL_AMT1",
        "BILL_AMT2",
        "BILL_AMT3",
        "BILL_AMT4",
        "BILL_AMT5",
        "BILL_AMT6",

        "PAY_AMT1",
        "PAY_AMT2",
        "PAY_AMT3",
        "PAY_AMT4",
        "PAY_AMT5",
        "PAY_AMT6",

        "信用額度使用率",
        "六個月還款帳單比",
        "PAY平均延遲",
        "PAY最大延遲",
        "PAY延遲次數",
        "PAY嚴重延遲次數",

        "最近帳單平均",
        "最近還款平均",

        "帳單變化量",
        "還款變化量",

        "帳單變化率",
        "還款變化率",

        "帳單波動",
        "還款波動",
    ]

    X = df[features]
    y = df[target].astype(int)

    # ========================================================
    # 3. 切分資料
    # ========================================================

    print("\n[3/6] 切分訓練集與測試集")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"訓練集：{len(X_train):,}")
    print(f"測試集：{len(X_test):,}")

    # ========================================================
    # 4. 建立 Decision Tree
    # ========================================================

    print("\n[4/6] 建立 Decision Tree")

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    # ========================================================
    # 5. Cross Validation
    # ========================================================

    print("\n[5/6] 執行 5-Fold Cross Validation")

    cv = StratifiedKFold(
        n_splits=CV_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    cv_pr_auc = cv_results[
        "test_pr_auc"
    ].mean()

    cv_roc_auc = cv_results[
        "test_roc_auc"
    ].mean()

    cv_precision = cv_results[
        "test_precision"
    ].mean()

    cv_recall = cv_results[
        "test_recall"
    ].mean()

    cv_f1 = cv_results[
        "test_f1"
    ].mean()

    print(
        f"CV PR-AUC     : {cv_pr_auc:.6f}"
    )

    print(
        f"CV ROC-AUC    : {cv_roc_auc:.6f}"
    )

    print(
        f"CV Precision  : {cv_precision:.6f}"
    )

    print(
        f"CV Recall     : {cv_recall:.6f}"
    )

    print(
        f"CV F1         : {cv_f1:.6f}"
    )

    # ========================================================
    # 6. 訓練與測試
    # ========================================================

    print("\n[6/6] 訓練模型並評估測試集")

    model.fit(
        X_train,
        y_train,
    )

    y_train_prob = model.predict_proba(
        X_train
    )[:, 1]

    y_test_prob = model.predict_proba(
        X_test
    )[:, 1]

    y_test_pred = (
        y_test_prob >= 0.5
    ).astype(int)

    train_pr_auc = average_precision_score(
        y_train,
        y_train_prob,
    )

    train_roc_auc = roc_auc_score(
        y_train,
        y_train_prob,
    )

    test_pr_auc = average_precision_score(
        y_test,
        y_test_prob,
    )

    test_roc_auc = roc_auc_score(
        y_test,
        y_test_prob,
    )

    test_precision = precision_score(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    test_recall = recall_score(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    test_f1 = f1_score(
        y_test,
        y_test_pred,
        zero_division=0,
    )

    print(
        f"Test PR-AUC   : {test_pr_auc:.6f}"
    )

    print(
        f"Test ROC-AUC  : {test_roc_auc:.6f}"
    )

    print(
        f"Test Precision: {test_precision:.6f}"
    )

    print(
        f"Test Recall   : {test_recall:.6f}"
    )

    print(
        f"Test F1       : {test_f1:.6f}"
    )

    # ========================================================
    # Gap
    # ========================================================

    pr_auc_gap = (
        cv_pr_auc
        - test_pr_auc
    )

    roc_auc_gap = (
        cv_roc_auc
        - test_roc_auc
    )

    precision_gap = (
        cv_precision
        - test_precision
    )

    recall_gap = (
        cv_recall
        - test_recall
    )

    f1_gap = (
        cv_f1
        - test_f1
    )

    # ========================================================
    # Metrics
    # ========================================================

    metrics = pd.DataFrame(
        [
            {
                "Model": "Decision Tree",

                "CV_PR_AUC": cv_pr_auc,
                "CV_ROC_AUC": cv_roc_auc,
                "CV_Precision": cv_precision,
                "CV_Recall": cv_recall,
                "CV_F1": cv_f1,

                "Train_PR_AUC": train_pr_auc,
                "Train_ROC_AUC": train_roc_auc,

                "Test_PR_AUC": test_pr_auc,
                "Test_ROC_AUC": test_roc_auc,
                "Test_Precision": test_precision,
                "Test_Recall": test_recall,
                "Test_F1": test_f1,

                "PR_AUC_Gap": pr_auc_gap,
                "ROC_AUC_Gap": roc_auc_gap,
                "Precision_Gap": precision_gap,
                "Recall_Gap": recall_gap,
                "F1_Gap": f1_gap,
            }
        ]
    )

    metrics.to_csv(
        OUTPUT_DIR / "model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # CV 結果
    # ========================================================

    cv_output = pd.DataFrame(
        {
            "PR_AUC": cv_results["test_pr_auc"],
            "ROC_AUC": cv_results["test_roc_auc"],
            "Precision": cv_results["test_precision"],
            "Recall": cv_results["test_recall"],
            "F1": cv_results["test_f1"],
        }
    )

    cv_output.to_csv(
        OUTPUT_DIR / "cv_results.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # Feature Importance
    # ========================================================

    feature_importance = pd.DataFrame(
        {
            "Feature": features,
            "Importance": model.feature_importances_,
        }
    ).sort_values(
        "Importance",
        ascending=False,
    )

    feature_importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    importance_plot = (
        feature_importance
        .head(15)
        .sort_values("Importance")
    )
    figure, axis = plt.subplots(
        figsize=(10, 7)
    )
    bars = axis.barh(
        importance_plot["Feature"],
        importance_plot["Importance"],
        color="#0d9488",
    )
    axis.set_title("Decision Tree Feature Importance")
    axis.set_xlabel("Feature Importance")
    axis.set_ylabel("Feature")
    for bar, value in zip(
        bars,
        importance_plot["Importance"],
    ):
        axis.text(
            value,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            ha="left",
        )
    axis.grid(
        axis="x",
        alpha=0.25,
    )
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIR / "feature_importance.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

# ========================================================
# Confusion Matrix
# ========================================================

    cm = confusion_matrix(
        y_test,
        y_test_pred,
    )

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    image = ax.imshow(
        cm,
        interpolation="nearest",
    )

    ax.set_title(
        "Decision Tree Confusion Matrix"
    )

    ax.set_xlabel(
        "預測結果"
    )

    ax.set_ylabel(
        "實際結果"
    )

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["未違約", "違約"]
    )

    ax.set_yticklabels(
        ["未違約", "違約"]
    )

    for i in range(2):

        for j in range(2):

            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                fontsize=13,
                fontweight="bold",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "confusion_matrix.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()
# ========================================================
# ROC Curve
# ========================================================

    fpr, tpr, _ = roc_curve(
        y_test,
        y_test_prob,
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC-AUC = {test_roc_auc:.4f}",
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random",
    )

    plt.text(
        0.60,
        0.20,
        f"ROC-AUC = {test_roc_auc:.4f}",
        fontsize=12,
        bbox=dict(
            boxstyle="round",
            alpha=0.8,
        ),
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Decision Tree ROC Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "roc_curve.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()

# ========================================================
# PR Curve
# ========================================================

    precision, recall, _ = (
        precision_recall_curve(
            y_test,
            y_test_prob,
        )
    )

    plt.figure(
        figsize=(7, 5)
    )

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"PR-AUC = {test_pr_auc:.4f}",
    )

    plt.text(
        0.55,
        0.80,
        f"PR-AUC = {test_pr_auc:.4f}",
        fontsize=12,
        bbox=dict(
            boxstyle="round",
            alpha=0.8,
        ),
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Decision Tree Precision-Recall Curve"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "pr_curve.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


    # ========================================================
    # 完成
    # ========================================================

    print()
    print("=" * 70)
    print("✓ Decision Tree 完成")
    print(f"輸出資料夾：{OUTPUT_DIR}")
    print("=" * 70)


# ============================================================
# 執行主程式
# ============================================================

if __name__ == "__main__":
    main()