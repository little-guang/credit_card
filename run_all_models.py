from __future__ import annotations

import json
# ============================================================
# 匯入套件
# ============================================================

from pathlib import Path
import shutil
import subprocess
import sys

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


# ============================================================
# 基本設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = (
    BASE_DIR
    / "output"
    / "model_comparison"
)

REPORT_DATA_FILE = (
    BASE_DIR
    / "docs"
    / "model-comparison.json"
)

REPORT_ASSET_DIR = (
    BASE_DIR
    / "docs"
    / "assets"
)

DATA_FILE = (
    BASE_DIR
    / "data"
    / "clean"
    / "credit_card_clean.csv"
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
        "PingFang TC",
        "Noto Sans CJK TC",
        "Source Han Sans TC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Arial Unicode MS",
        "Yu Gothic UI",
        "Meiryo UI",
    ]

    available_fonts = {
        font.name
        for font in fm.fontManager.ttflist
    }

    matched = [
        font_name
        for font_name in font_candidates
        if font_name in available_fonts
    ]

    if matched:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = matched + [
            "DejaVu Sans",
            "Arial",
        ]
        plt.rcParams["axes.unicode_minus"] = False
        print(f"✓ Matplotlib 中文字型：{matched[0]}")
        return

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "DejaVu Sans",
        "Arial",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    print("⚠ 找不到指定中文字型，將使用預設 sans-serif 字型，中文可能仍需安裝支援字型")


# ============================================================
# 三個模型
# ============================================================

MODELS = [

    {
        "name": "Logistic Regression",
        "slug": "logistic",

        "script":
            BASE_DIR
            / "train_logistic.py",

        "output":
            BASE_DIR
            / "output"
            / "logistic"
            / "model_metrics.csv",

        "output_dir":
            BASE_DIR
            / "output"
            / "logistic",
    },

    {
        "name": "Decision Tree",
        "slug": "decision_tree",

        "script":
            BASE_DIR
            / "train_decision_tree.py",

        "output":
            BASE_DIR
            / "output"
            / "decision_tree"
            / "model_metrics.csv",

        "output_dir":
            BASE_DIR
            / "output"
            / "decision_tree",
    },

    {
        "name": "XGBoost",
        "slug": "xgboost",

        "script":
            BASE_DIR
            / "train_xgboost.py",

        "output":
            BASE_DIR
            / "output"
            / "xgboost"
            / "model_metrics.csv",

        "output_dir":
            BASE_DIR
            / "output"
            / "xgboost",
    },
]

METRIC_COLUMNS = [
    "Model",
    "CV_PR_AUC",
    "CV_ROC_AUC",
    "CV_Precision",
    "CV_Recall",
    "CV_F1",
    "Train_PR_AUC",
    "Train_ROC_AUC",
    "Test_PR_AUC",
    "Test_ROC_AUC",
    "Test_Precision",
    "Test_Recall",
    "Test_F1",
    "PR_AUC_Gap",
    "ROC_AUC_Gap",
    "Precision_Gap",
    "Recall_Gap",
    "F1_Gap",
]


# ============================================================
# 執行模型
# ============================================================

def run_model(model_info):

    name = model_info["name"]

    script = model_info["script"]

    output_file = model_info["output"]

    print()
    print("=" * 70)
    print(f"執行：{name}")
    print("=" * 70)

    if not script.exists():

        raise FileNotFoundError(
            f"找不到程式：\n{script}"
        )

    # --------------------------------------------------------
    # 刪除舊的 Metrics
    #
    # 避免模型這次沒有成功執行，
    # 卻讀到上一次留下來的舊結果。
    # --------------------------------------------------------

    if output_file.exists():

        output_file.unlink()

    # --------------------------------------------------------
    # 執行模型
    # --------------------------------------------------------

    result = subprocess.run(
        [
            sys.executable,
            str(script),
        ],
        cwd=BASE_DIR,
    )

    if result.returncode != 0:

        raise RuntimeError(
            f"{name} 執行失敗"
        )

    # --------------------------------------------------------
    # 檢查結果
    # --------------------------------------------------------

    if not output_file.exists():

        raise FileNotFoundError(
            f"{name} 執行結束，"
            f"但找不到：\n"
            f"{output_file}"
        )

    print()
    print(
        f"✓ {name} 完成"
    )


# ============================================================
# 讀取結果
# ============================================================

def load_model_results():

    result_list = []

    for model_info in MODELS:

        file_path = (
            model_info["output"]
        )

        df = pd.read_csv(
            file_path
        )

        if df.empty:

            raise ValueError(
                f"結果檔是空的：\n"
                f"{file_path}"
            )

        missing_columns = set(METRIC_COLUMNS).difference(df.columns)

        if missing_columns:
            raise ValueError(
                f"{model_info['name']} 指標欄位不完整："
                f"{sorted(missing_columns)}"
            )

        if len(df) != 1 or df.loc[0, "Model"] != model_info["name"]:
            raise ValueError(
                f"{model_info['name']} 指標檔格式或模型名稱不符："
                f"{file_path}"
            )

        metrics = df[METRIC_COLUMNS[1:]].apply(
            pd.to_numeric,
            errors="raise",
        )

        if metrics.isna().any().any():
            raise ValueError(
                f"{model_info['name']} 指標含缺失值："
                f"{file_path}"
            )

        result_list.append(
            df[METRIC_COLUMNS]
        )

    return pd.concat(
        result_list,
        ignore_index=True,
    )


def publish_report_charts():
    comparison_charts = [
        ("model_comparison_metrics.png", "三模型測試指標比較"),
        ("model_comparison_auc.png", "交叉驗證與測試集 AUC 比較"),
        ("model_comparison_ranking.png", "模型綜合分數排名"),
    ]
    model_chart_files = [
        ("confusion_matrix.png", "混淆矩陣"),
        ("roc_curve.png", "ROC 曲線"),
        ("pr_curve.png", "Precision-Recall 曲線"),
        ("feature_importance.png", "特徵重要度"),
    ]

    REPORT_ASSET_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison_manifest = []
    for filename, title in comparison_charts:
        source = OUTPUT_DIR / filename
        if not source.exists():
            raise FileNotFoundError(
                f"找不到模型比較圖，無法更新網站：\n{source}"
            )

        shutil.copy2(
            source,
            REPORT_ASSET_DIR / filename,
        )
        comparison_manifest.append(
            {
                "title": title,
                "src": f"assets/{filename}",
            }
        )

    model_manifest = []
    for model_info in MODELS:
        model_charts = []
        for filename, title in model_chart_files:
            source = model_info["output_dir"] / filename
            if not source.exists():
                if filename == "feature_importance.png":
                    continue
                raise FileNotFoundError(
                    f"{model_info['name']} 缺少網站圖表：\n{source}"
                )

            report_filename = (
                f"{model_info['slug']}_{filename}"
            )
            shutil.copy2(
                source,
                REPORT_ASSET_DIR / report_filename,
            )
            model_charts.append(
                {
                    "title": title,
                    "src": f"assets/{report_filename}",
                }
            )

        model_manifest.append(
            {
                "name": model_info["name"],
                "charts": model_charts,
            }
        )

    return {
        "comparison": comparison_manifest,
        "models": model_manifest,
    }


# ============================================================
# 建立比較表
# ============================================================

def create_model_comparison(df):

    comparison_columns = [

        "Model",

        "CV_PR_AUC",
        "CV_ROC_AUC",

        "Test_PR_AUC",
        "Test_ROC_AUC",

        "Test_Precision",
        "Test_Recall",
        "Test_F1",

        "PR_AUC_Gap",
        "ROC_AUC_Gap",

        "Precision_Gap",
        "Recall_Gap",
        "F1_Gap",
    ]

    comparison = (
        df[
            comparison_columns
        ]
        .copy()
    )

    # --------------------------------------------------------
    # Overall Score
    # --------------------------------------------------------

    comparison["Overall_Score"] = (

        comparison["Test_PR_AUC"]
        * 0.35

        + comparison["Test_Recall"]
        * 0.25

        + comparison["Test_F1"]
        * 0.20

        + comparison["Test_ROC_AUC"]
        * 0.20
    )

    # --------------------------------------------------------
    # 個別排名
    # --------------------------------------------------------

    comparison["PR_AUC_Rank"] = (
        comparison["Test_PR_AUC"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    comparison["ROC_AUC_Rank"] = (
        comparison["Test_ROC_AUC"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    comparison["Recall_Rank"] = (
        comparison["Test_Recall"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    comparison["F1_Rank"] = (
        comparison["Test_F1"]
        .rank(
            ascending=False,
            method="min",
        )
    )

    comparison["Overall_Rank"] = (
        comparison["Overall_Score"]
        .rank(
            ascending=False,
            method="min",
        )
        .astype(int)
    )

    comparison = (
        comparison
        .sort_values(
            "Overall_Rank"
        )
        .reset_index(
            drop=True
        )
    )

    return comparison


# ============================================================
# 柱狀圖：每根柱子顯示數值
# ============================================================

def add_bar_labels(
    ax,
    bars,
    decimals=4,
):

    for bar in bars:

        value = bar.get_height()

        ax.text(
            bar.get_x()
            + bar.get_width() / 2,

            value,

            f"{value:.{decimals}f}",

            ha="center",
            va="bottom",

            fontsize=10,
        )


# ============================================================
# 圖一：Test Metrics
# ============================================================

def plot_test_metrics(
    comparison,
):

    metrics = [
        "Test_PR_AUC",
        "Test_ROC_AUC",
        "Test_Precision",
        "Test_Recall",
        "Test_F1",
    ]

    labels = [
        "PR-AUC",
        "ROC-AUC",
        "Precision",
        "Recall",
        "F1",
    ]

    models = (
        comparison["Model"]
        .tolist()
    )

    x = range(
        len(metrics)
    )

    width = 0.25

    _, ax = plt.subplots(
        figsize=(12, 6)
    )

    for i, model_name in enumerate(
        models
    ):

        values = (
            comparison.loc[
                comparison["Model"]
                == model_name,
                metrics,
            ]
            .iloc[0]
            .values
        )

        positions = [
            value
            + (i - 1) * width
            for value in x
        ]

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=model_name,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                value,

                f"{value:.4f}",

                ha="center",
                va="bottom",

                fontsize=8,
            )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        labels
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Model Comparison - Test Metrics"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "model_comparison_metrics.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 圖二：CV vs Test AUC
# ============================================================

def plot_auc_comparison(
    comparison,
):

    models = (
        comparison["Model"]
        .tolist()
    )

    x = range(
        len(models)
    )

    width = 0.18

    _, ax = plt.subplots(
        figsize=(11, 6)
    )

    metric_config = [

        (
            "CV_PR_AUC",
            "CV PR-AUC",
            -1.5,
        ),

        (
            "Test_PR_AUC",
            "Test PR-AUC",
            -0.5,
        ),

        (
            "CV_ROC_AUC",
            "CV ROC-AUC",
            0.5,
        ),

        (
            "Test_ROC_AUC",
            "Test ROC-AUC",
            1.5,
        ),
    ]

    for column, label, offset in metric_config:

        positions = [
            i
            + offset * width
            for i in x
        ]

        values = (
            comparison[column]
            .values
        )

        bars = ax.bar(
            positions,
            values,
            width=width,
            label=label,
        )

        for bar, value in zip(
            bars,
            values,
        ):

            ax.text(
                bar.get_x()
                + bar.get_width() / 2,

                value,

                f"{value:.4f}",

                ha="center",
                va="bottom",

                fontsize=8,
            )

    ax.set_xticks(
        list(x)
    )

    ax.set_xticklabels(
        models
    )

    ax.set_ylabel(
        "Score"
    )

    ax.set_title(
        "Model Comparison - CV vs Test AUC"
    )

    ax.set_ylim(
        0,
        1.05
    )

    ax.legend()

    ax.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "model_comparison_auc.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 圖三：Overall Score
# ============================================================

def plot_overall_score(
    comparison,
):

    plot_data = (
        comparison
        .sort_values(
            "Overall_Score",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 5)
    )

    bars = plt.barh(
        plot_data["Model"],
        plot_data["Overall_Score"],
    )

    plt.xlabel(
        "Overall Score"
    )

    plt.ylabel(
        "Model"
    )

    plt.title(
        "Model Ranking by Overall Score"
    )

    for bar, score, rank in zip(
        bars,
        plot_data["Overall_Score"],
        plot_data["Overall_Rank"],
    ):

        plt.text(
            score,
            bar.get_y()
            + bar.get_height() / 2,

            f"{score:.4f}  "
            f"(Rank {rank})",

            va="center",
            ha="left",

            fontsize=10,
        )

    plt.xlim(
        0,
        min(
            1.0,
            plot_data["Overall_Score"].max()
            + 0.10
        )
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR
        / "model_comparison_ranking.png",
        dpi=180,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# 建立文字摘要
# ============================================================

def create_summary(
    comparison,
):

    best_model = (
        comparison.iloc[0]
    )

    lines = []

    lines.append(
        "信用卡違約模型比較摘要"
    )

    lines.append(
        "=" * 70
    )

    lines.append("")

    lines.append(
        "【模型排名】"
    )

    lines.append("")

    for _, row in (
        comparison.iterrows()
    ):

        lines.append(
            f"{int(row['Overall_Rank'])}. "
            f"{row['Model']}"
        )

        lines.append(
            f"   Overall Score : "
            f"{row['Overall_Score']:.6f}"
        )

        lines.append(
            f"   Test PR-AUC   : "
            f"{row['Test_PR_AUC']:.6f}"
        )

        lines.append(
            f"   Test ROC-AUC  : "
            f"{row['Test_ROC_AUC']:.6f}"
        )

        lines.append(
            f"   Test Precision: "
            f"{row['Test_Precision']:.6f}"
        )

        lines.append(
            f"   Test Recall   : "
            f"{row['Test_Recall']:.6f}"
        )

        lines.append(
            f"   Test F1       : "
            f"{row['Test_F1']:.6f}"
        )

        lines.append(
            f"   PR-AUC Gap    : "
            f"{row['PR_AUC_Gap']:.6f}"
        )

        lines.append(
            f"   ROC-AUC Gap   : "
            f"{row['ROC_AUC_Gap']:.6f}"
        )

        lines.append("")

    lines.append(
        "【最佳模型】"
    )

    lines.append("")

    lines.append(
        f"模型："
        f"{best_model['Model']}"
    )

    lines.append(
        f"Overall Score："
        f"{best_model['Overall_Score']:.6f}"
    )

    lines.append("")

    lines.append(
        "【評估方式】"
    )

    lines.append("")

    lines.append(
        "Overall Score = "
        "Test PR-AUC × 0.35 "
        "+ Test Recall × 0.25 "
        "+ Test F1 × 0.20 "
        "+ Test ROC-AUC × 0.20"
    )

    lines.append("")

    lines.append(
        "Gap 主要用於觀察模型的泛化差距，"
        "目前沒有直接加入 Overall Score。"
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    return "\n".join(
        lines
    )


# ============================================================
# 主程式
# ============================================================

def main():

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    setup_chinese_font()

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"找不到清理後資料，請先執行 convert.py：\n{DATA_FILE}"
        )

    print("=" * 70)
    print("信用卡違約模型批次執行與比較")
    print("=" * 70)

    # ========================================================
    # 第一階段：執行三個模型
    # ========================================================

    for model_info in MODELS:

        run_model(
            model_info
        )

    # ========================================================
    # 第二階段：讀取模型結果
    # ========================================================

    print()
    print("=" * 70)
    print("三個模型皆完成，開始模型比較")
    print("=" * 70)

    results = (
        load_model_results()
    )

    # ========================================================
    # 第三階段：建立比較表
    # ========================================================

    comparison = (
        create_model_comparison(
            results
        )
    )

    # ========================================================
    # 儲存 CSV
    # ========================================================

    comparison_file = (
        OUTPUT_DIR
        / "model_comparison.csv"
    )

    comparison.to_csv(
        comparison_file,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 建立圖表
    # ========================================================

    print()
    print("建立模型比較圖...")

    plot_test_metrics(
        comparison
    )

    print(
        "✓ model_comparison_metrics.png"
    )

    plot_auc_comparison(
        comparison
    )

    print(
        "✓ model_comparison_auc.png"
    )

    plot_overall_score(
        comparison
    )

    print(
        "✓ model_comparison_ranking.png"
    )

    chart_manifest = publish_report_charts()

    target = pd.read_csv(
        DATA_FILE,
        usecols=["default_payment_next_month"],
    )["default_payment_next_month"]
    report_data = {
        "dataset": {
            "rows": int(len(target)),
            "default_count": int(target.sum()),
            "non_default_count": int((target == 0).sum()),
            "default_rate": float(target.mean()),
        },
        "models": json.loads(
            comparison.to_json(orient="records")
        ),
        "charts": chart_manifest,
    }
    REPORT_DATA_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    REPORT_DATA_FILE.write_text(
        json.dumps(
            report_data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # ========================================================
    # 建立摘要
    # ========================================================

    summary = (
        create_summary(
            comparison
        )
    )

    summary_file = (
        OUTPUT_DIR
        / "model_comparison_summary.txt"
    )

    summary_file.write_text(
        summary,
        encoding="utf-8",
    )

    # ========================================================
    # 顯示結果
    # ========================================================

    print()
    print("=" * 70)
    print("模型比較結果")
    print("=" * 70)

    display_columns = [

        "Model",

        "CV_PR_AUC",
        "CV_ROC_AUC",

        "Test_PR_AUC",
        "Test_ROC_AUC",

        "Test_Precision",
        "Test_Recall",
        "Test_F1",

        "PR_AUC_Gap",
        "ROC_AUC_Gap",

        "Overall_Score",
        "Overall_Rank",
    ]

    print(
        comparison[
            display_columns
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.6f}",
        )
    )

    best_model = (
        comparison.iloc[0]
    )

    print()
    print("=" * 70)

    print(
        f"最佳模型："
        f"{best_model['Model']}"
    )

    print(
        f"Overall Score："
        f"{best_model['Overall_Score']:.6f}"
    )

    print()

    print(
        f"✓ 比較結果："
        f"{comparison_file}"
    )

    print(
        f"✓ 網頁報告資料："
        f"{REPORT_DATA_FILE}"
    )

    print(
        f"✓ 摘要："
        f"{summary_file}"
    )

    print(
        f"✓ Test Metrics 圖："
        f"{OUTPUT_DIR / 'model_comparison_metrics.png'}"
    )

    print(
        f"✓ AUC 比較圖："
        f"{OUTPUT_DIR / 'model_comparison_auc.png'}"
    )

    print(
        f"✓ Overall Ranking 圖："
        f"{OUTPUT_DIR / 'model_comparison_ranking.png'}"
    )

    print("=" * 70)


# ============================================================
# 執行主程式
# ============================================================

if __name__ == "__main__":
    main()