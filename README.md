# 信用卡違約風險預測

使用 UCI「Default of Credit Card Clients」資料集，清理資料並比較 Logistic Regression、Decision Tree 與 XGBoost，預測客戶下個月是否違約。專案也提供資料探索、各模型評估產物，以及可部署到 GitHub Pages 的繁體中文報告網站。

> 本專案為研究與教學用途，不構成個人信用評分或授信建議。

## 資料集

- 來源：[UCI Machine Learning Repository — Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- 原始資料：30,000 筆客戶紀錄、23 個特徵、識別欄位 `ID` 及目標欄位。
- 預測目標：`default_payment_next_month`，`0` 代表未違約，`1` 代表違約。
- 特徵涵蓋信用額度、客戶基本資料、六期還款狀態、帳單金額與付款金額。

原始 Excel 檔應放在 `data/raw/default of credit card clients.xls`。資料集包含歷史客戶資料，分析結果不應直接推論為未來授信成效。

## 專案檔案

```text
.
├── convert.py                 # 驗證 Excel 欄位並清理為 CSV
├── explore_data.py            # 描述統計、類別計數與特徵分布圖
├── train_logistic.py          # Logistic Regression 訓練與評估
├── train_decision_tree.py     # Decision Tree 訓練與評估
├── train_xgboost.py           # XGBoost 訓練與評估
├── run_all_models.py          # 執行三個模型、比較結果並更新網站資料
├── data/
│   ├── raw/                   # 原始 UCI Excel 資料
│   └── clean/                 # convert.py 產生的標準化 CSV
├── docs/
│   ├── index.html             # GitHub Pages 報告網站
│   ├── model-comparison.json  # 由 run_all_models.py 更新的網站資料
│   └── assets/                # 報告網站顯示的圖表
├── output/                    # 模型與比較報告產物（執行時建立，不納入 Git）
├── pyproject.toml
└── uv.lock
```

## 環境安裝

需求：Python 3.12 以上及 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
```

## 執行

### 1. 清理資料

```bash
uv run python convert.py
```

讀取 `data/raw/default of credit card clients.xls`，驗證欄位、ID 唯一性及缺失值，輸出 `data/clean/credit_card_clean.csv`。

### 2. 探索資料（選用）

```bash
uv run python explore_data.py
```

報表寫入 `data/reports/`，包含數值摘要、類別計數 CSV 與數值特徵分布圖。若要顯示圖表視窗：

```bash
uv run python explore_data.py --show
```

執行探索分析時，數值特徵分布圖也會複製至 `docs/assets/`，供報告網站呈現。

### 3. 訓練與比較三個模型

```bash
uv run python run_all_models.py
```

此命令依序執行 Logistic Regression、Decision Tree 和 XGBoost，檢查每個模型是否成功產生一致且完整的指標，再輸出比較表、圖表、摘要及網站資料。模型比較圖以及各模型的混淆矩陣、ROC 曲線、PR 曲線和特徵重要度圖會複製至 `docs/assets/`；Decision Tree 的特徵重要度圖也會在此流程中產生。模型個別輸出在 `output/logistic/`、`output/decision_tree/`、`output/xgboost/`；比較產物在 `output/model_comparison/`。這些都是可重現的衍生檔，已由 `.gitignore` 排除。

也可以單獨執行模型，例如：

```bash
uv run python train_logistic.py
uv run python train_decision_tree.py
uv run python train_xgboost.py
```

單獨執行不會更新三模型比較表或網站 JSON。

### 4. 開啟報告網站

先依序執行 `convert.py`、`explore_data.py` 及 `run_all_models.py`，更新清理資料、探索分布圖、模型圖表和 `docs/model-comparison.json`；再於 GitHub 專案設定 **Settings → Pages → Deploy from a branch**，選擇主要分支及 `/docs` 資料夾。GitHub Pages 部署完成後，即可用 `https://<帳號>.github.io/<專案名稱>/` 分享含圖表的報告。

網站是純 HTML/CSS/JavaScript，不需要額外套件或建置步驟；更新報告時，請一併提交 `docs/model-comparison.json` 與 `docs/assets/` 中新產生或更新的圖表。

## 評估方式

- 以固定亂數種子 `27` 做分層 80/20 訓練／測試切分。
- 在訓練資料上使用 5-fold `StratifiedKFold` 交叉驗證。
- 分別比較 PR-AUC、ROC-AUC、Precision、Recall 與 F1；對類別不平衡資料，不應只看 Accuracy。
- 三個模型的 `*_Gap` 欄位統一為「交叉驗證平均 − 保留測試集」，方便以相同定義觀察差距。Gap 不是單獨判定過擬合的依據。
- 綜合排名採用專案設定的加權分數：

  ```text
  Overall Score =
      Test PR-AUC × 0.35
    + Test Recall × 0.25
    + Test F1 × 0.20
    + Test ROC-AUC × 0.20
  ```

  權重為本專案的比較設定，不是通用標準；若業務對漏判或誤判的成本不同，應重新設定閾值及評分方式。

## 參考資料

Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications.
