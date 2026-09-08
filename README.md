# 信用卡違約分析

本專題使用台灣信用卡客戶資料，分析客戶的信用額度、人口統計資料、過去付款紀錄、帳單金額與繳款金額，並探討信用卡客戶下個月違約的情形。

## 資料來源

本專題使用 UCI Machine Learning Repository 的資料集：

- [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
- 資料集作者：I-Cheng Yeh
- 捐贈日期：2016 年 1 月 25 日
- 資料筆數：30,000 筆
- 特徵數量：23 個
- 資料型態：整數（Integer）
- 資料集特性：多變量（Multivariate）
- 研究領域：商業（Business）
- 主要任務：分類（Classification）
- 缺失值：無
- 原始檔案：`default of credit card clients.xls`

## 研究背景與目的

本資料集的研究目的，是針對台灣信用卡客戶的下個月違約付款情形，使用六種資料探勘方法比較違約機率的預測準確度。

從風險管理的角度來看，預測違約機率比單純將客戶分類為「可信」或「不可信」更有價值。由於真實違約機率未知，原研究使用 Sorting Smoothing Method 估計實際違約機率，並將其作為反應變數，與模型預測的違約機率進行比較。

本專題將以此資料集進行資料清理、探索式資料分析與 SQL 分析，了解各項特徵與信用卡違約之間的關係，並建立可延伸至分類模型的分析資料。

## 目標變數

`default payment next month` 是二元目標變數：

- `1`：下個月違約付款（Yes）
- `0`：下個月未違約付款（No）

## 變數說明

UCI 官方資料將 23 個變數分為特徵（X1 至 X23）與 1 個目標變數（Y）。資料欄位經轉換腳本整理後，欄位名稱中的空格與句點會改為底線。

### 基本資料與信用額度

| 官方變數 | 欄位 | 說明 | 可能的資料內容 |
| --- | --- | --- | --- |
| X1 | `LIMIT_BAL` | 給定的信用額度，包含個人信用及家庭附加信用 | 新台幣金額，整數 |
| X2 | `SEX` | 性別 | `1`：男性；`2`：女性 |
| X3 | `EDUCATION` | 教育程度 | `1`：研究所；`2`：大學；`3`：高中；`4`：其他 |
| X4 | `MARRIAGE` | 婚姻狀況 | `1`：已婚；`2`：單身；`3`：其他 |
| X5 | `AGE` | 年齡 | 年數，整數 |

### 過去付款狀態

X6 至 X11 紀錄 2005 年 4 月至 9 月的過去每月付款狀態：

| 官方變數 | 欄位 | 月份 | 說明 |
| --- | --- | --- | --- |
| X6 | `PAY_0` | 2005 年 9 月 | 付款狀態 |
| X7 | `PAY_2` | 2005 年 8 月 | 付款狀態 |
| X8 | `PAY_3` | 2005 年 7 月 | 付款狀態 |
| X9 | `PAY_4` | 2005 年 6 月 | 付款狀態 |
| X10 | `PAY_5` | 2005 年 5 月 | 付款狀態 |
| X11 | `PAY_6` | 2005 年 4 月 | 付款狀態 |

付款狀態的官方編碼如下：

| 編碼 | 意義 |
| --- | --- |
| `-1` | 按時付款（Pay duly） |
| `1` | 延遲付款 1 個月 |
| `2` | 延遲付款 2 個月 |
| `3` 至 `8` | 延遲付款 3 至 8 個月 |
| `9` | 延遲付款 9 個月以上 |

### 帳單金額

X12 至 X17 為 2005 年 4 月至 9 月的帳單金額，單位為新台幣：

| 官方變數 | 欄位 | 月份 |
| --- | --- | --- |
| X12 | `BILL_AMT1` | 2005 年 9 月 |
| X13 | `BILL_AMT2` | 2005 年 8 月 |
| X14 | `BILL_AMT3` | 2005 年 7 月 |
| X15 | `BILL_AMT4` | 2005 年 6 月 |
| X16 | `BILL_AMT5` | 2005 年 5 月 |
| X17 | `BILL_AMT6` | 2005 年 4 月 |

### 過去繳款金額

X18 至 X23 為 2005 年 4 月至 9 月的過去繳款金額，單位為新台幣：

| 官方變數 | 欄位 | 月份 |
| --- | --- | --- |
| X18 | `PAY_AMT1` | 2005 年 9 月 |
| X19 | `PAY_AMT2` | 2005 年 8 月 |
| X20 | `PAY_AMT3` | 2005 年 7 月 |
| X21 | `PAY_AMT4` | 2005 年 6 月 |
| X22 | `PAY_AMT5` | 2005 年 5 月 |
| X23 | `PAY_AMT6` | 2005 年 4 月 |

### 目標變數

| 官方變數 | 轉換後欄位 | 說明 | 資料型態 |
| --- | --- | --- | --- |
| Y | `default_payment_next_month` | 下個月是否違約付款 | 二元值：`0` 或 `1` |

## 預計分析內容

1. 檢查資料型態、資料筆數、缺失值與客戶識別碼。
2. 計算整體違約率，並比較不同信用額度、年齡、教育程度與婚姻狀況的違約率。
3. 分析 2005 年 4 月至 9 月付款狀態與下個月違約之間的關係。
4. 比較帳單金額、過去繳款金額與違約情形的差異。
5. 使用 SQL 彙整客戶群組的違約人數與違約率。
6. 將分析結果延伸至分類模型與違約機率預測。

## 專案檔案

```text
.
├── convert.py                 # 將 UCI 原始 Excel 轉換並清理為 CSV
├── credit_card_raw.sql        # 原始資料表建立與匯入腳本
├── credit_card_clean .sql     # 清理後資料表相關 SQL
├── credit_card_analysis .sql  # 分析查詢 SQL
├── credit_card.csv            # 清理後的信用卡客戶資料
├── pyproject.toml              # Python 專案設定與相依套件
└── src/credit_card/            # Python 套件原始碼
```

## 執行資料轉換

請從 UCI 資料集頁面下載 `default of credit card clients.xls`，放在專案根目錄，接著執行：

```bash
python convert.py
```

執行後會產生 `credit_card.csv`。轉換腳本會讀取 Excel 的欄位名稱、清理欄位名稱，並檢查資料筆數、欄位、缺失值及重複的 `ID`。

## 原始研究

Yeh, I.-C., & Lien, C.-H. (2009). *The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients*. Expert Systems with Applications.

## 授權

本資料集依 [Creative Commons Attribution 4.0 International（CC BY 4.0）](https://creativecommons.org/licenses/by/4.0/) 授權。使用或修改資料時，應適當標示資料來源。

## 使用技術

- Python
- pandas
- SQL
- UCI Machine Learning Repository

## 注意事項

本資料集適合用於教學、探索式資料分析與信用風險建模練習。分析結果不應直接作為真實金融授信或個人信用決策的唯一依據。
