import pandas as pd

input_file = "default of credit card clients.xls"
output_file = "credit_card.csv"

# 讀取 Excel
df = pd.read_excel(
    input_file,
    header=1
)

# 清理欄位名稱
df.columns = [
    col.strip().replace(".", "_").replace(" ", "_")
    for col in df.columns
]

# 顯示資料基本資訊
print("資料筆數:", len(df))
print("欄位數:", len(df.columns))

print("\n欄位名稱:")
print(df.columns.tolist())

print("\n缺失值:")
print(df.isna().sum())

print("\nID 重複數:")
print(df["ID"].duplicated().sum())

# 輸出 CSV
df.to_csv(
    output_file,
    index=False,
    encoding="utf-8-sig"
)

print("\nCSV 已建立:", output_file)