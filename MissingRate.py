import pandas as pd
import numpy as np

# ========= 1. 读取Excel =========
file_path = "实验数据.xlsx"
df = pd.read_excel(file_path, sheet_name="model_data")

# ========= 2. 统一伪缺失值 =========
df = df.replace(["NA", "N/A", "null", "NULL", ""], np.nan)
df = df.replace(r"^\s+$", np.nan, regex=True)   # 只有空格也算缺失

# ========= 3. 行缺失率统计 =========
row_missing_count = df.isna().sum(axis=1)
row_missing_rate = df.isna().mean(axis=1)

row_result = pd.DataFrame({
    "行号": df.index + 1,
    "行缺失个数": row_missing_count,
    "总列数": df.shape[1],
    "行缺失率": row_missing_rate,
    "行缺失率(%)": row_missing_rate * 100
})

# 也可附加到原始数据中
df_with_row_missing = df.copy()
df_with_row_missing["行缺失个数"] = row_missing_count
df_with_row_missing["总列数"] = df.shape[1]
df_with_row_missing["行缺失率"] = row_missing_rate
df_with_row_missing["行缺失率(%)"] = row_missing_rate * 100

# ========= 4. 列缺失率统计 =========
col_missing_count = df.isna().sum(axis=0)
col_missing_rate = df.isna().mean(axis=0)

col_result = pd.DataFrame({
    "变量名": df.columns,
    "列缺失个数": col_missing_count.values,
    "总行数": df.shape[0],
    "列缺失率": col_missing_rate.values,
    "列缺失率(%)": col_missing_rate.values * 100
}).sort_values(by="列缺失率(%)", ascending=False)

# ========= 5. 打印结果 =========
print("行缺失率统计：")
print(row_result.to_string(index=False))

print("\n列缺失率统计：")
print(col_result.to_string(index=False))

# ========= 6. 导出结果 =========

output_file = "实验数据-缺失率.xlsx"
with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    row_result.to_excel(writer, sheet_name="行缺失率", index=False)
    col_result.to_excel(writer, sheet_name="列缺失率", index=False)
    df_with_row_missing.to_excel(writer, sheet_name="原始数据_含行缺失率", index=False)

print(f"\n已保存到：{output_file}")
