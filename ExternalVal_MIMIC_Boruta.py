import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings("ignore")


# =========================================================
# 1. 参数设置
# =========================================================
INPUT_FILE = "mice_external_validation_outputs/mimic_imputed.xlsx"   # MIMIC填补后Excel文件
SHEET_NAME = 0                      # 可写 0 或工作表名
OUTPUT_DIR = "ExternalVal_boruta_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 如果为 None，则默认最后一列是标签
LABEL_COL = None

# 不参与 Boruta 特征筛选的列，如编号、患者ID等。
EXCLUDE_COLS = [""]

# 是否在“Boruta粗筛后数据”中保留上述编号列，便于样本追溯。
# 这些列仅保留在输出文件中，不会进入 Boruta 或随机森林。
KEEP_EXCLUDED_COLS_IN_OUTPUT = True

RANDOM_STATE = 42


# =========================================================
# 2. 工具函数
# =========================================================
def to_binary_label(y: pd.Series) -> pd.Series:
    unique_vals = pd.Series(y).dropna().unique().tolist()

    if set(unique_vals) <= {0, 1}:
        return y.astype(int)

    if len(unique_vals) == 2:
        mapping = {sorted(unique_vals)[0]: 0, sorted(unique_vals)[1]: 1}
        return y.map(mapping).astype(int)

    raise ValueError(f"标签列必须为二分类，当前取值为：{unique_vals}")


# =========================================================
# 3. 读取数据
# =========================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# 确定标签列
if LABEL_COL is None:
    if df.shape[1] < 2:
        raise ValueError("表格至少需要2列：特征列和标签列。")
    label_name = df.columns[-1]
else:
    if LABEL_COL not in df.columns:
        raise ValueError(f"未找到标签列：{LABEL_COL}")
    label_name = LABEL_COL

# 确定不参与 Boruta 的列
existing_exclude_cols = [
    c for c in EXCLUDE_COLS
    if c in df.columns and c != label_name
]
missing_exclude_cols = [
    c for c in EXCLUDE_COLS
    if c not in df.columns
]

if missing_exclude_cols:
    print(f"提示：以下排除列未在数据中找到，将自动忽略：{missing_exclude_cols}")

# 标签列和排除列均不进入特征矩阵
feature_names = [
    c for c in df.columns
    if c != label_name and c not in existing_exclude_cols
]

if not feature_names:
    raise ValueError("排除标签列和编号列后，没有可用于 Boruta 的特征。")

X = df[feature_names].copy()
y = df[label_name].copy()

# 强制转数值
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# 检查是否仍有缺失
if X.isna().sum().sum() > 0:
    raise ValueError("特征数据中仍存在缺失值，请先确认 Excel 已完成缺失值填补。")

# 标签转 0/1
y = to_binary_label(y)

# 检查特征类型
non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric_cols:
    raise ValueError(f"存在非数值型特征，请先编码或删除：{non_numeric_cols}")

print("样本数：", len(X))
print("参与 Boruta 的特征数：", X.shape[1])
print("标签列：", label_name)
print("不参与 Boruta 的列：", existing_exclude_cols if existing_exclude_cols else "无")


# =========================================================
# 4. Boruta 粗筛
# =========================================================
# Boruta 背后的随机森林
rf_for_boruta = RandomForestClassifier(
    n_estimators=1000,
    criterion="gini",
    max_depth=None,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

boruta_selector = BorutaPy(
    estimator=rf_for_boruta,
    n_estimators="auto",
    perc=90, #默认100
    alpha=0.05,
    two_step=True,
    max_iter=100,
    random_state=RANDOM_STATE,
    verbose=2
)

boruta_selector.fit(X.values, y.values)


# =========================================================
# 5. 重新拟合随机森林，计算特征重要性（用于画图）
# =========================================================
rf_importance_model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=5,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_importance_model.fit(X, y)
rf_importances = rf_importance_model.feature_importances_


# =========================================================
# 6. 整理筛选结果
# =========================================================
result_df = pd.DataFrame({
    "feature": feature_names,
    "selected": boruta_selector.support_,
    "tentative": boruta_selector.support_weak_,
    "rank": boruta_selector.ranking_,
    "rf_importance": rf_importances
})

# 为了更方便看结果，排序一下
result_df = result_df.sort_values(
    by=["selected", "tentative", "rf_importance"],
    ascending=[False, False, False]
).reset_index(drop=True)

# 入选特征
selected_features = result_df.loc[result_df["selected"], "feature"].tolist()

# 若一个都没选中，可查看 tentative
tentative_features = result_df.loc[result_df["tentative"], "feature"].tolist()

print("\n正式入选特征数：", len(selected_features))
print("暂定特征数：", len(tentative_features))


# =========================================================
# 7. 保存结果
# =========================================================
result_file = os.path.join(OUTPUT_DIR, "mimic_boruta_results.xlsx")
result_df.to_excel(result_file, index=False)

selected_file = os.path.join(OUTPUT_DIR, "mimic_boruta_selected_features.xlsx")
pd.DataFrame({"selected_features": selected_features}).to_excel(selected_file, index=False)

tentative_file = os.path.join(OUTPUT_DIR, "mimic_boruta_tentative_features.xlsx")
pd.DataFrame({"tentative_features": tentative_features}).to_excel(tentative_file, index=False)

# 粗筛后的数据
if len(selected_features) > 0:
    output_cols = selected_features + [label_name]

    # 编号列仅用于样本追溯，不参与 Boruta；可选择保留在输出文件中
    if KEEP_EXCLUDED_COLS_IN_OUTPUT:
        output_cols = existing_exclude_cols + output_cols

    # 去重并保持原顺序
    output_cols = list(dict.fromkeys(output_cols))

    selected_df = df[output_cols].copy()
    selected_data_file = os.path.join(OUTPUT_DIR, "mimic_boruta_selected_data.xlsx")
    selected_df.to_excel(selected_data_file, index=False)
else:
    selected_data_file = None


# =========================================================
# 8. 特征重要性图
# =========================================================
plot_df = result_df.sort_values("rf_importance", ascending=True).copy()

colors = []
for _, row in plot_df.iterrows():
    if row["selected"]:
        colors.append("tab:red")      # 正式入选
    elif row["tentative"]:
        colors.append("gold")         # 暂定
    else:
        colors.append("lightgray")    # 未入选

plt.figure(figsize=(10, max(6, 0.35 * len(plot_df))))
plt.barh(plot_df["feature"], plot_df["rf_importance"], color=colors)
plt.xlabel("Random Forest Importance")
plt.ylabel("Feature")
plt.title("Boruta Feature Screening on Imputed MIMIC Data")
plt.tight_layout()

plot_file = os.path.join(OUTPUT_DIR, "mimic_boruta_importance.png")
plt.savefig(plot_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 9. 汇总说明
# =========================================================
summary_df = pd.DataFrame([{
    "n_samples": len(X),
    "n_features_entered_boruta": X.shape[1],
    "excluded_cols": ", ".join(existing_exclude_cols),
    "n_excluded_cols": len(existing_exclude_cols),
    "n_selected_features": len(selected_features),
    "n_tentative_features": len(tentative_features),
    "label_col": label_name,
    "result_file": result_file,
    "selected_features_file": selected_file,
    "tentative_features_file": tentative_file,
    "selected_data_file": selected_data_file if selected_data_file is not None else "",
    "importance_plot": plot_file
}])

summary_file = os.path.join(OUTPUT_DIR, "mimic_boruta_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

print("\n完成。")
print(f"Boruta结果表：{result_file}")
print(f"正式入选特征列表：{selected_file}")
print(f"暂定特征列表：{tentative_file}")
if selected_data_file is not None:
    print(f"Boruta粗筛后数据：{selected_data_file}")
print(f"重要性图：{plot_file}")
print(f"汇总说明：{summary_file}")
