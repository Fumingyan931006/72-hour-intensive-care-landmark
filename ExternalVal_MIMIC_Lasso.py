# 对填补好的MIMIC外部验证实验数据进行Boruta特征粗筛后，进行LASSO精筛

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

warnings.filterwarnings("ignore")


# =========================================================
# 1. 参数设置
# =========================================================
INPUT_FILE = "ExternalVal_boruta_outputs/mimic_boruta_selected_data.xlsx"   # Boruta粗筛后的数据
SHEET_NAME = 0
OUTPUT_DIR = "ExternalVal_lasso_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 如果为 None，则默认最后一列是标签
LABEL_COL = None

# 若有不参与建模的ID列，可填在这里；没有就留空
ID_COLS = ["number"]

# Logistic L1 的 C 搜索范围
Cs = np.logspace(-3, 2, 50)

# 内部交叉验证折数
INNER_CV = 5

# 评分指标
SCORING = "roc_auc"

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


def get_label_col(df: pd.DataFrame, configured_label_col=None) -> str:
    return df.columns[-1] if configured_label_col is None else configured_label_col


def prepare_xy(df: pd.DataFrame, label_col: str, id_cols=None):
    if id_cols is None:
        id_cols = []

    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]
    X = df[feature_cols].copy()
    y = to_binary_label(df[label_col].copy())

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"存在非数值型特征，请先编码或删除：{non_numeric_cols}")

    return X, y, feature_cols


# =========================================================
# 3. 读取数据
# =========================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

label_col = get_label_col(df, LABEL_COL)
X, y, feature_names = prepare_xy(df, label_col, ID_COLS)

# 检查缺失
if X.isna().sum().sum() > 0:
    raise ValueError("数据中仍存在缺失值，请先确认 Boruta 粗筛后的数据已完成缺失值处理。")

print("样本数：", len(X))
print("Boruta后特征数：", X.shape[1])
print("标签列：", label_col)


# =========================================================
# 4. 标准化
# =========================================================
scaler = StandardScaler()
X_scaled = pd.DataFrame(
    scaler.fit_transform(X),
    columns=feature_names,
    index=X.index
)


# =========================================================
# 5. LASSO（L1 Logistic RegressionCV）
# =========================================================
lasso_cv = LogisticRegressionCV(
    Cs=Cs,
    cv=INNER_CV,
    penalty="l1",
    solver="saga",
    scoring=SCORING,
    max_iter=10000,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

lasso_cv.fit(X_scaled, y)

best_C = float(lasso_cv.C_[0])
coef = lasso_cv.coef_.ravel()
abs_coef = np.abs(coef)

result_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coef,
    "abs_coef": abs_coef,
    "selected": abs_coef > 1e-8
}).sort_values(["selected", "abs_coef"], ascending=[False, False]).reset_index(drop=True)

selected_features = result_df.loc[result_df["selected"], "feature"].tolist()

# 如果一个都没选中，保底选绝对系数最大的1个
used_fallback = False
if len(selected_features) == 0:
    selected_features = result_df.sort_values("abs_coef", ascending=False)["feature"].head(1).tolist()
    used_fallback = True
    result_df["selected_fallback"] = result_df["feature"].isin(selected_features)
else:
    result_df["selected_fallback"] = False

print("LASSO正式入选特征数：", len(selected_features))
print("最优 C：", best_C)


# =========================================================
# 6. 保存筛选结果
# =========================================================
result_file = os.path.join(OUTPUT_DIR, "mimic_lasso_results.xlsx")
selected_file = os.path.join(OUTPUT_DIR, "mimic_lasso_selected_features.xlsx")
selected_data_file = os.path.join(OUTPUT_DIR, "mimic_lasso_selected_data.xlsx")

result_df.to_excel(result_file, index=False)
pd.DataFrame({"selected_features": selected_features}).to_excel(selected_file, index=False)

selected_df = df[selected_features + [label_col]].copy()
selected_df.to_excel(selected_data_file, index=False)


# =========================================================
# 7. 系数路径图
# =========================================================
coefs = []

for c in Cs:
    clf = LogisticRegression(
        penalty="l1",
        solver="saga",
        C=c,
        max_iter=10000,
        random_state=RANDOM_STATE
    )
    clf.fit(X_scaled, y)
    coefs.append(clf.coef_.ravel())

coefs = np.array(coefs)
logCs = np.log10(Cs)

plt.figure(figsize=(12, 8))

# 只给系数变化幅度较大的特征加图例，避免过于拥挤
max_abs_coef = np.max(np.abs(coefs), axis=0)
top_idx = np.argsort(max_abs_coef)[::-1][:20]
top_idx_set = set(top_idx.tolist())

for i, feat in enumerate(feature_names):
    if i in top_idx_set:
        plt.plot(logCs, coefs[:, i], linewidth=1.5, label=feat)
    else:
        plt.plot(logCs, coefs[:, i], linewidth=1.0, alpha=0.4)

plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.axvline(np.log10(best_C), color="red", linestyle="--", linewidth=1.2, label=f"Best C={best_C:.4f}")

plt.xlabel("log10(C)")
plt.ylabel("Coefficient")
plt.title("LASSO Coefficient Path (L1 Logistic Regression)")
plt.legend(fontsize=8, loc="best", ncol=2)
plt.tight_layout()

coef_path_file = os.path.join(OUTPUT_DIR, "mimic_lasso_coef_path.png")
plt.savefig(coef_path_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 8. CV评分曲线
# =========================================================
# scores_ 是一个 dict，key 为类别标签
# 对二分类模型，常取正类那一项；若不确定，取第一个
score_dict = lasso_cv.scores_
score_key = list(score_dict.keys())[0]
score_matrix = score_dict[score_key]   # shape: [n_folds, n_Cs]

mean_scores = np.mean(score_matrix, axis=0)
std_scores = np.std(score_matrix, axis=0)

plt.figure(figsize=(8, 6))
plt.plot(logCs, mean_scores, linewidth=2, label="Mean CV score")
plt.fill_between(logCs, mean_scores - std_scores, mean_scores + std_scores, alpha=0.2)

plt.axvline(np.log10(best_C), color="red", linestyle="--", linewidth=1.2, label=f"Best C={best_C:.4f}")

plt.xlabel("log10(C)")
plt.ylabel(SCORING)
plt.title("LASSO CV Performance Curve")
plt.legend(loc="best")
plt.tight_layout()

cv_curve_file = os.path.join(OUTPUT_DIR, "mimic_lasso_cv_curve.png")
plt.savefig(cv_curve_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 9. 汇总信息
# =========================================================
summary_df = pd.DataFrame([{
    "n_samples": len(X),
    "n_boruta_features": X.shape[1],
    "n_lasso_selected_features": len(selected_features),
    "best_C": best_C,
    "used_fallback": used_fallback,
    "label_col": label_col,
    "result_file": result_file,
    "selected_features_file": selected_file,
    "selected_data_file": selected_data_file,
    "coef_path_file": coef_path_file,
    "cv_curve_file": cv_curve_file
}])

summary_file = os.path.join(OUTPUT_DIR, "mimic_lasso_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

print("\n完成。")
print(f"LASSO结果表：{result_file}")
print(f"LASSO入选特征：{selected_file}")
print(f"LASSO精筛后数据：{selected_data_file}")
print(f"系数路径图：{coef_path_file}")
print(f"CV评分曲线：{cv_curve_file}")
print(f"汇总说明：{summary_file}")