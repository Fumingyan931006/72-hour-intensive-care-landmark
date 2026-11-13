import os
import joblib
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge


# =========================================================
# 1. 参数设置
# =========================================================
MIMIC_FILE = "MIMIC外部验证实验数据.xlsx"          # MIMIC 全部数据
SPHCC_FILE = "SPHCC外部验证实验数据.xlsx"         # SPHCC 外部验证数据

MIMIC_SHEET = 0
EICU_SHEET = 0

OUTPUT_DIR = "mice_external_validation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 不参与 MICE 拟合/插补的列
# 例如病例ID、标签列、明显泄漏变量等
EXCLUDE_COLS = [""]

# 若某些“伪缺失值”在 Excel 中以字符串形式出现，可在这里补充
PSEUDO_MISSING = ["NA", "N/A", "null", "NULL", ""]


# =========================================================
# 2. 读取数据
# =========================================================
mimic_df = pd.read_excel(MIMIC_FILE, sheet_name=MIMIC_SHEET)
SPHCC_df = pd.read_excel(SPHCC_FILE, sheet_name=SPHCC_SHEET)

# 统一伪缺失
mimic_df = mimic_df.replace(PSEUDO_MISSING, np.nan)
mimic_df = mimic_df.replace(r"^\s+$", np.nan, regex=True)

SPHCC_df = SPHCC_df.replace(PSEUDO_MISSING, np.nan)
SPHCC_df = SPHCC_df.replace(r"^\s+$", np.nan, regex=True)

# 统一列名格式
mimic_df.columns = mimic_df.columns.map(str).str.strip()
SPHCC_df.columns = SPHCC_df.columns.map(str).str.strip()


# =========================================================
# 3. 选择参与插补的列
#    只对数值型特征做 MICE
# =========================================================
candidate_cols = [c for c in mimic_df.columns if c not in EXCLUDE_COLS]

mimic_numeric_cols = mimic_df[candidate_cols].select_dtypes(include=[np.number]).columns.tolist()

if len(mimic_numeric_cols) == 0:
    raise ValueError("MIMIC 中没有可用于 MICE 的数值型特征。")

# SPHCC 必须至少包含同名列
missing_in_SPHCC = [c for c in mimic_numeric_cols if c not in SPHCC_df.columns]
if missing_in_SPHCC:
    raise ValueError(f"SPHCC 缺少以下 MIMIC 最终特征列，无法直接外部验证：{missing_in_SPHCC}")

# 只取与 MIMIC 一致的数值列，顺序保持一致
feature_cols = mimic_numeric_cols

print("用于 MICE 的特征数：", len(feature_cols))
print("特征列表：", feature_cols)


# =========================================================
# 4. 提取 MIMIC / eICU 特征矩阵
# =========================================================
X_mimic = mimic_df[feature_cols].copy()
X_SPHCC = SPHCC_df[feature_cols].copy()

# 强制转数值
for c in feature_cols:
    X_mimic[c] = pd.to_numeric(X_mimic[c], errors="coerce")
    X_SPHCC[c] = pd.to_numeric(X_SPHCC[c], errors="coerce")


# =========================================================
# 5. 在 MIMIC 全部数据上拟合最终 MICE
# =========================================================
mice_imputer = IterativeImputer(
    estimator=BayesianRidge(),
    max_iter=10,
    random_state=42,
    sample_posterior=False
)

X_mimic_imputed = pd.DataFrame(
    mice_imputer.fit_transform(X_mimic),
    columns=feature_cols,
    index=X_mimic.index
)

# 应用于 SPHCC：只 transform，不重新 fit
X_SPHCC_imputed = pd.DataFrame(
    mice_imputer.transform(X_SPHCC),
    columns=feature_cols,
    index=X_SPHCC.index
)


# =========================================================
# 6. 回填到原始表结构中
# =========================================================
mimic_imputed_df = mimic_df.copy()
SPHCC_imputed_df = SPHCC_df.copy()

mimic_imputed_df[feature_cols] = X_mimic_imputed
SPHCC_imputed_df[feature_cols] = X_SPHCC_imputed


# =========================================================
# 7. 保存结果
# =========================================================
mimic_outfile = os.path.join(OUTPUT_DIR, "mimic_imputed.xlsx")
SPHCC_outfile = os.path.join(OUTPUT_DIR, "SPHCC_imputed.xlsx")
imputer_file = os.path.join(OUTPUT_DIR, "mice_imputer_mimic.joblib")

mimic_imputed_df.to_excel(mimic_outfile, index=False)
SPHCC_imputed_df.to_excel(SPHCC_outfile, index=False)
joblib.dump(mice_imputer, imputer_file)

print("\n完成。")
print(f"MIMIC 插补后文件：{mimic_outfile}")
print(f"SPHCC 插补后文件：{SPHCC_outfile}")
print(f"MICE 插补器：{imputer_file}")
