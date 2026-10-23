import os
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedKFold


INPUT_FILE = "实验数据.xlsx"   # 实验数据Excel文件
SHEET_NAME = 0                   # 工作表名或序号，0表示第1个工作表
OUTPUT_DIR = "folder"     # 输出目录
N_SPLITS = 5
RANDOM_STATE = 42

PSEUDO_MISSING = ["NA", "N/A", "null", "NULL", ""]


df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)
df.columns = df.columns.map(str).str.strip()

df = df.replace(PSEUDO_MISSING, np.nan)
df = df.replace(r"^\s+$", np.nan, regex=True)

if df.shape[1] < 2:
    raise ValueError("Excel 至少需要两列：特征列 + 最后一列标签列。")

label_col = df.columns[-1]
feature_cols = df.columns[:-1].tolist()

X = df[feature_cols].copy()
y = df[label_col].copy()

for col in feature_cols:
    X[col] = pd.to_numeric(X[col], errors="ignore")

numeric_feature_cols = X.select_dtypes(include=[np.number]).columns.tolist()
non_numeric_feature_cols = [c for c in feature_cols if c not in numeric_feature_cols]

if len(numeric_feature_cols) == 0:
    raise ValueError("没有检测到数值型特征列，无法执行 MICE。")

os.makedirs(OUTPUT_DIR, exist_ok=True)

summary_rows = []

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        random_state=RANDOM_STATE,
        sample_posterior=True
    )

    X_train_num_imputed = pd.DataFrame(
        imputer.fit_transform(X_train[numeric_feature_cols]),
        columns=numeric_feature_cols,
        index=X_train.index
    )

    X_test_num_imputed = pd.DataFrame(
        imputer.transform(X_test[numeric_feature_cols]),
        columns=numeric_feature_cols,
        index=X_test.index
    )

    if non_numeric_feature_cols:
        X_train_imputed = pd.concat(
            [X_train[non_numeric_feature_cols], X_train_num_imputed],
            axis=1
        )
        X_test_imputed = pd.concat(
            [X_test[non_numeric_feature_cols], X_test_num_imputed],
            axis=1
        )
        X_train_imputed = X_train_imputed[feature_cols]
        X_test_imputed = X_test_imputed[feature_cols]
    else:
        X_train_imputed = X_train_num_imputed[feature_cols]
        X_test_imputed = X_test_num_imputed[feature_cols]

    train_imputed_df = X_train_imputed.copy()
    train_imputed_df[label_col] = y_train.values

    test_imputed_df = X_test_imputed.copy()
    test_imputed_df[label_col] = y_test.values

    train_path = os.path.join(OUTPUT_DIR, f"fold{fold}_train_imputed.xlsx")
    test_path = os.path.join(OUTPUT_DIR, f"fold{fold}_test_imputed.xlsx")
    index_path = os.path.join(OUTPUT_DIR, f"fold{fold}_indices.xlsx")

    train_imputed_df.to_excel(train_path, index=False)
    test_imputed_df.to_excel(test_path, index=False)

    pd.DataFrame({
        "train_index": pd.Series(train_idx),
        "test_index": pd.Series(test_idx)
    }).to_excel(index_path, index=False)

    summary_rows.append({
        "fold": fold,
        "train_n": len(train_idx),
        "test_n": len(test_idx),
        "train_file": train_path,
        "test_file": test_path,
        "index_file": index_path
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_excel(os.path.join(OUTPUT_DIR, "imputed_folds_summary.xlsx"), index=False)

print("处理完成。")
print(f"标签列：{label_col}")
print(f"数值型特征列数：{len(numeric_feature_cols)}")
print(f"非数值型特征列数：{len(non_numeric_feature_cols)}")
print(f"结果已保存到文件夹：{os.path.abspath(OUTPUT_DIR)}")
