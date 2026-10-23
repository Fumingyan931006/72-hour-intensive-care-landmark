import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

# =========================
# 1. 可选依赖
# =========================
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# =========================
# 2. 参数设置
# =========================
# LASSO精筛后的5折数据目录
LASSO_DIR = "lasso_outputs"

# 总表文件：SOFA 和 SAPSII 从这里直接读取
FULL_DATA_FILE = "SAPSII_Sofa2.xlsx"
FULL_DATA_SHEET = 0

# 输出目录
OUTPUT_DIR = "roc_10model_outputs5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 折数
N_FOLDS = 5

# 标签列：None表示默认最后一列
LABEL_COL = None

# ID列（如果有）
ID_COLS: List[str] = [""]

# 基线评分列名
SOFA_COL = "sofa_first"
SAPSII_COL = "sapsii"

# 如果评分方向与阳性标签相反，可设为 True
FLIP_SOFA = False
FLIP_SAPSII = False

RANDOM_STATE = 42


# =========================
# 3. 工具函数
# =========================
def get_label_col(df: pd.DataFrame, configured_label_col=None) -> str:
    return df.columns[-1] if configured_label_col is None else configured_label_col


def to_binary_label(y: pd.Series) -> pd.Series:
    unique_vals = pd.Series(y).dropna().unique().tolist()
    if set(unique_vals) == {0, 1} or set(unique_vals) == {0} or set(unique_vals) == {1}:
        return y.astype(int)
    if len(unique_vals) == 2:
        mapping = {sorted(unique_vals)[0]: 0, sorted(unique_vals)[1]: 1}
        return y.map(mapping).astype(int)
    raise ValueError(f"标签列必须为二分类，当前取值为：{unique_vals}")


def prepare_xy(df: pd.DataFrame, label_col: str, id_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]
    X = df[feature_cols].copy()
    y = to_binary_label(df[label_col].copy())

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"存在非数值型特征，请先编码或删除：{non_numeric_cols}")

    return X, y, feature_cols


def build_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {}

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=130,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )

        try:
            models["XGBoost_Sigmoid"] = CalibratedClassifierCV(
                estimator=XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1
                ),
                method="sigmoid",
                cv=5
            )
        except TypeError:
            models["XGBoost_Sigmoid"] = CalibratedClassifierCV(
                base_estimator=XGBClassifier(
                    n_estimators=120,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_state,
                    n_jobs=-1
                ),
                method="sigmoid",
                cv=5
            )

    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )

    models["SVM"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state))
    ])

    models["KNN"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, weights="distance"))
    ])

    models["BPNN"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=1000,
            random_state=random_state
        ))
    ])

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=100,  # 原本为100
        max_depth=2,  # 限制树深度 原本为4
        min_samples_split=40,  # 节点至少20个样本才允许继续分裂
        min_samples_leaf=40,  # 每个叶节点至少10个样本
        max_features=0.3,  # 每次分裂只考虑部分特征 原本为"sqrt"
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1
    )

    models["LogisticRegression"] = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l1",
            C=0.01,
            solver="liblinear",
            max_iter=1000,
            random_state=random_state
        ))
    ])

    return models


def fit_and_predict_proba(model, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> np.ndarray:
    model.fit(X_train, y_train)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        score = model.decision_function(X_test)
        score = (score - np.min(score)) / (np.max(score) - np.min(score) + 1e-12)
        return score
    raise ValueError("模型既没有 predict_proba，也没有 decision_function。")


# =========================
# 4. 读取总表中的 SOFA / SAPSII
# =========================
full_df = pd.read_excel(FULL_DATA_FILE, sheet_name=FULL_DATA_SHEET)
full_df = full_df.replace(["NA", "N/A", "null", "NULL", ""], np.nan)
full_df = full_df.replace(r"^\s+$", np.nan, regex=True)

full_label_col = get_label_col(full_df, LABEL_COL)

required_cols = [full_label_col, SOFA_COL, SAPSII_COL]
missing_cols = [c for c in required_cols if c not in full_df.columns]
if missing_cols:
    raise ValueError(f"总表中缺少必要列：{missing_cols}")

score_df = full_df[[full_label_col, SOFA_COL, SAPSII_COL]].dropna().copy()
score_y = to_binary_label(score_df[full_label_col])
sofa_score = score_df[SOFA_COL].astype(float).copy()
sapsii_score = score_df[SAPSII_COL].astype(float).copy()

if FLIP_SOFA:
    sofa_score = -sofa_score
if FLIP_SAPSII:
    sapsii_score = -sapsii_score


# =========================
# 5. 基于5折LASSO数据训练8个模型并拼接预测
# =========================
models = build_models(RANDOM_STATE)
if not HAS_XGB:
    print("Warning: xgboost 未安装，XGBoost 和 XGBoost_Sigmoid 将被跳过。")
if not HAS_LGBM:
    print("Warning: lightgbm 未安装，LightGBM 将被跳过。")

all_model_pred = {model_name: {"y_true": [], "y_prob": []} for model_name in models.keys()}

for fold_id in range(1, N_FOLDS + 1):
    print(f"\n========== Fold {fold_id} ==========")

    train_file = os.path.join(LASSO_DIR, f"fold{fold_id}", f"fold{fold_id}_train_lasso_selected.xlsx")
    test_file = os.path.join(LASSO_DIR, f"fold{fold_id}", f"fold{fold_id}_test_lasso_selected.xlsx")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"未找到文件：{train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"未找到文件：{test_file}")

    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)

    label_col = get_label_col(train_df, LABEL_COL)

    X_train, y_train, _ = prepare_xy(train_df, label_col, ID_COLS)
    X_test, y_test, _ = prepare_xy(test_df, label_col, ID_COLS)

    for model_name, model in models.items():
        print(f"  Training {model_name} ...")
        y_prob = fit_and_predict_proba(model, X_train, y_train, X_test)

        all_model_pred[model_name]["y_true"].extend(y_test.tolist())
        all_model_pred[model_name]["y_prob"].extend(y_prob.tolist())


# =========================
# 6. 计算10条ROC并绘图
# =========================
plt.figure(figsize=(10, 8))
auc_rows = []

# 8个模型
for model_name, pred_dict in all_model_pred.items():
    y_true = np.array(pred_dict["y_true"])
    y_prob = np.array(pred_dict["y_prob"])

    auc_value = roc_auc_score(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.plot(fpr, tpr, linewidth=2, label=f"{model_name} (AUC={auc_value:.4f})")

    auc_rows.append({
        "model": model_name,
        "AUC": auc_value,
        "type": "ML_5fold"
    })

    pd.DataFrame({
        "y_true": y_true,
        "y_prob": y_prob
    }).to_excel(os.path.join(OUTPUT_DIR, f"{model_name}_5fold_predictions.xlsx"), index=False)

# SOFA
sofa_auc = roc_auc_score(score_y, sofa_score)
sofa_fpr, sofa_tpr, _ = roc_curve(score_y, sofa_score)
plt.plot(sofa_fpr, sofa_tpr, linewidth=2, label=f"SOFA2 (AUC={sofa_auc:.4f})")
auc_rows.append({"model": "SOFA", "AUC": sofa_auc, "type": "ClinicalScore"})

pd.DataFrame({
    "y_true": score_y,
    "score": sofa_score
}).to_excel(os.path.join(OUTPUT_DIR, "SOFA_scores_from_full_data.xlsx"), index=False)

# SAPSII
sapsii_auc = roc_auc_score(score_y, sapsii_score)
sapsii_fpr, sapsii_tpr, _ = roc_curve(score_y, sapsii_score)
plt.plot(sapsii_fpr, sapsii_tpr, linewidth=2, label=f"SAPSII (AUC={sapsii_auc:.4f})")
auc_rows.append({"model": "SAPSII", "AUC": sapsii_auc, "type": "ClinicalScore"})

pd.DataFrame({
    "y_true": score_y,
    "score": sapsii_score
}).to_excel(os.path.join(OUTPUT_DIR, "SAPSII_scores_from_full_data.xlsx"), index=False)

# 对角线
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves of 8 ML Models + SOFA2 + SAPSII")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()

roc_fig_path = os.path.join(OUTPUT_DIR, "fivefold_8models_plus_SOFA2_SAPSII.png")
plt.savefig(roc_fig_path, dpi=300, bbox_inches="tight")
plt.close()

# AUC汇总
auc_df = pd.DataFrame(auc_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
auc_excel_path = os.path.join(OUTPUT_DIR, "fivefold_8models_plus_SOFA2_SAPSII_auc_summary.xlsx")
auc_df.to_excel(auc_excel_path, index=False)

print("\n完成。")
print(f"ROC图已保存：{roc_fig_path}")
print(f"AUC汇总已保存：{auc_excel_path}")
