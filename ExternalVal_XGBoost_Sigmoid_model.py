import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    accuracy_score,
    brier_score_loss
)
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("请先安装 xgboost：pip install xgboost") from e


# =========================================================
# 1. 参数设置
# =========================================================
# MIMIC：LASSO精筛后的全部训练数据
MIMIC_FILE = "ExternalVal_lasso_outputs/mimic_lasso_selected_data.xlsx"

# EICU：外部验证数据（应包含与MIMIC最终特征同名列）
EICU_FILE = "EICU_lasso_selected_data.xlsx"

MIMIC_SHEET = 0
EICU_SHEET = 0

OUTPUT_DIR = "ExternalVal_xgboost_sigmoid_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 如果为 None，则默认最后一列是标签
LABEL_COL = None

# 预测阈值
THRESHOLD = 0.5

# DCA 阈值范围，可按临床需要调整
THRESHOLDS = np.arange(0.05, 0.31, 0.01)

# Calibration curve 分箱数
N_BINS = 10

# 常见伪缺失值
PSEUDO_MISSING = ["NA", "N/A", "null", "NULL", "", " ", "None", "nan"]

RANDOM_STATE = 42


# =========================================================
# 2. 工具函数
# =========================================================
def get_label_col(df: pd.DataFrame, configured_label_col=None) -> str:
    return df.columns[-1] if configured_label_col is None else configured_label_col


def clean_label_series(y: pd.Series) -> pd.Series:
    y = y.copy()
    y = y.replace([np.inf, -np.inf], np.nan)
    y = y.astype(str).str.strip()
    y = y.replace({
        "": np.nan,
        "nan": np.nan,
        "None": np.nan,
        "NA": np.nan,
        "N/A": np.nan,
        "null": np.nan,
        "NULL": np.nan
    })

    mapping_dict = {
        "0": 0, "1": 1,
        0: 0, 1: 1,
        "yes": 1, "no": 0,
        "Yes": 1, "No": 0,
        "true": 1, "false": 0,
        "True": 1, "False": 0,
        "dead": 1, "alive": 0,
        "Dead": 1, "Alive": 0,
        "死亡": 1, "存活": 0,
        "阳性": 1, "阴性": 0
    }

    y = y.map(lambda v: mapping_dict[v] if v in mapping_dict else v)
    y = pd.to_numeric(y, errors="coerce")
    return y


def to_binary_label(y: pd.Series) -> pd.Series:
    y = clean_label_series(y)
    valid_unique = pd.Series(y).dropna().unique().tolist()

    if len(valid_unique) == 0:
        raise ValueError("标签列在清洗后全部为空。")

    if set(valid_unique) <= {0, 1}:
        return y

    raise ValueError(f"标签列清洗后仍不是二分类 0/1，当前有效取值为：{valid_unique}")


def build_base_xgb(random_state=42):
    return XGBClassifier(
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
    )


def build_xgb_sigmoid_model(random_state=42):
    base = build_base_xgb(random_state=random_state)
    try:
        calibrated = CalibratedClassifierCV(
            estimator=base,
            method="sigmoid",
            cv=5
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=base,
            method="sigmoid",
            cv=5
        )
    return calibrated


def calc_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_div(a, b):
        return float(a / b) if b != 0 else np.nan

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Sensitivity": safe_div(tp, tp + fn),
        "Specificity": safe_div(tn, tn + fp),
        "PPV": safe_div(tp, tp + fp),
        "NPV": safe_div(tn, tn + fn),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "threshold": threshold
    }


def preprocess_feature_label(df: pd.DataFrame, label_col: str):
    X = df.drop(columns=[label_col]).copy()
    y = df[label_col].copy()

    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    y = to_binary_label(y)

    valid_mask = y.notna()
    dropped_n = (~valid_mask).sum()
    if dropped_n > 0:
        print(f"警告：标签列中发现 {dropped_n} 个缺失/非法值，这些样本将被删除。")

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy().astype(int)

    if X.isna().sum().sum() > 0:
        missing_detail = X.isna().sum()
        missing_detail = missing_detail[missing_detail > 0].sort_values(ascending=False)
        raise ValueError(
            "特征数据中仍存在缺失值，请先完成预处理。缺失列统计如下：\n"
            f"{missing_detail.to_string()}"
        )

    return X, y


def calibration_intercept_slope(y_true, y_prob):
    eps = 1e-12
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(fit_intercept=True, solver="lbfgs")
    lr.fit(logit_p, y_true)

    slope = float(lr.coef_[0][0])
    intercept = float(lr.intercept_[0])
    return intercept, slope


def calculate_net_benefit_model(y_true, y_prob, thresholds):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    N = len(y_true)
    net_benefits = []

    for pt in thresholds:
        y_pred = (y_prob >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        nb = (tp / N) - (fp / N) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)


def calculate_net_benefit_all(y_true, thresholds):
    y_true = np.asarray(y_true)
    prevalence = np.mean(y_true)
    net_benefits = []

    for pt in thresholds:
        nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)


# =========================================================
# 3. 读取 MIMIC 数据
# =========================================================
mimic_df = pd.read_excel(MIMIC_FILE, sheet_name=MIMIC_SHEET)
mimic_df = mimic_df.replace(PSEUDO_MISSING, np.nan)
mimic_df = mimic_df.replace(r"^\s+$", np.nan, regex=True)
mimic_df.columns = mimic_df.columns.map(str).str.strip()

mimic_label_col = get_label_col(mimic_df, LABEL_COL)
X_mimic, y_mimic = preprocess_feature_label(mimic_df, mimic_label_col)
feature_names = X_mimic.columns.tolist()

print("MIMIC 样本数：", len(X_mimic))
print("MIMIC 特征数：", X_mimic.shape[1])
print("MIMIC 标签列：", mimic_label_col)
print("MIMIC 标签分布：")
print(y_mimic.value_counts(dropna=False))


# =========================================================
# 4. 读取 EICU 数据，并按 MIMIC 特征对齐
# =========================================================
eicu_df = pd.read_excel(EICU_FILE, sheet_name=EICU_SHEET)
eicu_df = eicu_df.replace(PSEUDO_MISSING, np.nan)
eicu_df = eicu_df.replace(r"^\s+$", np.nan, regex=True)
eicu_df.columns = eicu_df.columns.map(str).str.strip()

eicu_label_col = get_label_col(eicu_df, LABEL_COL)

missing_in_eicu = [c for c in feature_names if c not in eicu_df.columns]
if missing_in_eicu:
    raise ValueError(f"EICU 缺少以下最终特征列，无法外部验证：{missing_in_eicu}")

eicu_df = eicu_df[feature_names + [eicu_label_col]].copy()
X_eicu, y_eicu = preprocess_feature_label(eicu_df, eicu_label_col)

print("EICU 样本数：", len(X_eicu))
print("EICU 特征数：", X_eicu.shape[1])
print("EICU 标签列：", eicu_label_col)
print("EICU 标签分布：")
print(y_eicu.value_counts(dropna=False))


# =========================================================
# 5. 训练最终 XGBoost_Sigmoid 模型
# =========================================================
base_xgb = build_base_xgb(random_state=RANDOM_STATE)
base_xgb.fit(X_mimic, y_mimic)

base_model_file = os.path.join(OUTPUT_DIR, "final_base_xgb.joblib")
joblib.dump(base_xgb, base_model_file)

xgb_sigmoid_model = build_xgb_sigmoid_model(random_state=RANDOM_STATE)
xgb_sigmoid_model.fit(X_mimic, y_mimic)

final_model_file = os.path.join(OUTPUT_DIR, "final_xgboost_sigmoid_model.joblib")
joblib.dump(xgb_sigmoid_model, final_model_file)

print(f"已保存底层 XGBoost 模型：{base_model_file}")
print(f"已保存最终 XGBoost_Sigmoid 模型：{final_model_file}")


# =========================================================
# 6. 在 EICU 上外部验证
# =========================================================
y_prob_eicu = xgb_sigmoid_model.predict_proba(X_eicu)[:, 1]
y_pred_eicu = (y_prob_eicu >= THRESHOLD).astype(int)

metrics = calc_metrics(y_eicu.values, y_prob_eicu, threshold=THRESHOLD)
brier = brier_score_loss(y_eicu, y_prob_eicu)
cal_intercept, cal_slope = calibration_intercept_slope(y_eicu.values, y_prob_eicu)

# 保存预测结果
pred_df = X_eicu.copy()
pred_df[eicu_label_col] = y_eicu.values
pred_df["pred_prob_xgboost_sigmoid"] = y_prob_eicu
pred_df["pred_label_xgboost_sigmoid"] = y_pred_eicu

pred_file = os.path.join(OUTPUT_DIR, "eicu_external_predictions.xlsx")
pred_df.to_excel(pred_file, index=False)

# =========================================================
# 7. ROC 曲线
# =========================================================
fpr, tpr, _ = roc_curve(y_eicu, y_prob_eicu)

roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_data_file = os.path.join(OUTPUT_DIR, "eicu_external_roc_data.xlsx")
roc_df.to_excel(roc_data_file, index=False)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"XGBoost_Sigmoid (AUC={metrics['AUC']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("External Validation ROC Curve on EICU")
plt.legend(loc="lower right")
plt.tight_layout()

roc_fig_file = os.path.join(OUTPUT_DIR, "eicu_external_roc_curve.png")
plt.savefig(roc_fig_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 8. Calibration curve
# =========================================================
prob_true, prob_pred = calibration_curve(
    y_eicu,
    y_prob_eicu,
    n_bins=N_BINS,
    strategy="quantile"
)

calibration_df = pd.DataFrame({
    "mean_predicted_probability": prob_pred,
    "observed_event_rate": prob_true
})
calibration_data_file = os.path.join(OUTPUT_DIR, "eicu_external_calibration_data.xlsx")
calibration_df.to_excel(calibration_data_file, index=False)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="XGBoost_Sigmoid")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed probability")
plt.title(
    f"Calibration Curve on EICU\n"
    f"Brier={brier:.4f}, Intercept={cal_intercept:.4f}, Slope={cal_slope:.4f}"
)
plt.legend(loc="upper left")
plt.tight_layout()

calibration_fig_file = os.path.join(OUTPUT_DIR, "eicu_external_calibration_curve.png")
plt.savefig(calibration_fig_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 9. DCA 曲线
# =========================================================
nb_model = calculate_net_benefit_model(y_eicu.values, y_prob_eicu, THRESHOLDS)
nb_all = calculate_net_benefit_all(y_eicu.values, THRESHOLDS)
nb_none = np.zeros_like(THRESHOLDS)

dca_df = pd.DataFrame({
    "threshold_probability": THRESHOLDS,
    "net_benefit_model": nb_model,
    "net_benefit_treat_all": nb_all,
    "net_benefit_treat_none": nb_none
})
dca_data_file = os.path.join(OUTPUT_DIR, "eicu_external_dca_data.xlsx")
dca_df.to_excel(dca_data_file, index=False)

plt.figure(figsize=(8, 6))
plt.plot(THRESHOLDS, nb_model, linewidth=2.5, marker="o", markersize=3,
         label="XGBoost_Sigmoid", zorder=3)
plt.plot(THRESHOLDS, nb_all, linestyle="--", linewidth=1.8,
         label="Treat All", zorder=2)
plt.plot(THRESHOLDS, nb_none, linestyle="--", linewidth=1.8,
         label="Treat None", zorder=1)

plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis on EICU")
plt.xlim(THRESHOLDS.min(), THRESHOLDS.max())

y_min = min(nb_model.min(), nb_all.min(), nb_none.min())
y_max = max(nb_model.max(), nb_all.max(), nb_none.max())
plt.ylim(y_min - 0.02, y_max + 0.02)

plt.grid(alpha=0.3, linestyle="--")
plt.legend(loc="best")
plt.tight_layout()

dca_fig_file = os.path.join(OUTPUT_DIR, "eicu_external_dca_curve.png")
plt.savefig(dca_fig_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 10. 保存综合指标
# =========================================================
metrics_full = metrics.copy()
metrics_full["Brier_score"] = brier
metrics_full["Calibration_intercept"] = cal_intercept
metrics_full["Calibration_slope"] = cal_slope

metrics_df = pd.DataFrame([metrics_full])
metrics_file = os.path.join(OUTPUT_DIR, "eicu_external_validation_metrics.xlsx")
metrics_df.to_excel(metrics_file, index=False)


# =========================================================
# 11. 汇总说明
# =========================================================
summary_df = pd.DataFrame([{
    "mimic_n_samples": len(X_mimic),
    "eicu_n_samples": len(X_eicu),
    "n_features": len(feature_names),
    "label_col_mimic": mimic_label_col,
    "label_col_eicu": eicu_label_col,
    "base_model_file": base_model_file,
    "final_model_file": final_model_file,
    "predictions_file": pred_file,
    "metrics_file": metrics_file,
    "roc_data_file": roc_data_file,
    "roc_figure_file": roc_fig_file,
    "calibration_data_file": calibration_data_file,
    "calibration_figure_file": calibration_fig_file,
    "dca_data_file": dca_data_file,
    "dca_figure_file": dca_fig_file
}])

summary_file = os.path.join(OUTPUT_DIR, "external_validation_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

print("\n完成。")
print(f"EICU 外部预测结果：{pred_file}")
print(f"EICU 外部验证指标：{metrics_file}")
print(f"EICU ROC 图：{roc_fig_file}")
print(f"EICU Calibration 图：{calibration_fig_file}")
print(f"EICU DCA 图：{dca_fig_file}")
print(f"汇总说明：{summary_file}")
