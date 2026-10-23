import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("请先安装 xgboost：pip install xgboost") from e


# =========================================================
# 1. 参数设置
# =========================================================
# LASSO精筛后的5折数据目录
LASSO_DIR = "lasso_outputs"

# 输出目录
OUTPUT_DIR = "xgboost_sigmoid_oof_calibration_dca"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交叉验证折数
N_FOLDS = 5

# 标签列：若为 None，则默认最后一列是标签
LABEL_COL = None

# ID列：保留在原表中但不参与建模
ID_COLS = [""]   # 例如 ["patient_id"]

# Calibration curve 分箱数
N_BINS = 10

# DCA 阈值
THRESHOLDS = np.arange(0.05, 0.31, 0.01)

RANDOM_STATE = 42


# =========================================================
# 2. 工具函数
# =========================================================
def get_label_col(df: pd.DataFrame, configured_label_col=None) -> str:
    return df.columns[-1] if configured_label_col is None else configured_label_col


def to_binary_label(y: pd.Series) -> pd.Series:
    unique_vals = pd.Series(y).dropna().unique().tolist()
    if set(unique_vals) <= {0, 1}:
        return y.astype(int)

    if len(unique_vals) == 2:
        mapping = {sorted(unique_vals)[0]: 0, sorted(unique_vals)[1]: 1}
        return y.map(mapping).astype(int)

    raise ValueError(f"标签列必须为二分类，当前取值为：{unique_vals}")


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


def build_xgb_sigmoid_model(random_state=42):
    base_xgb = XGBClassifier(
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

    # 兼容不同 sklearn 版本
    try:
        calibrated = CalibratedClassifierCV(
            estimator=base_xgb,
            method="sigmoid",
            cv=5
        )
    except TypeError:
        calibrated = CalibratedClassifierCV(
            base_estimator=base_xgb,
            method="sigmoid",
            cv=5
        )

    # 加一个简单中位数插补更稳妥
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", calibrated)
    ])
    return model


def calculate_net_benefit_model(y_true, y_prob, thresholds):
    """
    模型净获益：
    NB = TP/N - FP/N * (pt / (1-pt))
    """
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
    """
    Treat-all 策略净获益
    """
    y_true = np.asarray(y_true)
    prevalence = np.mean(y_true)
    net_benefits = []

    for pt in thresholds:
        nb = prevalence - (1 - prevalence) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.array(net_benefits)


def calibration_intercept_slope(y_true, y_prob):
    """
    计算校准截距和校准斜率
    常用近似：
    - 截距：logit(p) -> y，只含截距项
    - 斜率：logit(p) -> y，拟合逻辑回归系数
    """
    eps = 1e-12
    p = np.clip(np.asarray(y_prob), eps, 1 - eps)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    # 斜率
    slope_model = LogisticRegression(fit_intercept=True, solver="lbfgs")
    slope_model.fit(logit_p, y_true)
    slope = float(slope_model.coef_[0][0])
    intercept = float(slope_model.intercept_[0])

    return intercept, slope


# =========================================================
# 3. 逐折训练并收集 OOF 预测
# =========================================================
all_y_true = []
all_y_prob = []
fold_rows = []

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

    X_train, y_train, feature_cols = prepare_xy(train_df, label_col, ID_COLS)
    X_test, y_test, _ = prepare_xy(test_df, label_col, ID_COLS)

    model = build_xgb_sigmoid_model(random_state=RANDOM_STATE + fold_id)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]

    fold_auc = roc_auc_score(y_test, y_prob)
    fold_brier = brier_score_loss(y_test, y_prob)

    print(f"Fold {fold_id} AUC   = {fold_auc:.4f}")
    print(f"Fold {fold_id} Brier = {fold_brier:.4f}")

    all_y_true.extend(y_test.tolist())
    all_y_prob.extend(y_prob.tolist())

    fold_rows.append({
        "fold": fold_id,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "AUC": fold_auc,
        "Brier_score": fold_brier
    })

    # 保存每折预测
    fold_pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_prob": y_prob
    })
    fold_pred_df.to_excel(
        os.path.join(OUTPUT_DIR, f"fold{fold_id}_xgboost_sigmoid_predictions.xlsx"),
        index=False
    )

# 汇总 OOF
oof_df = pd.DataFrame({
    "y_true": all_y_true,
    "y_prob": all_y_prob
})
oof_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_oof_predictions.xlsx")
oof_df.to_excel(oof_file, index=False)

fold_summary_df = pd.DataFrame(fold_rows)
fold_summary_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_fold_summary.xlsx")
fold_summary_df.to_excel(fold_summary_file, index=False)


# =========================================================
# 4. 基于 OOF 概率做 Calibration curve
# =========================================================
y_true = np.asarray(all_y_true)
y_prob = np.asarray(all_y_prob)

overall_auc = roc_auc_score(y_true, y_prob)
overall_brier = brier_score_loss(y_true, y_prob)
cal_intercept, cal_slope = calibration_intercept_slope(y_true, y_prob)

prob_true, prob_pred = calibration_curve(
    y_true,
    y_prob,
    n_bins=N_BINS,
    strategy="quantile"
)

calibration_df = pd.DataFrame({
    "mean_predicted_probability": prob_pred,
    "observed_event_rate": prob_true
})
calibration_data_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_calibration_data.xlsx")
calibration_df.to_excel(calibration_data_file, index=False)

plt.figure(figsize=(7, 6))
plt.plot(prob_pred, prob_true, marker="o", linewidth=2, label="XGBoost_Sigmoid")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed probability")
plt.title(
    f"Calibration Curve\n"
    f"AUC={overall_auc:.4f}, Brier={overall_brier:.4f}, "
    f"Intercept={cal_intercept:.4f}, Slope={cal_slope:.4f}"
)
plt.legend(loc="upper left")
plt.tight_layout()

calibration_fig = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_calibration_curve.png")
plt.savefig(calibration_fig, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 5. 基于 OOF 概率做 DCA
# =========================================================
nb_model = calculate_net_benefit_model(y_true, y_prob, THRESHOLDS)
nb_all = calculate_net_benefit_all(y_true, THRESHOLDS)
nb_none = np.zeros_like(THRESHOLDS)

dca_df = pd.DataFrame({
    "threshold_probability": THRESHOLDS,
    "net_benefit_model": nb_model,
    "net_benefit_treat_all": nb_all,
    "net_benefit_treat_none": nb_none
})
dca_data_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_dca_data.xlsx")
dca_df.to_excel(dca_data_file, index=False)

plt.figure(figsize=(8, 6))
plt.plot(THRESHOLDS, nb_model, linewidth=2, label="XGBoost_Sigmoid")
plt.plot(THRESHOLDS, nb_all, linestyle="--", linewidth=1.5, label="Treat All")
plt.plot(THRESHOLDS, nb_none, linestyle="--", linewidth=1.5, label="Treat None")
plt.xlabel("Threshold probability")
plt.ylabel("Net benefit")
plt.title("Decision Curve Analysis")

# 只显示更有临床意义的局部范围
plt.xlim(0.05, 0.30)
y_min = min(nb_model.min(), nb_all.min(), nb_none.min())
y_max = max(nb_model.max(), nb_all.max(), nb_none.max())
plt.ylim(y_min - 0.02, y_max + 0.02)

# 加网格，便于看细微差异
plt.grid(alpha=0.3, linestyle="--")

plt.legend(loc="upper right")
plt.tight_layout()

dca_fig = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_dca_curve.png")
plt.savefig(dca_fig, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 6. 额外保存 ROC 结果（可选）
# =========================================================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr})
roc_data_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_oof_roc_data.xlsx")
roc_df.to_excel(roc_data_file, index=False)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, linewidth=2, label=f"XGBoost_Sigmoid (AUC={overall_auc:.4f})")
plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Based on 5-fold OOF Probabilities")
plt.legend(loc="lower right")
plt.tight_layout()

roc_fig = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_oof_roc_curve.png")
plt.savefig(roc_fig, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 7. 汇总结果
# =========================================================
summary_df = pd.DataFrame([{
    "model": "XGBoost_Sigmoid",
    "n_samples": len(y_true),
    "event_rate": np.mean(y_true),
    "AUC_oof": overall_auc,
    "Brier_score_oof": overall_brier,
    "calibration_intercept": cal_intercept,
    "calibration_slope": cal_slope
}])
summary_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_oof_calibration_dca_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

print("\n完成。")
print(f"OOF预测结果: {oof_file}")
print(f"分折汇总: {fold_summary_file}")
print(f"Calibration curve 图: {calibration_fig}")
print(f"Calibration 数据: {calibration_data_file}")
print(f"DCA 图: {dca_fig}")
print(f"DCA 数据: {dca_data_file}")
print(f"汇总表: {summary_file}")
