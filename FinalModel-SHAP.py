import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
except Exception as e:
    raise ImportError("请先安装 xgboost：pip install xgboost") from e

try:
    import shap
except Exception as e:
    raise ImportError("请先安装 shap：pip install shap") from e


# =========================================================
# 1. 参数设置
# =========================================================
INPUT_FILE = "FinalModelData.xlsx"   # Excel文件名
SHEET_NAME = 0                       # 可写 0 或工作表名
OUTPUT_DIR = "final_xgboost_sigmoid_shap_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 如果为 None，则默认最后一列是标签，前面所有列都是特征
LABEL_COL = None

# 样本太多时，SHAP作图可抽样
MAX_SHAP_SAMPLES = 1000

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


# =========================================================
# 3. 读取数据
# =========================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

# 默认：最后一列是标签，前面所有列都是特征
if LABEL_COL is None:
    if df.shape[1] < 2:
        raise ValueError("表格至少需要2列：前面是特征，最后一列是标签。")
    X = df.iloc[:, :-1].copy()
    y = df.iloc[:, -1].copy()
    feature_names = X.columns.tolist()
    label_name = df.columns[-1]
else:
    if LABEL_COL not in df.columns:
        raise ValueError(f"未找到标签列：{LABEL_COL}")
    label_name = LABEL_COL
    feature_names = [c for c in df.columns if c != label_name]
    X = df[feature_names].copy()
    y = df[label_name].copy()

# 强制转数值
for c in X.columns:
    X[c] = pd.to_numeric(X[c], errors="coerce")

# 检查是否还有缺失
if X.isna().sum().sum() > 0:
    raise ValueError("特征数据中仍存在缺失值，请先确认Excel已完成缺失值填补。")

# 标签转0/1
y = to_binary_label(y)

print("样本数：", len(X))
print("特征数：", X.shape[1])
print("标签列：", label_name)


# =========================================================
# 4. 训练最终模型
# =========================================================
# 4.1 底层 XGBoost：用于 SHAP
base_xgb_for_shap = build_base_xgb(random_state=RANDOM_STATE)
base_xgb_for_shap.fit(X, y)

# 训练后立即保存底层模型
base_model_file = os.path.join(OUTPUT_DIR, "final_base_xgb_for_shap.joblib")
joblib.dump(base_xgb_for_shap, base_model_file)
print(f"已保存底层XGBoost模型：{base_model_file}")

# 4.2 XGBoost + Sigmoid：用于最终预测
xgb_sigmoid_model = build_xgb_sigmoid_model(random_state=RANDOM_STATE)
xgb_sigmoid_model.fit(X, y)

# 训练后立即保存最终模型
final_model_file = os.path.join(OUTPUT_DIR, "final_xgboost_sigmoid_model.joblib")
joblib.dump(xgb_sigmoid_model, final_model_file)
print(f"已保存最终XGBoost_Sigmoid模型：{final_model_file}")


# =========================================================
# 5. 最终模型预测结果
# =========================================================
final_prob = xgb_sigmoid_model.predict_proba(X)[:, 1]

pred_df = X.copy()
pred_df[label_name] = y.values
pred_df["final_xgboost_sigmoid_prob"] = final_prob

pred_file = os.path.join(OUTPUT_DIR, "final_model_predictions.xlsx")
pred_df.to_excel(pred_file, index=False)


# =========================================================
# 6. 计算 SHAP
# 说明：SHAP解释的是底层XGBoost，不是外层Sigmoid校准器
# =========================================================
if len(X) > MAX_SHAP_SAMPLES:
    X_shap = X.sample(n=MAX_SHAP_SAMPLES, random_state=RANDOM_STATE).copy()
else:
    X_shap = X.copy()

explainer = shap.TreeExplainer(base_xgb_for_shap)
shap_values = explainer.shap_values(X_shap)

# 兼容不同shap / xgboost版本
if isinstance(shap_values, list):
    shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

# 平均绝对SHAP值
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance_df = pd.DataFrame({
    "feature": X_shap.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

shap_importance_file = os.path.join(OUTPUT_DIR, "final_model_shap_importance.xlsx")
shap_importance_df.to_excel(shap_importance_file, index=False)


# =========================================================
# 7. SHAP summary plot
# =========================================================
plt.figure()
shap.summary_plot(shap_values, X_shap, max_display=X_shap.shape[1], show=False)
plt.tight_layout()
summary_png = os.path.join(OUTPUT_DIR, "final_model_shap_summary.png")
plt.savefig(summary_png, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 8. SHAP bar plot
# =========================================================
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", max_display=X_shap.shape[1], show=False)
plt.tight_layout()
bar_png = os.path.join(OUTPUT_DIR, "final_model_shap_bar.png")
plt.savefig(bar_png, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 9. 保存 SHAP 明细值
# =========================================================
shap_value_df = pd.DataFrame(shap_values, columns=X_shap.columns, index=X_shap.index)
shap_value_file = os.path.join(OUTPUT_DIR, "final_model_shap_values.xlsx")
shap_value_df.to_excel(shap_value_file, index=True)


# =========================================================
# 10. 汇总信息
# =========================================================
summary_df = pd.DataFrame([{
    "n_samples": len(X),
    "n_features": X.shape[1],
    "label_col": label_name,
    "shap_sample_size": len(X_shap),
    "base_model_file": base_model_file,
    "final_model_file": final_model_file,
    "note": "数据已完整，无需缺失值插补；SHAP解释的是底层XGBoost"
}])

summary_file = os.path.join(OUTPUT_DIR, "final_model_shap_summary_info.xlsx")
summary_df.to_excel(summary_file, index=False)

print("\n完成。")
print(f"最终预测结果: {pred_file}")
print(f"SHAP重要性表: {shap_importance_file}")
print(f"SHAP summary图: {summary_png}")
print(f"SHAP bar图: {bar_png}")
print(f"SHAP明细值: {shap_value_file}")
print(f"汇总说明: {summary_file}")
