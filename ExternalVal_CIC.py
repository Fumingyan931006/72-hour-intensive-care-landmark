import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# 1. 参数设置
# =========================================================
INPUT_FILE = "ExternalVal_xgboost_sigmoid_outputs/eicu_external_predictions.xlsx"   # 外部验证预测结果文件
SHEET_NAME = 0

# 真实标签列名
Y_TRUE_COL = "label"

# XGBoost_Sigmoid 外部验证预测概率列名
Y_PROB_COL = "pred_prob_xgboost_sigmoid"

OUTPUT_DIR = "ExternalVal_cic_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CIC 阈值范围
THRESHOLDS = np.arange(0.05, 0.51, 0.01)

# 标准化到每1000人
POPULATION_SIZE = 1000


# =========================================================
# 2. 读取数据
# =========================================================
df = pd.read_excel(INPUT_FILE, sheet_name=SHEET_NAME)

required_cols = [Y_TRUE_COL, Y_PROB_COL]
missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"缺少必要列：{missing_cols}")

df = df[required_cols].copy()
df = df.dropna()

y_true = pd.to_numeric(df[Y_TRUE_COL], errors="coerce")
y_prob = pd.to_numeric(df[Y_PROB_COL], errors="coerce")

valid_mask = y_true.notna() & y_prob.notna()
y_true = y_true.loc[valid_mask].astype(int).values
y_prob = y_prob.loc[valid_mask].astype(float).values

# 限制概率范围到 [0,1]
y_prob = np.clip(y_prob, 0, 1)

# 检查标签
if set(np.unique(y_true)) - {0, 1}:
    raise ValueError("真实标签必须是二分类 0/1。")

N = len(y_true)
print("样本数：", N)


# =========================================================
# 3. 计算 CIC 数据
# =========================================================
rows = []

for pt in THRESHOLDS:
    y_pred_high_risk = (y_prob >= pt).astype(int)

    # 被判定为高风险的人数
    n_high_risk = np.sum(y_pred_high_risk == 1)

    # 高风险中真正发生事件的人数（真阳性）
    n_high_risk_with_event = np.sum((y_pred_high_risk == 1) & (y_true == 1))

    # 标准化为每1000人
    n_high_risk_per_1000 = n_high_risk / N * POPULATION_SIZE
    n_event_among_high_risk_per_1000 = n_high_risk_with_event / N * POPULATION_SIZE

    rows.append({
        "threshold_probability": pt,
        "n_high_risk": n_high_risk,
        "n_high_risk_with_event": n_high_risk_with_event,
        "n_high_risk_per_1000": n_high_risk_per_1000,
        "n_high_risk_with_event_per_1000": n_event_among_high_risk_per_1000
    })

cic_df = pd.DataFrame(rows)

# 保存数据
cic_data_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_external_cic_data.xlsx")
cic_df.to_excel(cic_data_file, index=False)


# =========================================================
# 4. 绘制 CIC 图
# =========================================================
plt.figure(figsize=(8, 6))

plt.plot(
    cic_df["threshold_probability"],
    cic_df["n_high_risk_per_1000"],
    linewidth=2.5,
    label="Number high risk"
)

plt.plot(
    cic_df["threshold_probability"],
    cic_df["n_high_risk_with_event_per_1000"],
    linewidth=2.5,
    label="Number high risk with event"
)

plt.xlabel("Threshold probability")
plt.ylabel("Number of people per 1000")
plt.title("Clinical Impact Curve on External Validation Set")
plt.grid(alpha=0.3, linestyle="--")
plt.legend(loc="best")
plt.tight_layout()

cic_fig_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_external_cic_curve.png")
plt.savefig(cic_fig_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 5. 汇总说明
# =========================================================
summary_df = pd.DataFrame([{
    "n_samples": N,
    "population_standardized_to": POPULATION_SIZE,
    "threshold_min": THRESHOLDS.min(),
    "threshold_max": THRESHOLDS.max(),
    "prediction_file": INPUT_FILE,
    "cic_data_file": cic_data_file,
    "cic_figure_file": cic_fig_file
}])

summary_file = os.path.join(OUTPUT_DIR, "xgboost_sigmoid_external_cic_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

print("完成。")
print(f"CIC 数据：{cic_data_file}")
print(f"CIC 图：{cic_fig_file}")
print(f"汇总表：{summary_file}")
