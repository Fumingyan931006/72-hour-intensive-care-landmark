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

# XGBoost_Sigmoid 预测概率列名
Y_PROB_COL = "pred_prob_xgboost_sigmoid"

OUTPUT_DIR = "ExternalVal_risk_stratification_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 风险分层方式：
# "quantile" = 按分位数分层
# "custom"   = 按自定义阈值分层
STRATIFY_METHOD = "quantile"

# 分位数分层：3组（低/中/高）
N_GROUPS = 3

# 如果使用自定义阈值，例如 [0, 0.2, 0.5, 1.0]
CUSTOM_BINS = [0.0, 0.2, 0.5, 1.0]
CUSTOM_LABELS = ["Low risk", "Medium risk", "High risk"]


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
df = df.loc[valid_mask].copy()

df[Y_TRUE_COL] = df[Y_TRUE_COL].astype(int)
df[Y_PROB_COL] = df[Y_PROB_COL].astype(float)

# 限制概率范围
df[Y_PROB_COL] = np.clip(df[Y_PROB_COL], 0, 1)

if set(df[Y_TRUE_COL].unique()) - {0, 1}:
    raise ValueError("真实标签必须是二分类 0/1。")


# =========================================================
# 3. 风险分层
# =========================================================
if STRATIFY_METHOD == "quantile":
    # 按分位数分成低/中/高风险组
    if N_GROUPS == 3:
        labels = ["Low risk", "Medium risk", "High risk"]
    elif N_GROUPS == 4:
        labels = ["Q1", "Q2", "Q3", "Q4"]
    else:
        labels = [f"Group {i+1}" for i in range(N_GROUPS)]

    df["risk_group"] = pd.qcut(
        df[Y_PROB_COL],
        q=N_GROUPS,
        labels=labels,
        duplicates="drop"
    )

elif STRATIFY_METHOD == "custom":
    df["risk_group"] = pd.cut(
        df[Y_PROB_COL],
        bins=CUSTOM_BINS,
        labels=CUSTOM_LABELS,
        include_lowest=True,
        right=True
    )
else:
    raise ValueError("STRATIFY_METHOD 只能是 'quantile' 或 'custom'")


# =========================================================
# 4. 汇总统计
# =========================================================
summary_df = df.groupby("risk_group", observed=False).agg(
    n_samples=(Y_TRUE_COL, "count"),
    n_events=(Y_TRUE_COL, "sum"),
    event_rate=(Y_TRUE_COL, "mean"),
    mean_pred_prob=(Y_PROB_COL, "mean"),
    median_pred_prob=(Y_PROB_COL, "median")
).reset_index()

summary_df["event_rate_percent"] = summary_df["event_rate"] * 100
summary_df["mean_pred_prob_percent"] = summary_df["mean_pred_prob"] * 100

# 计算 RR 和 OR（以最低风险组为参照）
summary_df["RR_vs_lowest"] = np.nan
summary_df["OR_vs_lowest"] = np.nan

if len(summary_df) >= 2:
    ref_events = summary_df.loc[0, "n_events"]
    ref_total = summary_df.loc[0, "n_samples"]
    ref_nonevents = ref_total - ref_events
    ref_risk = ref_events / ref_total if ref_total > 0 else np.nan

    for i in range(len(summary_df)):
        events = summary_df.loc[i, "n_events"]
        total = summary_df.loc[i, "n_samples"]
        nonevents = total - events
        risk = events / total if total > 0 else np.nan

        # RR
        if pd.notna(ref_risk) and ref_risk > 0:
            summary_df.loc[i, "RR_vs_lowest"] = risk / ref_risk

        # OR
        if ref_events > 0 and ref_nonevents > 0 and events > 0 and nonevents > 0:
            odds = events / nonevents
            ref_odds = ref_events / ref_nonevents
            summary_df.loc[i, "OR_vs_lowest"] = odds / ref_odds

# 保存汇总表
summary_file = os.path.join(OUTPUT_DIR, "risk_stratification_summary.xlsx")
summary_df.to_excel(summary_file, index=False)


# =========================================================
# 5. 绘制风险分层柱状图：各组事件发生率
# =========================================================
plt.figure(figsize=(8, 6))
bars = plt.bar(summary_df["risk_group"].astype(str), summary_df["event_rate_percent"])

for bar, val in zip(bars, summary_df["event_rate_percent"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.1f}%",
        ha="center",
        va="bottom",
        fontsize=10
    )

plt.xlabel("Risk group")
plt.ylabel("Observed event rate (%)")
plt.title("Risk Stratification by XGBoost_Sigmoid on External Validation Set")
plt.tight_layout()

bar_fig_file = os.path.join(OUTPUT_DIR, "risk_stratification_barplot.png")
plt.savefig(bar_fig_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 6. 绘制风险分层对比图：平均预测概率 vs 实际事件率
# =========================================================
plt.figure(figsize=(8, 6))

x = np.arange(len(summary_df))

plt.plot(
    x,
    summary_df["mean_pred_prob_percent"],
    marker="o",
    linewidth=2,
    label="Mean predicted probability (%)"
)

plt.plot(
    x,
    summary_df["event_rate_percent"],
    marker="s",
    linewidth=2,
    label="Observed event rate (%)"
)

plt.xticks(x, summary_df["risk_group"].astype(str))
plt.xlabel("Risk group")
plt.ylabel("Percentage (%)")
plt.title("Predicted Risk vs Observed Event Rate")
plt.legend(loc="best")
plt.tight_layout()

line_fig_file = os.path.join(OUTPUT_DIR, "risk_stratification_lineplot.png")
plt.savefig(line_fig_file, dpi=300, bbox_inches="tight")
plt.close()


# =========================================================
# 7. 保存逐样本风险分层结果
# =========================================================
detail_file = os.path.join(OUTPUT_DIR, "risk_stratification_detail.xlsx")
df.to_excel(detail_file, index=False)


print("完成。")
print(f"风险分层汇总表：{summary_file}")
print(f"风险分层柱状图：{bar_fig_file}")
print(f"风险分层折线图：{line_fig_file}")
print(f"逐样本风险分层结果：{detail_file}")
