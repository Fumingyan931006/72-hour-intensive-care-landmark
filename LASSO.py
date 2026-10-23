import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score


# =========================================================
# 1. 参数设置
# =========================================================
# Boruta筛选后的5折数据目录
input_dir = "boruta_selected_folds"

# 输出目录
output_dir = "lasso_outputs"
os.makedirs(output_dir, exist_ok=True)

# 折数
n_folds = 5

# 标签列：
# 如果为 None，则默认最后一列为标签
label_col = None

# 如果有ID列想保留在输出文件中，但不参与LASSO
id_cols = [""]   # 例如 ["patient_id"]

# Logistic LASSO 的 C 搜索范围（越大正则越弱）
Cs = np.logspace(-4, -1, 40) #默认(-3,2,50)

# 内部CV折数（仅在训练折内部）
inner_cv = 5

# 评分方式
scoring = "roc_auc"

# 随机种子
random_state = 42


# =========================================================
# 2. 工具函数
# =========================================================
def get_label_col(df, configured_label_col=None):
    if configured_label_col is None:
        return df.columns[-1]
    return configured_label_col


def prepare_X_y(df, label_col, id_cols=None):
    if id_cols is None:
        id_cols = []

    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    # 检查是否全为数值型
    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"存在非数值型特征，请先编码或删除：{non_numeric_cols}")

    return X, y, feature_cols


def fit_lasso_logistic_cv(X_train, y_train, Cs, inner_cv=5, scoring="roc_auc", random_state=42):
    """
    用 L1 Logistic RegressionCV 做 LASSO 精筛
    """
    model = LogisticRegressionCV(
        Cs=Cs,
        cv=inner_cv,
        penalty="l1",
        solver="saga",
        scoring=scoring,
        max_iter=10000,
        random_state=random_state,
        n_jobs=-1,
        refit=True
    )
    model.fit(X_train, y_train)
    return model


def plot_lasso_path(
    X_train_scaled,
    y_train,
    feature_names,
    Cs,
    output_png,
    random_state=42,
    max_features_to_label=25
):
    """
    手动拟合一系列 L1 Logistic 模型，绘制系数路径图
    横轴：log10(C)
    纵轴：系数值
    """
    coefs = []

    for c in Cs:
        clf = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=c,
            max_iter=10000,
            random_state=random_state
        )
        clf.fit(X_train_scaled, y_train)
        coefs.append(clf.coef_.ravel())

    coefs = np.array(coefs)  # shape: [n_Cs, n_features]
    logCs = np.log10(Cs)

    plt.figure(figsize=(12, 8))

    # 若特征太多，只给绝对最大系数前 max_features_to_label 个特征加图例
    max_abs_coef = np.max(np.abs(coefs), axis=0)
    top_idx = np.argsort(max_abs_coef)[::-1][:max_features_to_label]
    top_idx_set = set(top_idx.tolist())

    for i, feat in enumerate(feature_names):
        if i in top_idx_set:
            plt.plot(logCs, coefs[:, i], linewidth=1.5, label=feat)
        else:
            plt.plot(logCs, coefs[:, i], linewidth=1.0, alpha=0.5)

    plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
    plt.xlabel("log10(C)")
    plt.ylabel("Coefficient")
    plt.title("LASSO Coefficient Path (L1 Logistic Regression)")
    if len(top_idx) > 0:
        plt.legend(fontsize=8, loc="best", ncol=2)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_lasso_results(
    fold_id,
    feature_names,
    model,
    scaler,
    X_train_df,
    y_train,
    X_test_df,
    y_test,
    train_raw_df,
    test_raw_df,
    label_col,
    id_cols,
    output_dir
):
    """
    保存：
    - LASSO结果表
    - 选中特征
    - 筛选后的训练/测试集
    - 内部性能
    """
    coef = model.coef_.ravel()
    abs_coef = np.abs(coef)

    result_df = pd.DataFrame({
        "feature": feature_names,
        "coef": coef,
        "abs_coef": abs_coef,
        "selected": abs_coef > 1e-8
    }).sort_values(["selected", "abs_coef"], ascending=[False, False]).reset_index(drop=True)

    selected_features = result_df.loc[result_df["selected"], "feature"].tolist()

    # 如果一个都没选中，保底取绝对系数最大的1个
    used_fallback = False
    if len(selected_features) == 0:
        selected_features = result_df.sort_values("abs_coef", ascending=False)["feature"].head(1).tolist()
        used_fallback = True
        result_df["selected_fallback"] = result_df["feature"].isin(selected_features)
    else:
        result_df["selected_fallback"] = False

    # 训练/测试集预测性能
    X_train_scaled = scaler.transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    train_prob = model.predict_proba(X_train_scaled)[:, 1]
    test_prob = model.predict_proba(X_test_scaled)[:, 1]

    try:
        train_auc = roc_auc_score(y_train, train_prob)
    except Exception:
        train_auc = np.nan

    try:
        test_auc = roc_auc_score(y_test, test_prob)
    except Exception:
        test_auc = np.nan

    # 生成筛选后的训练/测试集Excel
    keep_cols_train = []
    keep_cols_test = []

    for c in id_cols:
        if c in train_raw_df.columns:
            keep_cols_train.append(c)
        if c in test_raw_df.columns:
            keep_cols_test.append(c)

    keep_cols_train.extend(selected_features)
    keep_cols_test.extend(selected_features)

    keep_cols_train.append(label_col)
    keep_cols_test.append(label_col)

    # 去重
    keep_cols_train = list(dict.fromkeys(keep_cols_train))
    keep_cols_test = list(dict.fromkeys(keep_cols_test))

    train_selected_df = train_raw_df[keep_cols_train].copy()
    test_selected_df = test_raw_df[keep_cols_test].copy()

    # 输出路径
    fold_dir = os.path.join(output_dir, f"fold{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)

    result_file = os.path.join(fold_dir, f"fold{fold_id}_lasso_results.xlsx")
    selected_file = os.path.join(fold_dir, f"fold{fold_id}_selected_features.xlsx")
    train_selected_file = os.path.join(fold_dir, f"fold{fold_id}_train_lasso_selected.xlsx")
    test_selected_file = os.path.join(fold_dir, f"fold{fold_id}_test_lasso_selected.xlsx")
    perf_file = os.path.join(fold_dir, f"fold{fold_id}_lasso_performance.xlsx")

    result_df.to_excel(result_file, index=False)
    pd.DataFrame({"selected_features": selected_features}).to_excel(selected_file, index=False)
    train_selected_df.to_excel(train_selected_file, index=False)
    test_selected_df.to_excel(test_selected_file, index=False)

    perf_df = pd.DataFrame([{
        "fold": fold_id,
        "best_C": float(model.C_[0]),
        "n_selected_features": len(selected_features),
        "used_fallback": used_fallback,
        "train_auc": train_auc,
        "test_auc": test_auc
    }])
    perf_df.to_excel(perf_file, index=False)

    return {
        "fold": fold_id,
        "best_C": float(model.C_[0]),
        "n_selected_features": len(selected_features),
        "used_fallback": used_fallback,
        "train_auc": train_auc,
        "test_auc": test_auc,
        "result_file": result_file,
        "selected_file": selected_file,
        "train_selected_file": train_selected_file,
        "test_selected_file": test_selected_file,
        "performance_file": perf_file
    }


# =========================================================
# 3. 主流程
# =========================================================
summary_rows = []
selection_frequency = {}

for fold_id in range(1, n_folds + 1):
    print(f"\n========== Fold {fold_id} ==========")

    train_file = os.path.join(input_dir, f"fold{fold_id}_train_boruta_selected.xlsx")
    test_file = os.path.join(input_dir, f"fold{fold_id}_test_boruta_selected.xlsx")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"未找到文件：{train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"未找到文件：{test_file}")

    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)

    current_label_col = get_label_col(train_df, label_col)

    if current_label_col not in train_df.columns:
        raise ValueError(f"Fold {fold_id} 训练集缺少标签列：{current_label_col}")
    if current_label_col not in test_df.columns:
        raise ValueError(f"Fold {fold_id} 测试集缺少标签列：{current_label_col}")

    # 拆分特征和标签
    X_train_df, y_train, feature_names = prepare_X_y(train_df, current_label_col, id_cols=id_cols)
    X_test_df, y_test, _ = prepare_X_y(test_df, current_label_col, id_cols=id_cols)

    # 标准化：只在训练集拟合
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_df)
    X_test_scaled = scaler.transform(X_test_df)

    # LASSO(Logistic L1) 交叉验证拟合
    lasso_model = fit_lasso_logistic_cv(
        X_train_scaled,
        y_train,
        Cs=Cs,
        inner_cv=inner_cv,
        scoring=scoring,
        random_state=random_state
    )

    # 系数路径图
    fold_dir = os.path.join(output_dir, f"fold{fold_id}")
    os.makedirs(fold_dir, exist_ok=True)
    path_png = os.path.join(fold_dir, f"fold{fold_id}_lasso_coef_path.png")

    plot_lasso_path(
        X_train_scaled=X_train_scaled,
        y_train=y_train,
        feature_names=feature_names,
        Cs=Cs,
        output_png=path_png,
        random_state=random_state
    )

    # 保存结果
    fold_summary = save_lasso_results(
        fold_id=fold_id,
        feature_names=feature_names,
        model=lasso_model,
        scaler=scaler,
        X_train_df=X_train_df,
        y_train=y_train,
        X_test_df=X_test_df,
        y_test=y_test,
        train_raw_df=train_df,
        test_raw_df=test_df,
        label_col=current_label_col,
        id_cols=id_cols,
        output_dir=output_dir
    )

    fold_summary["coef_path_png"] = path_png
    summary_rows.append(fold_summary)

    # 统计入选频次
    selected_df = pd.read_excel(fold_summary["selected_file"])
    selected_features = selected_df["selected_features"].dropna().astype(str).tolist()
    for feat in selected_features:
        selection_frequency[feat] = selection_frequency.get(feat, 0) + 1

    print(f"Fold {fold_id} 完成")
    print(f"  最优 C: {fold_summary['best_C']}")
    print(f"  入选特征数: {fold_summary['n_selected_features']}")
    print(f"  训练AUC: {fold_summary['train_auc']:.4f}")
    print(f"  测试AUC: {fold_summary['test_auc']:.4f}")
    print(f"  系数路径图: {path_png}")

# =========================================================
# 4. 汇总输出
# =========================================================
summary_df = pd.DataFrame(summary_rows)
summary_file = os.path.join(output_dir, "lasso_all_folds_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

freq_df = pd.DataFrame({
    "feature": list(selection_frequency.keys()),
    "selected_times": list(selection_frequency.values())
}).sort_values("selected_times", ascending=False).reset_index(drop=True)

freq_file = os.path.join(output_dir, "lasso_feature_selection_frequency.xlsx")
freq_df.to_excel(freq_file, index=False)

print("\n全部完成。")
print(f"汇总表：{summary_file}")
print(f"频次表：{freq_file}")
