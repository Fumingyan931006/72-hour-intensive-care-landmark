import os
import pandas as pd

# =========================
# 1. 路径参数
# =========================
# 填补后的5折数据所在文件夹
imputed_dir = "folder"

# Boruta结果所在文件夹
boruta_dir = "boruta_outputs"

# 输出文件夹
output_dir = "boruta_selected_folds"
os.makedirs(output_dir, exist_ok=True)

# 交叉验证折数
n_folds = 5

# =========================
# 2. 可选参数
# =========================
# 如果有病例编号等需要保留，但不参与LASSO，可写在这里
id_cols = [""]

# 标签列：
# 如果为 None，则默认“最后一列是标签”
label_col = None

# Boruta结果表中的特征列名和筛选列名
feature_col_in_boruta = "feature"
selected_col_in_boruta = "selected"

# 如果某一折 Boruta 一个特征都没选出来：
# True = 使用保底列 selected_fallback_top_rf
# False = 报错停止
use_fallback_if_empty = True
fallback_col = "selected_fallback_top_rf"


# =========================
# 3. 主循环
# =========================
summary_rows = []

for fold_id in range(1, n_folds + 1):
    print(f"\n========== 处理 Fold {fold_id} ==========")

    # ---- 3.1 文件路径 ----
    train_file = os.path.join(imputed_dir, f"fold{fold_id}_train_imputed.xlsx")
    test_file = os.path.join(imputed_dir, f"fold{fold_id}_test_imputed.xlsx")
    boruta_file = os.path.join(boruta_dir, f"fold{fold_id}", f"fold{fold_id}_boruta_results.xlsx")

    if not os.path.exists(train_file):
        raise FileNotFoundError(f"未找到训练集文件: {train_file}")
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"未找到测试集文件: {test_file}")
    if not os.path.exists(boruta_file):
        raise FileNotFoundError(f"未找到Boruta结果文件: {boruta_file}")

    # ---- 3.2 读取数据 ----
    train_df = pd.read_excel(train_file)
    test_df = pd.read_excel(test_file)
    boruta_df = pd.read_excel(boruta_file)

    # ---- 3.3 标签列 ----
    if label_col is None:
        current_label_col = train_df.columns[-1]
    else:
        current_label_col = label_col

    if current_label_col not in train_df.columns:
        raise ValueError(f"Fold {fold_id} 训练集缺少标签列: {current_label_col}")
    if current_label_col not in test_df.columns:
        raise ValueError(f"Fold {fold_id} 测试集缺少标签列: {current_label_col}")

    # ---- 3.4 提取 Boruta 选中特征 ----
    if feature_col_in_boruta not in boruta_df.columns:
        raise ValueError(f"Fold {fold_id} Boruta结果缺少列: {feature_col_in_boruta}")
    if selected_col_in_boruta not in boruta_df.columns:
        raise ValueError(f"Fold {fold_id} Boruta结果缺少列: {selected_col_in_boruta}")

    selected_features = boruta_df.loc[
        boruta_df[selected_col_in_boruta] == True,
        feature_col_in_boruta
    ].astype(str).tolist()

    # 如果正式入选为空，是否启用保底特征
    used_fallback = False
    if len(selected_features) == 0:
        if use_fallback_if_empty and fallback_col in boruta_df.columns:
            selected_features = boruta_df.loc[
                boruta_df[fallback_col] == True,
                feature_col_in_boruta
            ].astype(str).tolist()
            used_fallback = True

        if len(selected_features) == 0:
            raise ValueError(f"Fold {fold_id} Boruta未选出任何特征，且保底特征也为空。")

    # ---- 3.5 检查选中特征是否存在于训练/测试集 ----
    missing_in_train = [f for f in selected_features if f not in train_df.columns]
    missing_in_test = [f for f in selected_features if f not in test_df.columns]

    if missing_in_train:
        raise ValueError(f"Fold {fold_id} 下列Boruta特征不在训练集中: {missing_in_train}")
    if missing_in_test:
        raise ValueError(f"Fold {fold_id} 下列Boruta特征不在测试集中: {missing_in_test}")

    # ---- 3.6 保留列 ----
    keep_cols = []

    # 先保留ID列（如果有）
    for c in id_cols:
        if c in train_df.columns and c in test_df.columns:
            keep_cols.append(c)

    # 再保留Boruta筛中的特征
    keep_cols.extend(selected_features)

    # 最后保留标签列
    keep_cols.append(current_label_col)

    # 去重，防止标签列或ID列和特征重复
    keep_cols = list(dict.fromkeys(keep_cols))

    # ---- 3.7 生成筛选后的训练/测试集 ----
    train_selected = train_df[keep_cols].copy()
    test_selected = test_df[keep_cols].copy()

    # ---- 3.8 保存输出 ----
    train_outfile = os.path.join(output_dir, f"fold{fold_id}_train_boruta_selected.xlsx")
    test_outfile = os.path.join(output_dir, f"fold{fold_id}_test_boruta_selected.xlsx")
    feature_outfile = os.path.join(output_dir, f"fold{fold_id}_selected_features.xlsx")

    train_selected.to_excel(train_outfile, index=False)
    test_selected.to_excel(test_outfile, index=False)

    pd.DataFrame({
        "selected_features": selected_features
    }).to_excel(feature_outfile, index=False)

    # ---- 3.9 汇总信息 ----
    summary_rows.append({
        "fold": fold_id,
        "n_train_samples": len(train_selected),
        "n_test_samples": len(test_selected),
        "n_selected_features": len(selected_features),
        "used_fallback": used_fallback,
        "label_col": current_label_col,
        "train_file": train_outfile,
        "test_file": test_outfile,
        "feature_file": feature_outfile
    })

    print(f"Fold {fold_id} 完成")
    print(f"  选中特征数: {len(selected_features)}")
    print(f"  训练集输出: {train_outfile}")
    print(f"  测试集输出: {test_outfile}")

# =========================
# 4. 保存汇总表
# =========================
summary_df = pd.DataFrame(summary_rows)
summary_file = os.path.join(output_dir, "boruta_selected_folds_summary.xlsx")
summary_df.to_excel(summary_file, index=False)

print("\n全部完成。")
print(f"汇总表已保存: {summary_file}")
