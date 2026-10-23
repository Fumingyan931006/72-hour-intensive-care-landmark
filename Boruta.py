import os
import warnings
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from boruta import BorutaPy
except ImportError as e:
    raise ImportError(
        "未安装 Boruta。请先运行: pip install Boruta"
    ) from e

# =========================
# 配置区
# =========================
INPUT_DIR = "folder"   # 保存5折填补后 train/test 数据的文件夹
OUTPUT_DIR = "boruta_outputs" # Boruta输出目录
LABEL_COL: Optional[str] = None  # 若为None，则默认最后一列为标签
DROP_COLS: List[str] = ['']        # 不参与Boruta的列，例如 patient_id
RANDOM_STATE = 42
MAX_ITER = 100
PERC = 90 # 默认100
ALPHA = 0.05
N_JOBS = -1


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def infer_label_col(df: pd.DataFrame, label_col: Optional[str]) -> str:
    if label_col is not None:
        if label_col not in df.columns:
            raise ValueError(f"指定的标签列不存在: {label_col}")
        return label_col
    return str(df.columns[-1])


def find_fold_pairs(input_dir: str) -> List[Tuple[int, str, str]]:
    pairs = []
    for i in range(1, 6):
        train_path = os.path.join(input_dir, f"fold{i}_train_imputed.xlsx")
        test_path = os.path.join(input_dir, f"fold{i}_test_imputed.xlsx")
        if os.path.exists(train_path) and os.path.exists(test_path):
            pairs.append((i, train_path, test_path))
    if not pairs:
        raise FileNotFoundError(
            f"未在 {input_dir} 中找到 fold1~fold5 的 train/test 文件。"
        )
    return pairs


def encode_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """仅用训练集生成哑变量列，并将测试集对齐到训练集列。"""
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()

    # 统一 bool
    for col in feature_cols:
        if X_train_raw[col].dtype == bool:
            X_train_raw[col] = X_train_raw[col].astype(int)
            X_test_raw[col] = X_test_raw[col].astype(int)

    # 数值列与非数值列
    numeric_cols = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    # 数值列直接保留
    X_train_num = X_train_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_train_raw.index)
    X_test_num = X_test_raw[numeric_cols].copy() if numeric_cols else pd.DataFrame(index=X_test_raw.index)

    # 分类列只根据训练集做dummy，再让测试集对齐
    if categorical_cols:
        X_train_cat = X_train_raw[categorical_cols].fillna("MISSING_CATEGORY").astype(str)
        X_test_cat = X_test_raw[categorical_cols].fillna("MISSING_CATEGORY").astype(str)

        X_train_cat = pd.get_dummies(X_train_cat, columns=categorical_cols, drop_first=False, dtype=float)
        X_test_cat = pd.get_dummies(X_test_cat, columns=categorical_cols, drop_first=False, dtype=float)
        X_test_cat = X_test_cat.reindex(columns=X_train_cat.columns, fill_value=0.0)
    else:
        X_train_cat = pd.DataFrame(index=X_train_raw.index)
        X_test_cat = pd.DataFrame(index=X_test_raw.index)

    X_train = pd.concat([X_train_num, X_train_cat], axis=1)
    X_test = pd.concat([X_test_num, X_test_cat], axis=1)

    # 防御性：全部转数值
    X_train = X_train.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X_test = X_test.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    return X_train, X_test


def encode_labels(y_train: pd.Series, y_test: pd.Series) -> Tuple[np.ndarray, np.ndarray, Optional[LabelEncoder]]:
    if pd.api.types.is_numeric_dtype(y_train):
        return y_train.to_numpy(), y_test.to_numpy(), None
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train.astype(str))
    # 测试集若含训练集中未见标签，会报错；二分类正常不会有这个问题
    y_test_enc = le.transform(y_test.astype(str))
    return y_train_enc, y_test_enc, le


def plot_importance(
    result_df: pd.DataFrame,
    save_path: str,
    top_n: Optional[int] = 30,
    title: str = "Boruta Selected Feature Importance"
) -> None:
    """仅绘制被 Boruta 正式保留(selected=True)的特征。

    若 Boruta 未正式保留任何特征，则使用保底列 `selected_fallback_top_rf=True`
    的特征作图，避免输出空图。
    """
    plot_df = result_df.loc[result_df["selected"]].copy()

    # 如果正式入选特征为空，则回退到保底特征
    if plot_df.empty and "selected_fallback_top_rf" in result_df.columns:
        plot_df = result_df.loc[result_df["selected_fallback_top_rf"]].copy()

    # 仍为空则不作图
    if plot_df.empty:
        plt.figure(figsize=(8, 4))
        plt.text(0.5, 0.5, "No selected features", ha="center", va="center", fontsize=12)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    plot_df = plot_df.sort_values("rf_importance", ascending=False).copy()
    if top_n is not None and len(plot_df) > top_n:
        plot_df = plot_df.head(top_n)

    colors = ["tab:red"] * len(plot_df)

    plt.figure(figsize=(10, max(6, 0.35 * len(plot_df))))
    plt.barh(plot_df["feature"], plot_df["rf_importance"], color=colors)
    plt.gca().invert_yaxis()
    plt.xlabel("Random Forest Importance")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_boruta_on_fold(fold_id: int, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fold_dir = os.path.join(OUTPUT_DIR, f"fold{fold_id}")
    ensure_dir(fold_dir)

    train_df = pd.read_excel(train_path)
    test_df = pd.read_excel(test_path)
    train_df.columns = train_df.columns.map(str).str.strip()
    test_df.columns = test_df.columns.map(str).str.strip()

    label_col = infer_label_col(train_df, LABEL_COL)
    if label_col not in test_df.columns:
        raise ValueError(f"fold{fold_id} 的测试集中未找到标签列: {label_col}")

    feature_cols = [c for c in train_df.columns if c not in set(DROP_COLS + [label_col])]
    if not feature_cols:
        raise ValueError(f"fold{fold_id} 没有可用于Boruta的特征列。")

    X_train, X_test = encode_features(train_df, test_df, feature_cols)
    y_train, y_test, _ = encode_labels(train_df[label_col], test_df[label_col])

    rf = RandomForestClassifier(
        n_estimators=1000,
        criterion="gini",
        max_depth=None,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        perc=PERC,
        alpha=ALPHA,
        two_step=True,
        max_iter=MAX_ITER,
        random_state=RANDOM_STATE,
        verbose=2,
    )

    boruta.fit(X_train.to_numpy(), y_train)

    # Boruta结果
    support = boruta.support_
    support_weak = boruta.support_weak_
    ranking = boruta.ranking_

    # 再拟合一个RF，用于画直观的重要性图
    rf.fit(X_train, y_train)
    rf_importance = rf.feature_importances_

    result_df = pd.DataFrame({
        "feature": X_train.columns,
        "selected": support,
        "tentative": support_weak,
        "rank": ranking,
        "rf_importance": rf_importance,
    }).sort_values(["selected", "rank", "rf_importance"], ascending=[False, True, False])

    selected_features = result_df.loc[result_df["selected"], "feature"].tolist()
    if len(selected_features) == 0:
        # 防止Boruta筛空：保底用RF重要性前10个
        selected_features = result_df.sort_values("rf_importance", ascending=False).head(min(10, len(result_df)))["feature"].tolist()
        result_df["selected_fallback_top_rf"] = result_df["feature"].isin(selected_features)
    else:
        result_df["selected_fallback_top_rf"] = False

    # 保存筛选结果
    result_path = os.path.join(fold_dir, f"fold{fold_id}_boruta_results.xlsx")
    result_df.to_excel(result_path, index=False)

    # 保存选中特征列表
    selected_df = pd.DataFrame({"selected_feature": selected_features})
    selected_path = os.path.join(fold_dir, f"fold{fold_id}_selected_features.xlsx")
    selected_df.to_excel(selected_path, index=False)

    # 保存筛选后的训练集和测试集
    train_selected = pd.concat([X_train[selected_features].copy(), train_df[[label_col]].reset_index(drop=True)], axis=1)
    test_selected = pd.concat([X_test[selected_features].copy(), test_df[[label_col]].reset_index(drop=True)], axis=1)
    train_selected.to_excel(os.path.join(fold_dir, f"fold{fold_id}_train_boruta_selected.xlsx"), index=False)
    test_selected.to_excel(os.path.join(fold_dir, f"fold{fold_id}_test_boruta_selected.xlsx"), index=False)

    # 保存重要性图
    plot_importance(
        result_df=result_df,
        save_path=os.path.join(fold_dir, f"fold{fold_id}_boruta_importance.png"),
        top_n=30,
        title=f"Fold {fold_id} Selected Features by Boruta"
    )

    # 汇总
    fold_summary = pd.DataFrame({
        "fold": [fold_id],
        "n_train": [len(train_df)],
        "n_test": [len(test_df)],
        "n_features_before_encoding": [len(feature_cols)],
        "n_features_after_encoding": [X_train.shape[1]],
        "n_selected_features": [len(selected_features)],
        "label_col": [label_col],
        "train_file": [train_path],
        "test_file": [test_path],
        "result_file": [result_path],
    })

    return result_df, fold_summary


def main() -> None:
    ensure_dir(OUTPUT_DIR)
    fold_pairs = find_fold_pairs(INPUT_DIR)

    all_results = []
    all_summaries = []

    for fold_id, train_path, test_path in fold_pairs:
        print(f"\n{'=' * 20} Fold {fold_id} {'=' * 20}")
        result_df, summary_df = run_boruta_on_fold(fold_id, train_path, test_path)
        result_df.insert(0, "fold", fold_id)
        all_results.append(result_df)
        all_summaries.append(summary_df)

    combined_results = pd.concat(all_results, axis=0, ignore_index=True)
    combined_summary = pd.concat(all_summaries, axis=0, ignore_index=True)

    combined_results.to_excel(os.path.join(OUTPUT_DIR, "boruta_all_folds_results.xlsx"), index=False)
    combined_summary.to_excel(os.path.join(OUTPUT_DIR, "boruta_all_folds_summary.xlsx"), index=False)

    # 特征在5折中的入选频次
    freq_df = (
        combined_results.groupby("feature", as_index=False)["selected"]
        .sum()
        .rename(columns={"selected": "selected_count"})
        .sort_values(["selected_count", "feature"], ascending=[False, True])
    )
    freq_df["selected_ratio"] = freq_df["selected_count"] / len(fold_pairs)
    freq_df.to_excel(os.path.join(OUTPUT_DIR, "boruta_feature_selection_frequency.xlsx"), index=False)

    print("\nBoruta粗筛完成。输出目录：", os.path.abspath(OUTPUT_DIR))


if __name__ == "__main__":
    main()
