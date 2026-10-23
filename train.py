import os
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings("ignore")

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


INPUT_DIR = "lasso_outputs"
OUTPUT_DIR = "model_train"
N_FOLDS = 5

LABEL_COL = None
ID_COLS: List[str] = [""]
THRESHOLD = 0.5
RANDOM_STATE = 42


def get_label_col(df: pd.DataFrame, configured_label_col=None) -> str:
    return df.columns[-1] if configured_label_col is None else configured_label_col


def prepare_xy(df: pd.DataFrame, label_col: str, id_cols: List[str]) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    feature_cols = [c for c in df.columns if c not in id_cols + [label_col]]
    X = df[feature_cols].copy()
    y = df[label_col].copy()

    non_numeric_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        raise ValueError(f"Found non-numeric columns. Encode or remove first: {non_numeric_cols}")

    unique_vals = pd.Series(y).dropna().unique().tolist()
    if set(unique_vals) == {0, 1} or set(unique_vals) == {0} or set(unique_vals) == {1}:
        y = y.astype(int)
    else:
        if len(unique_vals) == 2:
            mapping = {sorted(unique_vals)[0]: 0, sorted(unique_vals)[1]: 1}
            y = y.map(mapping).astype(int)
        else:
            raise ValueError(f"Label column must be binary. Got unique values: {unique_vals}")

    return X, y, feature_cols


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_div(a, b):
        return float(a / b) if b != 0 else np.nan

    return {
        "Accuracy": safe_div(tp + tn, tp + tn + fp + fn),
        "AUC": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else np.nan,
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


def build_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {}

    if HAS_XGB:
        xgb_base = XGBClassifier(
            n_estimators=120, #默认300
            max_depth=4,
            learning_rate=0.05, #默认0.05
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0, #默认0.0
            reg_lambda=1.0, #默认1.0
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
        models["XGBoost"] = xgb_base

        xgb_sigmoid = CalibratedClassifierCV(
            estimator=XGBClassifier(
                n_estimators=120, #默认300
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
            cv=5,
        )
        models["XGBoost_Sigmoid"] = xgb_sigmoid

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
        n_estimators=100, # 原本为100
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
            penalty="l1", #默认l2
            C=0.01, # 默认1.0
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
    raise ValueError("Model has neither predict_proba nor decision_function.")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    models = build_models(RANDOM_STATE)

    if not HAS_XGB:
        print("Warning: xgboost is not installed. XGBoost models will be skipped.")
    if not HAS_LGBM:
        print("Warning: lightgbm is not installed. LightGBM will be skipped.")

    all_rows = []

    for fold_id in range(1, N_FOLDS + 1):
        print(f"\n========== Fold {fold_id} ==========")

        train_file = os.path.join(INPUT_DIR, f"fold{fold_id}", f"fold{fold_id}_train_lasso_selected.xlsx")
        test_file = os.path.join(INPUT_DIR, f"fold{fold_id}", f"fold{fold_id}_test_lasso_selected.xlsx")

        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Missing: {train_file}")
        if not os.path.exists(test_file):
            raise FileNotFoundError(f"Missing: {test_file}")

        train_df = pd.read_excel(train_file)
        test_df = pd.read_excel(test_file)

        current_label_col = get_label_col(train_df, LABEL_COL)

        X_train, y_train, feature_cols = prepare_xy(train_df, current_label_col, ID_COLS)
        X_test, y_test, _ = prepare_xy(test_df, current_label_col, ID_COLS)

        fold_dir = os.path.join(OUTPUT_DIR, f"fold{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        fold_metrics = []

        for model_name, model in models.items():
            print(f"  Training {model_name} ...")
            try:
                y_prob = fit_and_predict_proba(model, X_train, y_train, X_test)
                metrics = calc_metrics(y_test.values, y_prob, threshold=THRESHOLD)
                metrics["fold"] = fold_id
                metrics["model"] = model_name
                metrics["n_features"] = X_train.shape[1]
                metrics["n_train"] = len(X_train)
                metrics["n_test"] = len(X_test)

                pred_df = pd.DataFrame({
                    "y_true": y_test.values,
                    "y_prob": y_prob,
                    "y_pred": (y_prob >= THRESHOLD).astype(int)
                })
                pred_file = os.path.join(fold_dir, f"fold{fold_id}_{model_name}_predictions.xlsx")
                pred_df.to_excel(pred_file, index=False)
                metrics["prediction_file"] = pred_file
                metrics["error"] = ""

                fold_metrics.append(metrics)
                all_rows.append(metrics)

            except Exception as e:
                err_row = {
                    "fold": fold_id,
                    "model": model_name,
                    "n_features": X_train.shape[1],
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "Accuracy": np.nan,
                    "AUC": np.nan,
                    "Sensitivity": np.nan,
                    "Specificity": np.nan,
                    "PPV": np.nan,
                    "NPV": np.nan,
                    "TP": np.nan,
                    "TN": np.nan,
                    "FP": np.nan,
                    "FN": np.nan,
                    "threshold": THRESHOLD,
                    "prediction_file": "",
                    "error": str(e)
                }
                fold_metrics.append(err_row)
                all_rows.append(err_row)
                print(f"    Failed: {e}")

        fold_metrics_df = pd.DataFrame(fold_metrics)
        fold_metrics_file = os.path.join(fold_dir, f"fold{fold_id}_all_model_metrics.xlsx")
        fold_metrics_df.to_excel(fold_metrics_file, index=False)
        print(f"  Saved fold metrics: {fold_metrics_file}")

    all_df = pd.DataFrame(all_rows)
    all_file = os.path.join(OUTPUT_DIR, "all_folds_all_models_metrics.xlsx")
    all_df.to_excel(all_file, index=False)

    metric_cols = ["Accuracy", "AUC", "Sensitivity", "Specificity", "PPV", "NPV"]
    summary_mean = all_df.groupby("model")[metric_cols].mean().reset_index()
    summary_std = all_df.groupby("model")[metric_cols].std().reset_index()
    summary_mean.columns = ["model"] + [f"{c}_mean" for c in metric_cols]
    summary_std.columns = ["model"] + [f"{c}_std" for c in metric_cols]

    summary_df = summary_mean.merge(summary_std, on="model", how="left")
    summary_df["n_folds"] = all_df.groupby("model")["fold"].nunique().values

    summary_file = os.path.join(OUTPUT_DIR, "fivefold_performance_summary.xlsx")
    summary_df.to_excel(summary_file, index=False)

    print("\nDone.")
    print(f"All metrics saved to: {os.path.abspath(all_file)}")
    print(f"Summary saved to: {os.path.abspath(summary_file)}")


if __name__ == "__main__":
    main()
