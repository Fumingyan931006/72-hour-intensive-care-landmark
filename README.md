# 72-hour intensive care landmark model

Analysis code for the study **"Mortality risk reassessment at 72 hours in patients with sepsis and pre-existing hypertension receiving vasoactive agents: model development and external validation."**

The scripts reproduce the modelling workflow described in the manuscript: five-fold patient-level cross-validation with fold-internal imputation and feature selection (Boruta followed by L1-penalized logistic regression), comparison of machine-learning algorithms, sigmoid calibration of XGBoost, decision-curve and risk-stratification analyses, SHAP interpretation, and external validation of the locked model in eICU-CRD.

## Data availability

No patient-level data are included in this repository. MIMIC-IV (v3.1) and eICU-CRD (v2.0) are available to credentialed users through PhysioNet under the respective data use agreements:

- MIMIC-IV v3.1: https://doi.org/10.13026/kpb9-mt58
- eICU-CRD v2.0: https://doi.org/10.13026/C2WM1R

Cohort definitions, the 72-hour landmark, the six 12-hour windows and the definitions of all candidate predictors are given in the manuscript and its Supplementary Material. The scripts start from analysis tables (Excel) that contain one row per patient, the candidate predictors and the binary outcome (in-hospital death after the landmark, coded 0/1).

## Requirements

Python 3.10 with the packages listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Input files

| File | Content |
| --- | --- |
| `实验数据.xlsx` | MIMIC-IV development cohort (candidate predictors + outcome) for internal cross-validation |
| `SAPSII_Sofa2.xlsx` | Outcome, `sofa_first` and `sapsii` for the clinical-score comparison |
| `MIMIC外部验证实验数据.xlsx` | Full MIMIC-IV development cohort used to build the locked model |
| `EICU外部验证实验数据.xlsx` | eICU-CRD external cohort with the same harmonized predictor names and units |
| `FinalModelData.xlsx` | Development cohort restricted to the final predictor set, for the SHAP analysis |

In each script, the outcome is taken from the last column unless `LABEL_COL` is set. Identifier columns should be listed in the corresponding `ID_COLS`, `DROP_COLS` or `EXCLUDE_COLS` setting so that they are not used as predictors.

## Workflow

### 1. Internal validation (MIMIC-IV, five-fold cross-validation)

```bash
python MissingRate.py            # optional: missingness summary
python 5-fold+MICE.py            # stratified 5-fold split; IterativeImputer fitted within each training fold
python Boruta.py                 # Boruta selection within each training fold
python DataAfterBoruta.py        # extract Boruta-selected predictors for each fold
python LASSO.py                  # L1-penalized logistic regression (inner 5-fold CV, AUC) within each fold
python train.py                  # train and evaluate the candidate algorithms on each fold
python CC-DCA.py                 # out-of-fold predictions of sigmoid-calibrated XGBoost: calibration, ROC, DCA
python ROCComparison.py          # ROC curves of the models versus SOFA and SAPS II
```

### 2. Locked model and external validation (MIMIC-IV → eICU-CRD)

Before running, set `EXCLUDE_COLS` in `ExternalVal_FillEICUbyMIMIC_Mice.py` to the outcome and identifier columns of your tables, so that the outcome is not used by the imputer.

```bash
python ExternalVal_FillEICUbyMIMIC_Mice.py     # fit imputer on MIMIC-IV, apply unchanged to eICU-CRD
python ExternalVal_MIMIC_Boruta.py             # Boruta on the full development cohort
python ExternalVal_MIMIC_Lasso.py              # L1 selection on the full development cohort -> final predictor set
python ExternalVal_XGBoost_Sigmoid_model.py    # fit calibrated XGBoost on MIMIC-IV, predict eICU-CRD without refitting
python ExternalVal_CIC.py                      # clinical impact curve
python ExternalVal_risk_stratification.py      # observed mortality across tertiles of external predicted risk
```

`ExternalVal_CIC.py` and `ExternalVal_risk_stratification.py` read the outcome column named in `Y_TRUE_COL` (default `label`); change it if your outcome column has another name.

### 3. Model interpretation

```bash
python FinalModel-SHAP.py        # SHAP values of the uncalibrated XGBoost refitted on the development cohort
```

## Main settings

| Component | Setting |
| --- | --- |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Imputation | `IterativeImputer` with `BayesianRidge`, `max_iter=10`; fitted on training data only |
| Boruta | random-forest base estimator, `perc=90`, `alpha=0.05`, `max_iter=100` |
| L1 logistic regression | `solver="saga"`, penalty chosen by inner five-fold CV maximizing ROC AUC |
| XGBoost (selected model) | `n_estimators=120`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9` |
| Calibration | `CalibratedClassifierCV(method="sigmoid", cv=5)` |
| Random seed | 42 |

## Outputs

Each script writes its tables (Excel) and figures (PNG, 300 dpi) to its own output folder, for example `model_train/`, `xgboost_sigmoid_oof_calibration_dca/`, `ExternalVal_xgboost_sigmoid_outputs/` and `final_xgboost_sigmoid_shap_outputs/`. The external-validation step also saves the fitted imputer and the calibrated model as `.joblib` files.

## Citation

If you use this code, please cite the associated article (citation details will be added after publication).

## Contact

Mingyan Fu, Shandong Public Health Clinical Center, Shandong University, Jinan, China (fmy10072017@163.com)
Corresponding author: Li Lin (17863801807@163.com)
