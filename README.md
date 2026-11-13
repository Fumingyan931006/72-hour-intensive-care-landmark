# 72-hour intensive care landmark model

Analysis code for the study **"Mortality risk reassessment at 72 hours in patients with sepsis and pre-existing hypertension receiving vasoactive agents: a retrospective model development and external validation study."**

This repository documents the modelling workflow described in the manuscript: five-fold patient-level cross-validation with fold-internal imputation and feature selection (Boruta followed by L1-penalized logistic regression), comparison of machine-learning algorithms, sigmoid calibration of XGBoost, decision-curve and risk-stratification analyses, SHAP interpretation, and **external validation in an independent, single-center cohort from Shandong Public Health Clinical Center (SPHCC), China**.

The model was developed using **MIMIC-IV (v3.1)** and applied to the SPHCC cohort **without retraining, recalibration, or changes to the fitted model**.

## Study cohorts

| Cohort | Source | Period | Patients | In-hospital deaths after the 72-hour landmark | Role |
| --- | --- | --- | ---: | ---: | --- |
| Development / internal validation | MIMIC-IV v3.1 | As available in MIMIC-IV | 1,019 | 338 (33.2%) | Model development and five-fold internal validation |
| External validation | Shandong Public Health Clinical Center (SPHCC), China | January 2018–September 2025 | 1,747 | 516 (29.5%) | Independent evaluation of the locked model |

The same eligibility and exclusion criteria were used for both cohorts. The prediction landmark was **72 hours after ICU admission**, based on information from six consecutive 12-hour windows during the first 72 hours. The outcome was subsequent in-hospital death after the landmark.

The selected sigmoid-calibrated XGBoost model achieved an internal AUC of **0.752** and an external AUC of **0.713** (95% CI **0.687–0.739**) in the SPHCC cohort.

## Data availability and ethics

**No patient-level data are included in this repository.**

- **MIMIC-IV v3.1** is accessible to credentialed users through PhysioNet, subject to the relevant data use agreement: https://doi.org/10.13026/kpb9-mt58
- **SPHCC external-validation data** are retrospective institutional patient records collected between January 2018 and September 2025. They are **not a public PhysioNet dataset**, and patient-level records are not distributed with this code. Access and any permitted reuse are subject to institutional, ethics, and data-protection requirements.

The study was approved by the Ethics Committee of Shandong Public Health Clinical Center (**GWLCZXEC-SOP-K-2025-93**). The approval covers the hospital-based retrospective analysis, and the requirement for informed consent was waived.

Cohort definitions, the 72-hour landmark, the six 12-hour windows, and candidate-predictor definitions are provided in the manuscript and its Supplementary Material. The analysis scripts expect prepared Excel tables containing one row per patient, harmonized candidate predictors, and the binary outcome (death after the landmark coded 0/1). The raw institutional dataset is not supplied.

## Requirements

Python 3.10 with the packages listed in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Input files

The filenames below follow the documented SPHCC workflow. **Ensure that the actual files and hard-coded paths in the Python scripts match these names before execution.**

| File | Content |
| --- | --- |
| `实验数据.xlsx` | MIMIC-IV development cohort (candidate predictors and outcome) for internal cross-validation |
| `SAPSII_Sofa2.xlsx` | Outcome, `sofa_first`, and `sapsii` for the clinical-score comparison |
| `MIMIC外部验证实验数据.xlsx` | Full MIMIC-IV development cohort used to fit the final, locked model (legacy filename) |
| `外部验证实验数据.xlsx` | **SPHCC hospital external-validation cohort**, with predictor names and model-input units harmonized to the development cohort |
| `FinalModelData.xlsx` | MIMIC-IV development cohort restricted to the final predictor set, for SHAP analysis |

The Excel files containing patient-level records should **not** be committed to a public repository. Confirm that all source-file paths in the scripts point to the intended MIMIC-IV and SPHCC tables before running the workflow.

In each script, the outcome is taken from the last column unless `LABEL_COL` is set. Identifier columns should be listed in the corresponding `ID_COLS`, `DROP_COLS`, or `EXCLUDE_COLS` setting so that they are not used as predictors. **The external outcome must be used only for evaluation, never for fitting the imputer, selecting predictors, calibrating probabilities, or training the model.**

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

### 2. Locked model and external validation (MIMIC-IV → SPHCC)

Fit the imputation, feature-selection, classifier, and sigmoid-calibration components using **MIMIC-IV development data only**. Apply the resulting locked pipeline directly to the **SPHCC external cohort**; do not retrain or recalibrate on hospital patients.

Use `ExternalVal_FillSPHCCbyMIMIC_Mice.py` for external imputation and `外部验证实验数据.xlsx` as the SPHCC input workbook. Ensure these filenames match the actual files and script paths.

Before running, set `EXCLUDE_COLS` in `ExternalVal_FillSPHCCbyMIMIC_Mice.py` to the outcome and identifier columns of the relevant tables, so that the outcome is not used by the imputer. Check script input/output paths and labels for consistency with the SPHCC cohort.

```bash
python ExternalVal_FillSPHCCbyMIMIC_Mice.py     # fit imputer on MIMIC-IV, apply unchanged to SPHCC
python ExternalVal_MIMIC_Boruta.py             # Boruta on the full MIMIC-IV development cohort
python ExternalVal_MIMIC_Lasso.py              # L1 selection on the full development cohort -> final predictor set
python ExternalVal_XGBoost_Sigmoid_model.py    # fit calibrated XGBoost on MIMIC-IV, predict SPHCC without refitting
python ExternalVal_CIC.py                      # clinical impact curve using SPHCC predictions and outcomes
python ExternalVal_risk_stratification.py      # observed mortality across tertiles of SPHCC predicted risk
```

`ExternalVal_CIC.py` and `ExternalVal_risk_stratification.py` read the outcome column named in `Y_TRUE_COL` (default `label`); change it if the actual outcome column has another name. Outcome labels are required for performance assessment but must not enter predictor preparation or model fitting.

### 3. Model interpretation

```bash
python FinalModel-SHAP.py        # SHAP values of uncalibrated XGBoost refitted on the MIMIC-IV development cohort
```

## Main settings

| Component | Setting |
| --- | --- |
| Cross-validation | `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` |
| Imputation | `IterativeImputer` with `BayesianRidge`, `max_iter=10`; fitted on development/training data only |
| Boruta | Random-forest base estimator, `perc=90`, `alpha=0.05`, `max_iter=100` |
| L1 logistic regression | `solver="saga"`, penalty chosen by inner five-fold CV maximizing ROC AUC |
| XGBoost (selected model) | `n_estimators=120`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.9`, `colsample_bytree=0.9` |
| Calibration | `CalibratedClassifierCV(method="sigmoid", cv=5)` fitted on development data only |
| Random seed | 42 |

## Outputs

Each script writes its tables (Excel) and figures (PNG, 300 dpi) to its own output folder, for example `model_train/`, `xgboost_sigmoid_oof_calibration_dca/`, `ExternalVal_xgboost_sigmoid_outputs/`, and `final_xgboost_sigmoid_shap_outputs/`. The external-validation step also saves the fitted imputer and calibrated model as `.joblib` files. Output folder names are script-specific and do not identify the clinical data source.

## Reproducibility notes

- Publicly accessible code and credentialed access to MIMIC-IV allow the development workflow to be examined, subject to its data use agreement.
- **Exact replication of the SPHCC external validation requires authorized access to the institutional cohort**; the hospital patient records are not supplied in this repository.
- This README corrects the **documentation of external-data provenance** and the names referenced in the instructions. The actual Python and Excel files were not supplied for this edit, so **their filesystem names and any source-code references have not been modified or verified**. Rename them and update paths before publishing or running the repository.

## Citation

If you use this code, please cite the associated article (citation details will be added after publication).

## Contact

Mingyan Fu, Shandong Public Health Clinical Center, Shandong University, Jinan, China (fmy10072017@163.com)  
Corresponding author: Li Lin (17863801807@163.com)
