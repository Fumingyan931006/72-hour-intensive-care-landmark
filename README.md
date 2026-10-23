# 脓毒症合并高血压研究：机器学习分析代码说明

本代码包提供基于 Excel 数据的二分类预测分析流程，包括缺失情况统计、五折分层交叉验证、迭代插补、Boruta 与 L1 Logistic 回归特征筛选、多模型比较、概率校准、决策曲线分析（DCA）、SHAP 解释，以及使用 MIMIC 开发数据和 eICU 数据进行外部验证。

本文依据 `code.zip` 中的 15 个 Python 脚本整理。压缩包未包含研究数据、预训练模型、依赖版本锁定文件或完整的数据字典。具体结局定义、观察时间窗、纳排标准和数据库版本应以研究方案及论文为准，不能仅由脚本确定。本文说明代码的实际实现，不代表已经复现论文结果。

## 1. 文件说明

| 脚本 | 功能 | 默认输出位置 |
| --- | --- | --- |
| `MissingRate.py` | 统计每行和每列的缺失数量及比例 | `实验数据-缺失率.xlsx` |
| `5-fold+MICE.py` | 五折分层划分；在各训练折拟合迭代插补器并应用于对应测试折 | `folder/` |
| `Boruta.py` | 在每个训练折中进行 Boruta 筛选并汇总入选频次 | `boruta_outputs/` |
| `DataAfterBoruta.py` | 根据 Boruta 结果提取各折训练和测试数据 | `boruta_selected_folds/` |
| `LASSO.py` | 使用 L1 LogisticRegressionCV 进一步筛选特征 | `lasso_outputs/` |
| `train.py` | 训练并评估多个分类模型，汇总五折指标 | `model_train/` |
| `ROCComparison.py` | 重新训练模型，绘制合并测试折 ROC，并比较 SOFA/SAPS II | `roc_10model_outputs5/` |
| `CC-DCA.py` | 生成校准 XGBoost 的折外预测、校准曲线、ROC 和 DCA | `xgboost_sigmoid_oof_calibration_dca/` |
| `FinalModel-SHAP.py` | 在指定完整数据上拟合模型，输出 SHAP 分析和模型文件 | `final_xgboost_sigmoid_shap_outputs/` |
| `ExternalVal_FillEICUbyMIMIC_Mice.py` | 在 MIMIC 拟合插补器，再应用于 eICU | `mice_external_validation_outputs/` |
| `ExternalVal_MIMIC_Boruta.py` | 在完整 MIMIC 开发数据中进行 Boruta 筛选 | `ExternalVal_boruta_outputs/` |
| `ExternalVal_MIMIC_Lasso.py` | 在 MIMIC Boruta 结果上进行 L1 Logistic 特征筛选 | 需要修改脚本中的绝对路径 |
| `ExternalVal_XGBoost_Sigmoid_model.py` | 在 MIMIC 拟合校准 XGBoost，在 eICU 评估 | `ExternalVal_xgboost_sigmoid_outputs/` |
| `ExternalVal_CIC.py` | 根据外部预测结果绘制临床影响曲线（CIC） | `ExternalVal_cic_outputs/` |
| `ExternalVal_risk_stratification.py` | 根据外部预测概率进行风险分层并汇总事件率 | `ExternalVal_risk_stratification_outputs/` |

压缩包中的 `__MACOSX/` 为 macOS 元数据，不参与分析。

## 2. 环境与依赖

代码使用 Python 及以下第三方包：

| 包 | 用途 |
| --- | --- |
| `numpy`、`pandas` | 数值计算及表格处理 |
| `openpyxl` | Excel 文件读写 |
| `matplotlib` | 绘图 |
| `scikit-learn` | 插补、特征标准化、分类模型、校准及评估 |
| `Boruta` | Boruta 特征筛选；导入名称为 `boruta` |
| `xgboost`、`lightgbm` | 梯度提升分类模型 |
| `shap` | SHAP 解释 |
| `joblib` | 保存模型及插补器 |

在解压后的代码目录中建立环境。Windows PowerShell 示例：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install numpy pandas openpyxl matplotlib scikit-learn Boruta xgboost lightgbm shap joblib
```

后文使用 `python` 表示该环境中的解释器；也可直接替换为 `.\.venv\Scripts\python.exe`。所有命令均应在代码目录中执行，相对路径以当前工作目录为基准。

原包没有记录 Python 或依赖的准确版本，上述命令只是依赖安装示例，并非已验证的复现环境。请优先使用作者原始环境；验证成功后记录版本：

```powershell
python --version
python -m pip freeze > requirements-lock.txt
```

`train.py` 和 `ROCComparison.py` 在无法导入 XGBoost 或 LightGBM 时会跳过对应模型；其他依赖 XGBoost 的脚本则要求成功安装该包。部分脚本使用 `CalibratedClassifierCV(estimator=...)`，部分提供旧参数名兼容分支，不能假定任意依赖版本均可运行。

## 3. 输入数据

### 3.1 数据格式

- 使用 `.xlsx` 表格，首行为列名，每行对应一个分析样本。
- 大多数脚本默认最后一列为二分类结局；有 `LABEL_COL` 或 `label_col` 参数的脚本可显式指定列名。`5-fold+MICE.py` 固定使用最后一列作为结局。
- 建议在输入数据中统一使用 `0` 表示非事件、`1` 表示事件。部分脚本会自动转换其他二分类标签，但转换规则不完全相同，应在输入前统一编码。
- 特征应采用一致的数值编码。MIMIC 与 eICU 中同名变量必须具有相同含义、单位和编码。
- 在相应的 `DROP_COLS`、`ID_COLS`、`id_cols` 或 `EXCLUDE_COLS` 中配置实际编号列。默认值 `[""]` 只是空字符串占位，并不会自动排除患者编号。
- 缺失标签、无穷值、整列缺失及类别样本过少等问题应在分析前检查。每个需要五折训练或校准的子集中，都应有足够的两类样本。

### 3.2 需要准备的文件

| 文件 | 用途及要求 |
| --- | --- |
| `实验数据.xlsx` | 内部验证原始数据；`MissingRate.py` 默认读取 `model_data` 工作表，`5-fold+MICE.py` 默认读取第一个工作表，需确认两者对应同一数据 |
| `SAPSII_Sofa2.xlsx` | ROC 评分比较，需包含结局、`sofa_first` 和 `sapsii` 列 |
| `FinalModelData.xlsx` | SHAP 脚本的独立输入；需自行准备完整、无缺失的最终特征和结局数据 |
| `MIMIC外部验证实验数据.xlsx` | 外部验证流程的 MIMIC 开发数据 |
| `EICU外部验证实验数据.xlsx` | 外部验证流程的 eICU 数据，需包含插补所需的同名特征列 |
| `EICU_lasso_selected_data.xlsx` | 外部建模脚本默认读取的 eICU 文件；该名称的文件不会由现有上游脚本自动生成，衔接方式见第 5 节 |

代码包未提供上述数据，也未提供从数据库提取队列的 SQL 或数据字典。

## 4. 内部五折验证

先检查输入路径、工作表、结局列及编号排除配置，再按顺序执行：

```powershell
# 可选：缺失情况统计
python "MissingRate.py"

# 主流程
python "5-fold+MICE.py"
python "Boruta.py"
python "DataAfterBoruta.py"
python "LASSO.py"
python "train.py"

# 校准、DCA 和 ROC 分析
python "CC-DCA.py"

# 准备好 SAPSII_Sofa2.xlsx 后执行
python "ROCComparison.py"
```

流程关系：

```text
实验数据.xlsx
  → 五折划分与训练折内插补：folder/
  → 每折 Boruta 筛选：boruta_outputs/
  → 按入选变量提取数据：boruta_selected_folds/
  → 每折 L1 Logistic 筛选：lasso_outputs/fold1/ ... fold5/
  ├→ 多模型性能评估：model_train/
  ├→ 校准 XGBoost 的折外校准、ROC 和 DCA
  └→ 模型与 SOFA/SAPS II 的 ROC 比较
```

### 实现参数

- **划分与插补：** `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`；数值特征使用 `IterativeImputer` 和 `BayesianRidge`，`max_iter=10`、`sample_posterior=True`。每折只在训练集拟合插补器，测试集调用 `transform`。非数值列保留，不进行该插补。
- **Boruta：** `perc=90`、`alpha=0.05`、`max_iter=100`。若某折未正式选中特征，则最多使用随机森林重要性前 10 个特征作为回退，并在结果中标记。
- **L1 Logistic 筛选：** 训练折标准化，`solver="saga"`，内部五折以 ROC AUC 选择 `C`；内部验证的搜索范围为 `np.logspace(-4, -1, 40)`。绝对系数大于 `1e-8` 的变量入选；若无入选变量，则回退到绝对系数最大的一个变量。导出的筛选数据保留原始尺度。
- **模型比较：** 包括 XGBoost、XGBoost + Sigmoid 校准、LightGBM、SVM、KNN、BPNN（MLP）、随机森林和 Logistic 回归，共 8 个模型配置；SOFA 和 SAPS II 为另外两个评分比较对象。
- **评估：** `train.py` 默认分类阈值为 `0.5`，计算 Accuracy、AUC、Sensitivity、Specificity、PPV、NPV 及混淆矩阵计数。汇总表给出各折均值和标准差。
- **校准与 DCA：** `CC-DCA.py` 重新训练模型并合并各测试折预测；默认 10 个校准分箱，DCA 阈值约为 `0.05–0.30`、步长 `0.01`。

主要结果包括：

- `folder/fold{n}_indices.xlsx`：原始数据的零基行位置索引。
- `boruta_outputs/boruta_feature_selection_frequency.xlsx`：Boruta 入选频次。
- `lasso_outputs/lasso_feature_selection_frequency.xlsx`：L1 Logistic 入选频次。
- `model_train/all_folds_all_models_metrics.xlsx`：逐折逐模型指标及错误信息。
- `model_train/fivefold_performance_summary.xlsx`：五折性能汇总。
- `xgboost_sigmoid_oof_calibration_dca/xgboost_sigmoid_oof_predictions.xlsx`：合并测试折预测。

`{n}` 表示折号 1–5。输出图像通常为 300 dpi PNG，表格为 Excel。

## 5. MIMIC 开发与 eICU 外部验证

### 5.1 运行前必须配置

**排除结局和编号列：** `ExternalVal_FillEICUbyMIMIC_Mice.py` 不会自动识别结局列；默认 `EXCLUDE_COLS = [""]` 会使数值型结局也进入插补特征。应将实际结局列和编号列明确加入排除列表，例如：

```python
# 示例列名，请按实际数据替换
EXCLUDE_COLS = ["label", "number"]
```

这样外部结局不会参与预测变量的插补。`ExternalVal_MIMIC_Boruta.py` 中也需配置不参与筛选的编号列；`ExternalVal_MIMIC_Lasso.py` 默认排除 `number`，需与实际数据一致。

**修改本机绝对路径：** `ExternalVal_MIMIC_Lasso.py` 中保留了 `/Users/...` 开头的作者本机路径。为衔接当前代码包，可改为：

```python
INPUT_FILE = "ExternalVal_boruta_outputs/mimic_boruta_selected_data.xlsx"
OUTPUT_DIR = "ExternalVal_lasso_outputs"
```

**衔接 eICU 插补结果：** 在 `ExternalVal_XGBoost_Sigmoid_model.py` 中将：

```python
EICU_FILE = "mice_external_validation_outputs/eicu_imputed.xlsx"
```

该脚本会按照 MIMIC 最终特征名提取并排列 eICU 列，因此无需在 eICU 上重新筛选特征。另一种方式是自行准备默认的 `EICU_lasso_selected_data.xlsx`。两种方式均要求特征完整、同名且无剩余缺失，并正确设置结局列。

**统一后处理标签名：** CIC 和风险分层脚本默认读取 `Y_TRUE_COL = "label"`。外部预测文件实际保留 eICU 的结局列名；若名称不同，需修改这两个脚本。概率列默认为 `pred_prob_xgboost_sigmoid`。

### 5.2 执行顺序

完成以上配置后，依次执行：

```powershell
python "ExternalVal_FillEICUbyMIMIC_Mice.py"
python "ExternalVal_MIMIC_Boruta.py"
python "ExternalVal_MIMIC_Lasso.py"
python "ExternalVal_XGBoost_Sigmoid_model.py"
python "ExternalVal_CIC.py"
python "ExternalVal_risk_stratification.py"
```

外部流程的插补器在全部 MIMIC 开发数据上拟合，使用 `sample_posterior=False`，eICU 只调用 `transform`。Boruta 和 L1 Logistic 特征筛选均在 MIMIC 中执行；外部流程的 L1 Logistic 搜索范围为 `np.logspace(-3, 2, 50)`，与内部流程不同。

外部验证模型使用 XGBoost（120 棵树、深度 4、学习率 0.05、`subsample=0.9`、`colsample_bytree=0.9`）及 `CalibratedClassifierCV(method="sigmoid", cv=5)`。模型拟合在 MIMIC 中完成，再对 eICU 预测。外部建模脚本会删除缺失或非法结局的样本并打印数量；剩余特征缺失会导致报错。

主要输出：

| 位置 | 内容 |
| --- | --- |
| `mice_external_validation_outputs/mice_imputer_mimic.joblib` | MIMIC 拟合的插补器 |
| `ExternalVal_lasso_outputs/mimic_lasso_selected_data.xlsx` | 最终开发数据，使用前述相对路径配置后生成 |
| `ExternalVal_xgboost_sigmoid_outputs/final_base_xgb.joblib` | 单独拟合的未校准 XGBoost |
| `ExternalVal_xgboost_sigmoid_outputs/final_xgboost_sigmoid_model.joblib` | 用于预测的校准模型 |
| `ExternalVal_xgboost_sigmoid_outputs/eicu_external_predictions.xlsx` | eICU 结局、预测概率及预测类别 |
| `ExternalVal_xgboost_sigmoid_outputs/eicu_external_validation_metrics.xlsx` | 外部验证性能指标 |
| `ExternalVal_xgboost_sigmoid_outputs/` | 外部 ROC、校准、DCA 图及对应数据和汇总 |
| `ExternalVal_cic_outputs/` | CIC 图、曲线数据和汇总 |
| `ExternalVal_risk_stratification_outputs/` | 风险分层明细、事件率汇总及图 |

CIC 默认阈值为约 `0.05–0.50`，按每 1000 人展示高风险人数及其中实际发生事件的人数。风险分层默认按 eICU 预测概率的分位数分成 3 组；代码另支持自定义分界值，示例为 `[0.0, 0.2, 0.5, 1.0]`。分位数组不是预先固定的临床风险阈值。

## 6. 最终模型与 SHAP

`FinalModel-SHAP.py` 是独立分支，需要先准备 `FinalModelData.xlsx`。该文件不会由内部五折流程自动生成；需明确最终特征集合及其来源，并确保所有特征均为无缺失的数值变量。

```powershell
python "FinalModel-SHAP.py"
```

主要输出位于 `final_xgboost_sigmoid_shap_outputs/`：

- `final_base_xgb_for_shap.joblib`：用于 SHAP 的单独拟合 XGBoost。
- `final_xgboost_sigmoid_model.joblib`：用于概率预测的校准模型。
- `final_model_predictions.xlsx`：输入数据上的预测概率。
- `final_model_shap_importance.xlsx`：平均绝对 SHAP 值。
- `final_model_shap_values.xlsx`：样本级 SHAP 值。
- `final_model_shap_summary.png`、`final_model_shap_bar.png`：SHAP 汇总图和重要性图。

SHAP 默认最多抽样 1000 个样本，随机种子为 42。解释对象是单独拟合的未校准 XGBoost，并非 Sigmoid 校准后模型的最终概率。该脚本对用于拟合的数据生成预测，不能将这些预测当作独立测试结果。

## 7. 复现时需要注意的实现细节

1. **“MICE”输出数量：** 当前脚本每折仅输出一份插补数据；没有生成多套插补数据并进行合并推断。内部插补和外部插补的后验采样设置不同。
2. **类别特征衔接：** `Boruta.py` 支持生成哑变量，但 `DataAfterBoruta.py` 从原始插补表按入选列名取列。如果入选名称是新生成的哑变量名，原表中可能没有对应列。完整默认流程应使用已统一数值编码的数据，或先调整这一衔接。
3. **外部 Boruta 筛空：** 若没有正式入选特征，外部 Boruta 脚本不会生成 `mimic_boruta_selected_data.xlsx`；此时需检查筛选结果，不能直接继续运行 LASSO。
4. **交叉验证层级：** 插补和特征筛选位于外层训练折内；但 LASSO 标准化在其内部 CV 前完成，Boruta 也不会在每个 LASSO 内部子折重新运行。不能将现有实现描述为所有预处理均在内层重新拟合的完全嵌套流程。
5. **ROC 比较会重新训练：** `ROCComparison.py` 不读取 `train.py` 的预测结果，且其中未校准 XGBoost 使用 130 棵树，而 `train.py` 为 120 棵。两者结果不应被假定完全一致。
6. **评分比较人群：** SOFA/SAPS II 曲线来自 `SAPSII_Sofa2.xlsx` 中结局及两评分均非缺失的样本，机器学习曲线来自合并测试折。代码未按患者 ID 自动匹配两者；比较前应确认样本及结局一致。
7. **校准统计量：** 校准截距和斜率由对预测概率 logit 拟合 `LogisticRegression` 得到，使用该类的默认正则化设置。README 不将其表述为已实现无惩罚、固定斜率的 calibration-in-the-large 估计。
8. **错误与覆盖：** `train.py` 可捕获单模型训练失败并继续运行，应检查结果表中的 `error` 列，不能仅凭出现汇总文件判断全部模型成功。重复执行会写入相同输出文件名，应另行保存需要保留的历史结果。

## 8. 复现状态与发布信息

本 README 已依据源代码核对文件名、主要参数、输入输出和运行依赖；由于代码包中未包含输入数据，本次未执行端到端训练，也未核验任何模型性能数值。

正式发布代码时，应补充原始运行环境版本、数据字典、队列及结局定义、最终特征确定方式，并注明论文引用信息、代码许可证和联系信息。原压缩包未提供这些信息，本文不代为推定。
