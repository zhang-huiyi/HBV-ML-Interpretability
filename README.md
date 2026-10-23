# HBV Multi-Outcome Machine Learning

Interpretability-driven feature engineering enables compact and accurate multi-outcome risk stratification at hospitalization for chronic hepatitis B.

## About

This repository contains the complete R analysis pipeline for the study of multi-outcome risk prediction in a chronic hepatitis B (HBV) cohort. Three clinical outcomes are modelled in parallel: disease non-progression, hepatocellular carcinoma (HCC) and liver cirrhosis.

The pipeline proceeds from raw data merging and preprocessing through the development and comparison of eight candidate algorithms, SHAP-based interpretation, two-stage interaction feature engineering, temporal validation, and a three-part robustness assessment.

## Repository Structure

Scripts are executed in numerical order (`00` → `08`).

| Script | Purpose |
| --- | --- |
| `01_data_preprocessing.R` | Data cleaning, patient-level train/test split, missing value imputation, baseline characteristics table, export of the modelling datasets |
| `02_baseline_models.R` | Training and hyperparameter tuning of ML algorithms (XGBoost, LightGBM, GBM, GLMNet, Conditional RF, RF, MLP, Logistic Regression) |
| `03_model_comparison.R` | DeLong tests for pairwise AUC comparison, ROC curve generation |
| `04_SHAP_analysis.R` | SHAP feature importance, TreeSHAP interaction analysis, waterfall decomposition |
| `05_SHAP_visualization.R` | SHAP summary plots, dependence plots, interaction heatmaps |
| `06_feature_engineering.R` | Two-stage feature engineering: Top-15 selection + interaction construction (product, log-product, log-ratio), with visualization of the results |
| `07_temporal_validation.R` | Temporal validation with a strict time-split cohort |
| `08_robustness_validation.R` | Robustness validation in three parts (see below) |

## Robustness Validation

`08_robustness_validation.R` performs the following analyses in the order in which they appear in the file:

- **A. Stability of SHAP feature importance and interaction features.** Patients are resampled with replacement; XGBoost is refitted and TreeSHAP is recomputed at every replicate (100 replicates for feature importance and 1000 replicates for interactions, the latter controlled by `INTERACTION_PATIENT_N`). Selection frequencies of the Top-15 features and the Top-4 interaction pairs are reported, and results are written to `Patient_Bootstrap_TreeSHAP_Results_B100_N1000_FINAL`.

- **B. Comparison of Top-15 feature selection methods.** SHAP, Gain, LASSO and Random Forest permutation importance are each used to select a Top-15 feature set. An XGBoost model is trained for every set under the same framework, evaluated on the same test set and compared by paired DeLong tests. Section B-1 covers SHAP, Gain and LASSO; section B-2 covers Random Forest permutation importance and reads the predictions produced by B-1, so the two sections must be run in order.

- **C. Internal robustness of the models.** A patient-level bootstrap (B = 2000) is performed on the internal test set to evaluate the stability of calibration and decision curve analysis for the Baseline, Reduced and Interaction models.

## Requirements

* R version 4.5.1

Each script installs any missing package at the top. Core packages by role:

| Role | Packages |
| --- | --- |
| Data I/O and wrangling | `data.table`, `readxl`, `writexl`, `openxlsx`, `reshape2`, `lubridate`, `tidyverse` (`dplyr`, `tidyr`, `stringr`, `tibble`, `forcats`) |
| Missing data and description | `mice`, `VIM`, `naniar`, `skimr` |
| Modelling and tuning | `xgboost`, `lightgbm`, `gbm`, `glmnet`, `nnet`, `partykit`, `ranger`, `caret`, `future`, `mlr3` family (`mlr3`, `mlr3learners`, `mlr3extralearners`, `mlr3tuning`, `mlr3verse`, `paradox`) |
| Evaluation and plotting | `pROC`, `PRROC`, `corrplot`, `gridExtra`, `ggplot2`, `scales`, `cowplot`, `patchwork`, `ggbeeswarm` |

## Data and Path Configuration

All paths point to the authors' local computing environment and must be adapted before running the pipeline.

* **Input data.** Raw cohort files and intermediate datasets (e.g. `dhbv.xlsx 2026.05.17.xlsx`, `train_preprocessed.xlsx`, `test_preprocessed.xlsx`) are configured at the top of the corresponding script.
* **Output.** Model objects, predictions, tables and figures are written to subdirectories under `J:/rtools45/`. `00_data_merging.R` writes the merged `dhbv.xlsx` and the summary table `统计结果.xlsx` to the working directory; the former must be moved to the input path configured in `01` and `08` before proceeding.
* No script calls `setwd()`, so the working directory does not need to be changed.

Cross-script dependencies — if the output root is changed, the paths below must be updated together:

| Producing script | Output consumed | Consuming script |
| --- | --- | --- |
| `01` | `train_preprocessed.xlsx`, `test_preprocessed.xlsx` | `02`, `06`, `08` |
| `01` | `train_preprocessed_with_id.rds`, `test_raw_patient_level.xlsx` | `08` |
| `02` | `XGBoost_Results` and the per-algorithm result directories | `03`, `04` |
| `04` | `SHAP_Results_TOPJOURNAL` | `05`, `06` |
| `08` | `Patient_Bootstrap_TreeSHAP_Results_B100_N1000_FINAL` | `05` |

`05_SHAP_visualization.R` must therefore be run after both `04_SHAP_analysis.R` and `08_robustness_validation.R`, and the directory names must stay consistent across the three scripts.

## Usage

Run the scripts in numerical order. Comments in the code are written in Chinese and annotate the logic of each step in detail.

```r
source("analysis_scripts/00_data_merging.R")
source("analysis_scripts/01_data_preprocessing.R")
# ... and so on through 08_robustness_validation.R
```

## Citation

Zhang HY, et al. Interpretability-driven feature engineering enables compact and accurate multi-outcome risk stratification at hospitalization for chronic hepatitis B.
