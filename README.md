# HBV-ML-Interpretability

Interpretability-driven feature engineering enables compact and accurate multi-outcome risk stratification in chronic hepatitis B .

## About

This repository contains the complete R analysis pipeline for the manuscript *"Interpretability-driven feature engineering enables compact and accurate multi-outcome risk stratification in chronic hepatitis B"*.

## Repository Structure

| Script | Purpose |
|--------|---------|
| `00_data_merging.R` | Raw data merging and initial variable consolidation |
| `01_data_preprocessing.R` | Data cleaning, missing value imputation, baseline table |
| `02_baseline_models.R` | Training and hyperparameter tuning of 8 ML algorithms (XGBoost, LightGBM, GBM, LR, GLMNet, CRF, RF, MLP) |
| `03_model_comparison.R` | DeLong tests for pairwise AUC comparison, ROC curve generation |
| `04_SHAP_analysis.R` | SHAP feature importance, interaction analysis, waterfall decomposition |
| `05_SHAP_visualization.R` | SHAP summary plots, dependence plots, interaction heatmaps |
| `06_feature_engineering.R` | Two-stage feature engineering: Top-15 selection + interaction construction (product, log-product, log-ratio) |
| `07_feature_eng_visualization.R` | Visualization of feature engineering results and interaction-enhanced model outputs |
| `08_temporal_validation.R` | Temporal validation with strict time-split cohort |
| `09_robustness_analysis.R` | Bootstrap stability analysis (AUC, feature importance, interactions) |

## Requirements

- R version 4.5.1
- Core packages: `caret`, `xgboost`, `lightgbm`, `randomForest`, `glmnet`, `gbm`, `nnet`, `shapviz`, `SHAPforxgboost`, `fastshap`, `ggplot2`, `patchwork`, `mlr3`, `pROC`, `data.table`, `tidyverse`

## Usage

Scripts should be run in numerical order (00 through 09). Data paths in the scripts reflect the local computing environment and should be updated to match the user's own directory structure. Code comments are in Chinese.

## Citation

Zhang HY, et al. Interpretability-driven feature engineering enables compact and accurate multi-outcome risk stratification in chronic hepatitis B. 
