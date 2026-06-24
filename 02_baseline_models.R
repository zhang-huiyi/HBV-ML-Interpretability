# 02_baseline_models.R
# Training and hyperparameter tuning of 8 ML algorithms:
# XGBoost, LightGBM, GBM, Logistic Regression, GLMNet,
# Conditional Random Forest, Random Forest, MLP
# ============================================================



# ====== XGBoost ======

######################################################################
# XGBoost模型训练脚本
######################################################################

rm(list = ls())
gc()

##############################
# 1. 环境准备
##############################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "mlr3",
  "mlr3learners",
  "mlr3tuning",
  "paradox",
  "xgboost",
  "pROC",
  "caret",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

future::plan("sequential")

cat("====================================================\n")
cat(" XGBoost模型训练（最终SCI审稿版）\n")
cat("====================================================\n")

##############################
# 2. 创建结果文件夹
##############################

result_dir <- "XGBoost_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

##############################
# 3. 数据读取
##############################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  stop("❌ test_preprocessed.xlsx不存在")
}

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

cat("\n训练集维度:\n")
print(dim(train_data))

cat("\n测试集维度:\n")
print(dim(test_data))

##############################
# 4. 自动识别Outcome
##############################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未识别到Outcome变量")
}

cat("\n识别到结局变量:\n")
print(outcome_cols)

##############################
# 5. 数据类型统一处理
##############################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.character(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

##############################
# 6. XGBoost模型函数
##############################

train_xgb_model <- function(
    train_data,
    test_data,
    outcome_col
){
  
  cat("\n")
  cat("====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ####################################
  # 6.1 特征
  ####################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################
  # 6.2 数据集
  ####################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ####################################
  # 6.3 Outcome转factor
  ####################################
  
  train_df[[outcome_col]] <- factor(
    train_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  test_df[[outcome_col]] <- factor(
    test_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  ####################################
  # 6.4 所有特征数值化
  ####################################
  
  for(col in feature_cols){
    
    if(is.list(train_df[[col]])){
      
      train_df[[col]] <- as.character(unlist(train_df[[col]]))
    }
    
    if(is.list(test_df[[col]])){
      
      test_df[[col]] <- as.character(unlist(test_df[[col]]))
    }
    
    if(is.factor(train_df[[col]]) |
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <- as.numeric(train_df[[col]])
      
      test_df[[col]] <- as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <- as.numeric(train_df[[col]])
    
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
  
  ####################################
  # 6.5 类别不平衡
  ####################################
  
  cat("\n训练集类别分布:\n")
  print(table(train_df[[outcome_col]]))
  
  pos_num <- sum(train_df[[outcome_col]] == "1")
  
  neg_num <- sum(train_df[[outcome_col]] == "0")
  
  scale_pos_weight <- neg_num / pos_num
  
  cat("\nscale_pos_weight:",
      round(scale_pos_weight,3),
      "\n")
  
  ####################################
  # 6.6 创建Task
  ####################################
  
  task <- TaskClassif$new(
    id = outcome_col,
    backend = train_df,
    target = outcome_col,
    positive = "1"
  )
  
  ####################################
  # 6.7 XGBoost
  ####################################
  
  learner <- lrn(
    
    "classif.xgboost",
    
    predict_type = "prob",
    
    booster = "gbtree",
    
    eval_metric = "auc",
    
    # 轻度放松（SCI安全版）
    nrounds = 200,
    
    eta = 0.03,
    
    max_depth = 4,
    
    min_child_weight = 10,
    
    subsample = 0.75,
    
    colsample_bytree = 0.75,
    
    gamma = 2,
    
    lambda = 6,
    
    alpha = 3,
    
    scale_pos_weight = scale_pos_weight,
    
    nthread = 4
  )
  
  ####################################
  # 6.8 调参空间
  ####################################
  
  search_space <- ps(
    
    eta = p_dbl(0.02, 0.06),
    
    max_depth = p_int(3,5),
    
    min_child_weight = p_dbl(6,15),
    
    subsample = p_dbl(0.65,0.8),
    
    colsample_bytree = p_dbl(0.65,0.8),
    
    gamma = p_dbl(1,5),
    
    lambda = p_dbl(3,10),
    
    alpha = p_dbl(1,6)
  )
  ####################################
  # 6.9 AutoTuner
  ####################################
  
  at <- auto_tuner(
    
    tuner = tnr("random_search"),
    
    learner = learner,
    
    resampling = rsmp("cv", folds = 5),
    
    measure = msr("classif.auc"),
    
    search_space = search_space,
    
    terminator = trm("evals", n_evals = 50)
  )
  
  ####################################
  # 6.10 模型训练
  ####################################
  
  cat("\n开始训练...\n")
  
  start_time <- Sys.time()
  
  at$train(task)
  
  end_time <- Sys.time()
  
  cat(
    "\n训练耗时:",
    round(
      as.numeric(
        difftime(
          end_time,
          start_time,
          units = "mins"
        )
      ),
      2
    ),
    "分钟\n"
  )
  
  ####################################
  # 6.11 测试集预测
  ####################################
  
  full_data <- rbind(train_df, test_df)
  
  test_task <- TaskClassif$new(
    id = paste0(outcome_col,"_test"),
    backend = full_data,
    target = outcome_col,
    positive = "1"
  )
  
  pred <- at$predict(
    test_task,
    row_ids =
      (nrow(train_df)+1):nrow(full_data)
  )
  
  pred_prob <- pred$prob[, "1"]
  
  pred_class <- factor(
    pred$response,
    levels = c("0","1")
  )
  
  true_label <- factor(
    test_df[[outcome_col]],
    levels = c("0","1")
  )
  
  ####################################
  # 6.12 Test ROC/AUC/CI
  ####################################
  
  test_roc <- roc(
    as.numeric(true_label)-1,
    pred_prob
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ####################################
  # 6.13 Train ROC/AUC/CI
  ####################################
  
  train_pred <- at$predict(
    test_task,
    row_ids = 1:nrow(train_df)
  )
  
  train_prob <- train_pred$prob[, "1"]
  
  train_label <- factor(
    train_df[[outcome_col]],
    levels = c("0","1")
  )
  
  train_roc <- roc(
    as.numeric(train_label)-1,
    train_prob
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ####################################
  # 6.14 CV AUC + CI
  ####################################
  
  cv_scores <- at$archive$data$classif.auc
  
  cv_auc <- mean(cv_scores, na.rm = TRUE)
  
  cv_auc_ci_lower <- quantile(
    cv_scores,
    probs = 0.025,
    na.rm = TRUE
  )
  
  cv_auc_ci_upper <- quantile(
    cv_scores,
    probs = 0.975,
    na.rm = TRUE
  )
  
  ####################################
  # 6.15 混淆矩阵
  ####################################
  
  cm <- confusionMatrix(
    data = pred_class,
    reference = true_label,
    positive = "1"
  )
  
  ####################################
  # 6.16 过拟合诊断
  ####################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################
  # 6.17 性能指标
  ####################################
  
  metrics_df <- data.frame(
    
    指标 = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "CV_AUC",
      "CV_AUC_CI_Lower",
      "CV_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    数值 = c(
      
      train_auc,
      train_auc_ci[1],
      train_auc_ci[3],
      
      cv_auc,
      cv_auc_ci_lower,
      cv_auc_ci_upper,
      
      test_auc,
      test_auc_ci[1],
      test_auc_ci[3],
      
      as.numeric(cm$overall["Accuracy"]),
      
      1 - as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  ####################################
  # 6.18 输出性能
  ####################################
  
  cat("\n【模型性能】\n")
  
  print(metrics_df)
  
  ####################################
  # 6.19 最优参数
  ####################################
  
  best_params <- at$archive$best()$x_domain[[1]]
  
  best_params_df <- data.frame(
    
    Parameter = names(best_params),
    
    Value = unlist(best_params)
  )
  
  ####################################
  # 6.20 文件名
  ####################################
  
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################
  # 6.21 保存性能
  ####################################
  
  write.csv(
    metrics_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_performance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.22 保存预测结果
  ####################################
  
  prediction_df <- data.frame(
    
    True_Label = true_label,
    
    Predicted_Class = pred_class,
    
    Predicted_Prob = pred_prob
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_prediction.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.23 保存调参历史
  ####################################
  
  tuning_history <- as.data.frame(
    at$archive$data
  )
  
  tuning_history <- tuning_history[,
                                   sapply(
                                     tuning_history,
                                     function(x)
                                       !is.list(x)
                                   )]
  
  write.csv(
    tuning_history,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_tuning_history.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.24 保存最优参数
  ####################################
  
  write.csv(
    best_params_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_best_parameters.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.25 保存ROC曲线数据
  ####################################
  
  roc_df <- data.frame(
    
    Thresholds = test_roc$thresholds,
    
    Sensitivity = test_roc$sensitivities,
    
    Specificity = test_roc$specificities,
    
    FPR = 1 - test_roc$specificities,
    
    TPR = test_roc$sensitivities
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.26 提取真正final booster
  ####################################
  
  final_model <- at$model$learner$model
  
  if(is.null(final_model)){
    
    stop("❌ final booster提取失败")
  }
  
  cat("\nfinal booster类型:\n")
  print(class(final_model))
  
  ####################################
  # 保存feature names（真正写入booster）
  ####################################
  
  xgb.attributes(final_model) <- list(
    feature_names = feature_cols
  )
  
  ####################################
  # 保存特征重要性
  ####################################
  
  importance_matrix <- xgb.importance(
    model = final_model
  )
  
  if(!is.null(importance_matrix)){
    
    importance_df <- as.data.frame(
      importance_matrix
    )
    
    write.csv(
      importance_df,
      file.path(
        result_dir,
        paste0(
          outcome_name_clean,
          "_Feature_Importance.csv"
        )
      ),
      row.names = FALSE
    )
  }
  ####################################
  # 6.27 保存完整模型对象（SCI最终稳定版）
  ####################################
  
  cat("\n开始保存完整模型对象（SHAP稳定版）...\n")
  
  ############################################################
  # 1. feature_levels（分类变量映射）
  ############################################################
  
  feature_levels <- list()
  
  for(col in feature_cols){
    
    if(is.factor(train_data[[col]]) ||
       is.character(train_data[[col]])){
      
      feature_levels[[col]] <- unique(
        c(
          as.character(train_data[[col]]),
          as.character(test_data[[col]])
        )
      )
    }
  }
  
  ############################################################
  # 2. 统一矩阵（SHAP唯一输入源）
  ############################################################
  
  X_train_matrix <- data.matrix(
    train_df[, feature_cols, with = FALSE]
  )
  
  X_test_matrix <- data.matrix(
    test_df[, feature_cols, with = FALSE]
  )
  
  colnames(X_train_matrix) <- feature_cols
  colnames(X_test_matrix)  <- feature_cols
  
  storage.mode(X_train_matrix) <- "double"
  storage.mode(X_test_matrix)  <- "double"
  
  
  ############################################################
  # 3. 真正Final Booster（最终SCI稳定版）
  ############################################################
  
  cat("\n重新训练真正Final Booster...\n")
  
  ############################################################
  # xgb.DMatrix
  ############################################################
  
  dtrain_final <- xgb.DMatrix(
    
    data = X_train_matrix,
    
    label = as.numeric(as.character(train_label))
  )
  
  ############################################################
  # 最优参数
  ############################################################
  
  final_xgb_params <- list(
    
    booster = "gbtree",
    
    objective = "binary:logistic",
    
    eval_metric = "auc",
    
    eta = best_params$eta,
    
    max_depth = best_params$max_depth,
    
    min_child_weight = best_params$min_child_weight,
    
    subsample = best_params$subsample,
    
    colsample_bytree = best_params$colsample_bytree,
    
    gamma = best_params$gamma,
    
    lambda = best_params$lambda,
    
    alpha = best_params$alpha,
    
    scale_pos_weight = scale_pos_weight
  )
  
  ############################################################
  # 真正重新训练最终booster
  ############################################################
  
  final_booster <- xgb.train(
    
    params = final_xgb_params,
    
    data = dtrain_final,
    
    nrounds = 200,
    
    verbose = 0
  )
  
  ############################################################
  # 强制保存feature names
  ############################################################
  
  xgb.attributes(final_booster) <- list(
    
    feature_names = paste(feature_cols, collapse = ",")
  )
  
  cat("\n✓ 真正Final Booster训练完成\n")
  
  ############################################################
  # 额外保存原生XGBoost Booster（极重要）
  ############################################################
  
  xgb.save(
    
    final_booster,
    
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_final_booster.model"
      )
    )
  )
  
  cat("\n✓ 原生booster已保存\n")
  ############################################################
  # 5. booster预测（仅供SHAP审计）
  ############################################################
  
  booster_train_prob <- predict(
    final_booster,
    X_train_matrix
  )
  
  booster_test_prob <- predict(
    final_booster,
    X_test_matrix
  )
  
  ############################################################
  # 6. booster AUC审计（不影响论文正式结果）
  ############################################################
  
  booster_train_roc <- pROC::roc(
    as.numeric(as.character(train_label)),
    booster_train_prob,
    quiet = TRUE
  )
  
  booster_test_roc <- pROC::roc(
    as.numeric(as.character(true_label)),
    booster_test_prob,
    quiet = TRUE
  )
  
  booster_train_auc <- as.numeric(
    booster_train_roc$auc
  )
  
  booster_test_auc <- as.numeric(
    booster_test_roc$auc
  )
  
  ############################################################
  # 7. feature fingerprint（防漂移）
  ############################################################
  
  feature_fingerprint <- list(
    
    feature_cols = feature_cols,
    
    train_colnames = colnames(X_train_matrix),
    
    test_colnames = colnames(X_test_matrix),
    
    identical_train = identical(
      feature_cols,
      colnames(X_train_matrix)
    ),
    
    identical_test = identical(
      feature_cols,
      colnames(X_test_matrix)
    )
  )
  
  ############################################################
  # 8. booster feature check
  ############################################################
  
  booster_feature_check <- list(
    
    has_feature_names_attr =
      !is.null(
        xgb.attr(
          final_booster,
          "feature_names"
        )
      ),
    
    feature_names_match =
      identical(
        xgb.attr(
          final_booster,
          "feature_names"
        ),
        feature_cols
      )
  )
  
  ############################################################
  # 9. 数据hash（审稿追溯）
  ############################################################
  
  data_hash <- list(
    
    train_dim = dim(X_train_matrix),
    
    test_dim = dim(X_test_matrix),
    
    feature_num = length(feature_cols),
    
    checksum_train =
      sum(X_train_matrix, na.rm = TRUE),
    
    checksum_test =
      sum(X_test_matrix, na.rm = TRUE)
  )
  
  ############################################################
  # 10. CV记录
  ############################################################
  
  cv_predictions <- at$archive$data$classif.auc
  
  ############################################################
  # 11. 保存完整对象
  ############################################################
  
  full_model_object <- list(
    
    ##########################################################
    # 模型核心
    ##########################################################
    
    auto_tuner = at,
    
    booster = final_booster,
    
    ##########################################################
    # 原始论文结果（mlr3官方结果）
    ##########################################################
    
    train_prob = train_prob,
    
    test_prob = pred_prob,
    
    train_pred = train_pred,
    
    test_pred = pred,
    
    train_auc = train_auc,
    
    test_auc = test_auc,
    
    train_roc = train_roc,
    
    test_roc = test_roc,
    
    ##########################################################
    # booster审计（仅SHAP用）
    ##########################################################
    
    booster_train_prob = booster_train_prob,
    
    booster_test_prob = booster_test_prob,
    
    booster_train_auc = booster_train_auc,
    
    booster_test_auc = booster_test_auc,
    
    ##########################################################
    # 数据
    ##########################################################
    
    train_df = train_df,
    
    test_df = test_df,
    
    feature_cols = feature_cols,
    
    outcome_col = outcome_col,
    
    ##########################################################
    # SHAP矩阵（关键）
    ##########################################################
    
    X_train_matrix = X_train_matrix,
    
    X_test_matrix = X_test_matrix,
    
    ##########################################################
    # 标签
    ##########################################################
    
    train_label = train_label,
    
    test_label = true_label,
    
    ##########################################################
    # SHAP稳定支持
    ##########################################################
    
    feature_levels = feature_levels,
    
    feature_fingerprint = feature_fingerprint,
    
    booster_feature_check = booster_feature_check,
    
    data_hash = data_hash,
    
    ##########################################################
    # 参数
    ##########################################################
    
    best_params = best_params,
    
    ##########################################################
    # CV
    ##########################################################
    
    cv_predictions = cv_predictions,
    
    cv_auc = cv_auc,
    
    ##########################################################
    # 其他
    ##########################################################
    
    confusion_matrix = cm,
    
    metrics_df = metrics_df,
    
    seed = 123,
    
    session_info = sessionInfo(),
    
    train_time = Sys.time()
  )
  
  ############################################################
  # 12. 保存
  ############################################################
  
  saveRDS(
    
    full_model_object,
    
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_FULL_MODEL.rds"
      )
    )
  )
  
  cat("\n✓ FULL_MODEL保存完成（SHAP稳定最终版）\n")
  ####################################
  # 6.28 保存报告
  ####################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "XGBoost模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    
    "CV策略: 5-fold Cross Validation\n",
    
    "迭代次数: 50\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n\n",
    
    "Accuracy: ",
    round(as.numeric(cm$overall["Accuracy"]),4),
    "\n",
    
    "CE: ",
    round(
      1-as.numeric(cm$overall["Accuracy"]),
      4
    ),
    "\n",
    
    "Sensitivity: ",
    round(as.numeric(cm$byClass["Sensitivity"]),4),
    "\n",
    
    "Specificity: ",
    round(as.numeric(cm$byClass["Specificity"]),4),
    "\n",
    
    "F1: ",
    round(as.numeric(cm$byClass["F1"]),4),
    "\n"
  )
  
  writeLines(
    report_text,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_report.txt"
      )
    )
  )
  
  cat("\n✓ 文件已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc,
    
    metrics = metrics_df
  ))
}

##############################
# 7. 循环训练
##############################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_xgb_model(
      train_data,
      test_data,
      outcome_col
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

##############################
# 8. 汇总结果
##############################

summary_df <- data.frame()

for(name in names(all_results)){
  
  if(is.null(all_results[[name]])) next
  
  tmp <- data.frame(
    
    Outcome = name,
    
    Train_AUC =
      all_results[[name]]$train_auc,
    
    CV_AUC =
      all_results[[name]]$cv_auc,
    
    Test_AUC =
      all_results[[name]]$test_auc
  )
  
  summary_df <- rbind(
    summary_df,
    tmp
  )
}

write.csv(
  summary_df,
  file.path(
    result_dir,
    "All_XGBoost_Results_Summary.csv"
  ),
  row.names = FALSE
)

##############################
# 9. 结束
##############################

cat("\n====================================================\n")
cat("✅ 所有XGBoost模型训练完成！\n")
cat("====================================================\n")

cat("\n结果保存位置:\n")
cat(result_dir, "\n")
print(class(at$model$learner$model))


# ====== LightGBM ======

######################################################################
# LightGBM模型训练脚本

######################################################################

rm(list = ls())
gc()

############################################################
# 1. 环境准备
############################################################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "lightgbm",
  "pROC",
  "caret",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

future::plan("sequential")

cat("====================================================\n")
cat(" LightGBM模型训练（最终SCI统一终稿版）\n")
cat("====================================================\n")

############################################################
# 2. 创建结果文件夹
############################################################

result_dir <- "LightGBM_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

############################################################
# 3. 数据读取
############################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  
  stop("❌ test_preprocessed.xlsx不存在")
}

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

cat("\n训练集维度:\n")
print(dim(train_data))

cat("\n测试集维度:\n")
print(dim(test_data))

############################################################
# 4. 自动识别Outcome
############################################################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未识别到Outcome变量")
}

cat("\n识别到结局变量:\n")
print(outcome_cols)

############################################################
# 5. 数据类型统一处理
############################################################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.character(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

############################################################
# 6. LightGBM模型函数
############################################################

train_lgb_model <- function(
    train_data,
    test_data,
    outcome_col,
    result_dir,
    n_evals = 50
){
  
  cat("\n")
  cat("====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ####################################################
  # 6.1 特征
  ####################################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################################
  # 6.2 数据集
  ####################################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ####################################################
  # 6.3 outcome转factor
  ####################################################
  
  train_df[[outcome_col]] <- factor(
    train_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  test_df[[outcome_col]] <- factor(
    test_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  ####################################################
  # 6.4 所有特征数值化
  ####################################################
  
  for(col in feature_cols){
    
    if(is.list(train_df[[col]])){
      
      train_df[[col]] <- as.character(unlist(train_df[[col]]))
    }
    
    if(is.list(test_df[[col]])){
      
      test_df[[col]] <- as.character(unlist(test_df[[col]]))
    }
    
    if(is.factor(train_df[[col]]) |
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <- as.numeric(train_df[[col]])
      
      test_df[[col]] <- as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <- as.numeric(train_df[[col]])
    
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
  
  ####################################################
  # 6.5 类别不平衡
  ####################################################
  
  cat("\n训练集类别分布:\n")
  print(table(train_df[[outcome_col]]))
  
  train_y <- as.numeric(train_df[[outcome_col]]) - 1
  
  test_y <- as.numeric(test_df[[outcome_col]]) - 1
  
  pos_num <- sum(train_y == 1)
  
  neg_num <- sum(train_y == 0)
  
  scale_pos_weight <- neg_num / pos_num
  
  cat("\nscale_pos_weight:",
      round(scale_pos_weight,3),
      "\n")
  
  ####################################################
  # 6.6 Matrix
  ####################################################
  
  train_x <- as.matrix(train_df[, feature_cols, with = FALSE])
  
  test_x <- as.matrix(test_df[, feature_cols, with = FALSE])
  
  lgb_train <- lgb.Dataset(
    data = train_x,
    label = train_y
  )
  
  ####################################################
  # 6.7 Random Search参数空间
  ####################################################
  
  search_grid <- data.frame(
    
    learning_rate =
      runif(n_evals, 0.01, 0.03),
    
    num_leaves =
      sample(seq(15,31,2), n_evals, replace = TRUE),
    
    max_depth =
      sample(2:5, n_evals, replace = TRUE),
    
    feature_fraction =
      runif(n_evals, 0.6, 0.8),
    
    bagging_fraction =
      runif(n_evals, 0.6, 0.8),
    
    min_data_in_leaf =
      sample(30:80, n_evals, replace = TRUE),
    
    lambda_l1 =
      runif(n_evals, 2, 15),
    
    lambda_l2 =
      runif(n_evals, 5, 20)
  )
  
  ####################################################
  # 6.8 Random Search + 5-fold CV
  ####################################################
  
  tuning_history <- data.frame()
  
  best_auc <- 0
  
  best_params <- NULL
  
  best_iter <- NULL
  
  cat("\n开始Random Search调参...\n")
  
  for(i in 1:nrow(search_grid)){
    
    cat("\nRandom Search Iteration:",
        i,
        "/",
        nrow(search_grid),
        "\n")
    
    params <- list(
      
      objective = "binary",
      
      metric = "auc",
      
      boosting_type = "gbdt",
      
      learning_rate =
        search_grid$learning_rate[i],
      
      num_leaves =
        search_grid$num_leaves[i],
      
      max_depth =
        search_grid$max_depth[i],
      
      feature_fraction =
        search_grid$feature_fraction[i],
      
      bagging_fraction =
        search_grid$bagging_fraction[i],
      
      bagging_freq = 5,
      
      min_data_in_leaf =
        search_grid$min_data_in_leaf[i],
      
      lambda_l1 =
        search_grid$lambda_l1[i],
      
      lambda_l2 =
        search_grid$lambda_l2[i],
      
      scale_pos_weight =
        scale_pos_weight,
      
      feature_pre_filter = FALSE,
      
      verbosity = -1,
      
      seed = 123
    )
    
    cv_model <- lgb.cv(
      
      params = params,
      
      data = lgb_train,
      
      nrounds = 200,
      
      nfold = 5,
      
      early_stopping_rounds = 20,
      
      stratified = TRUE,
      
      verbose = -1
    )
    
    current_auc <- max(
      unlist(
        cv_model$record_evals$valid$auc$eval
      )
    )
    
    current_iter <- cv_model$best_iter
    
    tuning_history <- rbind(
      
      tuning_history,
      
      cbind(
        Iteration = i,
        search_grid[i,],
        CV_AUC = current_auc,
        Best_Iteration = current_iter
      )
    )
    
    if(current_auc > best_auc){
      
      best_auc <- current_auc
      
      best_params <- params
      
      best_iter <- current_iter
    }
  }
  
  ####################################################
  # 6.9 最终模型
  ####################################################
  
  cat("\n开始最终模型训练...\n")
  
  final_model <- lgb.train(
    
    params = best_params,
    
    data = lgb_train,
    
    nrounds = best_iter,
    
    verbose = -1
  )
  
  ####################################################
  # 6.10 Train预测
  ####################################################
  
  train_prob <- predict(
    final_model,
    train_x
  )
  
  train_roc <- roc(
    train_y,
    train_prob
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ####################################################
  # 6.11 Test预测
  ####################################################
  
  test_prob <- predict(
    final_model,
    test_x
  )
  
  test_roc <- roc(
    test_y,
    test_prob
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ####################################################
  # 6.12 CV AUC
  ####################################################
  
  cv_auc <- best_auc
  
  cv_auc_ci_lower <- quantile(
    tuning_history$CV_AUC,
    0.025
  )
  
  cv_auc_ci_upper <- quantile(
    tuning_history$CV_AUC,
    0.975
  )
  
  ####################################################
  # 6.13 分类指标
  ####################################################
  
  pred_class <- ifelse(test_prob > 0.5,1,0)
  
  cm <- confusionMatrix(
    
    factor(pred_class, levels=c(0,1)),
    
    factor(test_y, levels=c(0,1)),
    
    positive = "1"
  )
  
  ####################################################
  # 6.14 Feature Importance
  ####################################################
  
  importance <- lgb.importance(final_model)
  
  ####################################################
  # 6.15 过拟合诊断
  ####################################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################################
  # 6.16 文件名
  ####################################################
  
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################################
  # 6.17 ROC数据
  ####################################################
  
  roc_df <- data.frame(
    
    specificity =
      rev(test_roc$specificities),
    
    sensitivity =
      rev(test_roc$sensitivities)
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.18 Feature Importance
  ####################################################
  
  write.csv(
    importance,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Feature_Importance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.19 Tuning History
  ####################################################
  
  write.csv(
    tuning_history,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Tuning_History.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.20 Best Parameters
  ####################################################
  
  best_params_df <- data.frame(
    
    Parameter = names(best_params),
    
    Value = unlist(best_params)
  )
  
  write.csv(
    best_params_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Best_Parameters.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.21 Prediction
  ####################################################
  
  prediction_df <- data.frame(
    
    True_Label = test_y,
    
    Predicted_Prob = test_prob,
    
    Predicted_Class = pred_class
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Prediction.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.22 Performance
  ####################################################
  
  metrics_df <- data.frame(
    
    Metric = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "CV_AUC",
      "CV_AUC_CI_Lower",
      "CV_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    Value = c(
      
      train_auc,
      train_auc_ci[1],
      train_auc_ci[3],
      
      cv_auc,
      cv_auc_ci_lower,
      cv_auc_ci_upper,
      
      test_auc,
      test_auc_ci[1],
      test_auc_ci[3],
      
      as.numeric(cm$overall["Accuracy"]),
      
      1-as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  write.csv(
    metrics_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Performance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.23 保存模型
  ####################################################
  
  saveRDS(
    final_model,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_LightGBM_Model.rds"
      )
    )
  )
  
  ####################################################
  # 6.24 最终报告
  ####################################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "LightGBM模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    
    "CV策略: 5-fold Cross Validation\n",
    
    "Random Search迭代次数: ",
    n_evals,
    "\n",
    
    "最终最佳迭代轮数: ",
    best_iter,
    "\n\n",
    
    "================ 最优参数 ================\n",
    
    paste(
      names(best_params),
      unlist(best_params),
      sep = " : ",
      collapse = "\n"
    ),
    
    "\n\n",
    
    "================ 模型性能 ================\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n\n",
    
    "================ 分类指标 ================\n\n",
    
    "Accuracy: ",
    round(as.numeric(cm$overall["Accuracy"]),4),
    "\n",
    
    "CE: ",
    round(
      1-as.numeric(cm$overall["Accuracy"]),
      4
    ),
    "\n",
    
    "Sensitivity: ",
    round(as.numeric(cm$byClass["Sensitivity"]),4),
    "\n",
    
    "Specificity: ",
    round(as.numeric(cm$byClass["Specificity"]),4),
    "\n",
    
    "PPV: ",
    round(as.numeric(cm$byClass["Pos Pred Value"]),4),
    "\n",
    
    "NPV: ",
    round(as.numeric(cm$byClass["Neg Pred Value"]),4),
    "\n",
    
    "F1 Score: ",
    round(as.numeric(cm$byClass["F1"]),4),
    "\n"
  )
  
  writeLines(
    
    report_text,
    
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Report.txt"
      )
    )
  )
  
  cat("\n✓ Report已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc,
    
    metrics = metrics_df
  ))
}

############################################################
# 7. 循环训练
############################################################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_lgb_model(
      
      train_data = train_data,
      
      test_data = test_data,
      
      outcome_col = outcome_col,
      
      result_dir = result_dir,
      
      n_evals = 50
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

############################################################
# 8. 汇总结果
############################################################

summary_df <- data.frame()

for(name in names(all_results)){
  
  if(is.null(all_results[[name]])) next
  
  tmp <- data.frame(
    
    Outcome = name,
    
    Train_AUC =
      all_results[[name]]$train_auc,
    
    CV_AUC =
      all_results[[name]]$cv_auc,
    
    Test_AUC =
      all_results[[name]]$test_auc
  )
  
  summary_df <- rbind(
    summary_df,
    tmp
  )
}

write.csv(
  
  summary_df,
  
  file.path(
    result_dir,
    "All_LightGBM_Results_Summary.csv"
  ),
  
  row.names = FALSE
)

############################################################
# 9. 结束
############################################################

cat("\n====================================================\n")
cat("✅ 所有LightGBM模型训练完成！\n")
cat("====================================================\n")

cat("\n结果保存位置:\n")
cat(result_dir, "\n")


# ====== GBM ======

######################################################################
# GBM模型训练脚本

######################################################################

rm(list = ls())
gc()

############################################################
# 1. 环境准备
############################################################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "gbm",
  "pROC",
  "caret",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

future::plan("sequential")

cat("====================================================\n")
cat(" GBM模型训练（最终SCI终稿版）\n")
cat("====================================================\n")

############################################################
# 2. 创建结果文件夹
############################################################

result_dir <- "GBM_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

############################################################
# 3. 数据读取
############################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  
  stop("❌ test_preprocessed.xlsx不存在")
}

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

cat("\n训练集维度:\n")
print(dim(train_data))

cat("\n测试集维度:\n")
print(dim(test_data))

############################################################
# 4. 自动识别Outcome
############################################################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未识别到Outcome变量")
}

cat("\n识别到结局变量:\n")
print(outcome_cols)

############################################################
# 5. 数据类型统一处理
############################################################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.character(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

############################################################
# 6. GBM模型函数
############################################################

train_gbm_model <- function(
    train_data,
    test_data,
    outcome_col,
    result_dir,
    n_evals = 50
){
  
  cat("\n")
  cat("====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ####################################################
  # 6.1 特征
  ####################################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################################
  # 6.2 数据集
  ####################################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ####################################################
  # 6.3 修复变量名
  ####################################################
  
  clean_feature_names <- make.names(
    feature_cols,
    unique = TRUE
  )
  
  setnames(
    train_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  setnames(
    test_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  feature_cols <- clean_feature_names
  
  ####################################################
  # 6.4 outcome必须numeric 0/1
  ####################################################
  
  train_df[[outcome_col]] <-
    as.numeric(as.character(train_df[[outcome_col]]))
  
  test_df[[outcome_col]] <-
    as.numeric(as.character(test_df[[outcome_col]]))
  
  ####################################################
  # 6.5 特征数值化
  ####################################################
  
  for(col in feature_cols){
    
    if(is.factor(train_df[[col]]) |
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <- as.numeric(train_df[[col]])
      
      test_df[[col]] <- as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <- as.numeric(train_df[[col]])
    
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
  
  ####################################################
  # 6.6 删除近零方差变量
  ####################################################
  
  nzv <- nearZeroVar(
    train_df[, feature_cols, with = FALSE]
  )
  
  if(length(nzv) > 0){
    
    remove_cols <- feature_cols[nzv]
    
    feature_cols <- setdiff(
      feature_cols,
      remove_cols
    )
    
    train_df <- train_df[
      ,
      c(feature_cols, outcome_col),
      with = FALSE
    ]
    
    test_df <- test_df[
      ,
      c(feature_cols, outcome_col),
      with = FALSE
    ]
  }
  
  ####################################################
  # 6.7 标签
  ####################################################
  
  train_y <- train_df[[outcome_col]]
  
  test_y <- test_df[[outcome_col]]
  
  ####################################################
  # 6.8 Random Search空间
  ####################################################
  
  search_grid <- data.frame(
    
    n.trees =
      sample(seq(100,400,50),
             n_evals,
             replace = TRUE),
    
    interaction.depth =
      sample(2:5,
             n_evals,
             replace = TRUE),
    
    shrinkage =
      runif(n_evals,
            0.01,
            0.03),
    
    n.minobsinnode =
      sample(30:80,
             n_evals,
             replace = TRUE),
    
    bag.fraction =
      runif(n_evals,
            0.6,
            0.8)
  )
  
  ####################################################
  # 6.9 Random Search + 5-fold CV
  ####################################################
  
  tuning_history <- data.frame()
  
  best_auc <- 0
  
  best_params <- NULL
  
  cat("\n开始Random Search调参...\n")
  
  folds <- createFolds(
    train_y,
    k = 5,
    list = TRUE
  )
  
  for(i in 1:nrow(search_grid)){
    
    cat("\nRandom Search Iteration:",
        i,
        "/",
        nrow(search_grid),
        "\n")
    
    current_params <- search_grid[i, ]
    
    fold_auc <- c()
    
    for(f in 1:5){
      
      valid_idx <- folds[[f]]
      
      train_idx <- setdiff(
        1:nrow(train_df),
        valid_idx
      )
      
      fold_train <- train_df[train_idx]
      
      fold_valid <- train_df[valid_idx]
      
      gbm_model <- gbm(
        
        formula =
          as.formula(
            paste0(
              "`",
              outcome_col,
              "` ~ ."
            )
          ),
        
        data = fold_train,
        
        distribution = "bernoulli",
        
        n.trees =
          current_params$n.trees,
        
        interaction.depth =
          current_params$interaction.depth,
        
        shrinkage =
          current_params$shrinkage,
        
        n.minobsinnode =
          current_params$n.minobsinnode,
        
        bag.fraction =
          current_params$bag.fraction,
        
        train.fraction = 1,
        
        cv.folds = 0,
        
        verbose = FALSE
      )
      
      valid_pred <- predict(
        gbm_model,
        newdata = fold_valid,
        n.trees = current_params$n.trees,
        type = "response"
      )
      
      valid_roc <- roc(
        fold_valid[[outcome_col]],
        valid_pred,
        quiet = TRUE
      )
      
      fold_auc <- c(
        fold_auc,
        as.numeric(valid_roc$auc)
      )
    }
    
    current_auc <- mean(fold_auc)
    
    tuning_history <- rbind(
      tuning_history,
      cbind(
        Iteration = i,
        current_params,
        CV_AUC = current_auc
      )
    )
    
    if(current_auc > best_auc){
      
      best_auc <- current_auc
      
      best_params <- current_params
    }
  }
  
  ####################################################
  # 6.10 最终模型
  ####################################################
  
  cat("\n开始最终模型训练...\n")
  
  final_model <- gbm(
    
    formula =
      as.formula(
        paste0(
          "`",
          outcome_col,
          "` ~ ."
        )
      ),
    
    data = train_df,
    
    distribution = "bernoulli",
    
    n.trees = best_params$n.trees,
    
    interaction.depth =
      best_params$interaction.depth,
    
    shrinkage =
      best_params$shrinkage,
    
    n.minobsinnode =
      best_params$n.minobsinnode,
    
    bag.fraction =
      best_params$bag.fraction,
    
    train.fraction = 1,
    
    cv.folds = 0,
    
    verbose = FALSE
  )
  
  ####################################################
  # 6.11 Train预测
  ####################################################
  
  train_prob <- predict(
    final_model,
    newdata = train_df,
    n.trees = best_params$n.trees,
    type = "response"
  )
  
  train_roc <- roc(
    train_y,
    train_prob,
    quiet = TRUE
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ####################################################
  # 6.12 Test预测
  ####################################################
  
  test_prob <- predict(
    final_model,
    newdata = test_df,
    n.trees = best_params$n.trees,
    type = "response"
  )
  
  test_roc <- roc(
    test_y,
    test_prob,
    quiet = TRUE
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ####################################################
  # 6.13 CV AUC
  ####################################################
  
  cv_auc <- best_auc
  
  cv_auc_ci_lower <- quantile(
    tuning_history$CV_AUC,
    0.025
  )
  
  cv_auc_ci_upper <- quantile(
    tuning_history$CV_AUC,
    0.975
  )
  
  ####################################################
  # 6.14 最佳Cutoff
  ####################################################
  
  best_cutoff <- as.numeric(
    coords(
      test_roc,
      "best",
      ret = "threshold"
    )
  )
  
  pred_class <- ifelse(
    test_prob > best_cutoff,
    1,
    0
  )
  
  ####################################################
  # 6.15 分类指标
  ####################################################
  
  cm <- confusionMatrix(
    
    factor(pred_class, levels=c(0,1)),
    
    factor(test_y, levels=c(0,1)),
    
    positive = "1"
  )
  
  ####################################################
  # 6.16 Feature Importance
  ####################################################
  
  importance_df <- summary(
    final_model,
    plotit = FALSE
  )
  
  ####################################################
  # 6.17 过拟合诊断
  ####################################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################################
  # 6.18 文件名
  ####################################################
  
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################################
  # 6.19 ROC数据
  ####################################################
  
  roc_df <- data.frame(
    
    specificity =
      rev(test_roc$specificities),
    
    sensitivity =
      rev(test_roc$sensitivities)
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.20 Feature Importance
  ####################################################
  
  write.csv(
    importance_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Feature_Importance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.21 Tuning History
  ####################################################
  
  write.csv(
    tuning_history,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Tuning_History.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.22 Best Parameters
  ####################################################
  
  best_params_df <- data.frame(
    
    Parameter = names(best_params),
    
    Value = unlist(best_params)
  )
  
  write.csv(
    best_params_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Best_Parameters.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.23 Prediction
  ####################################################
  
  prediction_df <- data.frame(
    
    True_Label = test_y,
    
    Predicted_Prob = test_prob,
    
    Predicted_Class = pred_class
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Prediction.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.24 Performance
  ####################################################
  
  metrics_df <- data.frame(
    
    Metric = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "CV_AUC",
      "CV_AUC_CI_Lower",
      "CV_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Best_Cutoff",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    Value = c(
      
      train_auc,
      train_auc_ci[1],
      train_auc_ci[3],
      
      cv_auc,
      cv_auc_ci_lower,
      cv_auc_ci_upper,
      
      test_auc,
      test_auc_ci[1],
      test_auc_ci[3],
      
      best_cutoff,
      
      as.numeric(cm$overall["Accuracy"]),
      
      1-as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  write.csv(
    metrics_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Performance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.25 保存模型
  ####################################################
  
  saveRDS(
    final_model,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_GBM_Model.rds"
      )
    )
  )
  
  ####################################################
  # 6.26 最终报告
  ####################################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "GBM模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    "CV策略: 5-fold Cross Validation\n",
    "Random Search迭代次数: ",
    n_evals,
    "\n\n",
    
    "================ 最优参数 ================\n",
    
    paste(
      names(best_params),
      unlist(best_params),
      sep = " : ",
      collapse = "\n"
    ),
    
    "\n\n",
    
    "================ 模型性能 ================\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n\n"
  )
  
  writeLines(
    
    report_text,
    
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Report.txt"
      )
    )
  )
  
  cat("\n✓ Report已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc,
    
    metrics = metrics_df
  ))
}

############################################################
# 7. 循环训练
############################################################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_gbm_model(
      
      train_data = train_data,
      test_data = test_data,
      outcome_col = outcome_col,
      result_dir = result_dir,
      n_evals = 50
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

############################################################
# 8. 汇总结果
############################################################

summary_df <- data.frame()

for(name in names(all_results)){
  
  if(is.null(all_results[[name]])) next
  
  tmp <- data.frame(
    
    Outcome = name,
    
    Train_AUC =
      all_results[[name]]$train_auc,
    
    CV_AUC =
      all_results[[name]]$cv_auc,
    
    Test_AUC =
      all_results[[name]]$test_auc
  )
  
  summary_df <- rbind(
    summary_df,
    tmp
  )
}

write.csv(
  
  summary_df,
  
  file.path(
    result_dir,
    "All_GBM_Results_Summary.csv"
  ),
  
  row.names = FALSE
)

############################################################
# 9. 结束
############################################################

cat("\n====================================================\n")
cat("✅ 所有GBM模型训练完成！\n")
cat("====================================================\n")

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

# ====== Logistic Regression ======

######################################################################
# Logistic Regression模型训练脚本
######################################################################

rm(list = ls())
gc()

##############################
# 1. 环境准备
##############################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "pROC",
  "caret"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

cat("====================================================\n")
cat(" Logistic Regression模型训练（最终SCI审稿版）\n")
cat("====================================================\n")

##############################
# 2. 创建结果文件夹
##############################

result_dir <- "Logistic_Regression_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

##############################
# 3. 数据读取
##############################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  
  stop("❌ test_preprocessed.xlsx不存在")
}

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

cat("\n训练集维度:\n")
print(dim(train_data))

cat("\n测试集维度:\n")
print(dim(test_data))

##############################
# 4. 自动识别Outcome
##############################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未识别到Outcome变量")
}

cat("\n识别到结局变量:\n")
print(outcome_cols)

##############################
# 5. 数据类型统一处理
##############################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

##############################
# 6. Logistic Regression模型函数
##############################

train_logistic_model <- function(
    train_data,
    test_data,
    outcome_col
){
  
  cat("\n")
  cat("====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ####################################
  # 6.1 特征
  ####################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################
  # 6.2 数据集
  ####################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ####################################
  # 6.3 Outcome数值化
  ####################################
  
  train_df[[outcome_col]] <-
    as.numeric(as.character(train_df[[outcome_col]]))
  
  test_df[[outcome_col]] <-
    as.numeric(as.character(test_df[[outcome_col]]))
  
  ####################################
  # 6.4 特征名清洗
  ####################################
  
  clean_feature_names <- make.names(
    feature_cols,
    unique = TRUE
  )
  
  setnames(
    train_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  setnames(
    test_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  feature_cols <- clean_feature_names
  
  ####################################
  # 6.5 所有特征数值化
  ####################################
  
  for(col in feature_cols){
    
    if(is.list(train_df[[col]])){
      
      train_df[[col]] <- as.character(unlist(train_df[[col]]))
    }
    
    if(is.list(test_df[[col]])){
      
      test_df[[col]] <- as.character(unlist(test_df[[col]]))
    }
    
    if(is.factor(train_df[[col]]) |
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <- as.numeric(train_df[[col]])
      
      test_df[[col]] <- as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <- as.numeric(train_df[[col]])
    
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
  
  ####################################
  # 6.6 标准化
  # 仅基于训练集拟合
  # 防止信息泄露
  ####################################
  
  preproc <- preProcess(
    train_df[, feature_cols, with = FALSE],
    method = c("center", "scale")
  )
  
  train_scaled <- predict(
    preproc,
    train_df[, feature_cols, with = FALSE]
  )
  
  test_scaled <- predict(
    preproc,
    test_df[, feature_cols, with = FALSE]
  )
  
  train_df[, (feature_cols) := train_scaled]
  
  test_df[, (feature_cols) := test_scaled]
  
  ####################################
  # 6.7 构建公式
  ####################################
  
  safe_outcome <- paste0("`", outcome_col, "`")
  
  safe_features <- paste0("`", feature_cols, "`")
  
  formula_str <- paste(
    safe_outcome,
    "~",
    paste(safe_features, collapse = " + ")
  )
  
  formula_obj <- as.formula(formula_str)
  
  ####################################
  # 6.8 5-fold CV
  ####################################
  
  folds <- createFolds(
    train_df[[outcome_col]],
    k = 5,
    list = TRUE
  )
  
  ####################################
  # 6.9 CV性能评估
  ####################################
  
  cat("\n开始5-fold CV评估...\n")
  
  cv_auc_vector <- c()
  
  for(f in 1:5){
    
    cat(
      "\nCV Fold:",
      f,
      "/5\n"
    )
    
    valid_idx <- folds[[f]]
    
    train_idx <- setdiff(
      1:nrow(train_df),
      valid_idx
    )
    
    fold_train <- train_df[train_idx]
    
    fold_valid <- train_df[valid_idx]
    
    glm_model_cv <- glm(
      formula_obj,
      data = fold_train,
      family = binomial()
    )
    
    valid_prob <- predict(
      glm_model_cv,
      newdata = fold_valid,
      type = "response"
    )
    
    valid_roc <- roc(
      fold_valid[[outcome_col]],
      valid_prob,
      quiet = TRUE
    )
    
    cv_auc_vector <- c(
      cv_auc_vector,
      as.numeric(valid_roc$auc)
    )
  }
  
  cv_auc <- mean(cv_auc_vector)
  
  cv_auc_ci_lower <- quantile(
    cv_auc_vector,
    0.025
  )
  
  cv_auc_ci_upper <- quantile(
    cv_auc_vector,
    0.975
  )
  
  ####################################
  # 6.10 最终模型训练
  ####################################
  
  cat("\n开始最终模型训练...\n")
  
  final_model <- glm(
    formula_obj,
    data = train_df,
    family = binomial()
  )
  
  ####################################
  # 6.11 Train预测
  ####################################
  
  train_prob <- predict(
    final_model,
    newdata = train_df,
    type = "response"
  )
  
  train_roc <- roc(
    train_df[[outcome_col]],
    train_prob
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ####################################
  # 6.12 Test预测
  ####################################
  
  test_prob <- predict(
    final_model,
    newdata = test_df,
    type = "response"
  )
  
  test_roc <- roc(
    test_df[[outcome_col]],
    test_prob
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ####################################
  # 6.13 分类结果
  ####################################
  
  pred_class <- ifelse(
    test_prob > 0.5,
    1,
    0
  )
  
  cm <- confusionMatrix(
    factor(pred_class, levels = c(0,1)),
    factor(test_df[[outcome_col]], levels = c(0,1)),
    positive = "1"
  )
  
  ####################################
  # 6.14 过拟合诊断
  ####################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################
  # 6.15 性能指标
  ####################################
  
  metrics_df <- data.frame(
    
    指标 = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "CV_AUC",
      "CV_AUC_CI_Lower",
      "CV_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    数值 = c(
      
      train_auc,
      train_auc_ci[1],
      train_auc_ci[3],
      
      cv_auc,
      cv_auc_ci_lower,
      cv_auc_ci_upper,
      
      test_auc,
      test_auc_ci[1],
      test_auc_ci[3],
      
      as.numeric(cm$overall["Accuracy"]),
      
      1 - as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  ####################################
  # 6.16 输出性能
  ####################################
  
  cat("\n【模型性能】\n")
  
  print(metrics_df)
  
  ####################################
  # 6.17 特征重要性
  ####################################
  
  coef_df <- data.frame(
    
    Feature = names(coef(final_model)),
    
    Coefficient = coef(final_model)
  )
  
  coef_df$Abs_Coefficient <-
    abs(coef_df$Coefficient)
  
  coef_df <- coef_df[
    order(-coef_df$Abs_Coefficient),
  ]
  
  ####################################
  # 6.18 文件名
  ####################################
  
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################
  # 6.19 保存性能
  ####################################
  
  write.csv(
    metrics_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_performance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.20 保存预测结果
  ####################################
  
  prediction_df <- data.frame(
    
    True_Label = test_df[[outcome_col]],
    
    Predicted_Class = pred_class,
    
    Predicted_Prob = test_prob
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_prediction.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.21 保存ROC数据
  ####################################
  
  roc_df <- data.frame(
    
    Thresholds = test_roc$thresholds,
    
    Sensitivity = test_roc$sensitivities,
    
    Specificity = test_roc$specificities,
    
    FPR = 1 - test_roc$specificities,
    
    TPR = test_roc$sensitivities
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.22 保存CV结果
  ####################################
  
  cv_df <- data.frame(
    
    Fold = 1:5,
    
    CV_AUC = cv_auc_vector
  )
  
  write.csv(
    cv_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_CV_Results.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.23 保存特征重要性
  ####################################
  
  write.csv(
    coef_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Feature_Importance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 6.24 保存模型
  ####################################
  
  saveRDS(
    final_model,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_logistic_model.rds"
      )
    )
  )
  
  ####################################
  # 6.25 保存报告
  ####################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "Logistic Regression模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "模型类型: Traditional Logistic Regression (glm)\n",
    
    "调参方法: None\n",
    
    "原因: 传统Logistic Regression不存在需要优化的超参数\n\n",
    
    "CV策略: 5-fold Cross Validation\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n\n",
    
    "Accuracy: ",
    round(as.numeric(cm$overall["Accuracy"]),4),
    "\n",
    
    "CE: ",
    round(
      1-as.numeric(cm$overall["Accuracy"]),
      4
    ),
    "\n",
    
    "Sensitivity: ",
    round(as.numeric(cm$byClass["Sensitivity"]),4),
    "\n",
    
    "Specificity: ",
    round(as.numeric(cm$byClass["Specificity"]),4),
    "\n",
    
    "F1: ",
    round(as.numeric(cm$byClass["F1"]),4),
    "\n"
  )
  
  writeLines(
    report_text,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_report.txt"
      )
    )
  )
  
  cat("\n✓ 文件已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc,
    
    metrics = metrics_df
  ))
}

##############################
# 7. 循环训练
##############################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_logistic_model(
      train_data,
      test_data,
      outcome_col
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

##############################
# 8. 汇总结果
##############################

summary_df <- data.frame()

for(name in names(all_results)){
  
  if(is.null(all_results[[name]])) next
  
  tmp <- data.frame(
    
    Outcome = name,
    
    Train_AUC =
      all_results[[name]]$train_auc,
    
    CV_AUC =
      all_results[[name]]$cv_auc,
    
    Test_AUC =
      all_results[[name]]$test_auc
  )
  
  summary_df <- rbind(
    summary_df,
    tmp
  )
}

write.csv(
  summary_df,
  file.path(
    result_dir,
    "All_Logistic_Regression_Results_Summary.csv"
  ),
  row.names = FALSE
)

##############################
# 9. 结束
##############################

cat("\n====================================================\n")
cat("✅ 所有Logistic Regression模型训练完成！\n")
cat("====================================================\n")

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

# ====== GLMnet模型 ======

rm(list = ls())
gc()

############################################################
# GLMnet模型
############################################################

############################
# 1. 加载包
############################
required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "glmnet",
  "pROC",
  "caret",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

############################
# 2. 设置
############################
set.seed(123)

future::plan("sequential")

cat("====================================================\n")
cat(" GLMnet模型训练（最终SCI统一终稿版）\n")
cat("====================================================\n")

############################
# 3. 结果文件夹
############################
result_dir <- "GLMnet_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

############################
# 4. 导入数据
############################
train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

############################
# 5. Outcome列
############################
outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

############################
# 6. 修复数据类型
############################
fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

############################################################
# 7. GLMnet训练函数
############################################################
train_glmnet_model <- function(
    train_data,
    test_data,
    outcome_col,
    result_dir,
    n_evals = 50
){
  
  cat("\n====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ##########################################################
  # 特征
  ##########################################################
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ##########################################################
  # 数据
  ##########################################################
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ##########################################################
  # 修正变量名
  ##########################################################
  clean_feature_names <- make.names(
    feature_cols,
    unique = TRUE
  )
  
  setnames(
    train_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  setnames(
    test_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  feature_cols <- clean_feature_names
  
  ##########################################################
  # Outcome
  ##########################################################
  train_df[[outcome_col]] <-
    as.numeric(as.character(train_df[[outcome_col]]))
  
  test_df[[outcome_col]] <-
    as.numeric(as.character(test_df[[outcome_col]]))
  
  ##########################################################
  # 特征转数值
  ##########################################################
  for(col in feature_cols){
    
    if(is.factor(train_df[[col]]) ||
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <-
        as.numeric(train_df[[col]])
      
      test_df[[col]] <-
        as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <-
      as.numeric(train_df[[col]])
    
    test_df[[col]] <-
      as.numeric(test_df[[col]])
  }
  
  ##########################################################
  # 删除零方差变量
  ##########################################################
  nzv <- nearZeroVar(
    train_df[, feature_cols, with = FALSE]
  )
  
  if(length(nzv) > 0){
    
    remove_cols <- feature_cols[nzv]
    
    feature_cols <- setdiff(
      feature_cols,
      remove_cols
    )
    
    train_df <- train_df[, c(feature_cols, outcome_col), with = FALSE]
    
    test_df <- test_df[, c(feature_cols, outcome_col), with = FALSE]
  }
  
  ##########################################################
  # y（关键修复）
  ##########################################################
  train_y <- as.numeric(
    unlist(train_df[[outcome_col]])
  )
  
  test_y <- as.numeric(
    unlist(test_df[[outcome_col]])
  )
  
  ##########################################################
  # matrix（关键修复）
  ##########################################################
  train_matrix <- as.matrix(
    train_df[, feature_cols, with = FALSE]
  )
  
  test_matrix <- as.matrix(
    test_df[, feature_cols, with = FALSE]
  )
  
  storage.mode(train_matrix) <- "double"
  
  storage.mode(test_matrix) <- "double"
  
  ##########################################################
  # 检查维度（关键修复）
  ##########################################################
  cat("\n===== CHECK DIM =====\n")
  
  cat("train_matrix rows:", nrow(train_matrix), "\n")
  
  cat("train_y length:", length(train_y), "\n")
  
  cat("test_matrix rows:", nrow(test_matrix), "\n")
  
  cat("test_y length:", length(test_y), "\n")
  
  ##########################################################
  # Random Search
  ##########################################################
  search_grid <- data.frame(
    
    alpha =
      runif(n_evals, 0, 1),
    
    lambda =
      10^runif(n_evals, -4, 0)
  )
  
  ##########################################################
  # CV
  ##########################################################
  tuning_history <- data.frame()
  
  best_auc <- 0
  
  best_params <- NULL
  
  folds <- createFolds(
    train_y,
    k = 5,
    list = TRUE
  )
  
  cat("\n开始Random Search调参...\n")
  
  for(i in 1:nrow(search_grid)){
    
    cat("\nRandom Search Iteration:",
        i,
        "/",
        nrow(search_grid),
        "\n")
    
    current_params <- search_grid[i,]
    
    fold_auc <- c()
    
    for(f in 1:5){
      
      valid_idx <- folds[[f]]
      
      train_idx <- setdiff(
        1:nrow(train_matrix),
        valid_idx
      )
      
      x_train <- train_matrix[train_idx, ]
      
      y_train <- train_y[train_idx]
      
      x_valid <- train_matrix[valid_idx, ]
      
      y_valid <- train_y[valid_idx]
      
      glmnet_model <- glmnet(
        
        x = x_train,
        
        y = y_train,
        
        family = "binomial",
        
        alpha = current_params$alpha,
        
        lambda = current_params$lambda,
        
        standardize = TRUE
      )
      
      valid_pred <- predict(
        
        glmnet_model,
        
        newx = x_valid,
        
        s = current_params$lambda,
        
        type = "response"
      )
      
      valid_roc <- roc(
        y_valid,
        as.vector(valid_pred),
        quiet = TRUE
      )
      
      fold_auc <- c(
        fold_auc,
        as.numeric(valid_roc$auc)
      )
    }
    
    current_auc <- mean(fold_auc)
    
    tuning_history <- rbind(
      tuning_history,
      cbind(
        Iteration = i,
        current_params,
        CV_AUC = current_auc
      )
    )
    
    if(current_auc > best_auc){
      
      best_auc <- current_auc
      
      best_params <- as.list(current_params)
    }
  }
  
  ##########################################################
  # 最终模型
  ##########################################################
  cat("\n开始最终模型训练...\n")
  
  final_model <- glmnet(
    
    x = train_matrix,
    
    y = train_y,
    
    family = "binomial",
    
    alpha = best_params$alpha,
    
    lambda = best_params$lambda,
    
    standardize = TRUE
  )
  
  ##########################################################
  # Train
  ##########################################################
  train_prob <- predict(
    
    final_model,
    
    newx = train_matrix,
    
    s = best_params$lambda,
    
    type = "response"
  )
  
  train_prob <- as.vector(train_prob)
  
  train_roc <- roc(
    train_y,
    train_prob,
    quiet = TRUE
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ##########################################################
  # Test
  ##########################################################
  test_prob <- predict(
    
    final_model,
    
    newx = test_matrix,
    
    s = best_params$lambda,
    
    type = "response"
  )
  
  test_prob <- as.vector(test_prob)
  
  test_roc <- roc(
    test_y,
    test_prob,
    quiet = TRUE
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ##########################################################
  # CV
  ##########################################################
  cv_auc <- best_auc
  
  cv_auc_ci_lower <- quantile(
    tuning_history$CV_AUC,
    0.025
  )
  
  cv_auc_ci_upper <- quantile(
    tuning_history$CV_AUC,
    0.975
  )
  
  ##########################################################
  # cutoff
  ##########################################################
  best_cut <- coords(
    test_roc,
    "best",
    ret = "threshold"
  )[[1]]
  
  pred_class <- ifelse(
    test_prob > best_cut,
    1,
    0
  )
  
  ##########################################################
  # confusion matrix
  ##########################################################
  cm <- confusionMatrix(
    factor(pred_class, levels = c(0,1)),
    factor(test_y, levels = c(0,1)),
    positive = "1"
  )
  
  ##########################################################
  # importance
  ##########################################################
  coef_matrix <- as.matrix(
    coef(
      final_model,
      s = best_params$lambda
    )
  )
  
  importance_df <- data.frame(
    Feature = rownames(coef_matrix),
    Coefficient = coef_matrix[,1]
  )
  
  importance_df$Abs_Coefficient <-
    abs(importance_df$Coefficient)
  
  importance_df <- importance_df[
    order(-importance_df$Abs_Coefficient),
  ]
  
  ##########################################################
  # 过拟合
  ##########################################################
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ##########################################################
  # 文件名
  ##########################################################
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ##########################################################
  # 保存 Feature Importance
  ##########################################################
  write.csv(
    importance_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Feature_Importance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # 保存 Tuning History
  ##########################################################
  write.csv(
    tuning_history,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Tuning_History.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # 保存 Prediction
  ##########################################################
  prediction_df <- data.frame(
    
    Actual = as.numeric(test_y),
    
    Predicted_Probability = as.vector(test_prob),
    
    Predicted_Class = as.numeric(pred_class)
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Predictions.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # 保存 ROC
  ##########################################################
  min_len <- min(
    length(test_roc$sensitivities),
    length(test_roc$specificities)
  )
  
  roc_df <- data.frame(
    Sensitivity = test_roc$sensitivities[1:min_len],
    Specificity = test_roc$specificities[1:min_len]
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Report
  ##########################################################
  report_text <- paste0(
    
    "====================================================\n",
    "GLMnet模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    
    "CV策略: 5-fold Cross Validation\n",
    
    "Random Search迭代次数: ",
    n_evals,
    "\n\n",
    
    "================ 最优参数 ================\n",
    
    paste(
      names(best_params),
      unlist(best_params),
      sep = " : ",
      collapse = "\n"
    ),
    
    "\n\n",
    
    "================ 模型性能 ================\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n"
  )
  
  writeLines(
    
    report_text,
    
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Report.txt"
      )
    )
  )
  
  cat("\n✓ Report已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc
  ))
}

############################################################
# 8. 训练所有Outcome
############################################################
all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_glmnet_model(
      
      train_data = train_data,
      
      test_data = test_data,
      
      outcome_col = outcome_col,
      
      result_dir = result_dir,
      
      n_evals = 50
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

cat("\n====================================================\n")
cat(" GLMnet全部模型训练完成\n")
cat("====================================================\n")


# ====== Conditional Random Forest ======

######################################################################
# Conditional Random Forest

######################################################################

rm(list = ls())
gc()

######################################################################
# Packages
######################################################################

pkgs <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "mlr3",
  "mlr3learners",
  "mlr3tuning",
  "mlr3verse",
  "mlr3extralearners",
  "paradox",
  "pROC",
  "caret",
  "partykit",
  "future"
)

for(pkg in pkgs){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

future::plan("sequential")

set.seed(123)

cat("====================================================\n")
cat(" Conditional Random Forest（最终SCI稳定版）\n")
cat("====================================================\n")

######################################################################
# 结果目录
######################################################################

result_dir <- "ConditionalRF_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir, recursive = TRUE)
}

######################################################################
# 数据路径
######################################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  
  stop("❌ test_preprocessed.xlsx不存在")
}

######################################################################
# 读取数据
######################################################################

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

names(train_data) <- make.names(names(train_data))

names(test_data) <- make.names(names(test_data))

######################################################################
# Outcome变量
######################################################################

outcome_cols <- grep(
  "^Outcome",
  names(train_data),
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未识别到Outcome变量")
}

cat("\n识别到Outcome变量:\n")

print(outcome_cols)

######################################################################
# 数据类型修正
######################################################################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- unlist(df[[col]])
    }
    
    if(is.character(df[[col]]) ||
       is.factor(df[[col]])){
      
      df[[col]] <- as.numeric(
        as.factor(df[[col]])
      )
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.numeric(df[[col]])
    }
    
    suppressWarnings({
      
      df[[col]] <- as.numeric(df[[col]])
    })
    
    df[[col]][is.na(df[[col]])] <- 0
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

gc()

######################################################################
# 安全保存函数
######################################################################

safe_write_csv <- function(df, path){
  
  tryCatch({
    
    write.csv(
      as.data.frame(df),
      path,
      row.names = FALSE
    )
    
    cat("✅ 已保存:", basename(path), "\n")
    
  }, error = function(e){
    
    cat("\n❌ 保存失败:\n")
    
    cat(path, "\n")
    
    cat(e$message, "\n")
  })
}

######################################################################
# Conditional RF训练函数
######################################################################

train_cforest_model <- function(
    
  train_data,
  test_data,
  outcome_col,
  result_dir,
  n_evals = 10
){
  
  cat("\n====================================================\n")
  
  cat("当前建模Outcome:", outcome_col, "\n")
  
  cat("====================================================\n")
  
  ####################################################################
  # 特征
  ####################################################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################################################
  # 数据集
  ####################################################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols,
                                 outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols,
                               outcome_col),
                             with = FALSE]
  
  ####################################################################
  # Outcome factor
  ####################################################################
  
  train_df[[outcome_col]] <- factor(
    train_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  test_df[[outcome_col]] <- factor(
    test_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  ####################################################################
  # 删除零方差变量
  ####################################################################
  
  nzv <- nearZeroVar(
    train_df[, feature_cols, with = FALSE]
  )
  
  if(length(nzv) > 0){
    
    remove_cols <- feature_cols[nzv]
    
    cat("\n删除零方差变量:\n")
    
    print(remove_cols)
    
    feature_cols <- setdiff(
      feature_cols,
      remove_cols
    )
  }
  
  ####################################################################
  # Task
  ####################################################################
  
  train_task_df <- copy(train_df)
  
  safe_outcome <- "target"
  
  setnames(
    train_task_df,
    outcome_col,
    safe_outcome
  )
  
  task <- TaskClassif$new(
    id = gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    backend = train_task_df,
    target = safe_outcome,
    positive = "1"
  )
  
  ####################################################################
  # learner
  ####################################################################
  
  learner <- lrn(
    "classif.cforest",
    predict_type = "prob"
  )
  
  ####################################################################
  # 搜索空间
  ####################################################################
  
  upper_mtry <- min(
    10,
    length(feature_cols)
  )
  
  if(upper_mtry < 2){
    
    upper_mtry <- 2
  }
  
  search_space <- ps(
    
    mtry = p_int(
      lower = 2,
      upper = upper_mtry
    ),
    
    ntree = p_int(
      lower = 60,
      upper = 120
    ),
    
    mincriterion = p_dbl(
      lower = 0.90,
      upper = 0.99
    )
  )
  
  ####################################################################
  # AutoTuner
  ####################################################################
  
  at <- auto_tuner(
    
    tuner = tnr("random_search"),
    
    learner = learner,
    
    resampling = rsmp(
      "cv",
      folds = 5
    ),
    
    measure = msr("classif.auc"),
    
    search_space = search_space,
    
    terminator = trm(
      "evals",
      n_evals = n_evals
    ),
    
    store_tuning_instance = FALSE
  )
  
  ####################################################################
  # 开始训练
  ####################################################################
  
  cat("\n开始Random Search调参...\n")
  
  start_time <- Sys.time()
  
  at$train(task)
  
  end_time <- Sys.time()
  
  cat(
    "\n训练耗时:",
    round(
      as.numeric(
        difftime(
          end_time,
          start_time,
          units = "mins"
        )
      ),
      2
    ),
    "分钟\n"
  )
  
  ####################################################################
  # Train预测
  ####################################################################
  
  train_pred <- at$predict(task)
  
  train_prob <- train_pred$prob[, "1"]
  
  train_truth <- factor(
    train_pred$truth,
    levels = c("0","1")
  )
  
  ####################################################################
  # Test预测
  ####################################################################
  
  test_task_df <- copy(test_df)
  
  setnames(
    test_task_df,
    outcome_col,
    safe_outcome
  )
  
  test_task <- TaskClassif$new(
    id = "test",
    backend = test_task_df,
    target = safe_outcome,
    positive = "1"
  )
  
  test_pred <- at$predict(test_task)
  
  test_prob <- test_pred$prob[, "1"]
  
  test_truth <- factor(
    test_pred$truth,
    levels = c("0","1")
  )
  
  ####################################################################
  # ROC
  ####################################################################
  
  train_roc <- roc(
    as.numeric(train_truth)-1,
    train_prob,
    quiet = TRUE
  )
  
  test_roc <- roc(
    as.numeric(test_truth)-1,
    test_prob,
    quiet = TRUE
  )
  
  train_auc <- as.numeric(
    auc(train_roc)
  )
  
  test_auc <- as.numeric(
    auc(test_roc)
  )
  
  ####################################################################
  # CI
  ####################################################################
  
  train_ci <- ci.auc(train_roc)
  
  test_ci <- ci.auc(test_roc)
  
  ####################################################################
  # confusion matrix
  ####################################################################
  
  pred_class <- factor(
    ifelse(test_prob > 0.5,
           "1",
           "0"),
    levels = c("0","1")
  )
  
  cm <- confusionMatrix(
    pred_class,
    test_truth,
    positive = "1"
  )
  
  ####################################################################
  # 过拟合诊断
  ####################################################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    
    auc_gap >= 0.10,
    
    "Severe Overfitting",
    
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################################################
  # 文件名
  ####################################################################
  
  clean_name <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################################################
  # 保存ROC（即时保存）
  ####################################################################
  
  roc_df <- data.frame(
    
    Thresholds = test_roc$thresholds,
    
    Sensitivity = test_roc$sensitivities,
    
    Specificity = test_roc$specificities,
    
    FPR = 1 - test_roc$specificities,
    
    TPR = test_roc$sensitivities
  )
  
  safe_write_csv(
    
    roc_df,
    
    file.path(
      result_dir,
      paste0(
        clean_name,
        "_ROC_Data.csv"
      )
    )
  )
  
  rm(roc_df)
  gc()
  
  ####################################################################
  # metrics（即时保存）
  ####################################################################
  
  metrics_df <- data.frame(
    
    指标 = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    数值 = c(
      
      train_auc,
      train_ci[1],
      train_ci[3],
      
      test_auc,
      test_ci[1],
      test_ci[3],
      
      as.numeric(cm$overall["Accuracy"]),
      
      1 - as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  safe_write_csv(
    
    metrics_df,
    
    file.path(
      result_dir,
      paste0(
        clean_name,
        "_Performance.csv"
      )
    )
  )
  
  rm(metrics_df)
  gc()
  
  ####################################################################
  # prediction（即时保存）
  ####################################################################
  
  prediction_df <- data.frame(
    
    True_Label = test_truth,
    
    Predicted_Class = pred_class,
    
    Predicted_Prob = test_prob
  )
  
  safe_write_csv(
    
    prediction_df,
    
    file.path(
      result_dir,
      paste0(
        clean_name,
        "_Prediction.csv"
      )
    )
  )
  
  rm(prediction_df)
  gc()
  
  ####################################################################
  # 最优参数（即时保存）
  ####################################################################
  
  best_params <- at$learner$param_set$values
  
  best_params_df <- data.frame(
    
    Parameter = names(best_params),
    
    Value = as.character(
      unlist(best_params)
    )
  )
  
  safe_write_csv(
    
    best_params_df,
    
    file.path(
      result_dir,
      paste0(
        clean_name,
        "_Best_Parameters.csv"
      )
    )
  )
  
  rm(best_params_df)
  gc()
  
  ####################################################################
  # 保存模型（更稳定）
  ####################################################################
  
  saveRDS(
    
    at$learner$model,
    
    file.path(
      result_dir,
      paste0(
        clean_name,
        "_ConditionalRF_Model.rds"
      )
    )
  )
  
  gc()
  
  ####################################################################
  # 保存报告（即时保存）
  ####################################################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "Conditional Random Forest模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    
    "CV策略: 5-fold Cross Validation\n",
    
    "迭代次数: 10\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_ci[1],4),
    " - ",
    round(train_ci[3],4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_ci[1],4),
    " - ",
    round(test_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n\n",
    
    "Accuracy: ",
    round(
      as.numeric(cm$overall["Accuracy"]),
      4
    ),
    "\n",
    
    "CE: ",
    round(
      1-as.numeric(cm$overall["Accuracy"]),
      4
    ),
    "\n",
    
    "Sensitivity: ",
    round(
      as.numeric(cm$byClass["Sensitivity"]),
      4
    ),
    "\n",
    
    "Specificity: ",
    round(
      as.numeric(cm$byClass["Specificity"]),
      4
    ),
    "\n",
    
    "F1: ",
    round(
      as.numeric(cm$byClass["F1"]),
      4
    ),
    "\n"
  )
  
  writeLines(
    
    report_text,
    
    file.path(
      result_dir,
      paste0(
        clean_name,
        "_Report.txt"
      )
    )
  )
  
  gc()
  
  ####################################################################
  # 清理内存
  ####################################################################
  
  rm(
    train_pred,
    test_pred,
    train_prob,
    test_prob,
    train_roc,
    test_roc,
    train_ci,
    test_ci,
    cm,
    at
  )
  
  gc()
  
  cat("\n✅ 完成:", outcome_col, "\n")
  
  return(list(
    
    train_auc = train_auc,
    
    test_auc = test_auc
  ))
}

######################################################################
# 循环运行
######################################################################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_cforest_model(
      
      train_data,
      
      test_data,
      
      outcome_col,
      
      result_dir,
      
      n_evals = 10
    ),
    
    error = function(e){
      
      cat("\n❌ Outcome失败:", outcome_col, "\n")
      
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
  
  gc()
}

######################################################################
# 完成
######################################################################

cat("\n====================================================\n")

cat(" Conditional Random Forest全部完成\n")

cat("====================================================\n")


# ====== Random Forest ======

######################################################################
# Random Forest模型训练脚本

######################################################################

rm(list = ls())
gc()

############################################################
# 1. 环境准备
############################################################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "ranger",
  "pROC",
  "caret",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

future::plan("sequential")

cat("====================================================\n")
cat(" Random Forest模型训练（最终SCI终稿稳定版）\n")
cat("====================================================\n")

############################################################
# 2. 创建结果文件夹
############################################################

result_dir <- "RandomForest_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

############################################################
# 3. 数据读取
############################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  
  stop("❌ test_preprocessed.xlsx不存在")
}

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

cat("\n训练集维度:\n")
print(dim(train_data))

cat("\n测试集维度:\n")
print(dim(test_data))

############################################################
# 4. 自动识别Outcome
############################################################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未识别到Outcome变量")
}

cat("\n识别到结局变量:\n")
print(outcome_cols)

############################################################
# 5. 数据类型统一处理
############################################################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.character(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

############################################################
# 6. RF模型函数
############################################################

train_rf_model <- function(
    train_data,
    test_data,
    outcome_col,
    result_dir,
    n_evals = 50
){
  
  cat("\n")
  cat("====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ####################################################
  # 6.1 特征
  ####################################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################################
  # 6.2 数据集
  ####################################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ####################################################
  # 6.3 修复变量名
  ####################################################
  
  clean_feature_names <- make.names(
    feature_cols,
    unique = TRUE
  )
  
  setnames(
    train_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  setnames(
    test_df,
    old = feature_cols,
    new = clean_feature_names
  )
  
  feature_cols <- clean_feature_names
  
  ####################################################
  # 6.4 outcome
  ####################################################
  
  train_df[[outcome_col]] <- factor(
    train_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  test_df[[outcome_col]] <- factor(
    test_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  ####################################################
  # 6.5 特征数值化
  ####################################################
  
  for(col in feature_cols){
    
    if(is.factor(train_df[[col]]) |
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <- as.numeric(train_df[[col]])
      test_df[[col]] <- as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <- as.numeric(train_df[[col]])
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
  
  ####################################################
  # 6.6 删除异常列
  ####################################################
  
  nzv <- nearZeroVar(
    train_df[, feature_cols, with = FALSE]
  )
  
  if(length(nzv) > 0){
    
    remove_cols <- feature_cols[nzv]
    
    feature_cols <- setdiff(
      feature_cols,
      remove_cols
    )
  }
  
  ####################################################
  # 6.7 类别分布
  ####################################################
  
  cat("\n训练集类别分布:\n")
  print(table(train_df[[outcome_col]]))
  
  train_y <- as.numeric(train_df[[outcome_col]]) - 1
  test_y <- as.numeric(test_df[[outcome_col]]) - 1
  
  ####################################################
  # 6.8 防过拟合参数空间（最终稳定版）
  ####################################################
  
  search_grid <- data.frame(
    
    num_trees =
      sample(seq(200,400,50),
             n_evals,
             replace = TRUE),
    
    mtry =
      sample(
        2:max(3, floor(sqrt(length(feature_cols)))),
        n_evals,
        replace = TRUE
      ),
    
    min_node_size =
      sample(30:80,
             n_evals,
             replace = TRUE),
    
    max_depth =
      sample(3:5,
             n_evals,
             replace = TRUE),
    
    sample_fraction =
      runif(n_evals,
            0.5,
            0.7)
  )
  
  ####################################################
  # 6.9 Random Search + 5-fold CV
  ####################################################
  
  tuning_history <- data.frame()
  
  best_auc <- 0
  
  best_params <- NULL
  
  cat("\n开始Random Search调参...\n")
  
  folds <- createFolds(
    train_y,
    k = 5,
    list = TRUE
  )
  
  for(i in 1:nrow(search_grid)){
    
    cat("\nRandom Search Iteration:",
        i,
        "/",
        nrow(search_grid),
        "\n")
    
    current_params <- search_grid[i, ]
    
    fold_auc <- c()
    
    for(f in 1:5){
      
      valid_idx <- folds[[f]]
      
      train_idx <- setdiff(
        1:nrow(train_df),
        valid_idx
      )
      
      fold_train <- train_df[train_idx]
      
      fold_valid <- train_df[valid_idx]
      
      rf_model <- ranger(
        
        formula =
          as.formula(
            paste0(
              "`",
              outcome_col,
              "` ~ ."
            )
          ),
        
        data = fold_train,
        
        probability = TRUE,
        
        num.trees =
          as.numeric(current_params$num_trees),
        
        mtry =
          as.numeric(current_params$mtry),
        
        min.node.size =
          as.numeric(current_params$min_node_size),
        
        max.depth =
          as.numeric(current_params$max_depth),
        
        sample.fraction =
          as.numeric(current_params$sample_fraction),
        
        importance = "permutation",
        
        classification = TRUE,
        
        seed = 123,
        
        verbose = FALSE
      )
      
      valid_pred <- predict(
        rf_model,
        data = fold_valid
      )$predictions[,2]
      
      valid_roc <- roc(
        as.numeric(fold_valid[[outcome_col]]) - 1,
        valid_pred,
        quiet = TRUE
      )
      
      fold_auc <- c(
        fold_auc,
        as.numeric(valid_roc$auc)
      )
    }
    
    current_auc <- mean(fold_auc)
    
    tuning_history <- rbind(
      tuning_history,
      cbind(
        Iteration = i,
        current_params,
        CV_AUC = current_auc
      )
    )
    
    if(current_auc > best_auc){
      
      best_auc <- current_auc
      
      best_params <- current_params
    }
  }
  
  ####################################################
  # 6.10 最终模型
  ####################################################
  
  cat("\n开始最终模型训练...\n")
  
  final_model <- ranger(
    
    formula =
      as.formula(
        paste0(
          "`",
          outcome_col,
          "` ~ ."
        )
      ),
    
    data = train_df,
    
    probability = TRUE,
    
    num.trees =
      as.numeric(best_params$num_trees),
    
    mtry =
      as.numeric(best_params$mtry),
    
    min.node.size =
      as.numeric(best_params$min_node_size),
    
    max.depth =
      as.numeric(best_params$max_depth),
    
    sample.fraction =
      as.numeric(best_params$sample_fraction),
    
    importance = "permutation",
    
    classification = TRUE,
    
    seed = 123,
    
    verbose = FALSE
  )
  
  ####################################################
  # 6.11 Train预测
  ####################################################
  
  train_prob <- predict(
    final_model,
    data = train_df
  )$predictions[,2]
  
  train_roc <- roc(
    train_y,
    train_prob,
    quiet = TRUE
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ####################################################
  # 6.12 Test预测
  ####################################################
  
  test_prob <- predict(
    final_model,
    data = test_df
  )$predictions[,2]
  
  test_roc <- roc(
    test_y,
    test_prob,
    quiet = TRUE
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ####################################################
  # 6.13 CV AUC
  ####################################################
  
  cv_auc <- best_auc
  
  cv_auc_ci_lower <- quantile(
    tuning_history$CV_AUC,
    0.025
  )
  
  cv_auc_ci_upper <- quantile(
    tuning_history$CV_AUC,
    0.975
  )
  
  ####################################################
  # 6.14 最佳Cutoff
  ####################################################
  
  best_cutoff <- as.numeric(
    coords(
      test_roc,
      "best",
      ret = "threshold"
    )
  )
  
  pred_class <- ifelse(
    test_prob > best_cutoff,
    1,
    0
  )
  
  ####################################################
  # 6.15 分类指标
  ####################################################
  
  cm <- confusionMatrix(
    
    factor(pred_class, levels=c(0,1)),
    
    factor(test_y, levels=c(0,1)),
    
    positive = "1"
  )
  
  ####################################################
  # 6.16 Feature Importance
  ####################################################
  
  importance_df <- data.frame(
    
    Feature =
      names(final_model$variable.importance),
    
    Importance =
      as.numeric(final_model$variable.importance)
  )
  
  importance_df <- importance_df[
    order(-importance_df$Importance),
  ]
  
  ####################################################
  # 6.17 过拟合诊断
  ####################################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################################
  # 6.18 文件名
  ####################################################
  
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################################
  # 6.19 ROC数据
  ####################################################
  
  roc_df <- data.frame(
    
    specificity =
      rev(test_roc$specificities),
    
    sensitivity =
      rev(test_roc$sensitivities)
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.20 Feature Importance
  ####################################################
  
  write.csv(
    importance_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Feature_Importance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.21 Tuning History
  ####################################################
  
  write.csv(
    tuning_history,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Tuning_History.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.22 Best Parameters
  ####################################################
  
  best_params_df <- data.frame(
    
    Parameter = names(best_params),
    
    Value = unlist(best_params)
  )
  
  write.csv(
    best_params_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Best_Parameters.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.23 Prediction
  ####################################################
  
  prediction_df <- data.frame(
    
    True_Label = test_y,
    
    Predicted_Prob = test_prob,
    
    Predicted_Class = pred_class
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Prediction.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.24 Performance
  ####################################################
  
  metrics_df <- data.frame(
    
    Metric = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "CV_AUC",
      "CV_AUC_CI_Lower",
      "CV_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Best_Cutoff",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    Value = c(
      
      train_auc,
      train_auc_ci[1],
      train_auc_ci[3],
      
      cv_auc,
      cv_auc_ci_lower,
      cv_auc_ci_upper,
      
      test_auc,
      test_auc_ci[1],
      test_auc_ci[3],
      
      best_cutoff,
      
      as.numeric(cm$overall["Accuracy"]),
      
      1-as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  write.csv(
    metrics_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Performance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################################
  # 6.25 保存模型
  ####################################################
  
  saveRDS(
    final_model,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_RF_Model.rds"
      )
    )
  )
  
  ####################################################
  # 6.26 最终报告
  ####################################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "Random Forest模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    "CV策略: 5-fold Cross Validation\n",
    "Random Search迭代次数: ",
    n_evals,
    "\n\n",
    
    "================ 最优参数 ================\n",
    
    paste(
      names(best_params),
      unlist(best_params),
      sep = " : ",
      collapse = "\n"
    ),
    
    "\n\n",
    
    "================ 模型性能 ================\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n\n",
    
    "================ 分类指标 ================\n\n",
    
    "Best Cutoff: ",
    round(best_cutoff,4),
    "\n",
    
    "Accuracy: ",
    round(as.numeric(cm$overall["Accuracy"]),4),
    "\n",
    
    "CE: ",
    round(
      1-as.numeric(cm$overall["Accuracy"]),
      4
    ),
    "\n",
    
    "Sensitivity: ",
    round(as.numeric(cm$byClass["Sensitivity"]),4),
    "\n",
    
    "Specificity: ",
    round(as.numeric(cm$byClass["Specificity"]),4),
    "\n",
    
    "PPV: ",
    round(as.numeric(cm$byClass["Pos Pred Value"]),4),
    "\n",
    
    "NPV: ",
    round(as.numeric(cm$byClass["Neg Pred Value"]),4),
    "\n",
    
    "F1 Score: ",
    round(as.numeric(cm$byClass["F1"]),4),
    "\n"
  )
  
  writeLines(
    
    report_text,
    
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_Report.txt"
      )
    )
  )
  
  cat("\n✓ Report已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc,
    
    metrics = metrics_df
  ))
}

############################################################
# 7. 循环训练
############################################################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_rf_model(
      
      train_data = train_data,
      test_data = test_data,
      outcome_col = outcome_col,
      result_dir = result_dir,
      n_evals = 50
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

############################################################
# 8. 汇总结果
############################################################

summary_df <- data.frame()

for(name in names(all_results)){
  
  if(is.null(all_results[[name]])) next
  
  tmp <- data.frame(
    
    Outcome = name,
    
    Train_AUC =
      all_results[[name]]$train_auc,
    
    CV_AUC =
      all_results[[name]]$cv_auc,
    
    Test_AUC =
      all_results[[name]]$test_auc
  )
  
  summary_df <- rbind(
    summary_df,
    tmp
  )
}

write.csv(
  
  summary_df,
  
  file.path(
    result_dir,
    "All_RF_Results_Summary.csv"
  ),
  
  row.names = FALSE
)

############################################################
# 9. 结束
############################################################

cat("\n====================================================\n")
cat("✅ 所有Random Forest模型训练完成！\n")
cat("====================================================\n")

cat("\n结果保存位置:\n")
cat(result_dir, "\n")


# ====== MLP ======

######################################################################
# MLP模型训练脚本
######################################################################

rm(list = ls())
gc()

##############################
# 1. 环境准备
##############################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "mlr3",
  "mlr3learners",
  "mlr3tuning",
  "paradox",
  "pROC",
  "caret",
  "future",
  "nnet"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

future::plan("sequential")

cat("====================================================\n")
cat(" MLP模型训练（最终SCI审稿版）\n")
cat("====================================================\n")

##############################
# 2. 创建结果文件夹
##############################

result_dir <- "MLP_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

##############################
# 3. 数据读取
##############################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

if(!file.exists(train_path)){
  stop("❌ train_preprocessed.xlsx不存在")
}

if(!file.exists(test_path)){
  stop("❌ test_preprocessed.xlsx不存在")
}

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

##############################
# 4. 合法化列名（MLP必须）
##############################

original_names <- names(train_data)

safe_names <- make.names(original_names)

name_mapping <- data.frame(
  Original = original_names,
  Safe = safe_names
)

names(train_data) <- safe_names
names(test_data) <- safe_names

cat("\n变量名合法化完成\n")

##############################
# 5. 自动识别Outcome
##############################

outcome_cols <- grep(
  "^Outcome",
  names(train_data),
  value = TRUE
)

cat("\n识别到Outcome变量:\n")
print(outcome_cols)

##############################
# 6. 数据类型统一
##############################

fix_data_types <- function(df){
  
  for(col in names(df)){
    
    if(is.list(df[[col]])){
      
      df[[col]] <- as.character(unlist(df[[col]]))
    }
    
    if(is.character(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
    
    if(is.logical(df[[col]])){
      
      df[[col]] <- as.factor(df[[col]])
    }
  }
  
  return(df)
}

train_data <- fix_data_types(train_data)

test_data <- fix_data_types(test_data)

##############################
# 7. MLP模型函数
##############################

train_mlp_model <- function(
    train_data,
    test_data,
    outcome_col
){
  
  cat("\n")
  cat("====================================================\n")
  cat("当前建模Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  ####################################
  # 7.1 特征
  ####################################
  
  feature_cols <- setdiff(
    names(train_data),
    outcome_cols
  )
  
  ####################################
  # 7.2 数据集
  ####################################
  
  train_df <- copy(train_data)[,
                               c(feature_cols, outcome_col),
                               with = FALSE]
  
  test_df <- copy(test_data)[,
                             c(feature_cols, outcome_col),
                             with = FALSE]
  
  ####################################
  # 7.3 Outcome转factor
  ####################################
  
  train_df[[outcome_col]] <- factor(
    train_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  test_df[[outcome_col]] <- factor(
    test_df[[outcome_col]],
    levels = c(0,1),
    labels = c("0","1")
  )
  
  ####################################
  # 7.4 所有特征数值化
  ####################################
  
  for(col in feature_cols){
    
    if(is.factor(train_df[[col]]) |
       is.character(train_df[[col]])){
      
      all_levels <- union(
        unique(as.character(train_df[[col]])),
        unique(as.character(test_df[[col]]))
      )
      
      train_df[[col]] <- factor(
        as.character(train_df[[col]]),
        levels = all_levels
      )
      
      test_df[[col]] <- factor(
        as.character(test_df[[col]]),
        levels = all_levels
      )
      
      train_df[[col]] <- as.numeric(train_df[[col]])
      
      test_df[[col]] <- as.numeric(test_df[[col]])
    }
    
    train_df[[col]] <- as.numeric(train_df[[col]])
    
    test_df[[col]] <- as.numeric(test_df[[col]])
  }
  
  ####################################
  # 7.5 删除零方差变量
  ####################################
  
  nzv <- nearZeroVar(
    train_df[, feature_cols, with = FALSE],
    saveMetrics = TRUE
  )
  
  remove_cols <- rownames(nzv)[nzv$zeroVar == TRUE]
  
  if(length(remove_cols) > 0){
    
    cat("\n删除零方差变量:\n")
    
    print(remove_cols)
    
    feature_cols <- setdiff(
      feature_cols,
      remove_cols
    )
  }
  
  ####################################
  # 7.6 标准化（仅train拟合）
  ####################################
  
  pre_obj <- preProcess(
    train_df[, feature_cols, with = FALSE],
    method = c("center", "scale")
  )
  
  train_x <- as.data.frame(
    predict(
      pre_obj,
      train_df[, feature_cols, with = FALSE]
    )
  )
  
  test_x <- as.data.frame(
    predict(
      pre_obj,
      test_df[, feature_cols, with = FALSE]
    )
  )
  
  ####################################
  # 7.7 最终数据集
  ####################################
  
  train_df_final <- data.frame(
    train_x,
    outcome = train_df[[outcome_col]]
  )
  
  test_df_final <- data.frame(
    test_x,
    outcome = test_df[[outcome_col]]
  )
  
  ####################################
  # 7.8 类别分布
  ####################################
  
  cat("\n训练集类别分布:\n")
  
  print(table(train_df_final$outcome))
  
  ####################################
  # 7.9 创建Task
  ####################################
  
  task <- TaskClassif$new(
    
    id = gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    
    backend = train_df_final,
    
    target = "outcome",
    
    positive = "1"
  )
  
  ####################################
  # 7.10 MLP Learner
  ####################################
  
  learner <- lrn(
    
    "classif.nnet",
    
    predict_type = "prob",
    
    trace = FALSE,
    
    MaxNWts = 20000,
    
    maxit = 300,
    
    decay = 0.01,
    
    size = 5
  )
  
  ####################################
  # 7.11 调参空间
  ####################################
  
  search_space <- ps(
    
    size = p_int(3,10),
    
    decay = p_dbl(0.001,0.1),
    
    maxit = p_int(100,500)
  )
  
  ####################################
  # 7.12 AutoTuner
  ####################################
  
  at <- auto_tuner(
    
    tuner = tnr("random_search"),
    
    learner = learner,
    
    resampling = rsmp("cv", folds = 5),
    
    measure = msr("classif.auc"),
    
    search_space = search_space,
    
    terminator = trm("evals", n_evals = 50)
  )
  
  ####################################
  # 7.13 模型训练
  ####################################
  
  cat("\n开始Random Search调参...\n")
  
  start_time <- Sys.time()
  
  at$train(task)
  
  end_time <- Sys.time()
  
  cat(
    "\n训练耗时:",
    round(
      as.numeric(
        difftime(
          end_time,
          start_time,
          units = "mins"
        )
      ),
      2
    ),
    "分钟\n"
  )
  
  ####################################
  # 7.14 测试集预测
  ####################################
  
  full_data <- rbind(
    train_df_final,
    test_df_final
  )
  
  test_task <- TaskClassif$new(
    
    id = paste0(
      gsub("[^A-Za-z0-9]",
           "_",
           outcome_col),
      "_test"
    ),
    
    backend = full_data,
    
    target = "outcome",
    
    positive = "1"
  )
  
  pred <- at$predict(
    test_task,
    row_ids =
      (nrow(train_df_final)+1):
      nrow(full_data)
  )
  
  pred_prob <- pred$prob[, "1"]
  
  pred_class <- factor(
    pred$response,
    levels = c("0","1")
  )
  
  true_label <- factor(
    test_df_final$outcome,
    levels = c("0","1")
  )
  
  ####################################
  # 7.15 Test ROC/AUC/CI
  ####################################
  
  test_roc <- roc(
    as.numeric(true_label)-1,
    pred_prob
  )
  
  test_auc <- as.numeric(test_roc$auc)
  
  test_auc_ci <- ci.auc(test_roc)
  
  ####################################
  # 7.16 Train ROC/AUC/CI
  ####################################
  
  train_pred <- at$predict(
    test_task,
    row_ids = 1:nrow(train_df_final)
  )
  
  train_prob <- train_pred$prob[, "1"]
  
  train_label <- factor(
    train_df_final$outcome,
    levels = c("0","1")
  )
  
  train_roc <- roc(
    as.numeric(train_label)-1,
    train_prob
  )
  
  train_auc <- as.numeric(train_roc$auc)
  
  train_auc_ci <- ci.auc(train_roc)
  
  ####################################
  # 7.17 CV AUC
  ####################################
  
  cv_scores <- at$archive$data$classif.auc
  
  cv_auc <- mean(cv_scores, na.rm = TRUE)
  
  cv_auc_ci_lower <- quantile(
    cv_scores,
    probs = 0.025,
    na.rm = TRUE
  )
  
  cv_auc_ci_upper <- quantile(
    cv_scores,
    probs = 0.975,
    na.rm = TRUE
  )
  
  ####################################
  # 7.18 混淆矩阵
  ####################################
  
  cm <- confusionMatrix(
    data = pred_class,
    reference = true_label,
    positive = "1"
  )
  
  ####################################
  # 7.19 过拟合诊断
  ####################################
  
  auc_gap <- train_auc - test_auc
  
  overfit_status <- ifelse(
    auc_gap >= 0.1,
    "Severe Overfitting",
    ifelse(
      auc_gap >= 0.05,
      "Mild Overfitting",
      "Good Generalization"
    )
  )
  
  ####################################
  # 7.20 性能指标
  ####################################
  
  metrics_df <- data.frame(
    
    指标 = c(
      
      "Train_AUC",
      "Train_AUC_CI_Lower",
      "Train_AUC_CI_Upper",
      
      "CV_AUC",
      "CV_AUC_CI_Lower",
      "CV_AUC_CI_Upper",
      
      "Test_AUC",
      "Test_AUC_CI_Lower",
      "Test_AUC_CI_Upper",
      
      "Accuracy",
      "CE",
      "Sensitivity",
      "Specificity",
      "PPV",
      "NPV",
      "F1"
    ),
    
    数值 = c(
      
      train_auc,
      train_auc_ci[1],
      train_auc_ci[3],
      
      cv_auc,
      cv_auc_ci_lower,
      cv_auc_ci_upper,
      
      test_auc,
      test_auc_ci[1],
      test_auc_ci[3],
      
      as.numeric(cm$overall["Accuracy"]),
      
      1 - as.numeric(cm$overall["Accuracy"]),
      
      as.numeric(cm$byClass["Sensitivity"]),
      
      as.numeric(cm$byClass["Specificity"]),
      
      as.numeric(cm$byClass["Pos Pred Value"]),
      
      as.numeric(cm$byClass["Neg Pred Value"]),
      
      as.numeric(cm$byClass["F1"])
    )
  )
  
  ####################################
  # 7.21 输出性能
  ####################################
  
  cat("\n【模型性能】\n")
  
  print(metrics_df)
  
  ####################################
  # 7.22 最优参数
  ####################################
  
  best_params <- at$archive$best()$x_domain[[1]]
  
  best_params_df <- data.frame(
    
    Parameter = names(best_params),
    
    Value = unlist(best_params)
  )
  
  ####################################
  # 7.23 文件名
  ####################################
  
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  ####################################
  # 7.24 保存性能
  ####################################
  
  write.csv(
    metrics_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_performance.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 7.25 保存预测
  ####################################
  
  prediction_df <- data.frame(
    
    True_Label = true_label,
    
    Predicted_Class = pred_class,
    
    Predicted_Prob = pred_prob
  )
  
  write.csv(
    prediction_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_prediction.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 7.26 保存调参历史
  ####################################
  
  tuning_history <- as.data.frame(
    at$archive$data
  )
  
  tuning_history <- tuning_history[,
                                   sapply(
                                     tuning_history,
                                     function(x)
                                       !is.list(x)
                                   )]
  
  write.csv(
    tuning_history,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_tuning_history.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 7.27 保存最优参数
  ####################################
  
  write.csv(
    best_params_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_best_parameters.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 7.28 保存ROC数据
  ####################################
  
  roc_df <- data.frame(
    
    Thresholds = test_roc$thresholds,
    
    Sensitivity = test_roc$sensitivities,
    
    Specificity = test_roc$specificities,
    
    FPR = 1 - test_roc$specificities,
    
    TPR = test_roc$sensitivities
  )
  
  write.csv(
    roc_df,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_ROC_Data.csv"
      )
    ),
    row.names = FALSE
  )
  
  ####################################
  # 7.29 保存模型
  ####################################
  
  saveRDS(
    at,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_mlp_model.rds"
      )
    )
  )
  
  ####################################
  # 7.30 保存报告
  ####################################
  
  report_text <- paste0(
    
    "====================================================\n",
    "MLP模型训练报告\n",
    "====================================================\n\n",
    
    "Outcome: ",
    outcome_col,
    "\n\n",
    
    "调参方法: Random Search\n",
    
    "CV策略: 5-fold Cross Validation\n",
    
    "迭代次数: 50\n\n",
    
    "Train AUC: ",
    round(train_auc,4),
    "\n",
    
    "Train AUC 95%CI: ",
    round(train_auc_ci[1],4),
    " - ",
    round(train_auc_ci[3],4),
    "\n\n",
    
    "CV AUC: ",
    round(cv_auc,4),
    "\n",
    
    "CV AUC 95%CI: ",
    round(cv_auc_ci_lower,4),
    " - ",
    round(cv_auc_ci_upper,4),
    "\n\n",
    
    "Test AUC: ",
    round(test_auc,4),
    "\n",
    
    "Test AUC 95%CI: ",
    round(test_auc_ci[1],4),
    " - ",
    round(test_auc_ci[3],4),
    "\n\n",
    
    "AUC Gap: ",
    round(auc_gap,4),
    "\n",
    
    "Diagnosis: ",
    overfit_status,
    "\n"
  )
  
  writeLines(
    report_text,
    file.path(
      result_dir,
      paste0(
        outcome_name_clean,
        "_report.txt"
      )
    )
  )
  
  cat("\n✓ 文件已保存\n")
  
  return(list(
    
    train_auc = train_auc,
    
    cv_auc = cv_auc,
    
    test_auc = test_auc,
    
    metrics = metrics_df
  ))
}

##############################
# 8. 循环训练
##############################

all_results <- list()

for(outcome_col in outcome_cols){
  
  result <- tryCatch(
    
    train_mlp_model(
      train_data,
      test_data,
      outcome_col
    ),
    
    error = function(e){
      
      cat("\n❌ 模型失败:", outcome_col, "\n")
      
      cat(e$message, "\n")
      
      return(NULL)
    }
  )
  
  all_results[[outcome_col]] <- result
}

##############################
# 9. 汇总结果
##############################

summary_df <- data.frame()

for(name in names(all_results)){
  
  if(is.null(all_results[[name]])) next
  
  tmp <- data.frame(
    
    Outcome = name,
    
    Train_AUC =
      all_results[[name]]$train_auc,
    
    CV_AUC =
      all_results[[name]]$cv_auc,
    
    Test_AUC =
      all_results[[name]]$test_auc
  )
  
  summary_df <- rbind(
    summary_df,
    tmp
  )
}

write.csv(
  summary_df,
  file.path(
    result_dir,
    "All_MLP_Results_Summary.csv"
  ),
  row.names = FALSE
)

##############################
# 10. 结束
##############################

cat("\n====================================================\n")
cat("✅ 所有MLP模型训练完成！\n")
cat("====================================================\n")

cat("\n结果保存位置:\n")
cat(result_dir, "\n")