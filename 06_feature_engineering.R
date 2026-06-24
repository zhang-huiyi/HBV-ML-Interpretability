# 06_feature_engineering.R - SHAP-based interaction feature construction


# ====== Product Interaction ======

######################################################################
# SCI Feature Engineering Script
# Top15 SHAP + SHAP Interaction Engineering
#
# 逻辑：
# Baseline模型（78特征）:
#   ✓ 不重新训练
#   ✓ 直接读取已有结果
#
# Model 1:
#   ✓ SHAP Top15特征重训练
#
# Model 2:
#   ✓ Top15 + SHAP交互特征
#
# 交互工程（SCI升级版）：
#   1. Product
#   2. Ratio
#   3. Log Interaction
#   4. Threshold Interaction
#
# 输出：
#   SCI_Feature_Engineering/
#       ├── Reduced_Model_Data/
#       ├── Interaction_Model_Data/
#       ├── Feature_Reports/
#       └── Interaction_Metadata/
######################################################################

rm(list = ls())
gc()

library(data.table)
library(readxl)

set.seed(123)

cat("====================================================\n")
cat("SCI Feature Engineering Pipeline\n")
cat("====================================================\n")

############################################################
# 1. 路径
############################################################

xgb_dir <- "J:/rtools45/XGBoost_Results"

shap_dir <- "J:/rtools45/SHAP_Results_TOPJOURNAL"

save_dir <- "J:/rtools45/手工4乘积交互Feature_Engineering"

dir.create(save_dir, showWarnings = FALSE)

dir.create(
  file.path(save_dir, "Reduced_Model_Data"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Interaction_Model_Data"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Feature_Reports"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Interaction_Metadata"),
  showWarnings = FALSE
)

############################################################
# 2. 读取原始数据
############################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path  <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

############################################################
# 3. Outcome
############################################################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  value = TRUE
)

print(outcome_cols)

############################################################
# 4. SHAP Top15提取函数
############################################################

extract_top15_features <- function(outcome_col){
  
  outcome_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  shap_file <- file.path(
    shap_dir,
    outcome_clean,
    "SHAP_Importance_Top15.csv"
  )
  
  cat("\n读取SHAP Top15:\n")
  cat(shap_file, "\n")
  
  if(!file.exists(shap_file)){
    
    stop(
      paste0(
        "❌ 文件不存在: ",
        shap_file
      )
    )
  }
  
  shap_imp <- fread(shap_file)
  
  top15 <- shap_imp$Feature[1:15]
  
  return(top15)
}

############################################################
# 5. 指定SCI交互组合（最终版）
############################################################

interaction_mapping <- list()

##########################################################
# Disease Non Progression
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Disease Non-Progression"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Absolute Lymphocyte Count",
    "Platelet Count"
  ),
  
  c(
    "Number of Hospitalizations",
    "Platelet Count"
  ),
  
  c(
    "Plateletcrit",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "C-Reactive Protein (CRP)",
    "Alpha-fetoprotein (AFP)"
  )
)

##########################################################
# Liver Cirrhosis
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Liver Cirrhosis"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Plateletcrit",
    "Platelet Count"
  ),
  
  c(
    "Absolute Lymphocyte Count",
    "Platelet Count"
  ),
  
  c(
    "Admission Interval",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "Cholinesterase",
    "Plateletcrit"
  )
)

##########################################################
# HCC
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Hepatocellular Carcinoma"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Direct Bilirubin",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "Number of Hospitalizations",
    "C-Reactive Protein (CRP)"
  ),
  
   c(
    "Age",
    "C-Reactive Protein (CRP)"
  ),
  
  c(
    "C-Reactive Protein (CRP)",
    "Alpha-fetoprotein (AFP)"
  )
  )

############################################################
# 6. 创建SCI交互特征
############################################################

create_interaction_features <- function(df, pairs){
  
  created_features <- c()
  
  for(pair in pairs){
    
    f1 <- pair[1]
    f2 <- pair[2]
    
    if(!(f1 %in% names(df))) next
    if(!(f2 %in% names(df))) next
    
    x1 <- as.numeric(df[[f1]])
    x2 <- as.numeric(df[[f2]])
    
    ########################################################
    # 名称
    ########################################################
    
    clean1 <- gsub("[^A-Za-z0-9]", "_", f1)
    clean2 <- gsub("[^A-Za-z0-9]", "_", f2)
    
    ########################################################
    # 1. Product
    ########################################################
    
    name_prod <- paste0(
      "INT_PROD_",
      clean1,
      "_",
      clean2
    )
    
    df[[name_prod]] <- x1 * x2
    
    created_features <- c(
      created_features,
      name_prod
    )
    
    
    
    
    
  }
  
  return(
    list(
      data = df,
      features = created_features
    )
  )
}

############################################################
# 7. 主循环
############################################################

feature_report <- list()

for(outcome_col in outcome_cols){
  
  cat("\n====================================================\n")
  cat("Processing:", outcome_col, "\n")
  cat("====================================================\n")
  
  ##########################################################
  # clean name
  ##########################################################
  
  outcome_clean <- paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    "_"
  )
  
  ##########################################################
  # Top15
  ##########################################################
  
  top15 <- extract_top15_features(
    outcome_clean
  )
  
  cat("\nTop15 Features:\n")
  print(top15)
  
  ##########################################################
  # Reduced Model Data
  ##########################################################
  
  reduced_train <- train_data[, c(
    top15,
    outcome_col
  ), with = FALSE]
  
  reduced_test <- test_data[, c(
    top15,
    outcome_col
  ), with = FALSE]
  
  ##########################################################
  # 保存 Reduced
  ##########################################################
  
  fwrite(
    reduced_train,
    file.path(
      save_dir,
      "Reduced_Model_Data",
      paste0(
        outcome_clean,
        "_Reduced_Train.csv"
      )
    )
  )
  
  fwrite(
    reduced_test,
    file.path(
      save_dir,
      "Reduced_Model_Data",
      paste0(
        outcome_clean,
        "_Reduced_Test.csv"
      )
    )
  )
  
  ##########################################################
  # Interaction Features
  ##########################################################
  
  interaction_pairs <- interaction_mapping[[outcome_clean]]
  
  ##########################################################
  # 用完整数据生成 interaction
  ##########################################################
  
  train_interaction <- create_interaction_features(
    copy(train_data),
    interaction_pairs
  )
  
  test_interaction <- create_interaction_features(
    copy(test_data),
    interaction_pairs
  )
  
  ##########################################################
  # 最终特征
  ##########################################################
  
  final_features <- c(
    top15,
    train_interaction$features,
    outcome_col
  )
  
  final_features <- unique(final_features)
  
  ##########################################################
  # 最终数据
  ##########################################################
  
  final_train <- train_interaction$data[
    ,
    final_features,
    with = FALSE
  ]
  
  final_test <- test_interaction$data[
    ,
    final_features,
    with = FALSE
  ]
  ##########################################################
  # 保存 Interaction
  ##########################################################
  
  fwrite(
    final_train,
    file.path(
      save_dir,
      "Interaction_Model_Data",
      paste0(
        outcome_clean,
        "_Interaction_Train.csv"
      )
    )
  )
  
  fwrite(
    final_test,
    file.path(
      save_dir,
      "Interaction_Model_Data",
      paste0(
        outcome_clean,
        "_Interaction_Test.csv"
      )
    )
  )
  
  ##########################################################
  # 保存Metadata
  ##########################################################
  
  meta <- data.frame(
    Interaction_Features =
      train_interaction$features
  )
  
  write.csv(
    meta,
    file.path(
      save_dir,
      "Interaction_Metadata",
      paste0(
        outcome_clean,
        "_Interaction_Metadata.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Feature Report
  ##########################################################
  
  report <- c(
    
    "====================================================",
    paste0("Outcome: ", outcome_col),
    "====================================================",
    "",
    
    "Top15 Features:",
    top15,
    "",
    
    "Interaction Pairs:"
  )
  
  for(pair in interaction_pairs){
    
    report <- c(
      report,
      paste0(
        pair[1],
        " × ",
        pair[2]
      )
    )
  }
  
  report <- c(
    report,
    "",
    "Generated Interaction Features:",
    train_interaction$features
  )
  
  writeLines(
    report,
    file.path(
      save_dir,
      "Feature_Reports",
      paste0(
        outcome_clean,
        "_Feature_Report.txt"
      )
    )
  )
  
  ##########################################################
  # 汇总
  ##########################################################
  
  feature_report[[outcome_clean]] <- list(
    top15 = top15,
    interaction_pairs = interaction_pairs,
    interaction_features =
      train_interaction$features
  )
  
  cat("\n✓ 完成:", outcome_col, "\n")
}

############################################################
# 8. 保存总对象
############################################################

saveRDS(
  feature_report,
  file.path(
    save_dir,
    "SCI_Feature_Engineering_Objects.rds"
  )
)

############################################################
# 9. 结束
############################################################

cat("\n====================================================\n")
cat("SCI特征工程完成\n")
cat("====================================================\n")

cat("\n保存位置:\n")
cat(save_dir, "\n")

##########################################################################











######################################################################
# SCI 三模型完整比较脚本（最终独立调参版）
#
# 核心：
# 1. Baseline使用原始结果
# 2. Top15独立50次Random Search
# 3. Interaction独立50次Random Search
# 4. 完全复制Baseline训练策略
# 5. 相同CV / 相同搜索空间 / 相同nrounds
# 6. SCI公平比较最终版
######################################################################

rm(list = ls())
gc()

######################################################################
# 1. package
######################################################################

required_packages <- c(
  "data.table",
  "pROC",
  "xgboost",
  "caret",
  "mlr3",
  "mlr3learners",
  "mlr3tuning",
  "paradox",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

future::plan("sequential")

set.seed(123)

cat("====================================================\n")
cat("SCI Three-Model Comparison Pipeline\n")
cat("Independent Hyperparameter Tuning Version\n")
cat("====================================================\n")

######################################################################
# 2. 路径
######################################################################

feature_dir <- "J:/rtools45/手工4乘积交互Feature_Engineering"

baseline_dir <- "J:/rtools45/XGBoost_Results"

save_dir <- "J:/rtools45/手工4乘积交互Model_Comparison_Independent_Tuning"

dir.create(save_dir, showWarnings = FALSE)

dir.create(
  file.path(save_dir, "Prediction_Files"),
  showWarnings = FALSE
)

######################################################################
# 3. outcome
######################################################################

outcomes <- c(
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma",
  "Outcome_Disease Non-Progression"
)

######################################################################
# 4. metric
######################################################################

calculate_metrics <- function(true, prob){
  
  pred <- ifelse(prob > 0.5, 1, 0)
  
  cm <- confusionMatrix(
    factor(pred),
    factor(true),
    positive = "1"
  )
  
  roc_obj <- roc(true, prob)
  
  auc_value <- as.numeric(auc(roc_obj))
  
  ci_auc <- ci.auc(roc_obj)
  
  f1 <- 2 * (
    cm$byClass["Sensitivity"] *
      cm$byClass["Precision"]
  ) / (
    cm$byClass["Sensitivity"] +
      cm$byClass["Precision"]
  )
  
  result <- data.frame(
    
    AUC = auc_value,
    
    CI_Lower = ci_auc[1],
    
    CI_Upper = ci_auc[3],
    
    Accuracy =
      cm$overall["Accuracy"],
    
    Sensitivity =
      cm$byClass["Sensitivity"],
    
    Specificity =
      cm$byClass["Specificity"],
    
    Precision =
      cm$byClass["Precision"],
    
    F1 = f1
  )
  
  return(
    list(
      metrics = result,
      roc = roc_obj
    )
  )
}

######################################################################
# 5. 独立调参训练函数
######################################################################

train_xgb_model <- function(
    train_df,
    test_df,
    outcome_col,
    model_name
){
  
  cat("\n----------------------------------------\n")
  cat("训练模型:", model_name, "\n")
  cat("----------------------------------------\n")
  
  ############################################################
  # feature
  ############################################################
  
  feature_cols <- setdiff(
    names(train_df),
    outcome_col
  )
  
  ############################################################
  # X / y
  ############################################################
  
  x_train <- train_df[
    ,
    feature_cols,
    with = FALSE
  ]
  
  y_train <- train_df[[outcome_col]]
  
  x_test <- test_df[
    ,
    feature_cols,
    with = FALSE
  ]
  
  y_test <- test_df[[outcome_col]]
  
  cat("\n训练集维度:\n")
  print(dim(x_train))
  
  cat("\n测试集维度:\n")
  print(dim(x_test))
  
  ############################################################
  # numeric
  ############################################################
  
  x_train <- data.matrix(x_train)
  x_test  <- data.matrix(x_test)
  
  ############################################################
  # scale_pos_weight
  ############################################################
  
  pos_num <- sum(y_train == 1)
  
  neg_num <- sum(y_train == 0)
  
  scale_pos_weight <- neg_num / pos_num
  
  cat("\nscale_pos_weight:\n")
  print(scale_pos_weight)
  
  ############################################################
  # task
  ############################################################
  
  train_task_df <- data.frame(
    x_train
  )
  
  train_task_df[[outcome_col]] <- factor(
    y_train,
    levels = c(0,1),
    labels = c("0","1")
  )
  
  task <- TaskClassif$new(
    
    id = model_name,
    
    backend = train_task_df,
    
    target = outcome_col,
    
    positive = "1"
  )
  
  ############################################################
  # learner
  ############################################################
  
  learner <- lrn(
    
    "classif.xgboost",
    
    predict_type = "prob",
    
    booster = "gbtree",
    
    eval_metric = "auc",
    
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
  
  ############################################################
  # search space
  ############################################################
  
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
  
  ############################################################
  # AutoTuner
  ############################################################
  
  at <- auto_tuner(
    
    tuner = tnr("random_search"),
    
    learner = learner,
    
    resampling = rsmp("cv", folds = 5),
    
    measure = msr("classif.auc"),
    
    search_space = search_space,
    
    terminator = trm("evals", n_evals = 50)
  )
  
  ############################################################
  # train
  ############################################################
  
  cat("\n开始50次随机搜索调参...\n")
  
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
  
  ############################################################
  # best params
  ############################################################
  
  best_params <- at$archive$best()$x_domain[[1]]
  
  cat("\n最优参数:\n")
  print(best_params)
  
  ############################################################
  # final booster
  ############################################################
  
  dtrain <- xgb.DMatrix(
    data = x_train,
    label = y_train
  )
  
  dtest <- xgb.DMatrix(
    data = x_test,
    label = y_test
  )
  
  final_params <- list(
    
    booster = "gbtree",
    
    objective = "binary:logistic",
    
    eval_metric = "auc",
    
    eta = best_params$eta,
    
    max_depth = best_params$max_depth,
    
    min_child_weight =
      best_params$min_child_weight,
    
    subsample =
      best_params$subsample,
    
    colsample_bytree =
      best_params$colsample_bytree,
    
    gamma =
      best_params$gamma,
    
    lambda =
      best_params$lambda,
    
    alpha =
      best_params$alpha,
    
    scale_pos_weight =
      scale_pos_weight,
    
    seed = 123
  )
  
  cat("\n最终训练参数:\n")
  print(final_params)
  
  ############################################################
  # final train
  ############################################################
  
  model <- xgb.train(
    
    params = final_params,
    
    data = dtrain,
    
    nrounds = 200,
    
    verbose = 0
  )
  
  cat("\n✓ 最终模型训练完成\n")
  
  ############################################################
  # predict
  ############################################################
  
  pred_prob <- predict(
    model,
    dtest
  )
  
  ############################################################
  # prediction df
  ############################################################
  
  prediction_df <- data.frame(
    
    True_Label = y_test,
    
    Predicted_Prob = pred_prob
  )
  ############################################################
  # Feature Importance (Gain)
  ############################################################
  
  importance <- xgb.importance(
    feature_names = feature_cols,
    model = model
  )
  ############################################################
  # tuning history
  ############################################################
  
  tuning_history <- as.data.frame(
    at$archive$data
  )
  
  tuning_history <- tuning_history[,
                                   sapply(
                                     tuning_history,
                                     function(x)
                                       !is.list(x)
                                   )]
  
  return(
    list(
      
      prediction = prediction_df,
      
      model = model,
      
      importance = importance,
      
      best_params = best_params,
      
      tuning_history = tuning_history
    )
  )
}

######################################################################
# 6. 主循环
######################################################################

all_results <- list()

summary_table <- data.frame()

for(outcome_col in outcomes){
  
  cat("\n====================================================\n")
  cat("Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  outcome_clean <- paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    "_"
  )
  
  ############################################################
  # Baseline
  ############################################################
  
  baseline_pred_file <- file.path(
    
    baseline_dir,
    
    paste0(
      outcome_clean,
      "prediction.csv"
    )
  )
  
  baseline_pred <- fread(
    baseline_pred_file
  )
  
  baseline_metrics <- calculate_metrics(
    
    baseline_pred$True_Label,
    
    baseline_pred$Predicted_Prob
  )
  
  ############################################################
  # Top15
  ############################################################
  
  top15_train <- fread(
    
    file.path(
      
      feature_dir,
      
      "Reduced_Model_Data",
      
      paste0(
        outcome_clean,
        "_Reduced_Train.csv"
      )
    )
  )
  
  top15_test <- fread(
    
    file.path(
      
      feature_dir,
      
      "Reduced_Model_Data",
      
      paste0(
        outcome_clean,
        "_Reduced_Test.csv"
      )
    )
  )
  
  top15_res <- train_xgb_model(
    
    top15_train,
    
    top15_test,
    
    outcome_col,
    
    "Top15_Model"
  )
  
  top15_metrics <- calculate_metrics(
    
    top15_res$prediction$True_Label,
    
    top15_res$prediction$Predicted_Prob
  )
  
  ############################################################
  # Interaction
  ############################################################
  
  int_train <- fread(
    
    file.path(
      
      feature_dir,
      
      "Interaction_Model_Data",
      
      paste0(
        outcome_clean,
        "_Interaction_Train.csv"
      )
    )
  )
  
  int_test <- fread(
    
    file.path(
      
      feature_dir,
      
      "Interaction_Model_Data",
      
      paste0(
        outcome_clean,
        "_Interaction_Test.csv"
      )
    )
  )
  
  int_res <- train_xgb_model(
    
    int_train,
    
    int_test,
    
    outcome_col,
    
    "Interaction_Model"
  )
  
  int_metrics <- calculate_metrics(
    
    int_res$prediction$True_Label,
    
    int_res$prediction$Predicted_Prob
  )
  
  ############################################################
  # DeLong
  ############################################################
  
  roc_base <- baseline_metrics$roc
  
  roc_top15 <- top15_metrics$roc
  
  roc_int <- int_metrics$roc
  
  delong_base_vs_top15 <- roc.test(
    
    roc_base,
    
    roc_top15,
    
    method = "delong"
  )
  
  delong_top15_vs_int <- roc.test(
    
    roc_top15,
    
    roc_int,
    
    method = "delong"
  )
  
  delong_base_vs_int <- roc.test(
    
    roc_base,
    
    roc_int,
    
    method = "delong"
  )
  
  delong_df <- data.frame(
    
    Comparison = c(
      
      "Baseline vs Top15",
      
      "Top15 vs Interaction",
      
      "Baseline vs Interaction"
    ),
    
    P_Value = c(
      
      delong_base_vs_top15$p.value,
      
      delong_top15_vs_int$p.value,
      
      delong_base_vs_int$p.value
    )
  )
  
  ############################################################
  # summary
  ############################################################
  
  tmp_summary <- rbind(
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Baseline",
      
      baseline_metrics$metrics
    ),
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Top15",
      
      top15_metrics$metrics
    ),
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Interaction",
      
      int_metrics$metrics
    )
  )
  
  summary_table <- rbind(
    
    summary_table,
    
    tmp_summary
  )
  ############################################################
  # 保存Gain
  ############################################################
  
  fwrite(
    
    top15_res$importance,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Top15_Gain.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$importance,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Gain.csv"
      )
    )
  )
  
  ############################################################
  # 仅保存交互特征Gain
  ############################################################
  
  interaction_gain <- int_res$importance[
    grepl("^INT_", Feature)
  ]
  
  fwrite(
    
    interaction_gain,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Feature_Gain.csv"
      )
    )
  )
  ############################################################
  # 保存结果
  ############################################################
  
  all_results[[outcome_clean]] <- list(
    
    baseline_metrics =
      baseline_metrics,
    
    top15_metrics =
      top15_metrics,
    
    interaction_metrics =
      int_metrics,
    
    top15_gain =
      top15_res$importance,
    
    interaction_gain =
      int_res$importance,
    
    top15_best_params =
      top15_res$best_params,
    
    interaction_best_params =
      int_res$best_params,
    
    delong =
      delong_df
  )
  
  ############################################################
  # 保存prediction
  ############################################################
  
  fwrite(
    
    top15_res$prediction,
    
    file.path(
      
      save_dir,
      
      "Prediction_Files",
      
      paste0(
        outcome_clean,
        "Top15_Prediction.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$prediction,
    
    file.path(
      
      save_dir,
      
      "Prediction_Files",
      
      paste0(
        outcome_clean,
        "Interaction_Prediction.csv"
      )
    )
  )
  
  ############################################################
  # 保存调参历史
  ############################################################
  
  fwrite(
    
    top15_res$tuning_history,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Top15_Tuning_History.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$tuning_history,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Tuning_History.csv"
      )
    )
  )
  
  cat("\n✓ 完成:", outcome_col, "\n")
}

######################################################################
# 7. 保存summary
######################################################################

write.csv(
  
  summary_table,
  
  file.path(
    
    save_dir,
    
    "SCI_Model_Performance_Summary.csv"
  ),
  
  row.names = FALSE
)

######################################################################
# 8. 保存RDS
######################################################################

saveRDS(
  
  all_results,
  
  file.path(
    
    save_dir,
    
    "SCI_Model_Comparison_Results.rds"
  )
)

######################################################################
# 9. 完成
######################################################################

cat("\n====================================================\n")
cat("SCI三模型独立调参比较完成\n")
cat("====================================================\n")

cat("\n结果保存于:\n")
cat(save_dir, "\n")


# ====== Log-Product Interaction ======

######################################################################
# SCI Feature Engineering Script
# Top15 SHAP + SHAP Interaction Engineering
#
# 逻辑：
# Baseline模型（78特征）:
#   ✓ 不重新训练
#   ✓ 直接读取已有结果
#
# Model 1:
#   ✓ SHAP Top15特征重训练
#
# Model 2:
#   ✓ Top15 + SHAP交互特征
#
# 交互工程（SCI升级版）：
#   1. Product
#   2. Ratio
#   3. Log Interaction
#   4. Threshold Interaction
#
# 输出：
#   SCI_Feature_Engineering/
#       ├── Reduced_Model_Data/
#       ├── Interaction_Model_Data/
#       ├── Feature_Reports/
#       └── Interaction_Metadata/
######################################################################

rm(list = ls())
gc()

library(data.table)
library(readxl)

set.seed(123)

cat("====================================================\n")
cat("SCI Feature Engineering Pipeline\n")
cat("====================================================\n")

############################################################
# 1. 路径
############################################################

xgb_dir <- "J:/rtools45/XGBoost_Results"

shap_dir <- "J:/rtools45/SHAP_Results_TOPJOURNAL"

save_dir <- "J:/rtools45/Log手工4乘积交互Feature_Engineering"

dir.create(save_dir, showWarnings = FALSE)

dir.create(
  file.path(save_dir, "Reduced_Model_Data"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Interaction_Model_Data"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Feature_Reports"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Interaction_Metadata"),
  showWarnings = FALSE
)

############################################################
# 2. 读取原始数据
############################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path  <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

############################################################
# 3. Outcome
############################################################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  value = TRUE
)

print(outcome_cols)

############################################################
# 4. SHAP Top15提取函数
############################################################

extract_top15_features <- function(outcome_col){
  
  outcome_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  shap_file <- file.path(
    shap_dir,
    outcome_clean,
    "SHAP_Importance_Top15.csv"
  )
  
  cat("\n读取SHAP Top15:\n")
  cat(shap_file, "\n")
  
  if(!file.exists(shap_file)){
    
    stop(
      paste0(
        "❌ 文件不存在: ",
        shap_file
      )
    )
  }
  
  shap_imp <- fread(shap_file)
  
  top15 <- shap_imp$Feature[1:15]
  
  return(top15)
}

############################################################
# 5. 指定SCI交互组合（最终版）
############################################################

interaction_mapping <- list()

##########################################################
# Disease Non Progression
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Disease Non-Progression"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Absolute Lymphocyte Count",
    "Platelet Count"
  ),
  
  c(
    "Number of Hospitalizations",
    "Platelet Count"
  ),
  
  c(
    "Plateletcrit",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "C-Reactive Protein (CRP)",
    "Alpha-fetoprotein (AFP)"
  )
)

##########################################################
# Liver Cirrhosis
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Liver Cirrhosis"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Plateletcrit",
    "Platelet Count"
  ),
  
  c(
    "Absolute Lymphocyte Count",
    "Platelet Count"
  ),
  
  c(
    "Admission Interval",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "Cholinesterase",
    "Plateletcrit"
  )
)

##########################################################
# HCC
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Hepatocellular Carcinoma"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Direct Bilirubin",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "Number of Hospitalizations",
    "C-Reactive Protein (CRP)"
  ),
  
  c(
    "Age",
    "C-Reactive Protein (CRP)"
  ),
  
  c(
    "C-Reactive Protein (CRP)",
    "Alpha-fetoprotein (AFP)"
  )
)

############################################################
# 6. 创建SCI交互特征
############################################################

create_interaction_features <- function(df, pairs){
  
  created_features <- c()
  
  for(pair in pairs){
    
    f1 <- pair[1]
    f2 <- pair[2]
    
    if(!(f1 %in% names(df))) next
    if(!(f2 %in% names(df))) next
    
    x1 <- as.numeric(df[[f1]])
    x2 <- as.numeric(df[[f2]])
    
    ########################################################
    # 名称
    ########################################################
    
    clean1 <- gsub("[^A-Za-z0-9]", "_", f1)
    clean2 <- gsub("[^A-Za-z0-9]", "_", f2)
    
    ########################################################
    #  Log Interaction
    ########################################################
    
    name_log <- paste0(
      "INT_LOG_",
      clean1,
      "_",
      clean2
    )
    
    df[[name_log]] <- log1p(abs(x1 * x2))
    
    created_features <- c(
      created_features,
      name_log
    )
    
    
    
    
    
  }
  
  return(
    list(
      data = df,
      features = created_features
    )
  )
}

############################################################
# 7. 主循环
############################################################

feature_report <- list()

for(outcome_col in outcome_cols){
  
  cat("\n====================================================\n")
  cat("Processing:", outcome_col, "\n")
  cat("====================================================\n")
  
  ##########################################################
  # clean name
  ##########################################################
  
  outcome_clean <- paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    "_"
  )
  
  ##########################################################
  # Top15
  ##########################################################
  
  top15 <- extract_top15_features(
    outcome_clean
  )
  
  cat("\nTop15 Features:\n")
  print(top15)
  
  ##########################################################
  # Reduced Model Data
  ##########################################################
  
  reduced_train <- train_data[, c(
    top15,
    outcome_col
  ), with = FALSE]
  
  reduced_test <- test_data[, c(
    top15,
    outcome_col
  ), with = FALSE]
  
  ##########################################################
  # 保存 Reduced
  ##########################################################
  
  fwrite(
    reduced_train,
    file.path(
      save_dir,
      "Reduced_Model_Data",
      paste0(
        outcome_clean,
        "_Reduced_Train.csv"
      )
    )
  )
  
  fwrite(
    reduced_test,
    file.path(
      save_dir,
      "Reduced_Model_Data",
      paste0(
        outcome_clean,
        "_Reduced_Test.csv"
      )
    )
  )
  
  ##########################################################
  # Interaction Features
  ##########################################################
  
  interaction_pairs <- interaction_mapping[[outcome_clean]]
  
  ##########################################################
  # 用完整数据生成 interaction
  ##########################################################
  
  train_interaction <- create_interaction_features(
    copy(train_data),
    interaction_pairs
  )
  
  test_interaction <- create_interaction_features(
    copy(test_data),
    interaction_pairs
  )
  
  ##########################################################
  # 最终特征
  ##########################################################
  
  final_features <- c(
    top15,
    train_interaction$features,
    outcome_col
  )
  
  final_features <- unique(final_features)
  
  ##########################################################
  # 最终数据
  ##########################################################
  
  final_train <- train_interaction$data[
    ,
    final_features,
    with = FALSE
  ]
  
  final_test <- test_interaction$data[
    ,
    final_features,
    with = FALSE
  ]
  ##########################################################
  # 保存 Interaction
  ##########################################################
  
  fwrite(
    final_train,
    file.path(
      save_dir,
      "Interaction_Model_Data",
      paste0(
        outcome_clean,
        "_Interaction_Train.csv"
      )
    )
  )
  
  fwrite(
    final_test,
    file.path(
      save_dir,
      "Interaction_Model_Data",
      paste0(
        outcome_clean,
        "_Interaction_Test.csv"
      )
    )
  )
  
  ##########################################################
  # 保存Metadata
  ##########################################################
  
  meta <- data.frame(
    Interaction_Features =
      train_interaction$features
  )
  
  write.csv(
    meta,
    file.path(
      save_dir,
      "Interaction_Metadata",
      paste0(
        outcome_clean,
        "_Interaction_Metadata.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Feature Report
  ##########################################################
  
  report <- c(
    
    "====================================================",
    paste0("Outcome: ", outcome_col),
    "====================================================",
    "",
    
    "Top15 Features:",
    top15,
    "",
    
    "Interaction Pairs:"
  )
  
  for(pair in interaction_pairs){
    
    report <- c(
      report,
      paste0(
        pair[1],
        " × ",
        pair[2]
      )
    )
  }
  
  report <- c(
    report,
    "",
    "Generated Interaction Features:",
    train_interaction$features
  )
  
  writeLines(
    report,
    file.path(
      save_dir,
      "Feature_Reports",
      paste0(
        outcome_clean,
        "_Feature_Report.txt"
      )
    )
  )
  
  ##########################################################
  # 汇总
  ##########################################################
  
  feature_report[[outcome_clean]] <- list(
    top15 = top15,
    interaction_pairs = interaction_pairs,
    interaction_features =
      train_interaction$features
  )
  
  cat("\n✓ 完成:", outcome_col, "\n")
}

############################################################
# 8. 保存总对象
############################################################

saveRDS(
  feature_report,
  file.path(
    save_dir,
    "SCI_Feature_Engineering_Objects.rds"
  )
)

############################################################
# 9. 结束
############################################################

cat("\n====================================================\n")
cat("SCI特征工程完成\n")
cat("====================================================\n")

cat("\n保存位置:\n")
cat(save_dir, "\n")

##########################################################################











######################################################################
# SCI 三模型完整比较脚本（最终独立调参版）
#
# 核心：
# 1. Baseline使用原始结果
# 2. Top15独立50次Random Search
# 3. Interaction独立50次Random Search
# 4. 完全复制Baseline训练策略
# 5. 相同CV / 相同搜索空间 / 相同nrounds
# 6. SCI公平比较最终版
######################################################################

rm(list = ls())
gc()

######################################################################
# 1. package
######################################################################

required_packages <- c(
  "data.table",
  "pROC",
  "xgboost",
  "caret",
  "mlr3",
  "mlr3learners",
  "mlr3tuning",
  "paradox",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

future::plan("sequential")

set.seed(123)

cat("====================================================\n")
cat("SCI Three-Model Comparison Pipeline\n")
cat("Independent Hyperparameter Tuning Version\n")
cat("====================================================\n")

######################################################################
# 2. 路径
######################################################################

feature_dir <- "J:/rtools45/Log手工4乘积交互Feature_Engineering"

baseline_dir <- "J:/rtools45/XGBoost_Results"

save_dir <- "J:/rtools45/Log手工4乘积交互Model_Comparison_Independent_Tuning"

dir.create(save_dir, showWarnings = FALSE)

dir.create(
  file.path(save_dir, "Prediction_Files"),
  showWarnings = FALSE
)

######################################################################
# 3. outcome
######################################################################

outcomes <- c(
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma",
  "Outcome_Disease Non-Progression"
)

######################################################################
# 4. metric
######################################################################

calculate_metrics <- function(true, prob){
  
  pred <- ifelse(prob > 0.5, 1, 0)
  
  cm <- confusionMatrix(
    factor(pred),
    factor(true),
    positive = "1"
  )
  
  roc_obj <- roc(true, prob)
  
  auc_value <- as.numeric(auc(roc_obj))
  
  ci_auc <- ci.auc(roc_obj)
  
  f1 <- 2 * (
    cm$byClass["Sensitivity"] *
      cm$byClass["Precision"]
  ) / (
    cm$byClass["Sensitivity"] +
      cm$byClass["Precision"]
  )
  
  result <- data.frame(
    
    AUC = auc_value,
    
    CI_Lower = ci_auc[1],
    
    CI_Upper = ci_auc[3],
    
    Accuracy =
      cm$overall["Accuracy"],
    
    Sensitivity =
      cm$byClass["Sensitivity"],
    
    Specificity =
      cm$byClass["Specificity"],
    
    Precision =
      cm$byClass["Precision"],
    
    F1 = f1
  )
  
  return(
    list(
      metrics = result,
      roc = roc_obj
    )
  )
}

######################################################################
# 5. 独立调参训练函数
######################################################################

train_xgb_model <- function(
    train_df,
    test_df,
    outcome_col,
    model_name
){
  
  cat("\n----------------------------------------\n")
  cat("训练模型:", model_name, "\n")
  cat("----------------------------------------\n")
  
  ############################################################
  # feature
  ############################################################
  
  feature_cols <- setdiff(
    names(train_df),
    outcome_col
  )
  
  ############################################################
  # X / y
  ############################################################
  
  x_train <- train_df[
    ,
    feature_cols,
    with = FALSE
  ]
  
  y_train <- train_df[[outcome_col]]
  
  x_test <- test_df[
    ,
    feature_cols,
    with = FALSE
  ]
  
  y_test <- test_df[[outcome_col]]
  
  cat("\n训练集维度:\n")
  print(dim(x_train))
  
  cat("\n测试集维度:\n")
  print(dim(x_test))
  
  ############################################################
  # numeric
  ############################################################
  
  x_train <- data.matrix(x_train)
  x_test  <- data.matrix(x_test)
  
  ############################################################
  # scale_pos_weight
  ############################################################
  
  pos_num <- sum(y_train == 1)
  
  neg_num <- sum(y_train == 0)
  
  scale_pos_weight <- neg_num / pos_num
  
  cat("\nscale_pos_weight:\n")
  print(scale_pos_weight)
  
  ############################################################
  # task
  ############################################################
  
  train_task_df <- data.frame(
    x_train
  )
  
  train_task_df[[outcome_col]] <- factor(
    y_train,
    levels = c(0,1),
    labels = c("0","1")
  )
  
  task <- TaskClassif$new(
    
    id = model_name,
    
    backend = train_task_df,
    
    target = outcome_col,
    
    positive = "1"
  )
  
  ############################################################
  # learner
  ############################################################
  
  learner <- lrn(
    
    "classif.xgboost",
    
    predict_type = "prob",
    
    booster = "gbtree",
    
    eval_metric = "auc",
    
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
  
  ############################################################
  # search space
  ############################################################
  
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
  
  ############################################################
  # AutoTuner
  ############################################################
  
  at <- auto_tuner(
    
    tuner = tnr("random_search"),
    
    learner = learner,
    
    resampling = rsmp("cv", folds = 5),
    
    measure = msr("classif.auc"),
    
    search_space = search_space,
    
    terminator = trm("evals", n_evals = 50)
  )
  
  ############################################################
  # train
  ############################################################
  
  cat("\n开始50次随机搜索调参...\n")
  
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
  
  ############################################################
  # best params
  ############################################################
  
  best_params <- at$archive$best()$x_domain[[1]]
  
  cat("\n最优参数:\n")
  print(best_params)
  
  ############################################################
  # final booster
  ############################################################
  
  dtrain <- xgb.DMatrix(
    data = x_train,
    label = y_train
  )
  
  dtest <- xgb.DMatrix(
    data = x_test,
    label = y_test
  )
  
  final_params <- list(
    
    booster = "gbtree",
    
    objective = "binary:logistic",
    
    eval_metric = "auc",
    
    eta = best_params$eta,
    
    max_depth = best_params$max_depth,
    
    min_child_weight =
      best_params$min_child_weight,
    
    subsample =
      best_params$subsample,
    
    colsample_bytree =
      best_params$colsample_bytree,
    
    gamma =
      best_params$gamma,
    
    lambda =
      best_params$lambda,
    
    alpha =
      best_params$alpha,
    
    scale_pos_weight =
      scale_pos_weight,
    
    seed = 123
  )
  
  cat("\n最终训练参数:\n")
  print(final_params)
  
  ############################################################
  # final train
  ############################################################
  
  model <- xgb.train(
    
    params = final_params,
    
    data = dtrain,
    
    nrounds = 200,
    
    verbose = 0
  )
  
  cat("\n✓ 最终模型训练完成\n")
  
  ############################################################
  # predict
  ############################################################
  
  pred_prob <- predict(
    model,
    dtest
  )
  
  ############################################################
  # prediction df
  ############################################################
  
  prediction_df <- data.frame(
    
    True_Label = y_test,
    
    Predicted_Prob = pred_prob
  )
  ############################################################
  # Feature Importance (Gain)
  ############################################################
  
  importance <- xgb.importance(
    feature_names = feature_cols,
    model = model
  )
  ############################################################
  # tuning history
  ############################################################
  
  tuning_history <- as.data.frame(
    at$archive$data
  )
  
  tuning_history <- tuning_history[,
                                   sapply(
                                     tuning_history,
                                     function(x)
                                       !is.list(x)
                                   )]
  
  return(
    list(
      
      prediction = prediction_df,
      
      model = model,
      
      importance = importance,
      
      best_params = best_params,
      
      tuning_history = tuning_history
    )
  )
}

######################################################################
# 6. 主循环
######################################################################

all_results <- list()

summary_table <- data.frame()

for(outcome_col in outcomes){
  
  cat("\n====================================================\n")
  cat("Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  outcome_clean <- paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    "_"
  )
  
  ############################################################
  # Baseline
  ############################################################
  
  baseline_pred_file <- file.path(
    
    baseline_dir,
    
    paste0(
      outcome_clean,
      "prediction.csv"
    )
  )
  
  baseline_pred <- fread(
    baseline_pred_file
  )
  
  baseline_metrics <- calculate_metrics(
    
    baseline_pred$True_Label,
    
    baseline_pred$Predicted_Prob
  )
  
  ############################################################
  # Top15
  ############################################################
  
  top15_train <- fread(
    
    file.path(
      
      feature_dir,
      
      "Reduced_Model_Data",
      
      paste0(
        outcome_clean,
        "_Reduced_Train.csv"
      )
    )
  )
  
  top15_test <- fread(
    
    file.path(
      
      feature_dir,
      
      "Reduced_Model_Data",
      
      paste0(
        outcome_clean,
        "_Reduced_Test.csv"
      )
    )
  )
  
  top15_res <- train_xgb_model(
    
    top15_train,
    
    top15_test,
    
    outcome_col,
    
    "Top15_Model"
  )
  
  top15_metrics <- calculate_metrics(
    
    top15_res$prediction$True_Label,
    
    top15_res$prediction$Predicted_Prob
  )
  
  ############################################################
  # Interaction
  ############################################################
  
  int_train <- fread(
    
    file.path(
      
      feature_dir,
      
      "Interaction_Model_Data",
      
      paste0(
        outcome_clean,
        "_Interaction_Train.csv"
      )
    )
  )
  
  int_test <- fread(
    
    file.path(
      
      feature_dir,
      
      "Interaction_Model_Data",
      
      paste0(
        outcome_clean,
        "_Interaction_Test.csv"
      )
    )
  )
  
  int_res <- train_xgb_model(
    
    int_train,
    
    int_test,
    
    outcome_col,
    
    "Interaction_Model"
  )
  
  int_metrics <- calculate_metrics(
    
    int_res$prediction$True_Label,
    
    int_res$prediction$Predicted_Prob
  )
  
  ############################################################
  # DeLong
  ############################################################
  
  roc_base <- baseline_metrics$roc
  
  roc_top15 <- top15_metrics$roc
  
  roc_int <- int_metrics$roc
  
  delong_base_vs_top15 <- roc.test(
    
    roc_base,
    
    roc_top15,
    
    method = "delong"
  )
  
  delong_top15_vs_int <- roc.test(
    
    roc_top15,
    
    roc_int,
    
    method = "delong"
  )
  
  delong_base_vs_int <- roc.test(
    
    roc_base,
    
    roc_int,
    
    method = "delong"
  )
  
  delong_df <- data.frame(
    
    Comparison = c(
      
      "Baseline vs Top15",
      
      "Top15 vs Interaction",
      
      "Baseline vs Interaction"
    ),
    
    P_Value = c(
      
      delong_base_vs_top15$p.value,
      
      delong_top15_vs_int$p.value,
      
      delong_base_vs_int$p.value
    )
  )
  
  ############################################################
  # summary
  ############################################################
  
  tmp_summary <- rbind(
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Baseline",
      
      baseline_metrics$metrics
    ),
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Top15",
      
      top15_metrics$metrics
    ),
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Interaction",
      
      int_metrics$metrics
    )
  )
  
  summary_table <- rbind(
    
    summary_table,
    
    tmp_summary
  )
  ############################################################
  # 保存Gain
  ############################################################
  
  fwrite(
    
    top15_res$importance,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Top15_Gain.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$importance,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Gain.csv"
      )
    )
  )
  
  ############################################################
  # 仅保存交互特征Gain
  ############################################################
  
  interaction_gain <- int_res$importance[
    grepl("^INT_", Feature)
  ]
  
  fwrite(
    
    interaction_gain,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Feature_Gain.csv"
      )
    )
  )
  ############################################################
  # 保存结果
  ############################################################
  
  all_results[[outcome_clean]] <- list(
    
    baseline_metrics =
      baseline_metrics,
    
    top15_metrics =
      top15_metrics,
    
    interaction_metrics =
      int_metrics,
    
    top15_gain =
      top15_res$importance,
    
    interaction_gain =
      int_res$importance,
    
    top15_best_params =
      top15_res$best_params,
    
    interaction_best_params =
      int_res$best_params,
    
    delong =
      delong_df
  )
  
  ############################################################
  # 保存prediction
  ############################################################
  
  fwrite(
    
    top15_res$prediction,
    
    file.path(
      
      save_dir,
      
      "Prediction_Files",
      
      paste0(
        outcome_clean,
        "Top15_Prediction.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$prediction,
    
    file.path(
      
      save_dir,
      
      "Prediction_Files",
      
      paste0(
        outcome_clean,
        "Interaction_Prediction.csv"
      )
    )
  )
  
  ############################################################
  # 保存调参历史
  ############################################################
  
  fwrite(
    
    top15_res$tuning_history,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Top15_Tuning_History.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$tuning_history,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Tuning_History.csv"
      )
    )
  )
  
  cat("\n✓ 完成:", outcome_col, "\n")
}

######################################################################
# 7. 保存summary
######################################################################

write.csv(
  
  summary_table,
  
  file.path(
    
    save_dir,
    
    "SCI_Model_Performance_Summary.csv"
  ),
  
  row.names = FALSE
)

######################################################################
# 8. 保存RDS
######################################################################

saveRDS(
  
  all_results,
  
  file.path(
    
    save_dir,
    
    "SCI_Model_Comparison_Results.rds"
  )
)

######################################################################
# 9. 完成
######################################################################

cat("\n====================================================\n")
cat("SCI三模型独立调参比较完成\n")
cat("====================================================\n")

cat("\n结果保存于:\n")
cat(save_dir, "\n")


# ====== Log-Ratio Interaction ======

######################################################################
# SCI Feature Engineering Script
# Top15 SHAP + SHAP Interaction Engineering
#
# 逻辑：
# Baseline模型（78特征）:
#   ✓ 不重新训练
#   ✓ 直接读取已有结果
#
# Model 1:
#   ✓ SHAP Top15特征重训练
#
# Model 2:
#   ✓ Top15 + SHAP交互特征
#
# 交互工程（SCI升级版）：
#   1. Product
#   2. Ratio
#   3. Log Interaction
#   4. Threshold Interaction
#
# 输出：
#   SCI_Feature_Engineering/
#       ├── Reduced_Model_Data/
#       ├── Interaction_Model_Data/
#       ├── Feature_Reports/
#       └── Interaction_Metadata/
######################################################################

rm(list = ls())
gc()

library(data.table)
library(readxl)

set.seed(123)

cat("====================================================\n")
cat("SCI Feature Engineering Pipeline\n")
cat("====================================================\n")

############################################################
# 1. 路径
############################################################

xgb_dir <- "J:/rtools45/XGBoost_Results"

shap_dir <- "J:/rtools45/SHAP_Results_TOPJOURNAL"

save_dir <- "J:/rtools45/比值手工4乘积交互Feature_Engineering"

dir.create(save_dir, showWarnings = FALSE)

dir.create(
  file.path(save_dir, "Reduced_Model_Data"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Interaction_Model_Data"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Feature_Reports"),
  showWarnings = FALSE
)

dir.create(
  file.path(save_dir, "Interaction_Metadata"),
  showWarnings = FALSE
)

############################################################
# 2. 读取原始数据
############################################################

train_path <- "C:/Users/user/Desktop/小论文/ML数据/train_preprocessed.xlsx"

test_path  <- "C:/Users/user/Desktop/小论文/ML数据/test_preprocessed.xlsx"

train_data <- as.data.table(read_excel(train_path))

test_data <- as.data.table(read_excel(test_path))

############################################################
# 3. Outcome
############################################################

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  value = TRUE
)

print(outcome_cols)

############################################################
# 4. SHAP Top15提取函数
############################################################

extract_top15_features <- function(outcome_col){
  
  outcome_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
  
  shap_file <- file.path(
    shap_dir,
    outcome_clean,
    "SHAP_Importance_Top15.csv"
  )
  
  cat("\n读取SHAP Top15:\n")
  cat(shap_file, "\n")
  
  if(!file.exists(shap_file)){
    
    stop(
      paste0(
        "❌ 文件不存在: ",
        shap_file
      )
    )
  }
  
  shap_imp <- fread(shap_file)
  
  top15 <- shap_imp$Feature[1:15]
  
  return(top15)
}

############################################################
# 5. 指定SCI交互组合（最终版）
############################################################

interaction_mapping <- list()

##########################################################
# Disease Non Progression
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Disease Non-Progression"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Absolute Lymphocyte Count",
    "Platelet Count"
  ),
  
  c(
    "Number of Hospitalizations",
    "Platelet Count"
  ),
  
  c(
    "Plateletcrit",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "C-Reactive Protein (CRP)",
    "Alpha-fetoprotein (AFP)"
  )
)

##########################################################
# Liver Cirrhosis
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Liver Cirrhosis"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Plateletcrit",
    "Platelet Count"
  ),
  
  c(
    "Absolute Lymphocyte Count",
    "Platelet Count"
  ),
  
  c(
    "Admission Interval",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "Cholinesterase",
    "Plateletcrit"
  )
)

##########################################################
# HCC
##########################################################

interaction_mapping[[
  paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      "Outcome_Hepatocellular Carcinoma"
    ),
    "_"
  )
]] <- list(
  
  c(
    "Direct Bilirubin",
    "Alpha-fetoprotein (AFP)"
  ),
  
  c(
    "Number of Hospitalizations",
    "C-Reactive Protein (CRP)"
  ),
  
  c(
    "Age",
    "C-Reactive Protein (CRP)"
  ),
  
  c(
    "C-Reactive Protein (CRP)",
    "Alpha-fetoprotein (AFP)"
  )
)

############################################################
# 6. 创建SCI交互特征
############################################################

create_interaction_features <- function(df, pairs){
  
  created_features <- c()
  eps <- 1e-6
  
  for(pair in pairs){
    
    f1 <- pair[1]
    f2 <- pair[2]
    
    if(!(f1 %in% names(df))) next
    if(!(f2 %in% names(df))) next
    
    x1 <- as.numeric(df[[f1]])
    x2 <- as.numeric(df[[f2]])
    
    clean1 <- gsub("[^A-Za-z0-9]", "_", f1)
    clean2 <- gsub("[^A-Za-z0-9]", "_", f2)
    
    #######################################################
    # ONLY INTERACTION: log-ratio (SCI standard)
    #######################################################
    
    name <- paste0(
      "INT_LOGRATIO_",
      clean1,
      "_",
      clean2
    )
    
    df[[name]] <- log1p((x1 + 1) / (x2 + 1))
    
    created_features <- c(created_features, name)
  }
  
  return(list(
    data = df,
    features = created_features
  ))
}
############################################################
# 7. 主循环
############################################################

feature_report <- list()

for(outcome_col in outcome_cols){
  
  cat("\n====================================================\n")
  cat("Processing:", outcome_col, "\n")
  cat("====================================================\n")
  
  ##########################################################
  # clean name
  ##########################################################
  
  outcome_clean <- paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    "_"
  )
  
  ##########################################################
  # Top15
  ##########################################################
  
  top15 <- extract_top15_features(
    outcome_clean
  )
  
  cat("\nTop15 Features:\n")
  print(top15)
  
  ##########################################################
  # Reduced Model Data
  ##########################################################
  
  reduced_train <- train_data[, c(
    top15,
    outcome_col
  ), with = FALSE]
  
  reduced_test <- test_data[, c(
    top15,
    outcome_col
  ), with = FALSE]
  
  ##########################################################
  # 保存 Reduced
  ##########################################################
  
  fwrite(
    reduced_train,
    file.path(
      save_dir,
      "Reduced_Model_Data",
      paste0(
        outcome_clean,
        "_Reduced_Train.csv"
      )
    )
  )
  
  fwrite(
    reduced_test,
    file.path(
      save_dir,
      "Reduced_Model_Data",
      paste0(
        outcome_clean,
        "_Reduced_Test.csv"
      )
    )
  )
  
  ##########################################################
  # Interaction Features
  ##########################################################
  
  interaction_pairs <- interaction_mapping[[outcome_clean]]
  
  ##########################################################
  # 用完整数据生成 interaction
  ##########################################################
  
  train_interaction <- create_interaction_features(
    copy(train_data),
    interaction_pairs
  )
  
  test_interaction <- create_interaction_features(
    copy(test_data),
    interaction_pairs
  )
  
  ##########################################################
  # 最终特征
  ##########################################################
  
  final_features <- c(
    top15,
    train_interaction$features,
    outcome_col
  )
  
  final_features <- unique(final_features)
  
  ##########################################################
  # 最终数据
  ##########################################################
  
  final_train <- train_interaction$data[
    ,
    final_features,
    with = FALSE
  ]
  
  final_test <- test_interaction$data[
    ,
    final_features,
    with = FALSE
  ]
  ##########################################################
  # 保存 Interaction
  ##########################################################
  
  fwrite(
    final_train,
    file.path(
      save_dir,
      "Interaction_Model_Data",
      paste0(
        outcome_clean,
        "_Interaction_Train.csv"
      )
    )
  )
  
  fwrite(
    final_test,
    file.path(
      save_dir,
      "Interaction_Model_Data",
      paste0(
        outcome_clean,
        "_Interaction_Test.csv"
      )
    )
  )
  
  ##########################################################
  # 保存Metadata
  ##########################################################
  
  meta <- data.frame(
    Interaction_Features =
      train_interaction$features
  )
  
  write.csv(
    meta,
    file.path(
      save_dir,
      "Interaction_Metadata",
      paste0(
        outcome_clean,
        "_Interaction_Metadata.csv"
      )
    ),
    row.names = FALSE
  )
  
  ##########################################################
  # Feature Report
  ##########################################################
  
  report <- c(
    
    "====================================================",
    paste0("Outcome: ", outcome_col),
    "====================================================",
    "",
    
    "Top15 Features:",
    top15,
    "",
    
    "Interaction Pairs:"
  )
  
  for(pair in interaction_pairs){
    
    report <- c(
      report,
      paste0(
        pair[1],
        " × ",
        pair[2]
      )
    )
  }
  
  report <- c(
    report,
    "",
    "Generated Interaction Features:",
    train_interaction$features
  )
  
  writeLines(
    report,
    file.path(
      save_dir,
      "Feature_Reports",
      paste0(
        outcome_clean,
        "_Feature_Report.txt"
      )
    )
  )
  
  ##########################################################
  # 汇总
  ##########################################################
  
  feature_report[[outcome_clean]] <- list(
    top15 = top15,
    interaction_pairs = interaction_pairs,
    interaction_features =
      train_interaction$features
  )
  
  cat("\n✓ 完成:", outcome_col, "\n")
}

############################################################
# 8. 保存总对象
############################################################

saveRDS(
  feature_report,
  file.path(
    save_dir,
    "SCI_Feature_Engineering_Objects.rds"
  )
)

############################################################
# 9. 结束
############################################################

cat("\n====================================================\n")
cat("SCI特征工程完成\n")
cat("====================================================\n")

cat("\n保存位置:\n")
cat(save_dir, "\n")

##########################################################################











######################################################################
# SCI 三模型完整比较脚本（最终独立调参版）
#
# 核心：
# 1. Baseline使用原始结果
# 2. Top15独立50次Random Search
# 3. Interaction独立50次Random Search
# 4. 完全复制Baseline训练策略
# 5. 相同CV / 相同搜索空间 / 相同nrounds
# 6. SCI公平比较最终版
######################################################################

rm(list = ls())
gc()

######################################################################
# 1. package
######################################################################

required_packages <- c(
  "data.table",
  "pROC",
  "xgboost",
  "caret",
  "mlr3",
  "mlr3learners",
  "mlr3tuning",
  "paradox",
  "future"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

future::plan("sequential")

set.seed(123)

cat("====================================================\n")
cat("SCI Three-Model Comparison Pipeline\n")
cat("Independent Hyperparameter Tuning Version\n")
cat("====================================================\n")

######################################################################
# 2. 路径
######################################################################

feature_dir <- "J:/rtools45/比值手工4乘积交互Feature_Engineering"

baseline_dir <- "J:/rtools45/XGBoost_Results"

save_dir <- "J:/rtools45/比值手工4乘积交互Model_Comparison_Independent_Tuning"

dir.create(save_dir, showWarnings = FALSE)

dir.create(
  file.path(save_dir, "Prediction_Files"),
  showWarnings = FALSE
)

######################################################################
# 3. outcome
######################################################################

outcomes <- c(
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma",
  "Outcome_Disease Non-Progression"
)

######################################################################
# 4. metric
######################################################################

calculate_metrics <- function(true, prob){
  
  pred <- ifelse(prob > 0.5, 1, 0)
  
  cm <- confusionMatrix(
    factor(pred),
    factor(true),
    positive = "1"
  )
  
  roc_obj <- roc(true, prob)
  
  auc_value <- as.numeric(auc(roc_obj))
  
  ci_auc <- ci.auc(roc_obj)
  
  f1 <- 2 * (
    cm$byClass["Sensitivity"] *
      cm$byClass["Precision"]
  ) / (
    cm$byClass["Sensitivity"] +
      cm$byClass["Precision"]
  )
  
  result <- data.frame(
    
    AUC = auc_value,
    
    CI_Lower = ci_auc[1],
    
    CI_Upper = ci_auc[3],
    
    Accuracy =
      cm$overall["Accuracy"],
    
    Sensitivity =
      cm$byClass["Sensitivity"],
    
    Specificity =
      cm$byClass["Specificity"],
    
    Precision =
      cm$byClass["Precision"],
    
    F1 = f1
  )
  
  return(
    list(
      metrics = result,
      roc = roc_obj
    )
  )
}

######################################################################
# 5. 独立调参训练函数
######################################################################

train_xgb_model <- function(
    train_df,
    test_df,
    outcome_col,
    model_name
){
  
  cat("\n----------------------------------------\n")
  cat("训练模型:", model_name, "\n")
  cat("----------------------------------------\n")
  
  ############################################################
  # feature
  ############################################################
  
  feature_cols <- setdiff(
    names(train_df),
    outcome_col
  )
  
  ############################################################
  # X / y
  ############################################################
  
  x_train <- train_df[
    ,
    feature_cols,
    with = FALSE
  ]
  
  y_train <- train_df[[outcome_col]]
  
  x_test <- test_df[
    ,
    feature_cols,
    with = FALSE
  ]
  
  y_test <- test_df[[outcome_col]]
  
  cat("\n训练集维度:\n")
  print(dim(x_train))
  
  cat("\n测试集维度:\n")
  print(dim(x_test))
  
  ############################################################
  # numeric
  ############################################################
  
  x_train <- data.matrix(x_train)
  x_test  <- data.matrix(x_test)
  
  ############################################################
  # scale_pos_weight
  ############################################################
  
  pos_num <- sum(y_train == 1)
  
  neg_num <- sum(y_train == 0)
  
  scale_pos_weight <- neg_num / pos_num
  
  cat("\nscale_pos_weight:\n")
  print(scale_pos_weight)
  
  ############################################################
  # task
  ############################################################
  
  train_task_df <- data.frame(
    x_train
  )
  
  train_task_df[[outcome_col]] <- factor(
    y_train,
    levels = c(0,1),
    labels = c("0","1")
  )
  
  task <- TaskClassif$new(
    
    id = model_name,
    
    backend = train_task_df,
    
    target = outcome_col,
    
    positive = "1"
  )
  
  ############################################################
  # learner
  ############################################################
  
  learner <- lrn(
    
    "classif.xgboost",
    
    predict_type = "prob",
    
    booster = "gbtree",
    
    eval_metric = "auc",
    
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
  
  ############################################################
  # search space
  ############################################################
  
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
  
  ############################################################
  # AutoTuner
  ############################################################
  
  at <- auto_tuner(
    
    tuner = tnr("random_search"),
    
    learner = learner,
    
    resampling = rsmp("cv", folds = 5),
    
    measure = msr("classif.auc"),
    
    search_space = search_space,
    
    terminator = trm("evals", n_evals = 50)
  )
  
  ############################################################
  # train
  ############################################################
  
  cat("\n开始50次随机搜索调参...\n")
  
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
  
  ############################################################
  # best params
  ############################################################
  
  best_params <- at$archive$best()$x_domain[[1]]
  
  cat("\n最优参数:\n")
  print(best_params)
  
  ############################################################
  # final booster
  ############################################################
  
  dtrain <- xgb.DMatrix(
    data = x_train,
    label = y_train
  )
  
  dtest <- xgb.DMatrix(
    data = x_test,
    label = y_test
  )
  
  final_params <- list(
    
    booster = "gbtree",
    
    objective = "binary:logistic",
    
    eval_metric = "auc",
    
    eta = best_params$eta,
    
    max_depth = best_params$max_depth,
    
    min_child_weight =
      best_params$min_child_weight,
    
    subsample =
      best_params$subsample,
    
    colsample_bytree =
      best_params$colsample_bytree,
    
    gamma =
      best_params$gamma,
    
    lambda =
      best_params$lambda,
    
    alpha =
      best_params$alpha,
    
    scale_pos_weight =
      scale_pos_weight,
    
    seed = 123
  )
  
  cat("\n最终训练参数:\n")
  print(final_params)
  
  ############################################################
  # final train
  ############################################################
  
  model <- xgb.train(
    
    params = final_params,
    
    data = dtrain,
    
    nrounds = 200,
    
    verbose = 0
  )
  
  cat("\n✓ 最终模型训练完成\n")
  
  ############################################################
  # predict
  ############################################################
  
  pred_prob <- predict(
    model,
    dtest
  )
  
  ############################################################
  # prediction df
  ############################################################
  
  prediction_df <- data.frame(
    
    True_Label = y_test,
    
    Predicted_Prob = pred_prob
  )
  ############################################################
  # Feature Importance (Gain)
  ############################################################
  
  importance <- xgb.importance(
    feature_names = feature_cols,
    model = model
  )
  ############################################################
  # tuning history
  ############################################################
  
  tuning_history <- as.data.frame(
    at$archive$data
  )
  
  tuning_history <- tuning_history[,
                                   sapply(
                                     tuning_history,
                                     function(x)
                                       !is.list(x)
                                   )]
  
  return(
    list(
      
      prediction = prediction_df,
      
      model = model,
      
      importance = importance,
      
      best_params = best_params,
      
      tuning_history = tuning_history
    )
  )
}

######################################################################
# 6. 主循环
######################################################################

all_results <- list()

summary_table <- data.frame()

for(outcome_col in outcomes){
  
  cat("\n====================================================\n")
  cat("Outcome:", outcome_col, "\n")
  cat("====================================================\n")
  
  outcome_clean <- paste0(
    gsub(
      "[^A-Za-z0-9]",
      "_",
      outcome_col
    ),
    "_"
  )
  
  ############################################################
  # Baseline
  ############################################################
  
  baseline_pred_file <- file.path(
    
    baseline_dir,
    
    paste0(
      outcome_clean,
      "prediction.csv"
    )
  )
  
  baseline_pred <- fread(
    baseline_pred_file
  )
  
  baseline_metrics <- calculate_metrics(
    
    baseline_pred$True_Label,
    
    baseline_pred$Predicted_Prob
  )
  
  ############################################################
  # Top15
  ############################################################
  
  top15_train <- fread(
    
    file.path(
      
      feature_dir,
      
      "Reduced_Model_Data",
      
      paste0(
        outcome_clean,
        "_Reduced_Train.csv"
      )
    )
  )
  
  top15_test <- fread(
    
    file.path(
      
      feature_dir,
      
      "Reduced_Model_Data",
      
      paste0(
        outcome_clean,
        "_Reduced_Test.csv"
      )
    )
  )
  
  top15_res <- train_xgb_model(
    
    top15_train,
    
    top15_test,
    
    outcome_col,
    
    "Top15_Model"
  )
  
  top15_metrics <- calculate_metrics(
    
    top15_res$prediction$True_Label,
    
    top15_res$prediction$Predicted_Prob
  )
  
  ############################################################
  # Interaction
  ############################################################
  
  int_train <- fread(
    
    file.path(
      
      feature_dir,
      
      "Interaction_Model_Data",
      
      paste0(
        outcome_clean,
        "_Interaction_Train.csv"
      )
    )
  )
  
  int_test <- fread(
    
    file.path(
      
      feature_dir,
      
      "Interaction_Model_Data",
      
      paste0(
        outcome_clean,
        "_Interaction_Test.csv"
      )
    )
  )
  
  int_res <- train_xgb_model(
    
    int_train,
    
    int_test,
    
    outcome_col,
    
    "Interaction_Model"
  )
  
  int_metrics <- calculate_metrics(
    
    int_res$prediction$True_Label,
    
    int_res$prediction$Predicted_Prob
  )
  
  ############################################################
  # DeLong
  ############################################################
  
  roc_base <- baseline_metrics$roc
  
  roc_top15 <- top15_metrics$roc
  
  roc_int <- int_metrics$roc
  
  delong_base_vs_top15 <- roc.test(
    
    roc_base,
    
    roc_top15,
    
    method = "delong"
  )
  
  delong_top15_vs_int <- roc.test(
    
    roc_top15,
    
    roc_int,
    
    method = "delong"
  )
  
  delong_base_vs_int <- roc.test(
    
    roc_base,
    
    roc_int,
    
    method = "delong"
  )
  
  delong_df <- data.frame(
    
    Comparison = c(
      
      "Baseline vs Top15",
      
      "Top15 vs Interaction",
      
      "Baseline vs Interaction"
    ),
    
    P_Value = c(
      
      delong_base_vs_top15$p.value,
      
      delong_top15_vs_int$p.value,
      
      delong_base_vs_int$p.value
    )
  )
  
  ############################################################
  # summary
  ############################################################
  
  tmp_summary <- rbind(
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Baseline",
      
      baseline_metrics$metrics
    ),
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Top15",
      
      top15_metrics$metrics
    ),
    
    cbind(
      
      Outcome = outcome_col,
      
      Model = "Interaction",
      
      int_metrics$metrics
    )
  )
  
  summary_table <- rbind(
    
    summary_table,
    
    tmp_summary
  )
  ############################################################
  # 保存Gain
  ############################################################
  
  fwrite(
    
    top15_res$importance,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Top15_Gain.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$importance,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Gain.csv"
      )
    )
  )
  
  ############################################################
  # 仅保存交互特征Gain
  ############################################################
  
  interaction_gain <- int_res$importance[
    grepl("^INT_", Feature)
  ]
  
  fwrite(
    
    interaction_gain,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Feature_Gain.csv"
      )
    )
  )
  ############################################################
  # 保存结果
  ############################################################
  
  all_results[[outcome_clean]] <- list(
    
    baseline_metrics =
      baseline_metrics,
    
    top15_metrics =
      top15_metrics,
    
    interaction_metrics =
      int_metrics,
    
    top15_gain =
      top15_res$importance,
    
    interaction_gain =
      int_res$importance,
    
    top15_best_params =
      top15_res$best_params,
    
    interaction_best_params =
      int_res$best_params,
    
    delong =
      delong_df
  )
  
  ############################################################
  # 保存prediction
  ############################################################
  
  fwrite(
    
    top15_res$prediction,
    
    file.path(
      
      save_dir,
      
      "Prediction_Files",
      
      paste0(
        outcome_clean,
        "Top15_Prediction.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$prediction,
    
    file.path(
      
      save_dir,
      
      "Prediction_Files",
      
      paste0(
        outcome_clean,
        "Interaction_Prediction.csv"
      )
    )
  )
  
  ############################################################
  # 保存调参历史
  ############################################################
  
  fwrite(
    
    top15_res$tuning_history,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Top15_Tuning_History.csv"
      )
    )
  )
  
  fwrite(
    
    int_res$tuning_history,
    
    file.path(
      
      save_dir,
      
      paste0(
        outcome_clean,
        "Interaction_Tuning_History.csv"
      )
    )
  )
  
  cat("\n✓ 完成:", outcome_col, "\n")
}

######################################################################
# 7. 保存summary
######################################################################

write.csv(
  
  summary_table,
  
  file.path(
    
    save_dir,
    
    "SCI_Model_Performance_Summary.csv"
  ),
  
  row.names = FALSE
)

######################################################################
# 8. 保存RDS
######################################################################

saveRDS(
  
  all_results,
  
  file.path(
    
    save_dir,
    
    "SCI_Model_Comparison_Results.rds"
  )
)

######################################################################
# 9. 完成
######################################################################

cat("\n====================================================\n")
cat("SCI三模型独立调参比较完成\n")
cat("====================================================\n")

cat("\n结果保存于:\n")
cat(save_dir, "\n")

