######################################################################
# 04_SHAP_analysis.R
# ----------------------------------------------------------------------------
# 用途：读取 02 训练好的各结局 XGBoost 模型，计算 TreeSHAP，输出论文所需的
#       SHAP 重要性、蜂群图与依赖图数据、标准化瀑布图数据，以及 TreeSHAP
#       交互的汇总与明细结果。
# 一致性校验：特征对齐校验、预测一致性校验、AUC 一致性校验，结果写入校验报告。
# 运行前提：先运行 02，确保各结局的完整模型与测试集预测已生成。
# 输出：save_dir 下的 SHAP 结果 CSV / RDS 与校验报告。
# 依赖包：xgboost, data.table, dplyr, tidyr, pROC
# ----------------------------------------------------------------------------
######################################################################

rm(list = ls())
gc()

set.seed(123)

######################################################################
# 1. Packages
######################################################################

required_packages <- c(
  "xgboost",
  "data.table",
  "dplyr",
  "tidyr",
  "pROC"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
  }
}

######################################################################
# 2. Paths
######################################################################

model_dir <- "J:/rtools45/XGBoost_Results"

save_dir <- "J:/rtools45/SHAP_Results_TOPJOURNAL"

if(!dir.exists(save_dir)){
  
  dir.create(save_dir, recursive = TRUE)
}

######################################################################
# 3. Load models
######################################################################

model_files <- list.files(
  
  model_dir,
  
  pattern = "_FULL_MODEL.rds$",
  
  full.names = TRUE
)

cat("====================================================\n")
cat(" TreeSHAP ANALYSIS PIPELINE\n")
cat("====================================================\n\n")

print(basename(model_files))

######################################################################
# 4. Main Loop
######################################################################

for(model_path in model_files){
  
  cat("\n====================================================\n")
  cat("MODEL:\n")
  cat("====================================================\n")
  cat(basename(model_path), "\n")
  
  tryCatch({
    
    ##################################################################
    # 4.1 Load model
    ##################################################################
    
    cat("\nLoading FULL_MODEL...\n")
    
    model_obj <- readRDS(model_path)
    
    booster <- model_obj$booster
    
    X_test <- model_obj$X_test_matrix
    
    X_train <- model_obj$X_train_matrix
    
    feature_cols <- model_obj$feature_cols
    
    y_test <- as.numeric(
      as.character(model_obj$test_label)
    )
    
    original_auc <- model_obj$test_auc
    
    ##################################################################
    # 4.2 Matrix standardization
    ##################################################################
    
    X_test <- as.matrix(X_test)
    X_train <- as.matrix(X_train)
    
    storage.mode(X_test) <- "double"
    storage.mode(X_train) <- "double"
    
    colnames(X_test) <- feature_cols
    colnames(X_train) <- feature_cols
    
    ##################################################################
    # 4.3 Feature lock
    ##################################################################
    
    stopifnot(
      identical(
        feature_cols,
        colnames(X_train)
      )
    )
    
    stopifnot(
      identical(
        feature_cols,
        colnames(X_test)
      )
    )
    
    xgb.attr(
      booster,
      "feature_names"
    ) <- feature_cols
    
    ##################################################################
    # 4.4 Create output folder
    ##################################################################
    
    outcome_name <- gsub(
      "FULL_MODEL.rds",
      "",
      basename(model_path)
    )
    
    out_dir <- file.path(
      save_dir,
      outcome_name
    )
    
    if(!dir.exists(out_dir)){
      
      dir.create(
        out_dir,
        recursive = TRUE
      )
    }
    
    ##################################################################
    # 4.5 Feature alignment validation
    ##################################################################
    
    cat("\n================================================\n")
    cat("Feature Alignment Validation\n")
    cat("================================================\n")
    
    align_train <- identical(
      feature_cols,
      colnames(X_train)
    )
    
    align_test <- identical(
      feature_cols,
      colnames(X_test)
    )
    
    print(align_train)
    print(align_test)
    
    if(!align_train | !align_test){
      
      stop("❌ Feature alignment failed")
    }
    
    ##################################################################
    # 4.6 Prediction validation
    ##################################################################
    
    cat("\n================================================\n")
    cat("Prediction Validation\n")
    cat("================================================\n")
    
    pred_test <- predict(
      booster,
      X_test
    )
    
    pred_diff <- max(
      abs(pred_test - model_obj$test_prob)
    )
    
    print(pred_diff)
    
    ##################################################################
    # 4.7 AUC validation
    ##################################################################
    
    cat("\n================================================\n")
    cat("AUC Validation\n")
    cat("================================================\n")
    
    roc_now <- roc(
      y_test,
      pred_test,
      quiet = TRUE
    )
    
    auc_now <- as.numeric(
      roc_now$auc
    )
    
    auc_diff <- abs(
      auc_now - original_auc
    )
    
    print(original_auc)
    print(auc_now)
    print(auc_diff)
    
    validation_status <- ifelse(
      auc_diff < 0.005,
      "PASS",
      "WARNING"
    )
    
    print(validation_status)
    
    ##################################################################
    # 4.8 Save validation report
    ##################################################################
    
    validation_df <- data.frame(
      
      Metric = c(
        "Train_Feature_Alignment",
        "Test_Feature_Alignment",
        "Original_AUC",
        "Current_AUC",
        "AUC_Difference",
        "Prediction_Max_Difference",
        "Validation_Status"
      ),
      
      Value = c(
        align_train,
        align_test,
        original_auc,
        auc_now,
        auc_diff,
        pred_diff,
        validation_status
      )
    )
    
    fwrite(
      validation_df,
      file.path(
        out_dir,
        "Validation_Report.csv"
      )
    )
    
    ##################################################################
    # 4.9 SHAP main effect
    ##################################################################
    
    cat("\nCalculating SHAP main effect...\n")
    
    shap_full <- predict(
      booster,
      X_train,
      predcontrib = TRUE
    )
    
    ##################################################################
    # bias
    ##################################################################
    
    bias_term <- shap_full[, ncol(shap_full)]
    
    ##################################################################
    # SHAP matrix
    ##################################################################
    
    shap_mat <- shap_full[, -ncol(shap_full)]
    
    colnames(shap_mat) <- feature_cols
    
    ##################################################################
    # 4.10 Save SHAP matrix
    ##################################################################
    
    shap_df <- as.data.frame(shap_mat)
    
    shap_df$BIAS <- bias_term
    
    fwrite(
      shap_df,
      file.path(
        out_dir,
        "SHAP_Main_Effect.csv"
      )
    )
    
    ##################################################################
    # 4.11 SHAP importance
    ##################################################################
    
    cat("Calculating SHAP importance...\n")
    
    shap_imp <- data.frame(
      
      Feature = feature_cols,
      
      MeanAbsSHAP = colMeans(
        abs(shap_mat)
      )
    )
    
    shap_imp <- shap_imp %>%
      arrange(desc(MeanAbsSHAP))
    
    fwrite(
      shap_imp,
      file.path(
        out_dir,
        "SHAP_Importance_All.csv"
      )
    )
    
    ##################################################################
    # Top15
    ##################################################################
    
    top15_imp <- head(
      shap_imp,
      15
    )
    
    fwrite(
      top15_imp,
      file.path(
        out_dir,
        "SHAP_Importance_Top15.csv"
      )
    )
    
    # 4.12 Beeswarm data
    ##################################################################
    
    cat("Saving beeswarm data...\n")
    
    beeswarm_df <- data.frame()
    
    for(feat in top15_imp$Feature){
      
      tmp <- data.frame(
        
        Feature = feat,
        
        SHAP = shap_mat[, feat],
        
        Feature_Value = X_train[, feat]
      )
      
      beeswarm_df <- rbind(
        beeswarm_df,
        tmp
      )
    }
    
    fwrite(
      beeswarm_df,
      file.path(
        out_dir,
        "SHAP_Beeswarm_Data.csv"
      )
    )
    
    ##################################################################
    # 4.13 Dependence data
    ##################################################################
    
    cat("Saving dependence data...\n")
    
    dep_dir <- file.path(
      out_dir,
      "Dependence_Data"
    )
    
    if(!dir.exists(dep_dir)){
      
      dir.create(
        dep_dir,
        recursive = TRUE
      )
    }
    
    for(feat in top15_imp$Feature){
      
      dep_df <- data.frame(
        
        Feature_Value = X_train[, feat],
        
        SHAP_Value = shap_mat[, feat]
      )
      
      fwrite(
        dep_df,
        file.path(
          dep_dir,
          paste0(
            gsub(
              "[^A-Za-z0-9]",
              "_",
              feat
            ),
            "_Dependence.csv"
          )
        )
      )
    }
    
    ##################################################################
    # 4.14 Standardized waterfall selection
    ##################################################################
    
    cat("\nSelecting representative samples...\n")
    
    pred_train <- predict(booster, X_train)
    pred_rank <- order(pred_train)
    
    low_id <- pred_rank[
      round(length(pred_rank) * 0.10)
    ]
    
    mid_id <- pred_rank[
      round(length(pred_rank) * 0.50)
    ]
    
    high_id <- pred_rank[
      round(length(pred_rank) * 0.90)
    ]
    
    selected_ids <- c(
      low_id,
      mid_id,
      high_id
    )
    
    waterfall_meta <- data.frame(
      
      Sample_Type = c(
        "Low_Risk_10th_Percentile",
        "Median_Risk_50th_Percentile",
        "High_Risk_90th_Percentile"
      ),
      
      Sample_ID = selected_ids,
      
      Predicted_Probability =
        pred_train[selected_ids]
    )
    
    fwrite(
      waterfall_meta,
      file.path(
        out_dir,
        "Waterfall_Sample_Selection.csv"
      )
    )
    
    ##################################################################
    # 4.15 Save waterfall data
    ##################################################################
    
    cat("Saving waterfall data...\n")
    
    waterfall_dir <- file.path(
      out_dir,
      "Waterfall_Data"
    )
    
    if(!dir.exists(waterfall_dir)){
      
      dir.create(
        waterfall_dir,
        recursive = TRUE
      )
    }
    
    for(i in 1:length(selected_ids)){
      
      sid <- selected_ids[i]
      
      wf_df <- data.frame(
        
        Feature = feature_cols,
        
        SHAP_Value = shap_mat[sid, ],
        
        Feature_Value = X_train[sid, ]
      )
      
      wf_df <- wf_df %>%
        arrange(desc(abs(SHAP_Value)))
      
      fwrite(
        wf_df,
        file.path(
          waterfall_dir,
          paste0(
            waterfall_meta$Sample_Type[i],
            ".csv"
          )
        )
      )
    }
    
    ##################################################################
    # 4.16 TRUE TreeSHAP interaction
    ##################################################################
    
    cat("\nCalculating TreeSHAP interaction...\n")
    
    inter_array <- predict(
      
      booster,
      
      X_train,
      
      predinteraction = TRUE
    )
    
    print(dim(inter_array))
    
    ##################################################################
    # Save interaction array
    ##################################################################
    
    saveRDS(
      inter_array,
      file.path(
        out_dir,
        "TreeSHAP_Interaction_Array_FULL.rds"
      )
    )
    
    ##################################################################
    # 4.17 Interaction summary
    ##################################################################
    
    cat("Building interaction summary...\n")
    
    interaction_summary <- data.frame()
    
    n_feat <- length(feature_cols)
    
    for(i in 1:n_feat){
      
      for(j in 1:n_feat){
        
        if(i >= j){
          
          next
        }
        
        interaction_values <- inter_array[, i, j]
        
        mean_abs_inter <- mean(
          abs(interaction_values)
        )
        
        tmp <- data.frame(
          
          Feature1 = feature_cols[i],
          
          Feature2 = feature_cols[j],
          
          MeanAbsInteraction =
            mean_abs_inter
        )
        
        interaction_summary <- rbind(
          interaction_summary,
          tmp
        )
      }
    }
    
    interaction_summary <- interaction_summary %>%
      arrange(desc(MeanAbsInteraction))
    
    fwrite(
      interaction_summary,
      file.path(
        out_dir,
        "TreeSHAP_Interaction_Summary_FULL.csv"
      )
    )
    
    ##################################################################
    # Top100
    ##################################################################
    
    fwrite(
      head(interaction_summary, 100),
      file.path(
        out_dir,
        "TreeSHAP_Interaction_Top100.csv"
      )
    )
    
    ##################################################################
    # 4.18 Interaction detail
    ##################################################################
    
    cat("Saving interaction details...\n")
    
    top_interaction <- head(
      interaction_summary,
      20
    )
    
    interaction_detail <- data.frame()
    
    for(k in 1:nrow(top_interaction)){
      
      f1 <- top_interaction$Feature1[k]
      f2 <- top_interaction$Feature2[k]
      
      i <- which(feature_cols == f1)
      j <- which(feature_cols == f2)
      
      tmp <- data.frame(
        
        SampleID = 1:nrow(X_train),
        
        Feature1 = f1,
        
        Feature2 = f2,
        
        InteractionValue =
          inter_array[, i, j]
      )
      
      interaction_detail <- rbind(
        interaction_detail,
        tmp
      )
    }
    
    fwrite(
      interaction_detail,
      file.path(
        out_dir,
        "TreeSHAP_Interaction_Detail_Top20.csv"
      )
    )
    
    # 4.19 Save raw feature values
    ##################################################################
    
    fwrite(
    
      as.data.frame(X_train),
      file.path(
        out_dir,
        "Feature_Raw_Value.csv"
      )
    )
    
    fwrite(
      as.data.frame(X_test),
    
      file.path(
        out_dir,
        "Feature_Raw_Value_Test.csv"
      )
    )
    ##################################################################
    # 4.20 Metadata
    ##################################################################
    
    meta <- list(
      
      Outcome = outcome_name,
      
      Sample_Num = nrow(X_train),
      
      Feature_Num = ncol(X_test),
      
      Validation_Status =
        validation_status,
      
      Original_AUC =
        original_auc,
      
      Current_AUC =
        auc_now,
      
      AUC_Difference =
        auc_diff,
      
      Prediction_Difference =
        pred_diff,
      
      SHAP_Method =
        "Exact TreeSHAP",
      
      Interaction_Method =
        "TreeSHAP Interaction",
      
      Waterfall_Strategy =
        "Percentile-based representative sample selection",
      
      Seed = 123,
      
      Save_Time = Sys.time()
    )
    
    saveRDS(
      meta,
      file.path(
        out_dir,
        "Analysis_Metadata.rds"
      )
    )
    
    ##################################################################
    # 4.21 COMPLETE
    ##################################################################
    
    cat("\n====================================================\n")
    cat("✅ TREE-SHAP COMPLETE:\n")
    cat(outcome_name, "\n")
    cat("====================================================\n")
    
    cat("\nTop5 SHAP Importance:\n")
    print(head(top15_imp, 5))
    
    cat("\nTop5 Interaction:\n")
    print(head(interaction_summary, 5))
    
    cat("\nValidation Status:\n")
    print(validation_status)
    
  }, error = function(e){
    
    cat("\n❌ SHAP FAILED:\n")
    
    cat(
      basename(model_path),
      "\n"
    )
    
    cat("\nERROR:\n")
    
    cat(e$message, "\n")
  })
}

######################################################################
# 5. END
######################################################################

cat("\n====================================================\n")
cat("✅ ALL TREE-SHAP ANALYSES COMPLETE\n")
cat("====================================================\n")

cat("\nResults saved to:\n")
cat(save_dir, "\n")


####################SHAP分析结束数据已保存#################################

