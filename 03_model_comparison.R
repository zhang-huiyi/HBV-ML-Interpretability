# 03_model_comparison.R - DeLong test, ROC curves, performance visualization


# ====== DeLong Test ======

######################################################################
# SCI Ultimate DeLong Test Pipeline
# 8 Machine Learning Models
# 完整SCI增强版（最终完整版）
######################################################################

rm(list = ls())
gc()

######################################################################
# 1. Packages
######################################################################

pkgs <- c(
  "data.table",
  "pROC",
  "ggplot2",
  "reshape2",
  "openxlsx",
  "dplyr",
  "stringr"
)

for(pkg in pkgs){
  
  if(!require(pkg, character.only = TRUE)){
    
    install.packages(pkg)
    
    library(pkg, character.only = TRUE)
    
  } else {
    
    library(pkg, character.only = TRUE)
  }
}

######################################################################
# 2. 模型路径
######################################################################

model_dirs <- list(
  
  XGBoost =
    "J:/rtools45/XGBoost_Results",
  
  LightGBM =
    "J:/rtools45/LightGBM_Results",
  
  GBM =
    "J:/rtools45/GBM_Results",
  
  Logistic_Regression =
    "J:/rtools45/Logistic_Regression_Results",
  
  GLMnet =
    "J:/rtools45/GLMnet_Results",
  
  ConditionalRF =
    "J:/rtools45/ConditionalRF_Results",
  
  RandomForest =
    "J:/rtools45/RandomForest_Results",
  
  MLP =
    "J:/rtools45/MLP_Results"
)

######################################################################
# 3. 输出目录
######################################################################

output_dir <- "J:/rtools45/DeLong_Results"

if(!dir.exists(output_dir)){
  
  dir.create(output_dir, recursive = TRUE)
}

######################################################################
# 4. Outcomes
######################################################################

outcomes <- c(
  
  "Outcome_Disease_Non_Progression",
  
  "Outcome_Liver_Cirrhosis",
  
  "Outcome_Hepatocellular_Carcinoma"
)

######################################################################
# 5. 自动识别文件
######################################################################

find_prediction_file <- function(model_name, outcome){
  
  if(model_name %in%
     c(
       "XGBoost",
       "LightGBM",
       "Logistic_Regression",
       "MLP"
     )){
    
    return(
      paste0(outcome, "_prediction.csv")
    )
  }
  
  if(model_name == "GLMnet"){
    
    return(
      paste0(outcome, "_Predictions.csv")
    )
  }
  
  if(model_name %in%
     c(
       "GBM",
       "ConditionalRF",
       "RandomForest"
     )){
    
    return(
      paste0(outcome, "_Prediction.csv")
    )
  }
  
  return(NULL)
}

######################################################################
# 总汇总对象
######################################################################

all_auc_summary <- data.frame()

all_best_models <- data.frame()

all_conclusions <- data.frame()

all_best_vs_others <- data.frame()

######################################################################
# 6. 开始分析
######################################################################

for(outcome in outcomes){
  
  cat("\n====================================================\n")
  
  cat("当前Outcome:", outcome, "\n")
  
  cat("====================================================\n")
  
  pred_list <- list()
  
  auc_summary <- data.frame()
  
  ####################################################################
  # 读取模型
  ####################################################################
  
  for(model_name in names(model_dirs)){
    
    file_name <- find_prediction_file(
      model_name,
      outcome
    )
    
    file_path <- file.path(
      model_dirs[[model_name]],
      file_name
    )
    
    cat("\n读取模型:", model_name, "\n")
    
    if(!file.exists(file_path)){
      
      cat("❌ 文件不存在\n")
      
      next
    }
    
    df <- fread(file_path)
    
    ##################################################################
    # 自动识别列
    ##################################################################
    
    truth_col <- names(df)[
      tolower(names(df)) %in%
        c(
          "true_label",
          "truth",
          "actual",
          "observed",
          "label"
        )
    ][1]
    
    prob_col <- names(df)[
      tolower(names(df)) %in%
        c(
          "predicted_prob",
          "predicted_probability",
          "prob",
          "probability",
          "prediction",
          "pred"
        )
    ][1]
    
    if(is.na(truth_col) | is.na(prob_col)){
      
      cat("❌ 无法识别列名\n")
      
      print(names(df))
      
      next
    }
    
    ##################################################################
    # 数据
    ##################################################################
    
    truth <- as.numeric(
      as.character(df[[truth_col]])
    )
    
    prob <- as.numeric(
      df[[prob_col]]
    )
    
    valid_idx <- complete.cases(
      truth,
      prob
    )
    
    truth <- truth[valid_idx]
    
    prob <- prob[valid_idx]
    
    ##################################################################
    # ROC
    ##################################################################
    
    roc_obj <- roc(
      truth,
      prob,
      quiet = TRUE
    )
    
    pred_list[[model_name]] <- list(
      
      roc = roc_obj,
      
      truth = truth,
      
      prob = prob
    )
    
    ##################################################################
    # AUC
    ##################################################################
    
    ci_auc <- ci.auc(roc_obj)
    
    auc_value <- round(
      as.numeric(auc(roc_obj)),
      4
    )
    
    auc_summary <- rbind(
      
      auc_summary,
      
      data.frame(
        
        Outcome = str_replace(
          outcome,
          "Outcome_",
          ""
        ),
        
        Model = model_name,
        
        AUC = auc_value,
        
        CI_Lower = round(ci_auc[1], 4),
        
        CI_Upper = round(ci_auc[3], 4),
        
        AUC_95CI = paste0(
          auc_value,
          " (",
          round(ci_auc[1], 4),
          "-",
          round(ci_auc[3], 4),
          ")"
        )
      )
    )
  }
  
  ####################################################################
  # 排名
  ####################################################################
  
  auc_summary <- auc_summary %>%
    
    arrange(desc(AUC))
  
  auc_summary$Rank <- seq_len(
    nrow(auc_summary)
  )
  
  ####################################################################
  # 保存总AUC
  ####################################################################
  
  all_auc_summary <- rbind(
    all_auc_summary,
    auc_summary
  )
  
  ####################################################################
  # DeLong
  ####################################################################
  
  model_names <- names(pred_list)
  
  delong_results <- data.frame()
  
  for(i in 1:(length(model_names)-1)){
    
    for(j in (i+1):length(model_names)){
      
      model1 <- model_names[i]
      
      model2 <- model_names[j]
      
      roc1 <- pred_list[[model1]]$roc
      
      roc2 <- pred_list[[model2]]$roc
      
      test_result <- roc.test(
        
        roc1,
        roc2,
        
        method = "delong"
      )
      
      delong_results <- rbind(
        
        delong_results,
        
        data.frame(
          
          Outcome = str_replace(
            outcome,
            "Outcome_",
            ""
          ),
          
          Model_1 = model1,
          
          Model_2 = model2,
          
          AUC_1 = round(as.numeric(auc(roc1)),4),
          
          AUC_2 = round(as.numeric(auc(roc2)),4),
          
          AUC_Difference = round(
            as.numeric(auc(roc1)) -
              as.numeric(auc(roc2)),
            4
          ),
          
          Z = round(
            as.numeric(test_result$statistic),
            4
          ),
          
          P_Value = signif(
            test_result$p.value,
            4
          )
        )
      )
    }
  }
  
  ####################################################################
  # FDR
  ####################################################################
  
  delong_results$Adjusted_P <- p.adjust(
    
    delong_results$P_Value,
    
    method = "fdr"
  )
  
  ####################################################################
  # 显著性
  ####################################################################
  
  delong_results$Significance <- ifelse(
    
    delong_results$Adjusted_P < 0.001,
    "***",
    
    ifelse(
      delong_results$Adjusted_P < 0.01,
      "**",
      
      ifelse(
        delong_results$Adjusted_P < 0.05,
        "*",
        "NS"
      )
    )
  )
  
  ####################################################################
  # 最佳模型
  ####################################################################
  
  best_model <- auc_summary$Model[1]
  
  best_auc <- auc_summary$AUC[1]
  
  best_ci_lower <- auc_summary$CI_Lower[1]
  
  best_ci_upper <- auc_summary$CI_Upper[1]
  
  ####################################################################
  # 最佳模型比较
  ####################################################################
  
  best_compare <- delong_results %>%
    
    filter(
      Model_1 == best_model |
        Model_2 == best_model
    )
  
  ####################################################################
  # Best Model vs Others 表
  ####################################################################
  
  best_vs_others_table <- data.frame()
  
  for(i in 1:nrow(best_compare)){
    
    row_i <- best_compare[i, ]
    
    ##############################################################
    # 对比模型
    ##############################################################
    
    compared_model <- ifelse(
      
      row_i$Model_1 == best_model,
      
      row_i$Model_2,
      
      row_i$Model_1
    )
    
    ##############################################################
    # 对比AUC
    ##############################################################
    
    compared_auc <- ifelse(
      
      row_i$Model_1 == best_model,
      
      row_i$AUC_2,
      
      row_i$AUC_1
    )
    
    ##############################################################
    # Delta AUC
    ##############################################################
    
    delta_auc <- round(
      best_auc - compared_auc,
      4
    )
    
    ##############################################################
    # 构建表格
    ##############################################################
    
    best_vs_others_table <- rbind(
      
      best_vs_others_table,
      
      data.frame(
        
        Outcome = str_replace(
          outcome,
          "Outcome_",
          ""
        ),
        
        Best_Model = best_model,
        
        Compared_Model = compared_model,
        
        Best_AUC = best_auc,
        
        Compared_AUC = compared_auc,
        
        Delta_AUC = delta_auc,
        
        Adjusted_P = signif(
          row_i$Adjusted_P,
          4
        ),
        
        Significance =
          row_i$Significance
      )
    )
  }
  
  ####################################################################
  # 加入总表
  ####################################################################
  
  all_best_vs_others <- rbind(
    
    all_best_vs_others,
    
    best_vs_others_table
  )
  
  ####################################################################
  # 显著比较数量
  ####################################################################
  
  significant_count <- sum(
    best_compare$Adjusted_P < 0.05,
    na.rm = TRUE
  )
  
  total_compare <- nrow(best_compare)
  
  ####################################################################
  # 保存最佳模型总表
  ####################################################################
  
  all_best_models <- rbind(
    
    all_best_models,
    
    data.frame(
      
      Outcome = str_replace(
        outcome,
        "Outcome_",
        ""
      ),
      
      Best_Model = best_model,
      
      AUC = best_auc,
      
      CI_Lower = best_ci_lower,
      
      CI_Upper = best_ci_upper,
      
      Significant_Comparisons =
        paste0(
          significant_count,
          "/",
          total_compare
        )
    )
  )
  
  ####################################################################
  # Conclusion
  ####################################################################
  
  conclusion_text <- paste0(
    
    best_model,
    
    " achieved the highest discriminative performance for ",
    
    str_replace(
      outcome,
      "Outcome_",
      ""
    ),
    
    " with an AUC of ",
    
    best_auc,
    
    " (95% CI: ",
    
    best_ci_lower,
    
    "-",
    
    best_ci_upper,
    
    ")."
  )
  
  all_conclusions <- rbind(
    
    all_conclusions,
    
    data.frame(
      
      Outcome = str_replace(
        outcome,
        "Outcome_",
        ""
      ),
      
      Conclusion = conclusion_text
    )
  )
  
  ####################################################################
  # 显著结果
  ####################################################################
  
  significant_results <- delong_results %>%
    
    filter(
      Adjusted_P < 0.05
    )
  
  ####################################################################
  # P矩阵
  ####################################################################
  
  p_matrix <- matrix(
    
    NA,
    
    nrow = length(model_names),
    
    ncol = length(model_names)
  )
  
  rownames(p_matrix) <- model_names
  
  colnames(p_matrix) <- model_names
  
  for(k in 1:nrow(delong_results)){
    
    m1 <- delong_results$Model_1[k]
    
    m2 <- delong_results$Model_2[k]
    
    p <- delong_results$Adjusted_P[k]
    
    p_matrix[m1, m2] <- p
    
    p_matrix[m2, m1] <- p
  }
  
  diag(p_matrix) <- 0
  
  matrix_df <- as.data.frame(p_matrix)
  
  ####################################################################
  # Heatmap
  ####################################################################
  
  heat_df <- melt(p_matrix)
  
  colnames(heat_df) <- c(
    "Model1",
    "Model2",
    "P_Value"
  )
  
  p <- ggplot(
    
    heat_df,
    
    aes(
      x = Model1,
      y = Model2,
      fill = P_Value
    )
    
  ) +
    
    geom_tile() +
    
    geom_text(
      aes(
        label = round(P_Value,3)
      ),
      size = 4
    ) +
    
    scale_fill_gradient(
      low = "red",
      high = "blue",
      na.value = "white"
    ) +
    
    theme_minimal() +
    
    theme(
      axis.text.x =
        element_text(
          angle = 45,
          hjust = 1
        )
    ) +
    
    labs(
      title =
        paste0(
          outcome,
          " DeLong Test Heatmap"
        )
    )
  
  ggsave(
    
    file.path(
      output_dir,
      paste0(
        outcome,
        "_Heatmap.png"
      )
    ),
    
    p,
    
    width = 10,
    
    height = 8,
    
    dpi = 300
  )
  
  ####################################################################
  # Excel
  ####################################################################
  
  wb <- createWorkbook()
  
  addWorksheet(wb, "AUC_Summary")
  writeData(wb, "AUC_Summary", auc_summary)
  
  addWorksheet(wb, "AUC_Ranking")
  writeData(wb, "AUC_Ranking", auc_summary)
  
  addWorksheet(wb, "Best_Model_Comparison")
  writeData(
    wb,
    "Best_Model_Comparison",
    best_vs_others_table
  )
  
  addWorksheet(wb, "DeLong_Results")
  writeData(wb, "DeLong_Results", delong_results)
  
  addWorksheet(wb, "Significant_Results")
  writeData(wb, "Significant_Results", significant_results)
  
  addWorksheet(wb, "P_Matrix")
  writeData(
    wb,
    "P_Matrix",
    matrix_df,
    rowNames = TRUE
  )
  
  ####################################################################
  # 自动列宽
  ####################################################################
  
  for(sheet in names(wb)){
    
    setColWidths(
      wb,
      sheet,
      cols = 1:30,
      widths = "auto"
    )
  }
  
  ####################################################################
  # 保存Excel
  ####################################################################
  
  saveWorkbook(
    
    wb,
    
    file.path(
      output_dir,
      paste0(
        outcome,
        "_SCI_DeLong_Report.xlsx"
      )
    ),
    
    overwrite = TRUE
  )
  
  ####################################################################
  # 保存CSV
  ####################################################################
  
  fwrite(
    auc_summary,
    
    file.path(
      output_dir,
      paste0(
        outcome,
        "_AUC_Summary.csv"
      )
    )
  )
  
  fwrite(
    delong_results,
    
    file.path(
      output_dir,
      paste0(
        outcome,
        "_DeLong_Results.csv"
      )
    )
  )
  
  fwrite(
    best_vs_others_table,
    
    file.path(
      output_dir,
      paste0(
        outcome,
        "_Best_vs_Others.csv"
      )
    )
  )
  
  ####################################################################
  # 输出
  ####################################################################
  
  cat("\n最佳模型:", best_model, "\n")
  
  cat("AUC:", best_auc, "\n")
  
  cat("显著比较:",
      significant_count,
      "/",
      total_compare,
      "\n")
  
  cat("\n✅ 完成:", outcome, "\n")
}

######################################################################
# 7. 全局总表输出
######################################################################

######################################################################
# 总AUC表
######################################################################

fwrite(
  
  all_auc_summary,
  
  file.path(
    output_dir,
    "ALL_Outcomes_AUC_Summary.csv"
  )
)

######################################################################
# 最佳模型表
######################################################################

fwrite(
  
  all_best_models,
  
  file.path(
    output_dir,
    "ALL_Outcomes_Best_Models.csv"
  )
)

######################################################################
# Best vs Others总表
######################################################################

fwrite(
  
  all_best_vs_others,
  
  file.path(
    output_dir,
    "ALL_Best_vs_Others.csv"
  )
)

######################################################################
# 论文结论表
######################################################################

fwrite(
  
  all_conclusions,
  
  file.path(
    output_dir,
    "ALL_Outcomes_Conclusions.csv"
  )
)

######################################################################
# 总Excel
######################################################################

wb_total <- createWorkbook()

addWorksheet(
  wb_total,
  "ALL_AUC_Summary"
)

writeData(
  wb_total,
  "ALL_AUC_Summary",
  all_auc_summary
)

addWorksheet(
  wb_total,
  "Best_Models"
)

writeData(
  wb_total,
  "Best_Models",
  all_best_models
)

addWorksheet(
  wb_total,
  "Best_vs_Others"
)

writeData(
  wb_total,
  "Best_vs_Others",
  all_best_vs_others
)

addWorksheet(
  wb_total,
  "Conclusions"
)

writeData(
  wb_total,
  "Conclusions",
  all_conclusions
)

for(sheet in names(wb_total)){
  
  setColWidths(
    wb_total,
    sheet,
    cols = 1:30,
    widths = "auto"
  )
}

saveWorkbook(
  
  wb_total,
  
  file.path(
    output_dir,
    "SCI_Total_DeLong_Summary.xlsx"
  ),
  
  overwrite = TRUE
)

######################################################################
# 完成
######################################################################

cat("\n====================================================\n")

cat("SCI完整版DeLong分析完成！\n")

cat("结果目录:\n")

cat(output_dir, "\n")

cat("====================================================\n")
# ====== ROC & XGBoost Visualization ======



##########################################################
# SCI顶刊级 ROC 曲线最终正式版（Nature终极版）
##########################################################
# 功能：
#
# 1. 自动读取8个模型ROC数据
# 2. 自动读取 Test AUC
# 3. 自动绘制三个结局ROC图
# 4. 自动高亮最佳模型
# 5. Nature风格玻璃辉光效果
# 6. publication-ready TIFF/PDF/PNG
# 7. 图例内置
# 8. Nature Medicine 风格

##########################################################

rm(list = ls())
gc()

##########################################################
# 1. 加载R包
##########################################################

pkgs <- c(
  "data.table",
  "dplyr",
  "ggplot2",
  "patchwork",
  "stringr",
  "scales"
)

installed <- rownames(installed.packages())

for(p in pkgs){
  
  if(!(p %in% installed)){
    install.packages(p)
  }
  
  library(p, character.only = TRUE)
}

##########################################################
# 2. 输出目录
##########################################################

output_dir <- "SCI_Publication_ROC_Nature"

if(!dir.exists(output_dir)){
  dir.create(output_dir, recursive = TRUE)
}

##########################################################
# 3. Nature Medicine 顶刊蓝系配色
##########################################################

model_colors <- c(
  
  ####################################################
  # 深海军蓝（主模型）
  ####################################################
  "XGBoost" ="#C62828",
  
  ####################################################
  # 皇家蓝
  ####################################################
  "LightGBM" = "#ff6b78",
  
  ####################################################
  # 湖蓝
  ####################################################
  "GBM" = "#FB8C00",
  
  ####################################################
  # 冰川青
  ####################################################
  "ConditionalRF" = "#f8d042",
  
  ####################################################
  # 青绿色
  ####################################################
  "MLP" = "#70d75c", 
  
  ####################################################
  # 石板灰蓝
  ####################################################
  "RandomForest" = "#47c4af",
  
  ####################################################
  # 深紫灰
  ####################################################
  "Logistic" = "#2058a2",
  
  ####################################################
  # 唯一暖色（高级砖红）
  ####################################################
  "GLMnet" =  "#72bcd5"
)
##########################################################
# 4. 图例显示名称
##########################################################

model_display_names <- c(
  
  "XGBoost" = "XGBoost",
  "LightGBM" = "LightGBM",
  "RandomForest" = "RF",
  "GBM" = "GBM",
  "GLMnet" = "GLMnet",
  "Logistic" = "LR",
  "ConditionalRF" = "CRF",
  "MLP" = "MLP"
)

##########################################################
# 5. 基础路径
##########################################################

base_path <- "C:/Users/user/Desktop/小论文/ML数据"

##########################################################
# 6. 自动创建ROC路径
##########################################################

outcomes <- c(
  "Outcome_Disease_Non_Progression",
  "Outcome_Hepatocellular_Carcinoma",
  "Outcome_Liver_Cirrhosis"
)

models <- list(
  
  XGBoost = "XGBoost_Results",
  LightGBM = "LightGBM_Results",
  RandomForest = "RandomForest_Results",
  GBM = "GBM_Results",
  GLMnet = "GLMnet_Results",
  Logistic = "Logistic_Regression_Results",
  ConditionalRF = "ConditionalRF_Results",
  MLP = "MLP_Results"
)

roc_files <- list()

for(outcome in outcomes){
  
  roc_files[[outcome]] <- list()
  
  for(model in names(models)){
    
    roc_files[[outcome]][[model]] <-
      file.path(
        base_path,
        models[[model]],
        paste0(outcome, "_ROC_Data.csv")
      )
  }
}

##########################################################
# 7. 提取 Test AUC
##########################################################

extract_test_auc <- function(performance_file){
  
  df <- fread(performance_file)
  
  ########################################################
  # 长表格式
  ########################################################
  
  if(
    ("指标" %in% colnames(df) & "数值" %in% colnames(df)) |
    ("Metric" %in% colnames(df) & "Value" %in% colnames(df))
  ){
    
    if("指标" %in% colnames(df)){
      
      metric_col <- "指标"
      value_col <- "数值"
      
    } else {
      
      metric_col <- "Metric"
      value_col <- "Value"
    }
    
    metric_values <- setNames(
      as.character(df[[value_col]]),
      df[[metric_col]]
    )
    
    get_metric <- function(x){
      
      if(x %in% names(metric_values)){
        
        return(as.numeric(metric_values[x]))
        
      } else {
        
        return(NA)
      }
    }
    
    auc <- get_metric("Test_AUC")
    
  } else {
    
    ####################################################
    # 宽表格式
    ####################################################
    
    get_col <- function(x){
      
      if(x %in% colnames(df)){
        
        return(as.numeric(df[[x]][1]))
        
      } else {
        
        return(NA)
      }
    }
    
    auc <- get_col("Test_AUC")
  }
  
  return(round(auc, 3))
}

##########################################################
# 8. ROC绘图函数
##########################################################

plot_roc_comparison <- function(outcome_name, file_list){
  
  cat("\n=================================================\n")
  cat("Processing:", outcome_name, "\n")
  cat("=================================================\n")
  
  roc_all <- data.frame()
  
  auc_results <- data.frame()
  
  ######################################################
  # 循环读取模型
  ######################################################
  
  for(model_name in names(file_list)){
    
    cat("Reading:", model_name, "\n")
    
    roc_file <- file_list[[model_name]]
    
    ####################################################
    # 读取ROC数据
    ####################################################
    
    roc_df <- fread(
      roc_file,
      data.table = FALSE,
      encoding = "UTF-8"
    )
    
    ####################################################
    # 统一列名
    ####################################################
    
    colnames(roc_df) <- tolower(trimws(colnames(roc_df)))
    
    ####################################################
    # 自动识别ROC列
    ####################################################
    
    if(all(c("fpr", "tpr") %in% colnames(roc_df))){
      
      roc_df <- data.frame(
        FPR = as.numeric(roc_df$fpr),
        TPR = as.numeric(roc_df$tpr)
      )
      
    } else if(all(c("specificity", "sensitivity") %in% colnames(roc_df))){
      
      roc_df <- data.frame(
        FPR = 1 - as.numeric(roc_df$specificity),
        TPR = as.numeric(roc_df$sensitivity)
      )
      
    } else {
      
      stop(
        paste0(
          "Cannot identify ROC columns in ",
          model_name
        )
      )
    }
    
    ####################################################
    # 清理异常值
    ####################################################
    
    roc_df <- roc_df %>%
      filter(
        !is.na(FPR),
        !is.na(TPR)
      ) %>%
      filter(
        FPR >= 0,
        FPR <= 1,
        TPR >= 0,
        TPR <= 1
      ) %>%
      arrange(FPR)
    
    ####################################################
    # 增加ROC起点终点
    ####################################################
    
    roc_df <- rbind(
      data.frame(FPR = 0, TPR = 0),
      roc_df,
      data.frame(FPR = 1, TPR = 1)
    )
    
    ####################################################
    # Performance文件路径
    ####################################################
    
    performance_file <- gsub(
      "_ROC_Data.csv",
      "_performance.csv",
      roc_file
    )
    
    ####################################################
    # 提取AUC
    ####################################################
    
    auc_value <- extract_test_auc(performance_file)
    
    ####################################################
    # 保存AUC结果
    ####################################################
    
    auc_results <- rbind(
      auc_results,
      data.frame(
        Model = model_name,
        AUC = auc_value
      )
    )
    
    ####################################################
    # 增加模型标签
    ####################################################
    
    roc_df$Model <- model_name
    
    roc_all <- rbind(roc_all, roc_df)
  }
  
  ######################################################
  # 按AUC降序排列
  ######################################################
  
  auc_results <- auc_results %>%
    arrange(desc(AUC))
  
  ######################################################
  # 最优模型
  ######################################################
  
  best_model <- auc_results$Model[1]
  
  ######################################################
  # 图例标签
  ######################################################
  
  legend_labels <- paste0(
    model_display_names[auc_results$Model],
    "\nAUC = ",
    sprintf("%.3f", auc_results$AUC)
  )
  
  names(legend_labels) <- auc_results$Model
  
  ######################################################
  # 设置模型顺序
  ######################################################
  
  roc_all$Model <- factor(
    roc_all$Model,
    levels = auc_results$Model
  )
  
  ######################################################
  # ROC主曲线粗细
  ######################################################
  #
  # 最佳模型：
  # 更粗、更突出
  #
  # 其他模型：
  # 保持纤细高级感
  #
  ######################################################
  
  roc_all$LineSize <- ifelse(
    roc_all$Model == best_model,
    
    ##################################################
    # 最佳模型线宽
    ##################################################
    
    1.8,
    
    ##################################################
    # 其他模型线宽
    ##################################################
    
    1.05
  )
  
  ######################################################
  # ROC透明度
  ######################################################
  #
  # 最佳模型：
  # 完全清晰
  #
  # 其他模型：
  # 半透明形成层次
  #
  ######################################################
  
  roc_all$Alpha <- ifelse(
    roc_all$Model == best_model,
    
    ##################################################
    # 最佳模型透明度
    ##################################################
    
    1,
    
    ##################################################
    # 其他模型透明度
    ##################################################
    
    0.68
  )
  
  ######################################################
  # 清理标题
  ######################################################
  
  clean_title <- outcome_name
  
  clean_title <- gsub("^Outcome_", "", clean_title)
  
  clean_title <- gsub("_", " ", clean_title)
  
  clean_title <- gsub(
    "Disease Non Progression",
    "Disease Non-Progression",
    clean_title
  )
  
  ######################################################
  # 开始绘图
  ######################################################
  
  p <- ggplot() +
    
    ####################################################
  # Nature Medicine 背景
  ####################################################
  #
  # fill：
  # 顶刊冷灰蓝背景
  #
  # alpha：
  # 建议保持1
  # 防止打印发灰
  #
  ####################################################
  
  annotate(
    "rect",
    xmin = 0,
    xmax = 1,
    ymin = 0,
    ymax = 1,
    fill = "#F5FAFC",
    alpha = 1
  ) +
    
    ####################################################
  # 最佳模型玻璃辉光层
  ####################################################
  #
  # linewidth：
  # 发光扩散宽度
  #
  # alpha：
  # 发光透明度
  #
  ####################################################
  
  geom_line(
    data = subset(roc_all, Model == best_model),
    
    aes(
      x = FPR,
      y = TPR
    ),
    
    ##################################################
    # 发光颜色
    ##################################################
    
    color = model_colors[best_model],
    
    ##################################################
    # 发光宽度
    ##################################################
    
    linewidth = 3.5,
    
    ##################################################
    # 发光透明度
    ##################################################
    
    alpha = 0.12,
    
    ##################################################
    # 圆角线条
    ##################################################
    
    lineend = "round"
  ) +
    
    ####################################################
  # ROC曲线白色描边层
  ####################################################
  #
  # 作用：
  # 增强曲线边界分离感
  #
  # 顶刊ROC核心技巧
  #
  ####################################################
  
  geom_line(
    data = roc_all,
    
    aes(
      x = FPR,
      y = TPR,
      group = Model
    ),
    
    ##################################################
    # 白色描边
    ##################################################
    
    color = "white",
    
    ##################################################
    # 描边粗细
    ##################################################
    
    linewidth = 2.8,
    
    ##################################################
    # 描边透明度
    ##################################################
    
    alpha = 0.85,
    
    ##################################################
    # 圆角线条
    ##################################################
    
    lineend = "round"
  ) +
    
    ####################################################
  # ROC主曲线
  ####################################################
  
  geom_line(
    data = roc_all,
    
    aes(
      x = FPR,
      y = TPR,
      color = Model,
      linewidth = LineSize,
      alpha = Alpha
    ),
    
    ##################################################
    # 圆角线条
    ##################################################
    
    lineend = "round"
  ) +
    
    ####################################################
  # 对角参考线
  ####################################################
  
  geom_abline(
    slope = 1,
    intercept = 0,
    linetype = "dashed",
    color = "#CBD5E1",
    linewidth = 0.75
  ) +
    
    ####################################################
  # 模型颜色
  ####################################################
  
  scale_color_manual(
    values = model_colors,
    labels = legend_labels
  ) +
    
    scale_alpha_identity() +
    
    scale_linewidth_identity() +
    
    ####################################################
  # X轴
  ####################################################
  
  scale_x_continuous(
    limits = c(0, 1),
    expand = c(0, 0),
    breaks = seq(0, 1, 0.2)
  ) +
    
    ####################################################
  # Y轴
  ####################################################
  
  scale_y_continuous(
    limits = c(0, 1),
    expand = c(0, 0),
    breaks = seq(0, 1, 0.2)
  ) +
    
    ####################################################
  # 保持正方形比例
  ####################################################
  
  coord_equal() +
    
    ####################################################
  # 标签
  ####################################################
  
  labs(
    title = clean_title,
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = NULL
  ) +
    
    ####################################################
  # Nature主题
  ####################################################
  
  theme_classic(base_size = 17) +
    
    theme(
      
      ##################################################
      # 标题
      ##################################################
      
      plot.title = element_text(
        hjust = 0.5,
        face = "bold",
        size = 28,
        color = "#0A2540"
      ),
      
      ##################################################
      # 坐标轴标题
      ##################################################
      
      axis.title = element_text(
        face = "bold",
        size = 25,
        color = "#1A1A1A"
      ),
      
      ##################################################
      # 坐标轴数字
      ##################################################
      
      axis.text = element_text(
        size = 20,
        color = "#333333"
      ),
      
      ##################################################
      # 坐标轴线
      ##################################################
      
      axis.line = element_line(
        linewidth = 0.7,
        color = "#6B7280"
      ),
      
      ##################################################
      # 刻度线
      ##################################################
      
      axis.ticks = element_line(
        linewidth = 0.6,
        color = "#6B7280"
      ),
      
      ##################################################
      # 图例位置
      ##################################################
      #
      # 第1个数：
      # 越大越往右
      #
      # 第2个数：
      # 越大越往上
      #
      ##################################################
      
      legend.position = c(0.8, 0.42),
      
      ##################################################
      # 图例背景（玻璃拟态）
      ##################################################
      
      legend.background = element_rect(
        
        ################################################
        # 半透明玻璃背景
        ################################################
        
        fill = alpha("#FFFFFF", 0.78),
        
        ################################################
        # 边框颜色
        ################################################
        
        color = "#D6DEE8",
        
        ################################################
        # 边框粗细
        ################################################
        
        linewidth = 0.6
      ),
      
      ##################################################
      # 去掉图例key背景
      ##################################################
      
      legend.key = element_blank(),
      
      ##################################################
      # 图例字体
      ##################################################
      
      legend.text = element_text(
        size = 18,
        color = "#1A1A1A",
        lineheight = 1.05
      ),
      
      ##################################################
      # 图例间距
      ##################################################
      
      legend.spacing.y = unit(1.5, "mm"),
      
      ##################################################
      # 面板背景
      ##################################################
      
      panel.background = element_rect(
        fill = "#F5F8FB",
        color = NA
      ),
      
      ##################################################
      # 去掉网格
      ##################################################
      
      panel.grid = element_blank(),
      
      ##################################################
      # 图边距
      ##################################################
      
      plot.margin = margin(
        4,
        4,
        4,
        4
      )
    )
  
  ######################################################
  # 保存TIFF
  ######################################################
  
  ggsave(
    filename = file.path(
      output_dir,
      paste0(clean_title, "_ROC.tiff")
    ),
    plot = p,
    width = 8.5,
    height = 7.5,
    dpi = 600,
    compression = "lzw",
    bg = "white"
  )
  
  ######################################################
  # 保存PNG
  ######################################################
  
  ggsave(
    filename = file.path(
      output_dir,
      paste0(clean_title, "_ROC.png")
    ),
    plot = p,
    width = 8.5,
    height = 7.5,
    dpi = 600,
    bg = "white"
  )
  
  ######################################################
  # 保存PDF
  ######################################################
  
  ggsave(
    filename = file.path(
      output_dir,
      paste0(clean_title, "_ROC.pdf")
    ),
    plot = p,
    width = 8.5,
    height = 7.5,
    bg = "white"
  )
  
  ######################################################
  # 导出AUC表
  ######################################################
  
  fwrite(
    auc_results,
    file.path(
      output_dir,
      paste0(clean_title, "_AUC.csv")
    )
  )
  
  cat("\nFinished:", clean_title, "\n")
  
  return(p)
}

##########################################################
# 9. 绘制三个ROC图
##########################################################

plot_list <- list()

for(outcome in names(roc_files)){
  
  plot_list[[outcome]] <- plot_roc_comparison(
    outcome_name = outcome,
    file_list = roc_files[[outcome]]
  )
}
##########################################################
# 10. 拼接总图（自定义顺序）
##########################################################

combined_plot <-
  plot_list[["Outcome_Disease_Non_Progression"]] +
  plot_list[["Outcome_Liver_Cirrhosis"]] +
  plot_list[["Outcome_Hepatocellular_Carcinoma"]] +
  plot_layout(ncol = 3)


##########################################################
# 11. 保存总图
##########################################################

ggsave(
  filename = file.path(
    output_dir,
    "Combined_ROC.tiff"
  ),
  plot = combined_plot,
  width = 24,
  height = 8,
  dpi = 600,
  compression = "lzw",
  bg = "white"
)

ggsave(
  filename = file.path(
    output_dir,
    "Combined_ROC.pdf"
  ),
  plot = combined_plot,
  width = 24,
  height = 8,
  bg = "white"
)

##########################################################
# 12. 完成
##########################################################

cat("\n====================================================\n")
cat("SCI Nature-style ROC figures completed!\n")
cat("Output folder:\n")
cat(output_dir, "\n")
cat("====================================================\n")












######################################################################
# XGBoost 可视化最终SCI拼图版（12图拼接专用）
#
# 最终用途：
# 12张小图拼接成SCI总图
#
# 包含：
# 1. Probability Distribution
# 2. Probability Boxplot
# 3. Top15 Feature Importance
#
# 已修复：
# ✓ 字体太小
# ✓ 拼图后不可读
# ✓ showtext冲突
# ✓ Arial冲突
# ✓ dpi缩放问题
# ✓ Windows查看器缩放问题
#
# 当前版本：
# ✓ 拼图专用超大字体
# ✓ Nature Medicine风格
# ✓ SCI投稿版
# ✓ 600 dpi
######################################################################

rm(list = ls())
gc()

######################################################################
# 1. 加载R包
######################################################################

pkgs <- c(
  "data.table",
  "ggplot2",
  "dplyr",
  "stringr",
  "scales"
)

installed <- rownames(installed.packages())

for(p in pkgs){
  
  if(!(p %in% installed)){
    install.packages(p)
  }
  
  library(p, character.only = TRUE)
}

######################################################################
# 2. 路径
######################################################################

# XGBoost结果路径
result_dir <- "C:/Users/user/Desktop/小论文/ML数据/XGBoost_Results"

# 总输出路径
base_output_dir <- "J:/rtools45"

# 图片保存文件夹
output_dir <- file.path(
  base_output_dir,
  "XGBoost_Visualization_Results"
)

if(!dir.exists(output_dir)){
  
  dir.create(
    output_dir,
    recursive = TRUE
  )
}

cat("结果保存路径：\n")
cat(output_dir, "\n")

######################################################################
# 3. Outcome
######################################################################

outcomes <- c(
  
  "Outcome_Disease_Non_Progression",
  
  "Outcome_Hepatocellular_Carcinoma",
  
  "Outcome_Liver_Cirrhosis"
)

######################################################################
# 4. SCI配色
######################################################################

# 阳性深蓝
positive_color <- "#1565C0"

# 阴性浅蓝
negative_color <- "#DCEBFA"

# 深色文字
nature_dark <- "#0A2540"

# 特征重要性蓝
bar_blue <- "#1976D2"

######################################################################
# 5. 标题清理
######################################################################

clean_title_function <- function(x){
  
  x <- gsub("^Outcome_", "", x)
  
  x <- gsub("_", " ", x)
  
  x <- gsub(
    "Disease Non Progression",
    "Disease Non-Progression",
    x
  )
  
  return(x)
}

######################################################################
# 6. 循环绘图
######################################################################

for(outcome in outcomes){
  
  cat("\n====================================================\n")
  cat("Processing:", outcome, "\n")
  cat("====================================================\n")
  
  ####################################################################
  # 标题
  ####################################################################
  
  clean_title <- clean_title_function(outcome)
  
  ####################################################################
  # 文件路径
  ####################################################################
  
  prediction_file <- file.path(
    result_dir,
    paste0(outcome, "_prediction.csv")
  )
  
  importance_file <- file.path(
    result_dir,
    paste0(outcome, "_Feature_Importance.csv")
  )
  
  ####################################################################
  # 文件检查
  ####################################################################
  
  if(!file.exists(prediction_file)){
    
    cat("缺少文件：\n")
    cat(prediction_file, "\n")
    
    next
  }
  
  if(!file.exists(importance_file)){
    
    cat("缺少文件：\n")
    cat(importance_file, "\n")
    
    next
  }
  
  ####################################################################
  # 读取预测结果
  ####################################################################
  
  pred_df <- fread(prediction_file)
  
  ####################################################################
  # 标签处理
  ####################################################################
  
  pred_df$True_Label <- as.character(pred_df$True_Label)
  
  pred_df$True_Label <- factor(
    
    pred_df$True_Label,
    
    levels = c("0","1"),
    
    labels = c(
      "Negative",
      "Positive"
    )
  )
  
  ####################################################################
  # 概率列
  ####################################################################
  
  prob_col <- grep(
    "Prob",
    colnames(pred_df),
    value = TRUE
  )[1]  # 取第一个匹配列，索引1
  
  pred_df$Probability <- pred_df[[prob_col]]
  
  ####################################################################
  ####################################################################
  # 1. Probability Distribution
  ####################################################################
  ####################################################################
  
  p_density <- ggplot(
    
    pred_df,
    
    aes(
      x = Probability,
      fill = True_Label
    )
    
  ) +
    
    ##################################################################
  # Density
  ##################################################################
  
  geom_density(
    
    alpha = 0.75,          # 透明度：0=完全透明，1=不透明；推荐范围0.5-0.9，可根据背景明暗调整
    linewidth = 1,       # 密度曲线边框粗细：单位mm，推荐1.2-2.5，拼图时宜粗不宜细
    adjust = 1,         # 带宽调整因子：>1使曲线更平滑，<1保留更多细节；推荐1-2
    color = "white"        # 曲线边框颜色（白色增强立体感）
  ) +
    
    ##################################################################
  # 颜色
  ##################################################################
  
  scale_fill_manual(
    
    values = c(
      
      "Negative" = negative_color,
      
      "Positive" = positive_color
    )
  ) +
    
    ##################################################################
  # X轴
  ##################################################################
  
  scale_x_continuous(
    
    limits = c(0,1),          # X轴显示范围（概率0-1）
    breaks = seq(0,1,0.2),    # 刻度间隔：0,0.2,0.4,0.6,0.8,1.0；可改为0.25或0.1
    expand = c(0.01,0.01)     # 轴两端扩展比例/数值：避免曲线贴边，可调至0.02-0.05
  ) +
    
    ##################################################################
  # 标签
  ##################################################################
  
  labs(
    
    title = clean_title,
    
    x = "Predicted Probability",
    
    y = "Density",
    
    fill = NULL
  ) +
    
    ##################################################################
  # SCI主题（拼图专用超大字体）
  ##################################################################
  
  theme_classic(base_size = 80) +   # 基础字体大小（所有文本的相对基准），拼图建议32-48
    
    theme(
      
      ################################################################
      # 全局字体
      ################################################################
      
      text = element_text(
        family = "sans"        # 字体族：sans(无衬线)/serif(衬线)/mono；也可指定如"Arial"
      ),
      
      ################################################################
      # 主标题
      ################################################################
      
      plot.title = element_text(
        
        hjust = 0.5,          # 标题水平对齐：0左，0.5中，1右
        face = "bold",        # 字体粗细：plain/bold/italic
        size = 90,            # 主标题字号（点），拼图推荐36-48
        color = nature_dark  , # 深灰色，可改为"black"
        
        #主标题边距
        margin = margin(
          t = 2,   # 上边距
          r = 0,    # 右边距
          b = 2,   # 下边距（最重要）
          l = 0
        )
      ),
      
      ################################################################
      # 坐标轴标题
      ################################################################
      
      axis.title = element_text(
        
        face = "bold",
        
        size = 85 ,            # 坐标轴标题字号，拼图推荐32-44
        margin = margin(t = 1)
      ),
      
      ################################################################
      # 坐标轴数字
      ################################################################
      
      axis.text = element_text(
        
        size = 80,            # 坐标轴刻度数字字号，拼图推荐28-38
        color = "#333333"     # 深灰色，可改为"black"
      ),
      
      axis.ticks = element_line(
        
        linewidth = 0.7,   # 刻度线粗细
        
        color = "#4A5568"
      ),
      ################################################################
      # 图例
      ################################################################
      
      legend.position = c(0.5,0.75),  # 图例位置：c(x,y) 坐标比例（0-1），左上为c(0.02,0.98)
      legend.text = element_text(
        
        size = 80             # 图例文字字号，拼图推荐26-34
      ),
      
      legend.background = element_rect(
        
        fill = alpha(NA,0.92),   # 图例背景：白色，透明度0.92（0-1）
        color = NA,            # 边框颜色
        linewidth = 0.5               # 边框粗细（mm）
      ),
      
      ################################################################
      # 坐标轴线
      ################################################################
      
      axis.line = element_line(
        
        linewidth = 0.7,      # 轴线粗细（mm），拼图建议1.2-1.8
        color = "#4A5568"     # 轴线颜色
      ),
      
      
      ################################################################
      # 背景
      ################################################################
      
      panel.background = element_rect(
        fill = "#F5FAFC",
        color = NA
      ),
      
      plot.background = element_rect(
        fill = "white",
        color = NA
      )
    )
  
  ####################################################################
  # 保存Density
  ####################################################################
  
  ggsave(
    
    filename = file.path(
      
      output_dir,
      
      paste0(clean_title, "_Density.png")
    ),
    
    plot = p_density,
    
    width = 5,      # 图片宽度（英寸），拼图专用建议4-6英寸
    height = 4.5,   # 图片高度（英寸），拼图专用建议4-5.5英寸
    dpi = 600,      # 分辨率：300为印刷最低，600为SCI推荐，1200超高但文件大
    bg = "white"    # 背景色：white/transparent
  )
  
  ####################################################################
  ####################################################################
  # 2. Probability Boxplot
  ####################################################################
  ####################################################################
  
  p_box <- ggplot(
    
    pred_df,
    
    aes(
      x = True_Label,
      y = Probability,
      fill = True_Label
    )
    
  ) +
    
    ##################################################################
  # 箱线图
  ##################################################################
  
  geom_boxplot(
    
    width = 0.58,        # 箱子宽度（相对比例）：0.5-0.8较合适
    alpha = 0.82,        # 填充透明度：0-1，推荐0.7-0.9
    linewidth = 1.2,     # 边框线宽（mm）
    outlier.shape = 21,  # 离群点形状：21为圆形带边框填充
    outlier.size = 4,    # 离群点大小（点），推荐3-5
    outlier.fill = "#1976D2",  # 离群点填充色
    color = "white"      # 箱子边框色（白色增强对比）
  ) +
    
    ##################################################################
  # 均值点
  ##################################################################
  
  stat_summary(
    
    fun = mean,          # 计算均值
    
    geom = "point",
    
    shape = 23,          # 菱形形状
    
    size = 6,            # 均值点大小（点），推荐5-7
    fill = "white",
    color = "#1A1A1A"
  ) +
    
    ##################################################################
  # 颜色
  ##################################################################
  
  scale_fill_manual(
    
    values = c(
      
      "Negative" = negative_color,
      
      "Positive" = positive_color
    )
  ) +
    
    ##################################################################
  # Y轴
  ##################################################################
  
  scale_y_continuous(
    
    limits = c(0,1),          # Y轴范围
    breaks = seq(0,1,0.2)     # 刻度间隔
  ) +
    
    ##################################################################
  # 标签
  ##################################################################
  
  labs(
    
    title = clean_title,
    
    x = "Actual Outcome",
    
    y = "Predicted Probability"
  ) +
    
    ##################################################################
  # SCI主题
  ##################################################################
  
  theme_classic(base_size = 80) +
    
    theme(
      
      text = element_text(
        family = "sans"
      ),
      
      ################################################################
      # 标题
      ################################################################
      
      plot.title = element_text(
        
        hjust = 0.5,
        
        face = "bold",
        
        size = 90,
        
        color = nature_dark
      ),
      
      ################################################################
      # 坐标轴标题
      ################################################################
      
      axis.title = element_text(
        
        face = "bold",
        
        size = 85
      ),
      
      axis.ticks = element_line(
        
        linewidth = 0.7,   # 刻度线粗细
        
        color = "#4A5568"
      ),
      ################################################################
      # 坐标数字
      ################################################################
      
      axis.text = element_text(
        
        size = 80,
        
        color = "#333333"
      ),
      
      ################################################################
      # 坐标轴
      ################################################################
      
      axis.line = element_line(
        
        linewidth = 0.7,
        
        color = "#4A5568"
      ),
      
      ################################################################
      # 背景
      ################################################################
      
      panel.background = element_rect(
        fill = "#F5FAFC",
        color = NA
      ),
      
      plot.background = element_rect(
        fill = "white",
        color = NA
      ),
      
      legend.position = "none"   # 不显示图例（因x轴标签已说明）
    )
  
  ####################################################################
  # 保存Boxplot
  ####################################################################
  
  ggsave(
    
    filename = file.path(
      
      output_dir,
      
      paste0(clean_title, "_Boxplot.png")
    ),
    
    plot = p_box,
    
    width = 5,
    
    height = 4.5,
    
    dpi = 600,
    
    bg = "white"
  )
  
  ####################################################################
  ####################################################################
  # 3. Top15 Feature Importance
  ####################################################################
  ####################################################################
  
  importance_df <- fread(importance_file)
  
  ####################################################################
  # Gain排序
  ####################################################################
  
  if("Gain" %in% colnames(importance_df)){
    
    importance_df <- importance_df %>%
      
      arrange(desc(Gain))
    
    importance_df$Importance <- importance_df$Gain
    
  } else {
    
    numeric_cols <- sapply(
      importance_df,
      is.numeric
    )
    
    importance_col <- names(
      importance_df
    )[numeric_cols][1]   # 取第一个数值列，索引1
    
    importance_df <- importance_df %>%
      
      arrange(
        desc(.data[[importance_col]])
      )
    
    importance_df$Importance <-
      importance_df[[importance_col]]
  }
  
  ####################################################################
  # Top15
  ####################################################################
  
  importance_df <- importance_df[1:15, ]   # 取前15行（Top15特征）
  
  ####################################################################
  # Feature列
  ####################################################################
  
  feature_col <- grep(
    
    "Feature|feature",
    
    colnames(importance_df),
    
    value = TRUE
  )[1]   # 取第一个匹配列，索引1
  
  importance_df$Feature_Name <-
    importance_df[[feature_col]]
  
  ####################################################################
  # 排序
  ####################################################################
  
  importance_df$Feature_Name <- factor(
    
    importance_df$Feature_Name,
    
    levels = rev(importance_df$Feature_Name)   # 反转顺序使重要性高的在顶部
  )
  
  ####################################################################
  # Importance Plot
  ####################################################################
  
  p_importance <- ggplot(
    
    importance_df,
    
    aes(
      x = Importance,
      y = Feature_Name
    )
    
  ) +
    
    ##################################################################
  # 条形图
  ##################################################################
  
  geom_col(
    
    fill = bar_blue,
    
    alpha = 0.9,          # 条形填充不透明度：接近1（几乎不透明）
    
    width = 0.7,          # 条形宽度比例：0.6-0.8较合适
    
    color = "white",       # 条形边框色
    
    linewidth = 0.7        # 边框粗细（mm）
  ) +
    
    ##################################################################
  # 数值标签
  ##################################################################
  
  geom_text(
    
    aes(
      label = round(Importance,3)   # 保留3位小数
    ),
    
    hjust = -0.1,          # 标签水平偏移：负值向左，正值向右
    size = 45             # 标签字号（点），拼图推荐8-11
  ) +
    
    ##################################################################
  # X轴扩展
  ##################################################################
  
  scale_x_continuous(
    
    expand = expansion(mult = c(0,0.2))   # X轴扩展：左端0扩展，右端扩展15%以容纳标签
  ) +
    
    ##################################################################
  # 标签
  ##################################################################
  
  labs(
    
    title = paste0(
      
      clean_title,"\nTop 15 Feature Importance"
    ),
    
    x = "Importance",
    
    y = NULL
  ) +
    
    ##################################################################
  # SCI主题
  ##################################################################
  
  theme_classic(base_size = 80) +
    
    theme(
      
      text = element_text(
        family = "sans"
      ),
      
      ################################################################
      # 标题
      ################################################################
      
      plot.title = element_text(
        
        hjust = 0.5,
        
        face = "bold",
        
        size = 80,            # 主标题字号（含换行），拼图推荐36-42
        color = nature_dark,
        
        lineheight = 1.1      # 行高倍数（当标题有换行时），推荐1.1-1.3
      ),
      
      ################################################################
      # 坐标轴标题
      ################################################################
      
      axis.title = element_text(
        
        face = "bold",
        
        size = 85            # 坐标轴标题字号
      ),
      
      ################################################################
      # 坐标数字
      ################################################################
      
      axis.text = element_text(
        
        size = 70,            # 坐标轴刻度字号（特征名和数值）
        color = "#333333"
      ),
      
      axis.ticks = element_line(
        
        linewidth = 0.7,   # 刻度线粗细
        
        color = "#4A5568"
      ),
      ################################################################
      # 坐标轴
      ################################################################
      
      axis.line = element_line(
        
        linewidth = 0.7,      # 轴线粗细
        
        color = "#4A5568"
      ),
      
      ################################################################
      # 网格线
      ################################################################
      
      panel.grid.major.x = element_line(
        
        color = "#E8EEF5",
        
        linewidth = 0.7       # 横向网格线粗细（增加可读性）
      ),
      
      panel.grid.major.y = element_blank(),  # 不显示纵向网格线
      
      ################################################################
      # 背景
      ################################################################
      
      panel.background = element_rect(
        fill = "#F5FAFC",
        color = NA
      ),
      
      plot.background = element_rect(
        fill = "white",
        color = NA
      )
    )
  
  ####################################################################
  # 保存Importance
  ####################################################################
  
  ggsave(
    
    filename = file.path(
      
      output_dir,
      
      paste0(clean_title, "_Top15_Importance.png")
    ),
    
    plot = p_importance,
    
    width = 5.5,      # 略宽于前两个图（因特征名可能较长）
    height = 4.8,     # 高度适当增加以容纳15行
    dpi = 600,
    
    bg = "white"
  )
  
  ####################################################################
  # 完成
  ####################################################################
  
  cat("\nFinished:", clean_title, "\n")
}

######################################################################
# 7. 完成
######################################################################

cat("\n====================================================\n")
cat("SCI Visualization Finished!\n")
cat("====================================================\n")

cat(output_dir, "\n")




##############################################################
#大图合并代码#
#########################################################
###############################################################################
# SCI 4×3 合成总图
# 风格照搬原始子图（颜色/图层/图例位置/背景 不变）
# 字体统一为合图可读大小（参考 SHAP V16）
# 标签 22pt bold
###############################################################################

rm(list = ls()); gc()
pkgs <- c("data.table","dplyr","ggplot2","cowplot","stringr","scales")
for(p in pkgs) if(!require(p, character.only=TRUE)){ install.packages(p); library(p, character.only=TRUE) }

base_path  <- "C:/Users/user/Desktop/小论文/ML数据"
xgb_dir    <- file.path(base_path, "XGBoost_Results")
output_dir <- file.path(base_path, "Combined_Figure_4x3_v3")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

outcome_order <- c("Outcome_Disease_Non_Progression",
                   "Outcome_Liver_Cirrhosis",
                   "Outcome_Hepatocellular_Carcinoma")

# ===== 配色 =====
model_colors <- c(XGBoost="#C62828",LightGBM="#ff6b78",GBM="#FB8C00",
                  ConditionalRF="#f8d042",MLP="#70d75c",RandomForest="#47c4af",
                  Logistic="#2058a2",GLMnet="#72bcd5")
model_display <- c(XGBoost="XGBoost",LightGBM="LightGBM",RandomForest="RF",
                   GBM="GBM",GLMnet="GLMnet",Logistic="LR",ConditionalRF="CRF",MLP="MLP")
models <- list(XGBoost="XGBoost_Results",LightGBM="LightGBM_Results",
               RandomForest="RandomForest_Results",GBM="GBM_Results",GLMnet="GLMnet_Results",
               Logistic="Logistic_Regression_Results",ConditionalRF="ConditionalRF_Results",MLP="MLP_Results")

positive_color <- "#1565C0"; negative_color <- "#DCEBFA"
nature_dark <- "#0A2540"; bar_blue <- "#1976D2"

clean_title <- function(x){ x<-gsub("^Outcome_","",x); x<-gsub("_"," ",x); gsub("Disease Non Progression","Disease Non-Progression",x) }
extract_auc <- function(f){
  df<-fread(f)
  if("指标"%in%colnames(df)){ mv<-setNames(as.character(df[["数值"]]),df[["指标"]]); auc<-as.numeric(mv["Test_AUC"]) }
  else if("Metric"%in%colnames(df)){ mv<-setNames(as.character(df[["Value"]]),df[["Metric"]]); auc<-as.numeric(mv["Test_AUC"]) }
  else auc<-as.numeric(df[["Test_AUC"]][1])
  round(auc,3)
}

# 标签
TAG <- theme(plot.tag.position=c(0.02,0.98), plot.tag=element_text(size=28,face="bold"))

###############################################################################
# ROC（图层/颜色/图例/玻璃辉光 照搬，字体统一）
###############################################################################
make_roc <- function(outcome){
  ct<-clean_title(outcome); roc_all<-data.frame(); auc_results<-data.frame()
  for(mn in names(models)){
    rf<-file.path(base_path,models[[mn]],paste0(outcome,"_ROC_Data.csv"))
    if(!file.exists(rf))next
    r<-fread(rf,data.table=FALSE); colnames(r)<-tolower(trimws(colnames(r)))
    if(all(c("fpr","tpr")%in%colnames(r))) r<-data.frame(FPR=as.numeric(r$fpr),TPR=as.numeric(r$tpr))
    else r<-data.frame(FPR=1-as.numeric(r$specificity),TPR=as.numeric(r$sensitivity))
    r<-r%>%filter(!is.na(FPR),!is.na(TPR),FPR>=0,FPR<=1,TPR>=0,TPR<=1)%>%arrange(FPR)
    r<-rbind(data.frame(FPR=0,TPR=0),r,data.frame(FPR=1,TPR=1))
    av<-extract_auc(gsub("_ROC_Data.csv","_performance.csv",rf))
    auc_results<-rbind(auc_results,data.frame(Model=mn,AUC=av)); r$Model<-mn; roc_all<-rbind(roc_all,r)
  }
  auc_results<-auc_results%>%arrange(desc(AUC)); bm<-auc_results$Model[1]
  ll<-paste0(model_display[auc_results$Model],"\nAUC = ",sprintf("%.3f",auc_results$AUC)); names(ll)<-auc_results$Model
  roc_all$Model<-factor(roc_all$Model,levels=auc_results$Model)
  roc_all$LS<-ifelse(roc_all$Model==bm,1.8,1.05); roc_all$AL<-ifelse(roc_all$Model==bm,1.0,0.68)
  
  ggplot()+
    annotate("rect",xmin=0,xmax=1,ymin=0,ymax=1,fill="#F5FAFC",alpha=1)+
    geom_line(data=subset(roc_all,Model==bm),aes(FPR,TPR),color=model_colors[bm],linewidth=3.5,alpha=0.12,lineend="round")+
    geom_line(data=roc_all,aes(FPR,TPR,group=Model),color="white",linewidth=2.8,alpha=0.85,lineend="round")+
    geom_line(data=roc_all,aes(FPR,TPR,color=Model,linewidth=LS,alpha=AL),lineend="round")+
    geom_abline(slope=1,intercept=0,linetype="dashed",color="#CBD5E1",linewidth=0.75)+
    scale_color_manual(values=model_colors,labels=ll)+scale_alpha_identity()+scale_linewidth_identity()+
    scale_x_continuous(limits=c(0,1),expand=c(0,0),breaks=seq(0,1,0.2))+
    scale_y_continuous(limits=c(0,1),expand=c(0,0),breaks=seq(0,1,0.2))+
    coord_equal()+
    labs(title=ct,x="False Positive Rate",y="True Positive Rate",color=NULL)+
    theme_classic(base_size=16)+
    theme(plot.title=element_text(hjust=0.5,face="bold",size=28,color="#0A2540"),
          axis.title=element_text(face="bold",size=24,color="#1A1A1A"),
          axis.text=element_text(size=20,color="#333333"),
          axis.line=element_line(linewidth=0.7,color="#6B7280"),
          axis.ticks=element_line(linewidth=0.6,color="#6B7280"),
          legend.position=c(0.8,0.42),
          legend.background=element_rect(fill=alpha("white",0.78),color="#D6DEE8",linewidth=0.6),
          legend.key=element_blank(),legend.text=element_text(size=16,color="#1A1A1A",lineheight=1.05),
          legend.spacing.y=unit(1.5,"mm"),
          panel.background=element_rect(fill="#F5F8FB",color=NA),panel.grid=element_blank(),
          plot.margin=margin(4,4,4,4))
}

###############################################################################
# 密度图（颜色/图层照搬）
###############################################################################
make_density <- function(pred_df,title){
  ggplot(pred_df,aes(Probability,fill=True_Label))+
    geom_density(alpha=0.75,linewidth=1,adjust=1,color="white")+
    scale_fill_manual(values=c("Negative"=negative_color,"Positive"=positive_color))+
    scale_x_continuous(limits=c(0,1),breaks=seq(0,1,0.2),expand=c(0.01,0.01))+
    labs(title=title,x="Predicted Probability",y="Density",fill=NULL)+
    theme_classic(base_size=22)+
    theme(text=element_text(family="sans"),
          plot.title=element_text(hjust=0.5,face="bold",size=28,color=nature_dark),
          axis.title=element_text(face="bold",size=24),
          axis.text=element_text(size=20,color="#333333"),
          axis.ticks=element_line(linewidth=0.7,color="#4A5568"),
          legend.position=c(0.5,0.78),legend.text=element_text(size=18),
          legend.background=element_rect(fill=alpha("white",0.92),color=NA),
          axis.line=element_line(linewidth=0.7,color="#4A5568"),
          panel.background=element_rect(fill="#F5FAFC",color=NA),
          plot.background=element_rect(fill="white",color=NA))
}

###############################################################################
# 箱线图（颜色/图层照搬）
###############################################################################
make_boxplot <- function(pred_df,title){
  ggplot(pred_df,aes(True_Label,Probability,fill=True_Label))+
    geom_boxplot(width=0.58,alpha=0.82,linewidth=1.2,
                 outlier.shape=21,outlier.size=4,outlier.fill="#1976D2",color="#1A1A1A")+
    stat_summary(fun=mean,geom="point",shape=23,size=6,fill="white",color="#1A1A1A")+
    scale_fill_manual(values=c("Negative"=negative_color,"Positive"=positive_color))+
    scale_y_continuous(limits=c(0,1),breaks=seq(0,1,0.2))+
    labs(title=title,x="Actual Outcome",y="Predicted Probability")+
    theme_classic(base_size=22)+
    theme(text=element_text(family="sans"),
          plot.title=element_text(hjust=0.5,face="bold",size=28,color=nature_dark),
          axis.title=element_text(face="bold",size=24),
          axis.text=element_text(size=20,color="#333333"),
          axis.ticks=element_line(linewidth=0.7,color="#4A5568"),
          axis.line=element_line(linewidth=0.7,color="#4A5568"),
          panel.background=element_rect(fill="#F5FAFC",color=NA),
          plot.background=element_rect(fill="white",color=NA),legend.position="none")
}

###############################################################################
# 特征重要性（颜色/图层照搬）
###############################################################################
make_importance <- function(imp_df,title){
  ggplot(imp_df,aes(Importance,Feature_Name))+
    geom_col(fill=bar_blue,alpha=0.9,width=0.7,color="white",linewidth=0.7)+
    geom_text(aes(label=round(Importance,3)),hjust=-0.1,size=4.2)+
    scale_x_continuous(expand=expansion(mult=c(0,0.2)))+
    labs(title=paste0(title,"\nTop 15 Feature Importance"),x="Importance",y=NULL)+
    theme_classic(base_size=22)+
    theme(text=element_text(family="sans"),
          plot.title=element_text(hjust=0.5,face="bold",size=28,color=nature_dark,lineheight=1.1),
          axis.title=element_text(face="bold",size=24),
          axis.text=element_text(size=18,color="#333333"),
          axis.ticks=element_line(linewidth=0.7,color="#4A5568"),
          axis.line=element_line(linewidth=0.7,color="#4A5568"),
          panel.grid.major.x=element_line(color="#E8EEF5",linewidth=0.5),
          panel.grid.major.y=element_blank(),
          panel.background=element_rect(fill="#F5FAFC",color=NA),
          plot.background=element_rect(fill="white",color=NA))
}

###############################################################################
cat("===== 生成 12 张子图 =====\n")
roc_p<-list(); imp_p<-list(); den_p<-list(); box_p<-list()
for(outcome in outcome_order){
  ct<-clean_title(outcome); cat("  ",ct,"\n")
  roc_p[[outcome]]<-make_roc(outcome)
  pf<-file.path(xgb_dir,paste0(outcome,"_prediction.csv"))
  imf<-file.path(xgb_dir,paste0(outcome,"_Feature_Importance.csv"))
  if(!file.exists(pf)||!file.exists(imf))next
  pred_df<-fread(pf)
  pred_df$True_Label<-factor(as.character(pred_df$True_Label),levels=c("0","1"),labels=c("Negative","Positive"))
  pc<-grep("Prob",colnames(pred_df),value=TRUE)[1]; pred_df$Probability<-pred_df[[pc]]
  imp_df<-fread(imf)
  if("Gain"%in%colnames(imp_df)){imp_df<-imp_df%>%arrange(desc(Gain)); imp_df$Importance<-imp_df$Gain}
  else{nc<-names(imp_df)[sapply(imp_df,is.numeric)][1]; imp_df<-imp_df%>%arrange(desc(.data[[nc]])); imp_df$Importance<-imp_df[[nc]]}
  imp_df<-imp_df[1:15,]; fc<-grep("Feature|feature",colnames(imp_df),value=TRUE)[1]
  imp_df$Feature_Name<-factor(imp_df[[fc]],levels=rev(imp_df[[fc]]))
  imp_p[[outcome]]<-make_importance(imp_df,ct)
  den_p[[outcome]]<-make_density(pred_df,ct)
  box_p[[outcome]]<-make_boxplot(pred_df,ct)
}

cat("===== 拼接 =====\n")
plotlist<-c(lapply(outcome_order,function(o)roc_p[[o]]),
            lapply(outcome_order,function(o)imp_p[[o]]),
            lapply(outcome_order,function(o)den_p[[o]]),
            lapply(outcome_order,function(o)box_p[[o]]))
plotlist[[1]]<-plotlist[[1]]+labs(tag="a")+TAG
plotlist[[4]]<-plotlist[[4]]+labs(tag="b")+TAG
plotlist[[7]]<-plotlist[[7]]+labs(tag="c")+TAG
plotlist[[10]]<-plotlist[[10]]+labs(tag="d")+TAG
combined<-cowplot::plot_grid(plotlist=plotlist,ncol=3,align="none")

base<-file.path(output_dir,"Combined_Figure_4x3")
ggsave(paste0(base,".tiff"),combined,width=28,height=30,dpi=600,compression="lzw",bg="white")
ggsave(paste0(base,".pdf"), combined,width=28,height=30,device="pdf",bg="white")
ggsave(paste0(base,".png"), combined,width=28,height=30,dpi=600,bg="white")
cat("  ->",basename(base),"\nDONE")
