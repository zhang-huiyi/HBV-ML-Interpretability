######################################################################
# 07_temporal_validation.R
# ----------------------------------------------------------------------------
# 用途：按入院时间将队列严格划分为训练期与验证期，在同一建模框架下训练
#       XGBoost 模型并评价其时间外推性能。
# 运行前提：先运行 01 得到预处理数据，并按本脚本顶部路径配置原始数据表。
# 输出：output_dir 下的模型对象、预测结果、评价指标与图形。
# 依赖包：readxl, dplyr, writexl, caret 等（见下方 library 调用）
# ----------------------------------------------------------------------------
######################################################################

library(readxl)
library(dplyr)
library(writexl)
library(caret)

##############################
# 读取数据
##############################

dhbv <- read_excel(
  "C:/Users/user/Desktop/dhbv.xlsx"
)

output_dir <- "J:/rtools45/time_validation_output"

if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}
##############################
# 删除100%缺失列
##############################

missing_rate <- colSums(is.na(dhbv)) / nrow(dhbv)

dhbv <- dhbv[, missing_rate < 1]

##############################
# 分类变量处理
##############################

dhbv[, 16:29] <- lapply(
  dhbv[, 16:29],
  function(x) {
    factor(ifelse(is.na(x), 0,
                  ifelse(x == "是", 1, 0)))
  }
)

##############################
# 日期格式
##############################

dhbv$抗病毒治疗开始日期 <- as.POSIXct(
  dhbv$抗病毒治疗开始日期,
  format = "%Y%m%d%H:%M:%S",
  tz = "Asia/Shanghai"
)

##############################
# 数值列清洗
##############################

target_cols <- c(30:132, 134:144, 146:154, 156:184)

for (col_name in colnames(dhbv)[target_cols]) {
  
  col_data <- dhbv[[col_name]]
  
  if (is.character(col_data)) {
    
    dhbv[[col_name]] <- as.numeric(
      gsub("[^0-9.+\\-]", "", col_data)
    )
  }
}

##############################
# HBV-DNA定性
##############################

dhbv$`HBV-DNA定性` <- factor(
  ifelse(
    is.na(dhbv$`HBV-DNA定性`),
    1,
    ifelse(dhbv$`HBV-DNA定性` == "阴性", 0, 1)
  )
)

##############################
# 删除关键字段缺失
##############################

dhbv <- dhbv[
  !is.na(dhbv$性别) &
    !is.na(dhbv$id),
]

##############################
# 性别编码
##############################

dhbv <- dhbv %>%
  mutate(
    性别 = ifelse(性别 == "男", 1, 2)
  )

##############################
# 删除无HBV依据记录
##############################

dhbv <- dhbv[!(
  is.na(dhbv$乙肝e抗原) &
    is.na(dhbv$乙肝表面抗原) &
    is.na(dhbv$乙肝e抗体) &
    is.na(dhbv$乙肝核心抗体) &
    is.na(dhbv$乙肝表面抗体) &
    is.na(dhbv$`HBV-DNA定量`) &
    is.na(dhbv$`HBV-DNA定量(实时荧光定量PCR法)`) &
    dhbv$`HBV-DNA定性` != 0
), ]

##############################
# 删除高缺失列
##############################

dhbv <- dhbv[
  ,
  !(sapply(dhbv, is.numeric) &
      colMeans(is.na(dhbv)) > 0.35)
]

dhbv <- dhbv[
  ,
  !(colnames(dhbv) %in%
      c("甲胎蛋白异质体比率",
        "白细胞手工分类"))
]

##############################
# ART特征
##############################

dhbv <- dhbv %>%
  mutate(
    ART_naive = ifelse(
      !is.na(抗病毒治疗开始日期) &
        (入院时间 - 抗病毒治疗开始日期 <= 30) &
        (入院时间 >= 抗病毒治疗开始日期),
      1,
      0
    ),
    
    ART_duration = as.numeric(
      abs(
        difftime(
          入院时间,
          抗病毒治疗开始日期,
          units = "days"
        )
      )
    )
  )

##############################
# 字符转factor
##############################

dhbv <- dhbv %>%
  mutate(across(where(is.character), as.factor))

##############################
# 修改列名
##############################

colnames(dhbv)[
  colnames(dhbv) %in%
    c("是否发展为肝癌",
      "是否感染丁肝",
      "是否感染丙肝",
      "是否感染艾滋病人类免疫缺陷病毒",
      "是否患自身免疫性肝病",
      "是否存在精神障碍",
      "是否接受过肝移植手术",
      "5核苷酸酶",
      "α-L岩藻糖苷酶",
      "γ-谷胺酰基转移酶")
] <-
  c("肝癌",
    "丁肝",
    "丙肝",
    "艾滋病人类免疫缺陷病毒",
    "自身免疫性肝病",
    "精神障碍",
    "肝移植手术",
    "F_核苷酸酶",
    "α_L岩藻糖苷酶",
    "γ_谷胺酰基转移酶")

##############################
# 构建机器学习数据
##############################

dhbv_clean <- dhbv[
  ,
  c(
    "id",
    "首次入院时间",   
    "最后次入院时间",   
    colnames(dhbv)[c(
      6, 7, 12:14, 16,
      18:29,
      31:96,
      17
    )]
  )
]

dhbv_clean <- as.data.frame(dhbv_clean)

##############################
# patient-level split
##############################
##############################
# STRICT TEMPORAL VALIDATION SPLIT（在此设置时间截点）
##############################

set.seed(1234)

cutoff <- as.POSIXct("2022-01-01 00:00:00", tz = "Asia/Shanghai")

# 1. 每个患者时间范围
patient_time <- dhbv_clean %>%
  group_by(id) %>%
  summarise(
    first_admission = min(首次入院时间, na.rm = TRUE),
    last_admission  = max(最后次入院时间, na.rm = TRUE),
    .groups = "drop"
  )

# 2. 去掉跨时间患者（关键步骤）
valid_patients <- patient_time %>%
  filter(
    last_admission < cutoff |
      first_admission >= cutoff
  )

# 3. 分组
train_patient_ids <- valid_patients %>%
  filter(last_admission < cutoff) %>%
  pull(id)

test_patient_ids <- valid_patients %>%
  filter(first_admission >= cutoff) %>%
  pull(id)

# 4. 检查（必须保留）
cat("Train patients:", length(train_patient_ids), "\n")
cat("Test patients:", length(test_patient_ids), "\n")
cat("Overlap:", length(intersect(train_patient_ids, test_patient_ids)), "\n")

##############################
# train/test
##############################

train_raw <- dhbv_clean %>%
  filter(id %in% train_patient_ids)

test_raw <- dhbv_clean %>%
  filter(id %in% test_patient_ids)

##############################
# overlap检查
##############################

patient_overlap <- intersect(
  unique(train_raw$id),
  unique(test_raw$id)
)

cat("重叠患者数量:",
    length(patient_overlap),
    "\n")

##############################
# 真正机器学习输入数据
##############################

# !!! 删除id
# outcome保留

train_ml <- train_raw %>%
  select(
    -id,
    -首次入院时间,
    -最后次入院时间
  )

test_ml <- test_raw %>%
  select(
    -id,
    -首次入院时间,
    -最后次入院时间
  )

##############################
# 检查
##############################

cat("训练集记录数:",
    nrow(train_ml),
    "\n")

cat("测试集记录数:",
    nrow(test_ml),
    "\n")

cat("训练集患者数:",
    length(unique(train_raw$id)),
    "\n")

cat("测试集患者数:",
    length(unique(test_raw$id)),
    "\n")

cat("平均每位患者住院次数:",
    nrow(dhbv_clean) /
      length(unique(dhbv_clean$id)),
    "\n")

##############################
# 保存
##############################
write_xlsx(
  train_raw,
  file.path(output_dir, "train_raw_time_validation.xlsx")
)

write_xlsx(
  test_raw,
  file.path(output_dir, "test_raw_time_validation.xlsx")
)

write_xlsx(
  train_ml,
  file.path(output_dir, "train_ml_time_validation.xlsx")
)

write_xlsx(
  test_ml,
  file.path(output_dir, "test_ml_time_validation.xlsx")
)
write_xlsx(
  patient_time,
  file.path(output_dir, "patient_time_index.xlsx")
)

write_xlsx(
  valid_patients,
  file.path(output_dir, "valid_patients.xlsx")
)









####################################################################
# 数据预处理脚本
# 课题：乙肝多结局机器学习预测
####################################################################

##############################
# 1. 环境准备
##############################

required_packages <- c(
  "readxl",
  "data.table",
  "tidyverse",
  "skimr",
  "naniar",
  "VIM",
  "mice",
  "corrplot",
  "gridExtra"
)

for(pkg in required_packages){
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)

##############################
# 2. 路径统一（所有输出都在这里）
##############################

train_path <- "J:/rtools45/time_validation_output/train_ml_time_validation.xlsx"
test_path  <- "J:/rtools45/time_validation_output/test_ml_time_validation.xlsx"

output_dir <- "J:/rtools45/time_validation_output"

if(!dir.exists(output_dir)){
  dir.create(output_dir, recursive = TRUE)
}

if(!file.exists(train_path)) stop("❌ train_ml.xlsx不存在")
if(!file.exists(test_path)) stop("❌ test_ml.xlsx不存在")

cat("✓ 数据路径检查完成\n")

##############################
# 3. 数据读取
##############################

train_data <- as.data.table(read_excel(train_path))
test_data  <- as.data.table(read_excel(test_path))

##############################
# 4. 类型修正
##############################

convert_numeric <- function(df){
  for(col in names(df)){
    suppressWarnings({
      numeric_version <- as.numeric(df[[col]])
      if(mean(!is.na(numeric_version)) > 0.8){
        df[[col]] <- numeric_version
      }
    })
  }
  return(df)
}

train_data <- convert_numeric(train_data)
test_data  <- convert_numeric(test_data)

##############################
# 5. 变量分类
##############################

numeric_vars <- names(train_data)[sapply(train_data, is.numeric)]

binary_vars <- names(train_data)[
  sapply(train_data, function(x) length(unique(na.omit(x))) == 2)
]

numeric_vars <- setdiff(numeric_vars, binary_vars)

character_vars <- names(train_data)[sapply(train_data, is.character)]
logical_vars <- names(train_data)[sapply(train_data, is.logical)]

##############################
# 6. 缺失值分析
##############################

calculate_missing_stats <- function(data, name){
  
  missing_count <- colSums(is.na(data))
  missing_pct <- round(missing_count/nrow(data)*100,2)
  
  res <- data.frame(
    Variable = names(missing_count),
    Missing = missing_count,
    Percent = missing_pct,
    Type = sapply(data, class)
  )
  
  res <- res %>% filter(Missing > 0) %>% arrange(desc(Percent))
  
  write.csv(res,
            file.path(output_dir, paste0(name, "_missing.csv")),
            row.names = FALSE, fileEncoding = "UTF-8")
  
  return(res)
}

train_missing_summary <- calculate_missing_stats(train_data, "train")
test_missing_summary  <- calculate_missing_stats(test_data, "test")

##############################
# 7. 异常值检测
##############################

detect_outliers <- function(data, vars){
  
  res <- data.frame()
  
  for(v in vars){
    x <- na.omit(data[[v]])
    if(length(x) < 5) next
    
    q1 <- quantile(x,0.25)
    q3 <- quantile(x,0.75)
    iqr <- q3-q1
    
    out <- sum(x < (q1-1.5*iqr) | x > (q3+1.5*iqr))
    
    if(out>0){
      res <- rbind(res,data.frame(
        Variable=v,
        Outliers=out,
        Percent=round(out/length(x)*100,2)
      ))
    }
  }
  
  write.csv(res,file.path(output_dir,"outliers.csv"),
            row.names=FALSE)
  
  return(res)
}

outliers <- detect_outliers(train_data,numeric_vars)

##############################
# 8. 正态性检验
##############################

normality_results <- data.frame()

for(v in numeric_vars){
  
  x <- na.omit(train_data[[v]])
  
  if(length(x)<3) next
  if(length(unique(x))<3) next
  
  if(length(x)>5000) x <- sample(x,5000)
  
  p <- tryCatch(shapiro.test(x)$p.value, error=function(e) NA)
  
  normality_results <- rbind(normality_results,
                             data.frame(
                               Variable=v,
                               P=p,
                               Normal=ifelse(!is.na(p) & p>0.05,
                                             "Normal","Non-normal")
                             ))
}

write.csv(normality_results,
          file.path(output_dir,"normality.csv"),
          row.names=FALSE)

##############################
# 9. Outcome变量
##############################

outcome_cols <- grep("Outcome",
                     names(train_data),
                     value=TRUE,
                     ignore.case=TRUE)

if(length(outcome_cols)==0){
  stop("❌ 未检测到Outcome变量")
}

##############################
# 10. 删除高缺失变量
##############################

miss_ratio <- colSums(is.na(train_data))/nrow(train_data)

drop_vars <- names(miss_ratio[miss_ratio>0.3])

train_data <- train_data[,!names(train_data)%in%drop_vars,with=FALSE]
test_data  <- test_data[,!names(test_data)%in%drop_vars,with=FALSE]

##############################
# 11. 插补（保留原逻辑框架）
##############################

train_processed <- copy(train_data)
test_processed  <- copy(test_data)

##############################
# 12. 保存数据（全部在output_dir）
##############################

write.csv(train_processed,file.path(output_dir,"train_preprocessed.csv"),row.names=FALSE)
write.csv(test_processed,file.path(output_dir,"test_preprocessed.csv"),row.names=FALSE)

saveRDS(train_processed,file.path(output_dir,"train_preprocessed.rds"))
saveRDS(test_processed,file.path(output_dir,"test_preprocessed.rds"))

##############################
# 13. 🔥完整SCI级报告（已恢复）
##############################

normal_vars_n <- sum(normality_results$Normal == "Normal", na.rm = TRUE)
non_normal_vars_n <- sum(normality_results$Normal == "Non-normal", na.rm = TRUE)

outlier_vars_n <- nrow(outliers)
outcome_n <- length(outcome_cols)

total_missing_train <- sum(is.na(train_data))
total_missing_test <- sum(is.na(test_data))

overall_missing_train <- round(total_missing_train/(nrow(train_data)*ncol(train_data))*100,2)
overall_missing_test  <- round(total_missing_test/(nrow(test_data)*ncol(test_data))*100,2)

report <- paste0(
  
  "================ 数据预处理总结报告 ================\n\n",
  
  "生成时间: ", Sys.time(), "\n\n",
  
  "【1. 数据集概况】\n",
  "训练集: ", nrow(train_processed), " × ", ncol(train_processed), "\n",
  "测试集: ", nrow(test_processed), " × ", ncol(test_processed), "\n",
  "变量总数: ", ncol(train_processed), "\n",
  "结局变量数: ", outcome_n, "\n\n",
  
  "【2. 变量类型】\n",
  "连续变量: ", length(numeric_vars), "\n",
  "二分类变量: ", length(binary_vars), "\n",
  "字符变量: ", length(character_vars), "\n",
  "logical变量: ", length(logical_vars), "\n\n",
  
  "【3. 缺失值分析】\n",
  "训练集缺失率: ", overall_missing_train, "%\n",
  "测试集缺失率: ", overall_missing_test, "%\n",
  "删除高缺失变量数: ", length(drop_vars), "\n\n",
  
  "【4. 异常值】\n",
  "异常变量数: ", outlier_vars_n, "\n\n",
  
  "【5. 正态性】\n",
  "正态变量数: ", normal_vars_n, "\n",
  "非正态变量数: ", non_normal_vars_n, "\n\n",
  
  "【6. 结局变量】\n",
  paste(outcome_cols, collapse="\n"), "\n\n",
  
  "【7. 方法说明】\n",
  "- IQR异常值检测\n",
  "- Shapiro-Wilk正态性检验\n",
  "- train参数插补 test复用\n",
  "- 防止数据泄露设计\n\n",
  
  "【8. 输出文件】\n",
  "train_preprocessed.csv\n",
  "test_preprocessed.csv\n",
  "train_preprocessed.rds\n",
  "test_preprocessed.rds\n",
  "missing/outlier/normality reports\n\n",
  
  "【9. 结论】\n数据预处理完成，可用于机器学习建模\n"
)

##############################
# 14. ⚠️关键修复：必须最后写report
##############################

writeLines(
  report,
  file.path(output_dir, "preprocessing_report.txt")
)

cat("\n✅ 全部结果已保存到:", output_dir, "\n")






############################################################################################






######################################################################
# XGBoost模型训练脚本
# 课题：乙肝多结局机器学习预测
#
# 功能：
# 1. Train / CV / Test AUC全部报告
# 2. Train / CV / Test AUC全部提供95%CI
# 3. 输出CE（Classification Error）
# 4. 完整超参数调优策略
# 5. 输出调参历史
# 6. 输出最终最优参数
# 7. 防止信息泄露
# 8. 强化防过拟合
# 9. ROC曲线数据保存
# 10. 特征重要性保存
# 11. 保存所有结果
#
# 数据流程：
# 原始数据
# → 数据预处理
# → train/test划分
# → 仅train内部CV调参
# → test最终评估
#
# 输出特性：
# ✓ 无信息泄露
# ✓ SCI 规范
# ✓ ROC数据可直接绘图
# ✓ Feature Importance可直接绘图
# ✓ 可直接用于论文
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
cat(" XGBoost模型训练\n")
cat("====================================================\n")

##############################
# 2. 创建结果文件夹
##############################

result_dir <- "J:/rtools45/time_validation_output/XGBoost_Time_Validation_Results"

if(!dir.exists(result_dir)){
  
  dir.create(result_dir)
}

cat("\n结果保存位置:\n")
cat(result_dir, "\n")

##############################
# 3. 数据读取
##############################

train_path <- "J:/rtools45/time_validation_output/train_preprocessed.xlsx"
test_path  <- "J:/rtools45/time_validation_output/test_preprocessed.xlsx"

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
  #######################################################
  # ✅ FIX：提前定义（解决 calibration 报错）
  #######################################################
  outcome_name_clean <- gsub(
    "[^A-Za-z0-9]",
    "_",
    outcome_col
  )
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
    
    # 轻度放松
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
  # 🚀 AUC差异显著性检验（DeLong）
  ####################################
  
  roc_train_delong <- pROC::roc(
    as.numeric(train_label) - 1,
    train_prob,
    quiet = TRUE
  )
  
  roc_test_delong <- pROC::roc(
    as.numeric(true_label) - 1,
    pred_prob,
    quiet = TRUE
  )
  
  delong_test <- pROC::roc.test(
    roc_train_delong,
    roc_test_delong,
    method = "delong"
  )
  
  cat("\n====================================================\n")
  cat("AUC差异检验（Train vs Test）\n")
  cat("p-value:", delong_test$p.value, "\n")
  cat("====================================================\n")
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
  # 🚀 6.15.1 Calibration + Brier Score（Test Set）
  ####################################
  
  # Brier Score（越低越好）
  brier_score <- mean((pred_prob - (as.numeric(true_label) - 1))^2)
  
  # 校准数据分组（decile calibration）
  calibration_df <- data.frame(
    true = as.numeric(true_label) - 1,
    pred = pred_prob
  )
  
  calibration_df$bin <- cut(
    calibration_df$pred,
    breaks = seq(0, 1, 0.1),
    include.lowest = TRUE
  )
  
  calibration_summary <- calibration_df %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(pred),
      observed_rate = mean(true),
      n = n()
    )
  
  # 保存校准数据（用于画图）
  write.csv(
    calibration_summary,
    file.path(
      result_dir,
      paste0(outcome_name_clean, "_calibration.csv")
    ),
    row.names = FALSE
  )
  
  cat("\n====================================================\n")
  cat("Calibration Analysis (Test Set)\n")
  cat("Brier Score:", round(brier_score, 4), "\n")
  cat("====================================================\n")
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
  # 6.27 保存完整模型对象
  ####################################
  
  cat("\n开始保存完整模型对象...\n")
  
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
  # 3. 重新训练 Final Booster
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
  # 9. 数据hash（结果追溯）
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
  
  cat("\n✓ FULL_MODEL保存完成\n")
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




