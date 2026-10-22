####################################################################
# 数据预处理脚本
# 课题：乙肝多结局机器学习预测
# 功能：
# 1. 数据导入
# 2. 数据类型修正
# 3. 缺失值分析
# 4. 异常值检测
# 5. 正态性检验
# 6. 结局类别分析
# 7. 缺失值插补（防数据泄露）
# 8. 基线特征表生成（SMD函数）
# 9. 保存机器学习可用数据
# 10. 生成完整预处理报告
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
  "gridExtra",
  "writexl"
)

for(pkg in required_packages){
  
  if(!require(pkg, character.only = TRUE)){
    install.packages(pkg)
    library(pkg, character.only = TRUE)
  }
}

set.seed(123)


library(readxl)
library(dplyr)
library(writexl)

##############################
# 读取数据
##############################

dhbv <- read_excel(
  "C:/Users/user/Desktop/小论文/乙肝三结局预测数据结果集合/dhbv.xlsx 2026.05.17.xlsx"
)

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
    "id",   # 保留patient_id
    
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

set.seed(1234)

patient_ids <- unique(dhbv_clean$id)

train_patient_ids <- sample(
  patient_ids,
  size = round(0.7 * length(patient_ids))
)

test_patient_ids <- setdiff(
  patient_ids,
  train_patient_ids
)

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
  select(-id)

test_ml <- test_raw %>%
  select(-id)

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
  "J:/rtools45/train_raw_patient_level.xlsx"
)

write_xlsx(
  test_raw,
  "J:/rtools45/test_raw_patient_level.xlsx"
)

write_xlsx(
  train_ml,
  "J:/rtools45/train_ml.xlsx"
)

write_xlsx(
  test_ml,
  "J:/rtools45/test_ml.xlsx"
)


##############################
# 原始数据清洗与训练/测试患者级划分
##############################

train_data <- as.data.table(train_ml)
test_data  <- as.data.table(test_ml)

cat("✓ 原始数据清洗与划分完成（train_data / test_data 已在内存中生成）
")

##############################
# 4. 自动修正数值变量类型
##############################

convert_numeric <- function(df){
  
  for(col in names(df)){
    
    suppressWarnings({
      
      numeric_version <- as.numeric(df[[col]])
      
      # 超过80%可转数值
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
# 5. 数据概览
##############################

cat("\n【第二步】数据概览\n")
cat(strrep("=", 60), "\n")

cat("\n【训练集】\n")
print(skim(train_data))

cat("\n【测试集】\n")
print(skim(test_data))

##############################
# 6. 变量分类
##############################

numeric_vars <- names(train_data)[
  sapply(train_data, is.numeric)
]

character_vars <- names(train_data)[
  sapply(train_data, is.character)
]

logical_vars <- names(train_data)[
  sapply(train_data, is.logical)
]

# 自动识别二分类变量
binary_vars <- names(train_data)[
  sapply(train_data, function(x){
    
    length(unique(na.omit(x))) == 2
  })
]

# 从连续变量中去除二分类变量
numeric_vars <- setdiff(numeric_vars, binary_vars)

cat("\n连续变量数:", length(numeric_vars), "\n")
cat("二分类变量数:", length(binary_vars), "\n")
cat("字符变量数:", length(character_vars), "\n")

##############################
# 7. 缺失值分析
##############################

cat("\n【第三步】缺失值分析\n")
cat(strrep("=", 60), "\n")

calculate_missing_stats <- function(data, data_name){
  
  cat("\n", data_name, "\n")
  cat(strrep("-", 60), "\n")
  
  missing_count <- colSums(is.na(data))
  
  missing_pct <- round(
    missing_count / nrow(data) * 100,
    2
  )
  
  missing_summary <- data.frame(
    变量名 = names(missing_count),
    缺失数 = missing_count,
    缺失率_percent = missing_pct,
    数据类型 = sapply(data, class)
  )
  
  missing_summary <- missing_summary %>%
    filter(缺失数 > 0) %>%
    arrange(desc(缺失率_percent))
  
  if(nrow(missing_summary) > 0){
    
    print(missing_summary)
    
  }else{
    
    cat("✓ 无缺失值\n")
  }
  
  cat(
    "\n总体缺失率:",
    round(
      sum(is.na(data)) /
        prod(dim(data)) * 100,
      2
    ),
    "%\n"
  )
  
  return(missing_summary)
}

train_missing_summary <- calculate_missing_stats(
  train_data,
  "训练集缺失值统计"
)

test_missing_summary <- calculate_missing_stats(
  test_data,
  "测试集缺失值统计"
)

write.csv(
  train_missing_summary,
  "J:/rtools45/训练集缺失值统计.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  test_missing_summary,
  "J:/rtools45/测试集缺失值统计.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)



##############################
# 9. 异常值检测
##############################

cat("\n【第四步】异常值检测\n")
cat(strrep("=", 60), "\n")

detect_outliers <- function(data, numeric_vars){
  
  result <- data.frame()
  
  for(var in numeric_vars){
    
    values <- data[[var]]
    
    values <- values[!is.na(values)]
    
    if(length(values) < 5) next
    
    Q1 <- quantile(values, 0.25)
    
    Q3 <- quantile(values, 0.75)
    
    IQR_val <- Q3 - Q1
    
    lower <- Q1 - 1.5 * IQR_val
    
    upper <- Q3 + 1.5 * IQR_val
    
    outliers <- sum(
      values < lower |
        values > upper
    )
    
    if(outliers > 0){
      
      result <- rbind(
        result,
        data.frame(
          变量 = var,
          异常值数 = outliers,
          异常值占比_percent =
            round(outliers / length(values) * 100, 2)
        )
      )
    }
  }
  
  return(result)
}

outlier_result <- detect_outliers(
  train_data,
  numeric_vars
)

print(outlier_result)

write.csv(
  outlier_result,
  "J:/rtools45/异常值统计.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

##############################
# 10. 正态性检验
##############################

cat("\n【第五步】正态性检验\n")
cat(strrep("=", 60), "\n")

normality_results <- data.frame()

for(var in numeric_vars){
  
  values <- train_data[[var]]
  
  values <- values[!is.na(values)]
  
  if(length(values) < 3) next
  
  if(length(unique(values)) < 3) next
  
  if(length(values) > 5000){
    
    values <- sample(values, 5000)
  }
  
  test_result <- tryCatch(
    
    shapiro.test(values),
    
    error = function(e) NULL
  )
  
  if(is.null(test_result)) next
  
  is_normal <- test_result$p.value > 0.05
  
  normality_results <- rbind(
    normality_results,
    data.frame(
      变量 = var,
      p值 = round(test_result$p.value, 5),
      正态性 =
        ifelse(is_normal,
               "正态",
               "非正态")
    )
  )
}

print(normality_results)

write.csv(
  normality_results,
  "J:/rtools45/正态性检验结果.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

##############################
# 11. 识别结局变量
##############################

cat("\n【第六步】结局变量分析\n")
cat(strrep("=", 60), "\n")

outcome_cols <- grep(
  "Outcome",
  names(train_data),
  ignore.case = TRUE,
  value = TRUE
)

if(length(outcome_cols) == 0){
  
  stop("❌ 未检测到Outcome结局变量")
}

cat("检测到结局变量:\n")
print(outcome_cols)

##############################
# 12. 结局类别统计
##############################

outcome_summary <- data.frame()

for(outcome in outcome_cols){
  
  cat("\n", outcome, "\n")
  cat(strrep("-", 60), "\n")
  
  train_table <- table(
    train_data[[outcome]]
  )
  
  print(train_table)
  
  train_pct <- round(
    prop.table(train_table) * 100,
    2
  )
  
  print(train_pct)
  
  imbalance_ratio <-
    min(train_pct) / max(train_pct)
  
  if(imbalance_ratio < 0.3){
    
    cat("⚠ 存在严重类别不平衡\n")
  }
  
  tmp <- data.frame(
    Outcome = outcome,
    类别 = names(train_table),
    样本数 = as.numeric(train_table),
    占比 = as.numeric(train_pct)
  )
  
  outcome_summary <- rbind(
    outcome_summary,
    tmp
  )
}

write.csv(
  outcome_summary,
  "J:/rtools45/结局变量分布.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

##############################
# 13. 缺失率>30%变量删除
##############################

missing_ratio <- colSums(
  is.na(train_data)
) / nrow(train_data)

high_missing_vars <- names(
  missing_ratio[
    missing_ratio > 0.3
  ]
)

cat("\n删除缺失率>30%的变量:\n")

print(high_missing_vars)

train_data <- train_data[
  ,
  !names(train_data) %in%
    high_missing_vars,
  with = FALSE
]

test_data <- test_data[
  ,
  !names(test_data) %in%
    high_missing_vars,
  with = FALSE
]

##############################
# 14. 使用训练集参数插补
##############################

cat("\n【第七步】缺失值插补\n")
cat(strrep("=", 60), "\n")

train_processed <- copy(train_data)
test_processed  <- copy(test_data)

impute_parameters <- list()

###################
# 连续变量
###################

for(var in numeric_vars){
  
  if(!(var %in% names(train_processed))) next
  
  train_values <- train_processed[[var]]
  
  non_missing <- train_values[
    !is.na(train_values)
  ]
  
  if(length(non_missing) < 3) next
  
  if(length(unique(non_missing)) < 3){
    
    impute_value <- median(
      non_missing,
      na.rm = TRUE
    )
    
  }else{
    
    test_result <- tryCatch(
      shapiro.test(
        sample(
          non_missing,
          min(length(non_missing), 5000)
        )
      ),
      error = function(e) NULL
    )
    
    if(is.null(test_result)){
      
      is_normal <- FALSE
      
    }else{
      
      is_normal <- test_result$p.value > 0.05
    }
    
    if(is_normal){
      
      impute_value <- mean(
        non_missing,
        na.rm = TRUE
      )
      
    }else{
      
      impute_value <- median(
        non_missing,
        na.rm = TRUE
      )
    }
  }
  
  impute_parameters[[var]] <- impute_value
  
  train_processed[[var]][
    is.na(train_processed[[var]])
  ] <- impute_value
  
  test_processed[[var]][
    is.na(test_processed[[var]])
  ] <- impute_value
  
  cat(
    "变量:",
    var,
    "插补值:",
    round(impute_value, 3),
    "\n"
  )
}

###################
# 分类变量
###################

categorical_vars <- c(
  binary_vars,
  character_vars,
  logical_vars
)

categorical_vars <- intersect(
  categorical_vars,
  names(train_processed)
)

for(var in categorical_vars){
  
  if(sum(is.na(train_processed[[var]])) > 0){
    
    mode_value <- names(
      sort(
        table(
          train_processed[[var]]
        ),
        decreasing = TRUE
      )
    )[1]
    
    train_processed[[var]][
      is.na(train_processed[[var]])
    ] <- mode_value
    
    test_processed[[var]][
      is.na(test_processed[[var]])
    ] <- mode_value
  }
}

##############################
# 15. 最终缺失值检查
##############################

cat("\n训练集最终缺失值:",
    sum(is.na(train_processed)),
    "\n")

cat("测试集最终缺失值:",
    sum(is.na(test_processed)),
    "\n")

##############################
# 16. 基线表生成 + P值检验 + SMD
##############################

cat("\n【第八步】基线表生成\n")
cat(strrep("=", 60), "\n")

baseline_table <- data.frame()

###################
# SMD函数
###################

calculate_smd_continuous <- function(train, test){
  
  train <- train[!is.na(train)]
  test  <- test[!is.na(test)]
  
  mean_train <- mean(train)
  mean_test  <- mean(test)
  
  sd_train <- sd(train)
  sd_test  <- sd(test)
  
  pooled_sd <- sqrt(
    (sd_train^2 + sd_test^2) / 2
  )
  
  if(pooled_sd == 0){
    return(0)
  }
  
  smd <- abs(mean_train - mean_test) / pooled_sd
  
  return(round(smd, 3))
}

calculate_smd_binary <- function(train, test){
  
  train <- as.numeric(as.character(train))
  test  <- as.numeric(as.character(test))
  
  p1 <- mean(train, na.rm = TRUE)
  p2 <- mean(test, na.rm = TRUE)
  
  pooled <- sqrt(
    (p1 * (1 - p1) + p2 * (1 - p2)) / 2
  )
  
  if(pooled == 0){
    return(0)
  }
  
  smd <- abs(p1 - p2) / pooled
  
  return(round(smd, 3))
}

###################
# 连续变量
###################

for(var in numeric_vars){
  
  if(!(var %in% names(train_processed))) next
  
  train_vals <- train_processed[[var]]
  test_vals  <- test_processed[[var]]
  
  train_vals_nonmiss <- train_vals[!is.na(train_vals)]
  test_vals_nonmiss  <- test_vals[!is.na(test_vals)]
  
  if(length(train_vals_nonmiss) < 3) next
  
  normal_row <- normality_results[
    normality_results$变量 == var,
  ]
  
  is_normal <- FALSE
  
  if(nrow(normal_row) > 0){
    is_normal <- normal_row$正态性 == "正态"
  }
  
  ###################
  # 统计描述
  ###################
  
  if(is_normal){
    
    train_stat <- paste0(
      round(mean(train_vals_nonmiss), 2),
      " ± ",
      round(sd(train_vals_nonmiss), 2)
    )
    
    test_stat <- paste0(
      round(mean(test_vals_nonmiss), 2),
      " ± ",
      round(sd(test_vals_nonmiss), 2)
    )
    
    ###################
    # t检验
    ###################
    
    p_value <- tryCatch(
      t.test(train_vals, test_vals)$p.value,
      error = function(e) NA
    )
    
  }else{
    
    train_stat <- paste0(
      round(median(train_vals_nonmiss), 2),
      " (",
      round(quantile(train_vals_nonmiss, 0.25), 2),
      "-",
      round(quantile(train_vals_nonmiss, 0.75), 2),
      ")"
    )
    
    test_stat <- paste0(
      round(median(test_vals_nonmiss), 2),
      " (",
      round(quantile(test_vals_nonmiss, 0.25), 2),
      "-",
      round(quantile(test_vals_nonmiss, 0.75), 2),
      ")"
    )
    
    ###################
    # Wilcoxon检验
    ###################
    
    p_value <- tryCatch(
      wilcox.test(train_vals, test_vals)$p.value,
      error = function(e) NA
    )
  }
  
  ###################
  # SMD
  ###################
  
  smd_value <- calculate_smd_continuous(
    train_vals_nonmiss,
    test_vals_nonmiss
  )
  
  baseline_table <- rbind(
    baseline_table,
    data.frame(
      Variable = var,
      Type = "Continuous Variable",
      Training_Set = train_stat,
      Testing_Set = test_stat,
      SMD = smd_value,
      P_Value = ifelse(
        is.na(p_value),
        "",
        format.pval(p_value, digits = 3, eps = 0.001)
      )
    )
  )
}

###################
# 分类变量
###################

for(var in categorical_vars){
  
  if(!(var %in% names(train_processed))) next
  
  train_tab <- table(train_processed[[var]])
  test_tab  <- table(test_processed[[var]])
  
  all_levels <- union(
    names(train_tab),
    names(test_tab)
  )
  
  ###################
  # 构建列联表
  ###################
  
  combined_data <- data.frame(
    group = c(
      rep("Train", length(train_processed[[var]])),
      rep("Test", length(test_processed[[var]]))
    ),
    value = c(
      as.character(train_processed[[var]]),
      as.character(test_processed[[var]])
    )
  )
  
  contingency_table <- table(
    combined_data$group,
    combined_data$value
  )
  
  ###################
  # 卡方/Fisher检验
  ###################
  
  p_value <- tryCatch({
    
    chi_test <- chisq.test(contingency_table)
    
    if(any(chi_test$expected < 5)){
      
      fisher.test(contingency_table)$p.value
      
    }else{
      
      chi_test$p.value
    }
    
  }, error = function(e) NA)
  
  ###################
  # 输出每个level
  ###################
  
  first_row <- TRUE
  
  for(level in all_levels){
    
    train_n <- ifelse(
      level %in% names(train_tab),
      train_tab[level],
      0
    )
    
    test_n <- ifelse(
      level %in% names(test_tab),
      test_tab[level],
      0
    )
    
    ###################
    # SMD
    ###################
    
    train_binary <- ifelse(
      as.character(train_processed[[var]]) == level,
      1,
      0
    )
    
    test_binary <- ifelse(
      as.character(test_processed[[var]]) == level,
      1,
      0
    )
    
    smd_value <- calculate_smd_binary(
      train_binary,
      test_binary
    )
    
    baseline_table <- rbind(
      baseline_table,
      data.frame(
        Variable = paste0(var, "=", level),
        Type = "Categorical Variable",
        Training_Set = paste0(
          train_n,
          " (",
          round(train_n /
                  nrow(train_processed) * 100, 1),
          "%)"
        ),
        Testing_Set = paste0(
          test_n,
          " (",
          round(test_n /
                  nrow(test_processed) * 100, 1),
          "%)"
        ),
        SMD = ifelse(first_row, smd_value, ""),
        P_Value = ifelse(
          first_row,
          format.pval(
            p_value,
            digits = 3,
            eps = 0.001
          ),
          ""
        )
      )
    )
    
    first_row <- FALSE
  }
}

###################
# 输出基线表
###################

print(baseline_table)

###################
# 保存基线表
###################

write.csv(
  baseline_table,
  "J:/rtools45/Baseline_Table_With_SMD.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

cat("✓ 基线表（含SMD）保存完成\n")

##############################
# 17. 保存预处理数据
##############################

cat("\n【第九步】保存数据\n")
cat(strrep("=", 60), "\n")

saveRDS(
  train_processed,
  "J:/rtools45/train_preprocessed.rds"
)

saveRDS(
  test_processed,
  "J:/rtools45/test_preprocessed.rds"
)

write.csv(
  train_processed,
  "J:/rtools45/train_preprocessed.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

write.csv(
  test_processed,
  "J:/rtools45/test_preprocessed.csv",
  row.names = FALSE,
  fileEncoding = "UTF-8"
)

# 供 02 与 06 直接读取的建模数据
write_xlsx(
  train_processed,
  "J:/rtools45/train_preprocessed.xlsx"
)

write_xlsx(
  test_processed,
  "J:/rtools45/test_preprocessed.xlsx"
)

cat("✓ 预处理数据保存完成\n")


##############################
# 18. 生成详细预处理报告
##############################

cat("\n【第十步】生成报告\n")
cat(strrep("=", 60), "\n")

###################
# 正态变量统计
###################

normal_vars_n <- sum(
  normality_results$正态性 == "正态"
)

non_normal_vars_n <- sum(
  normality_results$正态性 == "非正态"
)

###################
# 缺失值统计
###################

total_missing_train <- sum(is.na(train_data))

total_missing_test <- sum(is.na(test_data))

overall_missing_train <- round(
  total_missing_train /
    (nrow(train_data) * ncol(train_data)) * 100,
  2
)

overall_missing_test <- round(
  total_missing_test /
    (nrow(test_data) * ncol(test_data)) * 100,
  2
)

###################
# 异常值统计
###################

outlier_vars_n <- nrow(outlier_result)

###################
# Outcome统计
###################

outcome_n <- length(outcome_cols)

###################
# 生成报告
###################

report <- paste0(
  
  "================ 数据预处理总结报告 ================\n\n",
  
  "生成时间: ",
  Sys.time(),
  "\n\n",
  
  "【1. 数据集概况】\n",
  "--------------------------------------------------\n",
  
  "训练集维度: ",
  nrow(train_processed),
  " × ",
  ncol(train_processed),
  "\n",
  
  "测试集维度: ",
  nrow(test_processed),
  " × ",
  ncol(test_processed),
  "\n",
  
  "总变量数: ",
  ncol(train_processed),
  "\n",
  
  "结局变量数: ",
  outcome_n,
  "\n\n",
  
  "【2. 变量类型统计】\n",
  "--------------------------------------------------\n",
  
  "连续变量数: ",
  length(numeric_vars),
  "\n",
  
  "二分类变量数: ",
  length(binary_vars),
  "\n",
  
  "字符变量数: ",
  length(character_vars),
  "\n",
  
  "logical变量数: ",
  length(logical_vars),
  "\n\n",
  
  "【3. 缺失值分析】\n",
  "--------------------------------------------------\n",
  
  "训练集总缺失值数: ",
  total_missing_train,
  "\n",
  
  "测试集总缺失值数: ",
  total_missing_test,
  "\n",
  
  "训练集总体缺失率: ",
  overall_missing_train,
  "%\n",
  
  "测试集总体缺失率: ",
  overall_missing_test,
  "%\n",
  
  "缺失率 >30% 删除变量数: ",
  length(high_missing_vars),
  "\n",
  
  ifelse(
    length(high_missing_vars) > 0,
    
    paste0(
      "删除变量:\n",
      paste(high_missing_vars,
            collapse = ", "),
      "\n"
    ),
    
    "无缺失率 >30% 变量\n"
  ),
  
  "\n",
  
  "【4. 异常值分析】\n",
  "--------------------------------------------------\n",
  
  "异常值检测方法: IQR法（1.5×IQR）\n",
  
  "存在异常值的变量数: ",
  outlier_vars_n,
  "\n\n",
  
  "【5. 正态性检验】\n",
  "--------------------------------------------------\n",
  
  "正态变量数: ",
  normal_vars_n,
  "\n",
  
  "非正态变量数: ",
  non_normal_vars_n,
  "\n",
  
  "正态性检验方法: Shapiro-Wilk检验\n\n",
  
  "【6. 结局变量分布】\n",
  "--------------------------------------------------\n",
  
  "检测到结局变量:\n",
  
  paste(outcome_cols,
        collapse = "\n"),
  
  "\n\n",
  
  "类别不平衡已完成检测。\n\n",
  
  "【7. 缺失值插补策略】\n",
  "--------------------------------------------------\n",
  
  "1. 删除缺失率 >30% 变量\n",
  "2. 正态连续变量使用均值插补\n",
  "3. 非正态连续变量使用中位数插补\n",
  "4. 分类变量使用众数插补\n",
  "5. 测试集严格使用训练集参数插补\n",
  "6. 插补过程中避免数据泄露（data leakage）\n\n",
  
  "【8. 基线特征分析】\n",
  "--------------------------------------------------\n",
  
  "连续变量:\n",
  "- 正态分布: 均值 ± 标准差\n",
  "- 非正态分布: 中位数（四分位间距）\n\n",
  
  "分类变量:\n",
  "- 使用 n (%) 表示\n\n",
  
  "P值检验:\n",
  "- 正态变量: t检验\n",
  "- 非正态变量: Wilcoxon秩和检验\n",
  "- 分类变量: 卡方检验/Fisher精确检验\n\n",
  
  "【9. 输出文件】\n",
  "--------------------------------------------------\n",
  
  "1. train_preprocessed.csv / .rds / .xlsx\n",
  "2. test_preprocessed.csv / .rds / .xlsx\n",
  "5. 基线特征表_含P值.csv\n",
  "6. 训练集缺失值统计.csv\n",
  "7. 测试集缺失值统计.csv\n",
  "8. 正态性检验结果.csv\n",
  "9. 异常值统计.csv\n",
  "10. 结局变量分布.csv\n\n",
  
  "【10. 结论】\n",
  "--------------------------------------------------\n",
  
  "数据预处理完成。\n",
  "最终数据集可直接用于机器学习建模与验证。\n"
  
)

###################
# 输出报告
###################

cat(report)

writeLines(
  report,
  "预处理报告.txt"
)

cat("\n✅ 数据预处理完成！\n")














##############用于患者层面bootstrap验证的增加id的预处理数据集
######################################################################
# 生成三个结局共用的带患者ID预处理训练数据
######################################################################

library(readxl)
library(data.table)
library(writexl)

######################################################################
# 1. 文件路径
######################################################################

# 带患者ID、尚未预处理的训练数据
raw_train_path <-
  "J:/rtools45/train_raw_patient_level.xlsx"

# 原来的完整预处理训练数据，包含三个结局
processed_train_path <-
  "J:/rtools45/train_preprocessed.rds"

# 输出路径
output_rds_path <-
  "J:/rtools45/train_preprocessed_with_id.rds"

output_xlsx_path <-
  "J:/rtools45/train_preprocessed_with_id.xlsx"

######################################################################
# 2. 检查文件
######################################################################

if(!file.exists(raw_train_path)){
  stop(
    paste0(
      "❌ 找不到原始患者级训练数据：",
      raw_train_path
    )
  )
}

if(!file.exists(processed_train_path)){
  stop(
    paste0(
      "❌ 找不到原预处理训练数据：",
      processed_train_path
    )
  )
}

######################################################################
# 3. 读取数据
######################################################################

raw_train <- as.data.table(
  read_excel(raw_train_path)
)

processed_train <- as.data.table(
  readRDS(processed_train_path)
)

cat(
  "\n患者级原始训练数据维度：",
  nrow(raw_train),
  "×",
  ncol(raw_train),
  "\n"
)

cat(
  "原预处理训练数据维度：",
  nrow(processed_train),
  "×",
  ncol(processed_train),
  "\n"
)

######################################################################
# 4. 检查ID
######################################################################

if(!"id" %in% names(raw_train)){
  stop("❌ train_raw_patient_level.xlsx中没有id列")
}

if(any(is.na(raw_train$id))){
  stop("❌ 患者ID中存在缺失值")
}

cat(
  "唯一患者数：",
  uniqueN(raw_train$id),
  "\n"
)

######################################################################
# 5. 检查行数
######################################################################

if(nrow(raw_train) != nrow(processed_train)){
  
  stop(
    paste0(
      "❌ 原始训练数据与预处理训练数据行数不一致：",
      "\n原始训练数据 = ",
      nrow(raw_train),
      "\n预处理训练数据 = ",
      nrow(processed_train)
    )
  )
}

######################################################################
# 6. 检查三个结局
######################################################################

outcome_cols_all <- c(
  "Outcome_Disease Non-Progression",
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma"
)

missing_outcomes <- setdiff(
  outcome_cols_all,
  names(processed_train)
)

if(length(missing_outcomes) > 0){
  
  stop(
    paste0(
      "❌ 原预处理数据缺少以下结局：",
      paste(
        missing_outcomes,
        collapse = ", "
      )
    )
  )
}

cat("\n三个结局变量均存在：\n")
print(outcome_cols_all)

######################################################################
# 7. 添加ID
######################################################################

train_preprocessed_with_id <- copy(
  processed_train
)

train_preprocessed_with_id[
  ,
  id := raw_train$id
]

# 将ID放在第一列
setcolorder(
  train_preprocessed_with_id,
  c(
    "id",
    setdiff(
      names(train_preprocessed_with_id),
      "id"
    )
  )
)

######################################################################
# 8. 检查ID没有改变原数据
######################################################################

data_without_id <- copy(
  train_preprocessed_with_id
)

data_without_id[
  ,
  id := NULL
]

if(!identical(
  names(data_without_id),
  names(processed_train)
)){
  
  stop(
    "❌ 添加ID后，原预处理数据的变量名称或变量顺序发生变化"
  )
}

if(!isTRUE(
  all.equal(
    data_without_id,
    processed_train,
    check.attributes = TRUE
  )
)){
  
  stop(
    "❌ 添加ID后，删除ID不能完全恢复原预处理数据"
  )
}

cat(
  "\n✓ 加入ID后，原来的预处理数据没有发生变化\n"
)

######################################################################
# 9. 输出数据概况
######################################################################

cat(
  "\n最终数据行数：",
  nrow(train_preprocessed_with_id),
  "\n"
)

cat(
  "最终数据列数：",
  ncol(train_preprocessed_with_id),
  "\n"
)

cat(
  "唯一患者数：",
  uniqueN(train_preprocessed_with_id$id),
  "\n"
)

cat(
  "重复住院记录数：",
  nrow(train_preprocessed_with_id) -
    uniqueN(train_preprocessed_with_id$id),
  "\n"
)

cat("\n三个结局的分布：\n")

for(outcome_col in outcome_cols_all){
  
  cat(
    "\n",
    outcome_col,
    "：\n",
    sep = ""
  )
  
  print(
    table(
      train_preprocessed_with_id[[outcome_col]],
      useNA = "ifany"
    )
  )
}

######################################################################
# 10. 保存RDS
######################################################################

saveRDS(
  train_preprocessed_with_id,
  output_rds_path
)

######################################################################
# 11. 保存Excel
######################################################################

write_xlsx(
  as.data.frame(
    train_preprocessed_with_id
  ),
  output_xlsx_path
)

cat("\n============================================\n")
cat("✓ 三个结局共用的带ID训练数据生成成功\n")
cat("RDS：", output_rds_path, "\n")
cat("Excel：", output_xlsx_path, "\n")
cat("============================================\n")

