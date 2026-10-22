######################################################################
# 08_robustness_validation.R
# ----------------------------------------------------------------------------
# 用途：验证特征结论与模型性能的稳健性，分三部分按顺序执行：
#   A. SHAP 特征重要性与交互特征的稳定性：患者级有放回重抽样，
#      每次重新拟合 XGBoost 并重新计算 TreeSHAP（特征 B=100、交互 N=1000），
#      统计 Top15 特征与 Top4 交互对的选择频率；
#   B. Top-15 特征选择方法比较：SHAP、Gain、LASSO、
#      Random Forest Permutation importance 四种方法构建的模型在测试集上的性能；
#   C. 模型内部稳健性：患者级 Bootstrap（B=2000）评估三个模型的
#      Calibration 与 DCA 性能。
# 运行前提：先运行 01 得到建模数据、02 得到各结局模型、04 得到 SHAP 结果。
# 输出：各部分的输出目录，含选择频率、性能比较与图形。
# ----------------------------------------------------------------------------
######################################################################


###############################################################
# 生成带患者ID的预处理测试集
###############################################################

rm(list = ls())
gc()

library(data.table)
library(readxl)
library(writexl)

###############################################################
# 1. 文件路径
###############################################################

raw_test_path <- "J:/rtools45/test_raw_patient_level.xlsx"

preprocessed_test_path <- "J:/rtools45/test_preprocessed.csv"

output_rds_path <- "J:/rtools45/test_preprocessed_with_id.rds"

output_csv_path <- "J:/rtools45/test_preprocessed_with_id.csv"

output_xlsx_path <- "J:/rtools45/test_preprocessed_with_id.xlsx"

###############################################################
# 2. 读取原始测试数据
#    将id列按文本读取，避免长ID发生精度损失
###############################################################

raw_header <- readxl::read_excel(
  raw_test_path,
  n_max = 0,
  .name_repair = "unique"
)

raw_col_types <- rep(
  "guess",
  ncol(raw_header)
)

id_position <- which(
  tolower(names(raw_header)) == "id"
)

if (length(id_position) != 1L) {
  stop("原始测试数据中未找到唯一的id列。")
}

raw_col_types[id_position] <- "text"

raw_test <- as.data.table(
  readxl::read_excel(
    raw_test_path,
    col_types = raw_col_types,
    .name_repair = "unique"
  )
)

###############################################################
# 3. 读取预处理测试数据
###############################################################

test_preprocessed <- data.table::fread(
  preprocessed_test_path,
  check.names = FALSE
)

###############################################################
# 4. 基本检查
###############################################################

if (!("id" %in% tolower(names(raw_test)))) {
  stop("原始测试数据中不存在id列。")
}

actual_id_name <- names(raw_test)[
  which(tolower(names(raw_test)) == "id")[1]
]

patient_id <- as.character(
  raw_test[[actual_id_name]]
)

if (nrow(raw_test) != nrow(test_preprocessed)) {
  stop(
    paste0(
      "原始测试集与预处理测试集行数不一致：",
      "\n原始测试集 = ", nrow(raw_test),
      "\n预处理测试集 = ", nrow(test_preprocessed)
    )
  )
}

if (anyNA(patient_id) || any(trimws(patient_id) == "")) {
  stop("患者ID中存在缺失值或空值。")
}

if ("id" %in% tolower(names(test_preprocessed))) {
  stop("预处理测试集中已经存在id列，请勿重复添加。")
}

###############################################################
# 5. 验证行顺序
###############################################################

check_columns <- c(
  "Age",
  "Gender",
  "Number of Hospitalizations",
  "Outcome_Liver Cirrhosis"
)

for (variable in check_columns) {
  
  if (!(variable %in% names(raw_test))) {
    stop("原始测试集中缺少核对变量：", variable)
  }
  
  if (!(variable %in% names(test_preprocessed))) {
    stop("预处理测试集中缺少核对变量：", variable)
  }
  
  raw_value <- suppressWarnings(
    as.numeric(raw_test[[variable]])
  )
  
  processed_value <- suppressWarnings(
    as.numeric(test_preprocessed[[variable]])
  )
  
  same_position <- (
    (is.na(raw_value) & is.na(processed_value)) |
      (
        !is.na(raw_value) &
          !is.na(processed_value) &
          abs(raw_value - processed_value) < 1e-10
      )
  )
  
  mismatch_number <- sum(
    !same_position
  )
  
  cat(
    variable,
    "：不一致行数 =",
    mismatch_number,
    "\n"
  )
  
  if (mismatch_number > 0L) {
    stop(
      "行顺序核验失败：",
      variable,
      "存在",
      mismatch_number,
      "行不一致。不能直接添加患者ID。"
    )
  }
}

cat("✓ 原始测试集与预处理测试集行顺序核验通过\n")

###############################################################
# 6. 添加患者ID
###############################################################

test_preprocessed_with_id <- cbind(
  data.table(
    id = patient_id
  ),
  test_preprocessed
)

###############################################################
# 7. 最终核查
###############################################################

if (nrow(test_preprocessed_with_id) != 10874L) {
  stop(
    "带ID测试集记录数不是预期的10874条；实际为：",
    nrow(test_preprocessed_with_id)
  )
}

patient_number <- uniqueN(
  test_preprocessed_with_id$id
)

if (patient_number != 6600L) {
  stop(
    "唯一患者数不是预期的6600名；实际为：",
    patient_number
  )
}

if (anyDuplicated(names(test_preprocessed_with_id))) {
  stop("带ID测试集中存在重复列名。")
}

cat("\n============================================\n")
cat("测试集住院记录数：", nrow(test_preprocessed_with_id), "\n")
cat("测试集患者数：", patient_number, "\n")
cat("测试集变量数（含ID）：", ncol(test_preprocessed_with_id), "\n")
cat("============================================\n")

###############################################################
# 8. 保存
###############################################################

saveRDS(
  test_preprocessed_with_id,
  output_rds_path
)

data.table::fwrite(
  test_preprocessed_with_id,
  output_csv_path
)

writexl::write_xlsx(
  as.data.frame(test_preprocessed_with_id),
  output_xlsx_path
)

cat("\n============================================\n")
cat("✓ 带患者ID的预处理测试集生成成功\n")
cat("RDS：", output_rds_path, "\n")
cat("CSV：", output_csv_path, "\n")
cat("Excel：", output_xlsx_path, "\n")
cat("============================================\n")




#########################################################################
#                  A
##########################################################################
###########################################################################
# A. SHAP 特征重要性稳定性与交互特征稳定性
#
# 以患者为单位有放回重抽样，每次重新拟合 XGBoost 并重新计算 TreeSHAP。
# 特征重要性稳定性重复 100 次，交互特征稳定性重复 1000 次。
###########################################################################

######################################################################
# PATIENT-LEVEL BOOTSTRAP REFITTING FOR TREESHAP REPRODUCIBILITY
#
# 目的：
# 1. 患者级有放回Bootstrap抽样
# 2. 每次从头重新拟合XGBoost
# 3. 每次重新计算SHAP特征排名
# 4. 每次重新计算TreeSHAP交互排名
# 5. 统计Top 15特征及Top 4交互对的选择频率
#
# 注意：
# 本脚本用于shap的稳定性分析
# 不改变主要 XGBoost 模型与主要 SHAP 分析
######################################################################
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

# 带患者ID的原预处理训练数据
patient_data_path <-
  "J:/rtools45/train_preprocessed_with_id.rds"


save_dir <-
  "J:/rtools45/Patient_Bootstrap_TreeSHAP_Results_B100_N1000_FINAL"
############################################################
# 调试参数
############################################################

# 正式患者级Bootstrap次数
B_REFIT <- 100

# 普通SHAP使用每次完整Bootstrap样本
# TreeSHAP interaction在每次Bootstrap内部重新抽取1000名不同患者计算
INTERACTION_PATIENT_N <- 1000

# 原最终XGBoost模型使用200轮
NROUNDS_REFIT <- 200

# TRUE：只运行新的患者级Bootstrap稳定性分析
# 跳过原来的完整主SHAP分析，避免重新计算4小时的全量interaction
RUN_REFIT_BOOTSTRAP_ONLY <- TRUE

# 三个结局均在新的正式结果目录中从头运行。
# 新目录中的有效检查点仅用于意外中断后的断点续跑。

if(!dir.exists(save_dir)){
  
  dir.create(save_dir, recursive = TRUE)
}

######################################################################
# 2.1 读取带患者ID的原预处理训练数据
######################################################################

if(!file.exists(patient_data_path)){
  
  stop(
    paste0(
      "❌ 带ID训练数据不存在：",
      patient_data_path
    )
  )
}

train_with_id <- as.data.table(
  readRDS(patient_data_path)
)

if(!"id" %in% names(train_with_id)){
  
  stop("❌ train_preprocessed_with_id.rds中不存在id")
}

if(any(is.na(train_with_id$id))){
  
  stop("❌ 患者ID存在缺失")
}

cat(
  "\n带ID训练数据维度：",
  nrow(train_with_id),
  "×",
  ncol(train_with_id),
  "\n"
)

cat(
  "唯一患者数：",
  uniqueN(train_with_id$id),
  "\n"
)

############################################################
# 三个结局
############################################################

outcome_cols_all <- c(
  "Outcome_Disease Non-Progression",
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma"
)

if(!all(
  outcome_cols_all %in%
  names(train_with_id)
)){
  
  stop("❌ 带ID数据中的三个结局变量不完整")
}

############################################################
# 提前建立患者ID和行号的对应关系
# 避免每次Bootstrap反复搜索全部数据
############################################################

rows_by_patient <- split(
  seq_len(nrow(train_with_id)),
  as.character(train_with_id$id)
)

all_patient_ids <- names(
  rows_by_patient
)

# 固定保存每一训练行对应的患者ID。
# 三个结局共用这一行级ID映射，但各自的特征和结局值从对应FULL_MODEL的train_df读取。
patient_id_by_row <- as.character(
  train_with_id$id
)

cat(
  "患者级抽样单位数：",
  length(all_patient_ids),
  "\n"
)
######################################################################
# 3. Load models
######################################################################

model_files <- list.files(
  
  model_dir,
  
  pattern = "_FULL_MODEL.rds$",
  
  full.names = TRUE
)
############################################################
# 三个结局模型均进入测试/正式分析
############################################################

expected_model_files <- c(
  "Outcome_Disease_Non_Progression_FULL_MODEL.rds",
  "Outcome_Liver_Cirrhosis_FULL_MODEL.rds",
  "Outcome_Hepatocellular_Carcinoma_FULL_MODEL.rds"
)

missing_model_files <- setdiff(
  expected_model_files,
  basename(model_files)
)

if(length(missing_model_files) > 0){
  stop(
    paste0(
      "❌ 缺少以下FULL_MODEL文件：",
      paste(missing_model_files, collapse = ", ")
    )
  )
}

# 只保留预先指定的三个结局模型，避免目录中其他旧模型被误运行
model_files <- file.path(
  model_dir,
  expected_model_files
)

cat(
  "识别到三个结局的FULL_MODEL文件：\n"
)

print(basename(model_files))
cat("====================================================\n")
cat(" TreeSHAP ANALYSIS PIPELINE\n")
cat("====================================================\n\n")

print(basename(model_files))

######################################################################
# 3.1 患者级Bootstrap行号生成函数
######################################################################

get_patient_bootstrap_rows <- function(
    all_patient_ids,
    rows_by_patient
) {
  
  sampled_patient_ids <- sample(
    all_patient_ids,
    size = length(all_patient_ids),
    replace = TRUE
  )
  
  # 如果同一患者被抽中多次，其全部记录也重复纳入
  bootstrap_rows <- unlist(
    rows_by_patient[
      sampled_patient_ids
    ],
    use.names = FALSE
  )
  
  return(
    list(
      sampled_patient_ids =
        sampled_patient_ids,
      bootstrap_rows =
        bootstrap_rows
    )
  )
}

######################################################################
# 3.2 计算TreeSHAP交互排名函数
######################################################################

calculate_interaction_ranking <- function(
    interaction_array,
    feature_cols,
    bootstrap_number,
    outcome_col
) {
  
  p <- length(feature_cols)
  
  expected_dimension <- p + 1
  
  if(length(dim(interaction_array)) != 3){
    
    stop("❌ predinteraction结果不是三维数组")
  }
  
  if(
    dim(interaction_array)[2] !=
    expected_dimension ||
    dim(interaction_array)[3] !=
    expected_dimension
  ){
    
    stop(
      paste0(
        "❌ interaction维度错误：",
        paste(
          dim(interaction_array),
          collapse = " × "
        )
      )
    )
  }
  
  ##########################################################
  # 删除最后一个BIAS行和列
  ##########################################################
  
  interaction_array <- interaction_array[
    ,
    seq_len(p),
    seq_len(p),
    drop = FALSE
  ]
  
  ##########################################################
  # 对参考样本求平均绝对交互强度
  ##########################################################
  
  mean_abs_matrix <- apply(
    abs(interaction_array),
    c(2, 3),
    mean,
    na.rm = TRUE
  )
  
  ##########################################################
  # 只保留上三角，避免重复组合及自身交互
  ##########################################################
  
  pair_index <- which(
    upper.tri(
      mean_abs_matrix,
      diag = FALSE
    ),
    arr.ind = TRUE
  )
  
  interaction_ranking <- data.table(
    Feature1 =
      feature_cols[
        pair_index[, 1]
      ],
    Feature2 =
      feature_cols[
        pair_index[, 2]
      ],
    MeanAbsInteraction =
      mean_abs_matrix[
        pair_index
      ]
  )
  
  setorder(
    interaction_ranking,
    -MeanAbsInteraction
  )
  
  interaction_ranking[
    ,
    Rank := seq_len(.N)
  ]
  
  interaction_ranking[
    ,
    SelectedTop4 := Rank <= 4
  ]
  
  interaction_ranking[
    ,
    SelectedTop10 := Rank <= 10
  ]
  
  interaction_ranking[
    ,
    `:=`(
      Bootstrap = bootstrap_number,
      Outcome = outcome_col
    )
  ]
  
  return(interaction_ranking)
}
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
    
    ############################################################
    # 当前模型对应的结局
    ############################################################
    
    outcome_col <- model_obj$outcome_col
    
    if(is.null(outcome_col)){
      
      stop(
        paste0(
          "❌ FULL_MODEL中没有保存outcome_col：",
          basename(model_path)
        )
      )
    }

    ############################################################
    # 为当前模型建立“原模型train_df + 共同患者ID”的数据
    ############################################################

    if(is.null(model_obj$train_df)){
      stop(
        paste0(
          "❌ FULL_MODEL中没有保存train_df：",
          basename(model_path)
        )
      )
    }

    model_train_with_id <- as.data.table(
      copy(model_obj$train_df)
    )

    if(nrow(model_train_with_id) != length(patient_id_by_row)){
      stop(
        paste0(
          "❌ 当前模型训练行数与患者ID行数不一致：模型训练行数=",
          nrow(model_train_with_id),
          "；患者ID行数=",
          length(patient_id_by_row)
        )
      )
    }

    model_train_with_id[, id := patient_id_by_row]
    
    if(!outcome_col %in% names(model_train_with_id)){
      
      stop(
        paste0(
          "❌ 当前FULL_MODEL的train_df中找不到当前结局：",
          outcome_col
        )
      )
    }
    
    cat(
      "当前结局：",
      outcome_col,
      "\n"
    )

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
    # 核对带ID数据与原模型训练矩阵
    ##################################################################
    
    missing_features <- setdiff(
      feature_cols,
      names(model_train_with_id)
    )
    
    if(length(missing_features) > 0){
      
      stop(
        paste0(
          "❌ 当前FULL_MODEL的train_df中缺少原模型特征：",
          paste(
            missing_features,
            collapse = ", "
          )
        )
      )
    }
    
    # 必须与原XGBoost训练脚本保持一致：data.matrix()会按原方式
    # 将factor转换为内部整数编码；as.matrix()会先转成字符，造成数值错位。
    X_from_id_data <- data.matrix(
      model_train_with_id[
        ,
        ..feature_cols
      ]
    )
    
    storage.mode(X_from_id_data) <- "double"
    
    if(!identical(
      dim(X_from_id_data),
      dim(X_train)
    )){
      
      stop("❌ 带ID数据与原X_train维度不一致")
    }
    
    matrix_max_diff <- max(
      abs(
        X_from_id_data -
          X_train
      ),
      na.rm = TRUE
    )
    
    cat(
      "带ID数据与原训练矩阵最大差异：",
      matrix_max_diff,
      "\n"
    )
    
    if(matrix_max_diff > 1e-8){
      
      stop(
        paste0(
          "❌ 带ID数据与原模型训练矩阵不一致；最大差异：",
          matrix_max_diff
        )
      )
    }
    
    cat("✓ 带ID数据与原训练矩阵完全对应\n")
    
    rm(X_from_id_data)
    gc()
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
    # 4.X 患者级Bootstrap重新拟合稳定性分析
    ##################################################################
    
    if(RUN_REFIT_BOOTSTRAP_ONLY){
      
      cat("\n")
      cat("====================================================\n")
      cat("PATIENT-LEVEL BOOTSTRAP REFITTING\n")
      cat("Outcome:", outcome_col, "\n")
      cat("Bootstrap replicates:", B_REFIT, "\n")
      cat("SHAP data: complete bootstrap sample\n")
      cat(
        "Interaction patients per replicate:",
        INTERACTION_PATIENT_N,
        "\n"
      )
      cat("====================================================\n")
      
      ############################################################
      # 新分析单独输出
      ############################################################
      
      refit_out_dir <- file.path(
        save_dir,
        outcome_name,
        "Patient_Level_Refit_Bootstrap"
      )
      
      if(!dir.exists(refit_out_dir)){
        
        dir.create(
          refit_out_dir,
          recursive = TRUE
        )
      }
      
      ############################################################
      # 当前结局对应的可重复随机种子编号
      ############################################################

      outcome_seed <- match(
        outcome_col,
        outcome_cols_all
      )

      if(is.na(outcome_seed)){

        stop(
          paste0(
            "❌ 无法匹配当前结局：",
            outcome_col
          )
        )
      }
      
      ############################################################
      # 读取原模型最优超参数
      ############################################################
      
      best_params <- model_obj$best_params
      
      if(is.null(best_params)){
        
        stop("❌ FULL_MODEL中不存在best_params")
      }
      
      required_params <- c(
        "eta",
        "max_depth",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
        "gamma",
        "lambda",
        "alpha"
      )
      
      missing_params <- setdiff(
        required_params,
        names(best_params)
      )
      
      if(length(missing_params) > 0){
        
        stop(
          paste0(
            "❌ 缺少最优参数：",
            paste(
              missing_params,
              collapse = ", "
            )
          )
        )
      }
      
      ############################################################
      # 保存每次排名结果
      ############################################################
      
      feature_rank_list <- vector(
        "list",
        B_REFIT
      )
      
      interaction_rank_list <- vector(
        "list",
        B_REFIT
      )
      
      bootstrap_audit_list <- vector(
        "list",
        B_REFIT
      )

      interaction_patient_audit_list <- vector(
        "list",
        B_REFIT
      )

      ############################################################
      # 读取检查点并断点续跑
      ############################################################

      checkpoint_path <- file.path(
        refit_out_dir,
        "Bootstrap_Refit_Checkpoint.rds"
      )

      first_bootstrap_to_run <- 1L

      if(file.exists(checkpoint_path)){

        checkpoint_obj <- readRDS(
          checkpoint_path
        )

        checkpoint_completed <- as.integer(
          checkpoint_obj$completed_bootstrap
        )

        checkpoint_valid <-
          !is.na(checkpoint_completed) &&
          checkpoint_completed >= 1L &&
          checkpoint_completed <= B_REFIT &&
          identical(
            as.character(checkpoint_obj$outcome),
            as.character(outcome_col)
          ) &&
          identical(
            as.integer(checkpoint_obj$interaction_patient_n),
            as.integer(INTERACTION_PATIENT_N)
          )

        if(checkpoint_valid){

          completed_index <- seq_len(
            checkpoint_completed
          )

          feature_rank_list[completed_index] <-
            checkpoint_obj$feature_rank_list

          interaction_rank_list[completed_index] <-
            checkpoint_obj$interaction_rank_list

          bootstrap_audit_list[completed_index] <-
            checkpoint_obj$bootstrap_audit_list

          interaction_patient_audit_list[completed_index] <-
            checkpoint_obj$interaction_patient_audit_list

          first_bootstrap_to_run <-
            checkpoint_completed + 1L

          cat(
            "读取到有效检查点：已完成",
            checkpoint_completed,
            "/",
            B_REFIT,
            "次Bootstrap\n"
          )

          if(first_bootstrap_to_run > B_REFIT){
            cat(
              "全部Bootstrap已经计算完成；",
              "本次仅重新生成汇总文件。\n"
            )
          } else {
            cat(
              "将从Bootstrap",
              first_bootstrap_to_run,
              "继续运行。\n"
            )
          }
        } else {
          stop(
            paste0(
              "❌ 检测到检查点，但与当前结局或参数不一致。",
              "为防止覆盖已有结果，脚本已停止；",
              "不会自动从Bootstrap 1重新运行。检查点：",
              checkpoint_path
            )
          )
        }
      }

      if(first_bootstrap_to_run <= B_REFIT){
        bootstrap_indices <- seq.int(
          first_bootstrap_to_run,
          B_REFIT
        )
      } else {
        bootstrap_indices <- integer(0)
      }
      
      ############################################################
      # 患者级Bootstrap循环
      ############################################################
      
      for(b in bootstrap_indices){
        
        cat(
          "\nBootstrap",
          b,
          "/",
          B_REFIT,
          "\n"
        )
        
        start_time_b <- Sys.time()
        
        set.seed(
          20260821 +
            outcome_seed * 10000 +
            b
        )
        
        ##########################################################
        # A. 按患者ID有放回抽样
        ##########################################################
        
        bootstrap_sample <-
          get_patient_bootstrap_rows(
            all_patient_ids =
              all_patient_ids,
            rows_by_patient =
              rows_by_patient
          )
        
        boot_rows <-
          bootstrap_sample$bootstrap_rows
        
        sampled_patient_ids <-
          bootstrap_sample$sampled_patient_ids
        
        ##########################################################
        # B. 当前Bootstrap训练矩阵
        ##########################################################
        
        # 与原模型的X_train_matrix使用完全相同的转换规则
        X_boot <- data.matrix(
          model_train_with_id[
            boot_rows,
            ..feature_cols
          ]
        )
        
        y_boot <- as.numeric(
          as.character(
            model_train_with_id[[outcome_col]][boot_rows]
          )
        )
        
        storage.mode(X_boot) <- "double"
        
        colnames(X_boot) <- feature_cols

        ##########################################################
        # 本次Bootstrap中每条记录对应的患者ID
        ##########################################################

        id_boot <- as.character(
          model_train_with_id$id[
            boot_rows
          ]
        )

        if(length(id_boot) != nrow(X_boot)){

          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 的ID数量与X_boot行数不一致"
            )
          )
        }
        
        if(anyNA(X_boot)){
          
          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 的X_boot存在缺失值"
            )
          )
        }
        
        if(anyNA(y_boot)){
          
          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 的结局存在缺失值"
            )
          )
        }
        
        if(!all(
          unique(y_boot) %in% c(0, 1)
        )){
          
          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 的结局不是0/1"
            )
          )
        }
        
        ##########################################################
        # C. 本次Bootstrap重新计算类别权重
        #
        # 原主模型的scale_pos_weight也是根据训练数据计算，
        # 所以在每次患者级抽样后重新计算
        ##########################################################
        
        positive_n <- sum(
          y_boot == 1
        )
        
        negative_n <- sum(
          y_boot == 0
        )
        
        if(positive_n == 0 ||
           negative_n == 0){
          
          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 只包含一个结局类别"
            )
          )
        }
        
        scale_pos_weight_b <-
          negative_n /
          positive_n
        
        ##########################################################
        # D. 构建本次XGBoost参数
        ##########################################################
        
        params_b <- list(
          
          booster = "gbtree",
          
          objective =
            "binary:logistic",
          
          eval_metric = "auc",
          
          eta =
            as.numeric(
              best_params$eta
            ),
          
          max_depth =
            as.integer(
              best_params$max_depth
            ),
          
          min_child_weight =
            as.numeric(
              best_params$min_child_weight
            ),
          
          subsample =
            as.numeric(
              best_params$subsample
            ),
          
          colsample_bytree =
            as.numeric(
              best_params$colsample_bytree
            ),
          
          gamma =
            as.numeric(
              best_params$gamma
            ),
          
          lambda =
            as.numeric(
              best_params$lambda
            ),
          
          alpha =
            as.numeric(
              best_params$alpha
            ),
          
          scale_pos_weight =
            scale_pos_weight_b,
          
          nthread = 4
        )
        
        dtrain_boot <- xgb.DMatrix(
          data = X_boot,
          label = y_boot
        )
        
        ##########################################################
        # E. 每次从头重新拟合XGBoost
        #
        # 不能传入xgb_model = booster
        # 不能继续训练原模型
        ##########################################################
        
        model_b <- xgb.train(
          params = params_b,
          data = dtrain_boot,
          nrounds = NROUNDS_REFIT,
          verbose = 0
        )
        
        ##########################################################
        # F. 在本次完整Bootstrap样本上重新计算SHAP
        ##########################################################
        
        shap_full_b <- predict(
          model_b,
          X_boot,
          predcontrib = TRUE
        )
        
        shap_b <- shap_full_b[
          ,
          -ncol(shap_full_b),
          drop = FALSE
        ]
        
        colnames(shap_b) <-
          feature_cols
        
        mean_abs_shap_b <- colMeans(
          abs(shap_b),
          na.rm = TRUE
        )
        
        feature_ranking_b <- data.table(
          Feature =
            feature_cols,
          MeanAbsSHAP =
            as.numeric(
              mean_abs_shap_b
            )
        )
        
        setorder(
          feature_ranking_b,
          -MeanAbsSHAP
        )
        
        feature_ranking_b[
          ,
          Rank := seq_len(.N)
        ]
        
        feature_ranking_b[
          ,
          SelectedTop15 :=
            Rank <= 15
        ]
        
        feature_ranking_b[
          ,
          `:=`(
            Bootstrap = b,
            Outcome = outcome_col
          )
        ]
        
        feature_rank_list[[b]] <- feature_ranking_b
        
        cat("本次Top 15特征：\n")
        
        print(
          feature_ranking_b[
            1:15,
            .(
              Rank,
              Feature,
              MeanAbsSHAP
            )
          ]
        )
        
        ##########################################################
        # G. 从本次Bootstrap中重新抽取1000名不同患者
        ##########################################################

        unique_boot_patient_ids <- unique(
          id_boot
        )

        if(length(unique_boot_patient_ids) <
           INTERACTION_PATIENT_N){

          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 中的不同患者数少于 ",
              INTERACTION_PATIENT_N
            )
          )
        }

        set.seed(
          20260821 +
            outcome_seed * 10000 +
            b * 100 +
            50
        )

        interaction_patient_ids_b <- sample(
          unique_boot_patient_ids,
          size = INTERACTION_PATIENT_N,
          replace = FALSE
        )

        ##########################################################
        # 保留这1000名患者在本次Bootstrap中的全部记录
        # 若患者在Bootstrap中被重复抽中，其记录按权重重复保留
        ##########################################################

        interaction_row_mask <- (
          id_boot %in%
            interaction_patient_ids_b
        )

        X_interaction_b <- X_boot[
          interaction_row_mask,
          ,
          drop = FALSE
        ]

        if(nrow(X_interaction_b) == 0){

          stop(
            paste0(
              "❌ Bootstrap ",
              b,
              " 的interaction样本为空"
            )
          )
        }

        storage.mode(X_interaction_b) <- "double"
        colnames(X_interaction_b) <- feature_cols

        cat(
          "本次interaction患者数：",
          length(interaction_patient_ids_b),
          "\n"
        )

        cat(
          "本次interaction记录数：",
          nrow(X_interaction_b),
          "\n"
        )

        interaction_patient_audit_list[[b]] <- data.table(
          Bootstrap = b,
          Outcome = outcome_col,
          Patient_ID = interaction_patient_ids_b
        )

        ##########################################################
        # H. 使用本次新模型重新计算TreeSHAP interaction
        ##########################################################
        
        interaction_array_b <- predict(
          model_b,
          X_interaction_b,
          predinteraction = TRUE
        )
        
        cat(
          "Interaction维度：",
          paste(
            dim(interaction_array_b),
            collapse = " × "
          ),
          "\n"
        )
        
        interaction_ranking_b <-
          calculate_interaction_ranking(
            interaction_array =
              interaction_array_b,
            feature_cols =
              feature_cols,
            bootstrap_number =
              b,
            outcome_col =
              outcome_col
          )
        
        interaction_rank_list[[b]] <- interaction_ranking_b
        
        cat("本次Top 4交互：\n")
        
        print(
          interaction_ranking_b[
            1:4,
            .(
              Rank,
              Feature1,
              Feature2,
              MeanAbsInteraction
            )
          ]
        )
        
        ##########################################################
        # I. 保存本次审计信息
        ##########################################################
        
        end_time_b <- Sys.time()
        
        runtime_minutes <- as.numeric(
          difftime(
            end_time_b,
            start_time_b,
            units = "mins"
          )
        )
        
        bootstrap_audit_list[[b]] <- data.table(
          Bootstrap = b,
          Outcome = outcome_col,
          Sampled_Patient_Units =
            length(
              sampled_patient_ids
            ),
          Unique_Original_Patients =
            uniqueN(
              sampled_patient_ids
            ),
          Bootstrap_Records =
            length(
              boot_rows
            ),
          Positive_Records =
            positive_n,
          Negative_Records =
            negative_n,
          Scale_Pos_Weight =
            scale_pos_weight_b,
          SHAP_Records =
            nrow(
              X_boot
            ),
          Interaction_Unique_Patients =
            length(
              interaction_patient_ids_b
            ),
          Interaction_Records =
            nrow(
              X_interaction_b
            ),
          Runtime_Minutes =
            runtime_minutes
        )

        ##########################################################
        # 每次Bootstrap结束后保存检查点
        ##########################################################

        saveRDS(
          list(
            completed_bootstrap = b,
            outcome = outcome_col,
            feature_rank_list =
              feature_rank_list[seq_len(b)],
            interaction_rank_list =
              interaction_rank_list[seq_len(b)],
            bootstrap_audit_list =
              bootstrap_audit_list[seq_len(b)],
            interaction_patient_audit_list =
              interaction_patient_audit_list[seq_len(b)],
            B_planned = B_REFIT,
            interaction_patient_n =
              INTERACTION_PATIENT_N,
            seed = 20260821
          ),
          file.path(
            refit_out_dir,
            "Bootstrap_Refit_Checkpoint.rds"
          )
        )

        cat(
          "Bootstrap",
          b,
          "检查点保存完成\n"
        )
        
        ##########################################################
        # J. 每次完成后释放大对象
        ##########################################################
        
        rm(
          X_boot,
          y_boot,
          id_boot,
          dtrain_boot,
          model_b,
          shap_full_b,
          shap_b,
          X_interaction_b,
          interaction_array_b,
          interaction_ranking_b,
          interaction_row_mask,
          interaction_patient_ids_b,
          unique_boot_patient_ids
        )
        
        gc()
        
        cat(
          "Bootstrap",
          b,
          "完成；耗时",
          round(
            runtime_minutes,
            2
          ),
          "分钟\n"
        )
      }
      
      ############################################################
      # 合并100次或调试3次的结果
      ############################################################
      
      all_feature_ranks <- rbindlist(
        feature_rank_list,
        use.names = TRUE
      )
      
      all_interaction_ranks <- rbindlist(
        interaction_rank_list,
        use.names = TRUE
      )
      
      bootstrap_audit <- rbindlist(
        bootstrap_audit_list,
        use.names = TRUE
      )

      interaction_patient_audit <- rbindlist(
        interaction_patient_audit_list,
        use.names = TRUE
      )
      
      ############################################################
      # 汇总特征选择频率
      ############################################################
      
      feature_stability <- all_feature_ranks[
        ,
        .(
          Top5_Frequency =
            mean(Rank <= 5),
          
          Top10_Frequency =
            mean(Rank <= 10),
          
          Top15_Frequency =
            mean(SelectedTop15),
          
          Median_Rank =
            median(Rank),
          
          Rank_Q1 =
            quantile(
              Rank,
              0.25
            ),
          
          Rank_Q3 =
            quantile(
              Rank,
              0.75
            ),
          
          Median_MeanAbsSHAP =
            median(
              MeanAbsSHAP
            )
        ),
        by = .(
          Outcome,
          Feature
        )
      ]
      
      setorder(
        feature_stability,
        Outcome,
        -Top15_Frequency,
        Median_Rank,
        -Median_MeanAbsSHAP
      )
      
      ############################################################
      # 汇总交互组合选择频率
      ############################################################
      
      interaction_stability <-
        all_interaction_ranks[
          ,
          .(
            Top4_Frequency =
              mean(SelectedTop4),
            
            Top10_Frequency =
              mean(SelectedTop10),
            
            Median_Rank =
              median(Rank),
            
            Rank_Q1 =
              quantile(
                Rank,
                0.25
              ),
            
            Rank_Q3 =
              quantile(
                Rank,
                0.75
              ),
            
            Median_MeanAbsInteraction =
              median(
                MeanAbsInteraction
              )
          ),
          by = .(
            Outcome,
            Feature1,
            Feature2
          )
        ]
      
      setorder(
        interaction_stability,
        Outcome,
        -Top4_Frequency,
        Median_Rank,
        -Median_MeanAbsInteraction
      )
      
      ############################################################
      # 导出结果
      ############################################################
      
      fwrite(
        all_feature_ranks,
        file.path(
          refit_out_dir,
          "All_Bootstrap_Feature_Rankings.csv"
        )
      )
      
      fwrite(
        feature_stability,
        file.path(
          refit_out_dir,
          "Feature_Top15_Selection_Frequency.csv"
        )
      )
      
      fwrite(
        all_interaction_ranks,
        file.path(
          refit_out_dir,
          "All_Bootstrap_Interaction_Rankings.csv"
        )
      )
      
      fwrite(
        interaction_stability,
        file.path(
          refit_out_dir,
          "Interaction_Top4_Selection_Frequency.csv"
        )
      )
      
      fwrite(
        bootstrap_audit,
        file.path(
          refit_out_dir,
          "Bootstrap_Refit_Audit.csv"
        )
      )

      fwrite(
        interaction_patient_audit,
        file.path(
          refit_out_dir,
          "Bootstrap_Interaction_Sampled_Patients.csv"
        )
      )
      
      ############################################################
      # 保存R对象，便于后续继续分析
      ############################################################
      
      saveRDS(
        list(
          feature_stability =
            feature_stability,
          interaction_stability =
            interaction_stability,
          bootstrap_audit =
            bootstrap_audit,
          interaction_patient_audit =
            interaction_patient_audit,
          B =
            B_REFIT,
          shap_strategy =
            "Complete bootstrap sample",
          interaction_sampling_strategy =
            "1000 unique patients newly sampled within each bootstrap replicate",
          interaction_patient_n =
            INTERACTION_PATIENT_N,
          seed =
            20260821,
          outcome =
            outcome_col
        ),
        file.path(
          refit_out_dir,
          "Patient_Level_Refit_Stability_Result.rds"
        )
      )
      
      cat("\n")
      cat("患者级重新拟合稳定性分析完成：\n")
      cat(refit_out_dir, "\n")
      
      ############################################################
      # 跳过原来的主要SHAP分析
      # 避免重新计算全训练集interaction
      ############################################################
      
      next
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
    
    ##################################################################
    # 4.12 Bootstrap SHAP stability
    ##################################################################
    
    cat("Running SHAP bootstrap stability...\n")
    
    B <- 100
    
    boot_mat <- matrix(
      
      NA,
      
      nrow = length(feature_cols),
      
      ncol = B
    )
    
    for(b in 1:B){
      
      idx <- sample(
        1:nrow(shap_mat),
        replace = TRUE
      )
      
      boot_mat[, b] <- colMeans(
        abs(shap_mat[idx, ])
      )
    }
    
    shap_boot <- data.frame(
      
      Feature = feature_cols,
      
      Mean = rowMeans(boot_mat),
      
      SD = apply(
        boot_mat,
        1,
        sd
      ),
      
      CI_low = apply(
        boot_mat,
        1,
        quantile,
        0.025
      ),
      
      CI_high = apply(
        boot_mat,
        1,
        quantile,
        0.975
      )
    )
    
    shap_boot <- shap_boot %>%
      arrange(desc(Mean))
    
    fwrite(
      shap_boot,
      file.path(
        out_dir,
        "SHAP_Bootstrap_Stability.csv"
      )
    )
    
    ##################################################################
    # 4.13 Beeswarm data
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
    # 4.14 Dependence data
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
    # 4.15 Standardized waterfall selection
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
    # 4.16 Save waterfall data
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
    # 4.17 TRUE TreeSHAP interaction
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
    # 4.18 Interaction summary
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
    # 4.19 Interaction detail
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
    
    ##################################################################
    # 4.20 Interaction bootstrap stability
    ##################################################################
    
    cat("Running interaction bootstrap stability...\n")
    
    B_inter <- 100
    
    interaction_stability <- data.frame()
    
    for(k in 1:nrow(top_interaction)){
      
      f1 <- top_interaction$Feature1[k]
      f2 <- top_interaction$Feature2[k]
      
      i <- which(feature_cols == f1)
      j <- which(feature_cols == f2)
      
      boot_vals <- c()
      
      for(b in 1:B_inter){
        
        idx <- sample(
          1:nrow(inter_array),
          replace = TRUE
        )
        
        val <- mean(
          abs(
            inter_array[idx, i, j]
          )
        )
        
        boot_vals <- c(
          boot_vals,
          val
        )
      }
      
      tmp <- data.frame(
        
        Feature1 = f1,
        
        Feature2 = f2,
        
        MeanInteraction =
          mean(boot_vals),
        
        SD =
          sd(boot_vals),
        
        CI_low =
          quantile(boot_vals, 0.025),
        
        CI_high =
          quantile(boot_vals, 0.975),
        
        CV =
          sd(boot_vals) /
          mean(boot_vals)
      )
      
      interaction_stability <- rbind(
        interaction_stability,
        tmp
      )
    }
    
    interaction_stability <- interaction_stability %>%
      arrange(CV)
    
    fwrite(
      interaction_stability,
      file.path(
        out_dir,
        "Interaction_Stability_Bootstrap.csv"
      )
    )
    
    ##################################################################
    # 4.21 Save raw feature values
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
    # 4.22 Metadata
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
      
      Bootstrap_Iterations =
        100,
      
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
    # 4.23 COMPLETE
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
    
    cat("\n❌ PATIENT-LEVEL BOOTSTRAP FAILED:\n")
    
    cat(
      basename(model_path),
      "\n"
    )
    
    cat("\nERROR:\n")
    
    cat(conditionMessage(e), "\n")

    stop(e)
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



######################################################################
# 生成Supplementary Table 6：
# 原主分析Top 20交互＋患者级B100重拟合稳定性
######################################################################

rm(list = ls())
gc()

library(data.table)

######################################################################
# 1. 路径
######################################################################

primary_root <-
  "J:/rtools45/SHAP_Results_TOPJOURNAL"

bootstrap_root <-
  "J:/rtools45/Patient_Bootstrap_TreeSHAP_Results_B100_N1000_FINAL"

output_path <- file.path(
  bootstrap_root,
  "Supplementary_Table6_Top20_Interaction_Reproducibility_B100.csv"
)

######################################################################
# 2. 三个结局及文件夹
######################################################################

outcome_folders <- c(
  "Outcome_Disease_Non_Progression_",
  "Outcome_Liver_Cirrhosis_",
  "Outcome_Hepatocellular_Carcinoma_"
)

outcome_labels <- c(
  "Disease Non-Progression",
  "Liver Cirrhosis",
  "Hepatocellular Carcinoma"
)

######################################################################
# 3. 保存三个结局结果
######################################################################

all_outcome_tables <- list()

######################################################################
# 4. 逐结局处理
######################################################################

for(i in seq_along(outcome_folders)){
  
  outcome_folder <- outcome_folders[i]
  
  outcome_label <- outcome_labels[i]
  
  cat(
    "\n正在处理：",
    outcome_label,
    "\n"
  )
  
  ############################################################
  # 4.1 原完整模型交互汇总
  ############################################################
  
  primary_path <- file.path(
    primary_root,
    outcome_folder,
    "TreeSHAP_Interaction_Summary_FULL.csv"
  )
  
  if(!file.exists(primary_path)){
    
    stop(
      paste0(
        "找不到原主分析交互文件：",
        primary_path
      )
    )
  }
  
  primary_interactions <- fread(
    primary_path
  )
  
  required_primary_cols <- c(
    "Feature1",
    "Feature2",
    "MeanAbsInteraction"
  )
  
  missing_primary_cols <- setdiff(
    required_primary_cols,
    names(primary_interactions)
  )
  
  if(length(missing_primary_cols) > 0){
    
    stop(
      paste0(
        "原交互文件缺少变量：",
        paste(
          missing_primary_cols,
          collapse = ", "
        )
      )
    )
  }
  
  # 明确按照原完整模型交互强度降序排列
  setorder(
    primary_interactions,
    -MeanAbsInteraction
  )
  
  # 只保留原主分析Top 20
  primary_top20 <- copy(
    primary_interactions[
      1:20
    ]
  )
  
  primary_top20[
    ,
    Rank := seq_len(.N)
  ]
  
  ############################################################
  # 4.2 创建与特征方向无关的交互对识别码
  ############################################################
  
  primary_top20[
    ,
    PairKey := paste(
      pmin(
        Feature1,
        Feature2
      ),
      pmax(
        Feature1,
        Feature2
      ),
      sep = "|||"
    )
  ]
  
  ############################################################
  # 4.3 读取100次患者级Bootstrap全部交互排名
  ############################################################
  
  bootstrap_path <- file.path(
    bootstrap_root,
    outcome_folder,
    "Patient_Level_Refit_Bootstrap",
    "All_Bootstrap_Interaction_Rankings.csv"
  )
  
  if(!file.exists(bootstrap_path)){
    
    stop(
      paste0(
        "找不到Bootstrap交互排名文件：",
        bootstrap_path
      )
    )
  }
  
  bootstrap_interactions <- fread(
    bootstrap_path
  )
  
  required_bootstrap_cols <- c(
    "Bootstrap",
    "Outcome",
    "Feature1",
    "Feature2",
    "Rank",
    "MeanAbsInteraction"
  )
  
  missing_bootstrap_cols <- setdiff(
    required_bootstrap_cols,
    names(bootstrap_interactions)
  )
  
  if(length(missing_bootstrap_cols) > 0){
    
    stop(
      paste0(
        "Bootstrap交互文件缺少变量：",
        paste(
          missing_bootstrap_cols,
          collapse = ", "
        )
      )
    )
  }
  
  ############################################################
  # 4.4 检查Bootstrap次数
  ############################################################
  
  completed_bootstraps <- sort(
    unique(
      bootstrap_interactions$Bootstrap
    )
  )
  
  if(!identical(
    completed_bootstraps,
    1:100
  )){
    
    stop(
      paste0(
        outcome_label,
        "的Bootstrap编号不是完整的1–100。实际为：",
        paste(
          completed_bootstraps,
          collapse = ", "
        )
      )
    )
  }
  
  cat(
    "确认Bootstrap次数：",
    length(completed_bootstraps),
    "\n"
  )
  
  ############################################################
  # 4.5 创建交互对识别码
  ############################################################
  
  bootstrap_interactions[
    ,
    PairKey := paste(
      pmin(
        Feature1,
        Feature2
      ),
      pmax(
        Feature1,
        Feature2
      ),
      sep = "|||"
    )
  ]
  
  ############################################################
  # 4.6 汇总所有交互对的稳定性
  ############################################################
  
  interaction_stability <- bootstrap_interactions[
    ,
    .(
      Top4_Frequency =
        mean(
          Rank <= 4
        ),
      
      Top10_Frequency =
        mean(
          Rank <= 10
        ),
      
      Top20_Frequency =
        mean(
          Rank <= 20
        ),
      
      Median_Rank =
        median(
          Rank
        ),
      
      Rank_Q1 =
        quantile(
          Rank,
          0.25
        ),
      
      Rank_Q3 =
        quantile(
          Rank,
          0.75
        ),
      
      Median_MeanAbsInteraction =
        median(
          MeanAbsInteraction
        )
    ),
    by = .(
      PairKey
    )
  ]
  
  ############################################################
  # 4.7 将原Top 20与新稳定性结果合并
  ############################################################
  
  supplementary_table6 <- merge(
    primary_top20,
    interaction_stability,
    by = "PairKey",
    all.x = TRUE,
    sort = FALSE
  )
  
  ############################################################
  # 4.8 恢复原主分析排名顺序
  ############################################################
  
  setorder(
    supplementary_table6,
    Rank
  )
  
  ############################################################
  # 4.9 检查是否成功匹配全部20组
  ############################################################
  
  unmatched_pairs <- supplementary_table6[
    is.na(
      Top20_Frequency
    )
  ]
  
  if(nrow(unmatched_pairs) > 0){
    
    print(
      unmatched_pairs[
        ,
        .(
          Feature1,
          Feature2
        )
      ]
    )
    
    stop(
      paste0(
        outcome_label,
        "存在无法与Bootstrap结果匹配的交互对"
      )
    )
  }
  
  ############################################################
  # 4.10 将频率转换为次数和百分比
  ############################################################
  
  supplementary_table6[
    ,
    `:=`(
      Top4_n =
        round(
          Top4_Frequency * 100
        ),
      
      Top10_n =
        round(
          Top10_Frequency * 100
        ),
      
      Top20_n =
        round(
          Top20_Frequency * 100
        )
    )
  ]
  
  supplementary_table6[
    ,
    `Selected in top 4, n (%)` :=
      paste0(
        Top4_n,
        " (",
        round(
          Top4_Frequency * 100
        ),
        "%)"
      )
  ]
  
  supplementary_table6[
    ,
    `Selected in top 10, n (%)` :=
      paste0(
        Top10_n,
        " (",
        round(
          Top10_Frequency * 100
        ),
        "%)"
      )
  ]
  
  supplementary_table6[
    ,
    `Selected in top 20, n (%)` :=
      paste0(
        Top20_n,
        " (",
        round(
          Top20_Frequency * 100
        ),
        "%)"
      )
  ]
  
  supplementary_table6[
    ,
    `Bootstrap rank, median (Q1–Q3)` :=
      paste0(
        Median_Rank,
        " (",
        Rank_Q1,
        "–",
        Rank_Q3,
        ")"
      )
  ]
  
  ############################################################
  # 4.11 整理最终列
  ############################################################
  
  final_outcome_table <- supplementary_table6[
    ,
    .(
      Outcome =
        outcome_label,
      
      Rank,
      
      `Feature 1` =
        Feature1,
      
      `Feature 2` =
        Feature2,
      
      `Mean absolute interaction` =
        round(
          MeanAbsInteraction,
          4
        ),
      
      `Selected in top 4, n (%)`,
      
      `Selected in top 10, n (%)`,
      
      `Selected in top 20, n (%)`,
      
      `Bootstrap rank, median (Q1–Q3)`,
      
      `Bootstrap median mean absolute interaction` =
        round(
          Median_MeanAbsInteraction,
          4
        )
    )
  ]
  
  ############################################################
  # 4.12 保存当前结局结果
  ############################################################
  
  individual_output <- file.path(
    bootstrap_root,
    outcome_folder,
    "Patient_Level_Refit_Bootstrap",
    "Supplementary_Table6_Top20_B100.csv"
  )
  
  fwrite(
    final_outcome_table,
    individual_output
  )
  
  all_outcome_tables[[i]] <-
    final_outcome_table
  
  cat(
    "当前结局Top 20汇总已保存：",
    individual_output,
    "\n"
  )
}

######################################################################
# 5. 合并三个结局
######################################################################

final_table6 <- rbindlist(
  all_outcome_tables,
  use.names = TRUE
)

######################################################################
# 6. 最终检查
######################################################################

if(nrow(final_table6) != 60){
  
  stop(
    paste0(
      "最终表格应为60行，实际为：",
      nrow(final_table6)
    )
  )
}

if(any(is.na(final_table6))){
  
  warning(
    "最终Table 6中存在缺失值，请检查"
  )
}

######################################################################
# 7. 保存合并结果
######################################################################

fwrite(
  final_table6,
  output_path
)

cat("\n============================================\n")
cat("Supplementary Table 6的Top 20稳定性结果生成成功\n")
cat("三个结局：3 × 20 = 60行\n")
cat("结果路径：", output_path, "\n")
cat("============================================\n")
#########################################################################
#                  B
##########################################################################
###############################################################################
# B. Top-15 特征选择方法比较
#
# 比较四种 Top-15 特征选择方法所构建 XGBoost 模型的测试集性能：
# SHAP、Gain、LASSO、Random Forest Permutation importance。
#
#   B-1. SHAP / Gain / LASSO Top-15 的统一比较
#   B-2. Random Forest Permutation importance Top-15
#
# 使用说明：
# 1. SHAP Top-15 模型不重新训练，直接读取既有测试集预测结果。
# 2. Gain Top-15 由完整 78 变量 XGBoost 模型的 Gain 排序产生。
# 3. LASSO Top-15 与 RF Permutation Top-15 均仅使用训练集产生，测试集不参与选择。
# 4. 需重新训练的模型沿用主分析框架：5 折交叉验证、50 次随机搜索、
#    相同超参数搜索空间、训练集类别权重、nrounds = 200、阈值 0.50。
# 5. 各方法结果写入各自独立目录；B-2 读取 B-1 产出的预测结果，须按先后顺序运行。
###############################################################################

###############################################################################
# B-1. SHAP / Gain / LASSO Top-15 的统一比较
###############################################################################

rm(list = ls())
gc()

###############################################################################
# 1. Packages and reproducibility
###############################################################################

required_packages <- c(
  "data.table", "readxl", "pROC", "PRROC", "glmnet", "caret",
  "xgboost", "mlr3", "mlr3learners", "mlr3tuning", "paradox", "future"
)

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

future::plan("sequential")

GLOBAL_SEED <- 123L
OPERATING_THRESHOLD <- 0.50
set.seed(GLOBAL_SEED)

###############################################################################
# 2. Authoritative input paths and new output directory
###############################################################################

# 与主分析完全相同的预处理数据。
train_path <- "J:/rtools45/train_preprocessed.xlsx"
test_path  <- "J:/rtools45/test_preprocessed.xlsx"

# 完整78变量XGBoost模型产生的Gain排序。
full_xgb_dir <- "J:/rtools45/XGBoost_Results"

# SHAP Top-15 特征清单。
shap_importance_root <- "J:/rtools45/SHAP_Results_TOPJOURNAL"

# Top-15 缩减模型的测试集预测结果。

shap_prediction_dir <- paste0(
  "J:/rtools45/手工4乘积交互Model_Comparison_Independent_Tuning/",
  "Prediction_Files"
)

# SHAP / Gain / LASSO 比较结果输出目录。
save_dir <- "J:/rtools45/SHAP_Gain_LASSO_Top15_Combined_Comparison"

subdirs <- c(
  "Feature_Lists", "Models", "Predictions", "Performance",
  "DeLong", "Tuning_History", "Best_Parameters", "Checks"
)

dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)
for (d in subdirs) {
  dir.create(file.path(save_dir, d), recursive = TRUE, showWarnings = FALSE)
}

###############################################################################
# 3. Outcomes
###############################################################################

outcomes <- c(
  "Outcome_Disease Non-Progression",
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma"
)

clean_outcome_name <- function(outcome_col) {
  paste0(gsub("[^A-Za-z0-9]", "_", outcome_col), "_")
}

display_outcome_name <- function(outcome_col) {
  sub("^Outcome_", "", outcome_col)
}

###############################################################################
# 4. Read and validate the common training/test data
###############################################################################

for (p in c(train_path, test_path)) {
  if (!file.exists(p)) stop("找不到输入数据：", p)
}

train_all <- data.table::as.data.table(readxl::read_excel(train_path))
test_all  <- data.table::as.data.table(readxl::read_excel(test_path))

if (anyDuplicated(names(train_all)) || anyDuplicated(names(test_all))) {
  stop("训练集或测试集含重复列名。")
}
if (!identical(names(train_all), names(test_all))) {
  stop("训练集和测试集列名或列顺序不完全一致。")
}
if (!all(outcomes %in% names(train_all))) {
  stop("数据中缺少一个或多个预设结局变量。")
}

# 三个结局全部排除，禁止把其他结局作为预测变量。
candidate_features <- setdiff(names(train_all), outcomes)
if (length(candidate_features) != 78L) {
  stop("候选预测变量必须为78个；当前为：", length(candidate_features))
}

if (anyNA(train_all[, ..candidate_features]) ||
    anyNA(test_all[, ..candidate_features])) {
  stop("当前预处理数据仍含缺失值，不能复现原Top-15训练框架。")
}

###############################################################################
# 5. Common helpers
###############################################################################

standardize_prediction <- function(pred, source_label) {
  pred <- data.table::as.data.table(pred)
  
  true_col <- intersect(
    c("True_Label", "TrueLabel", "true_label", "Observed", "Label"),
    names(pred)
  )[1]
  prob_col <- intersect(
    c("Predicted_Prob", "PredictedProbability", "predicted_prob",
      "Probability", "Prob"),
    names(pred)
  )[1]
  
  if (is.na(true_col) || is.na(prob_col)) {
    stop(source_label, "缺少真实标签列或预测概率列。")
  }
  
  ans <- data.table::data.table(
    Row_Index = seq_len(nrow(pred)),
    True_Label = as.integer(pred[[true_col]]),
    Predicted_Prob = as.numeric(pred[[prob_col]])
  )
  
  if (!all(ans$True_Label %in% c(0L, 1L))) {
    stop(source_label, "的真实标签不是严格0/1编码。")
  }
  if (anyNA(ans$Predicted_Prob) || any(!is.finite(ans$Predicted_Prob)) ||
      any(ans$Predicted_Prob < 0 | ans$Predicted_Prob > 1)) {
    stop(source_label, "含无效预测概率。")
  }
  
  ans
}

verify_prediction_alignment <- function(pred, expected_labels, source_label) {
  if (nrow(pred) != length(expected_labels)) {
    stop(source_label, "与当前测试集行数不一致。")
  }
  if (!identical(as.integer(pred$True_Label), as.integer(expected_labels))) {
    stop(source_label, "与当前测试集真实标签或行顺序不一致。")
  }
  invisible(TRUE)
}

safe_logit <- function(prob, eps = 1e-6) {
  qlogis(pmin(pmax(as.numeric(prob), eps), 1 - eps))
}

calculate_metrics <- function(true, prob, threshold = OPERATING_THRESHOLD) {
  true <- as.integer(true)
  prob <- as.numeric(prob)
  
  pred <- ifelse(prob > threshold, 1L, 0L)
  
  cm <- caret::confusionMatrix(
    data = factor(pred, levels = c(0, 1)),
    reference = factor(true, levels = c(0, 1)),
    positive = "1"
  )
  
  roc_obj <- pROC::roc(
    response = true,
    predictor = prob,
    levels = c(0, 1),
    direction = "<",
    quiet = TRUE
  )
  ci_auc <- as.numeric(pROC::ci.auc(roc_obj, method = "delong"))
  
  pr_obj <- PRROC::pr.curve(
    scores.class0 = prob[true == 1L],
    scores.class1 = prob[true == 0L],
    curve = FALSE
  )
  
  lp <- safe_logit(prob)
  
  # 联合估计的校准截距和斜率。
  calibration_model <- tryCatch(
    glm(true ~ lp, family = binomial()),
    error = function(e) NULL
  )
  calibration_intercept <- if (is.null(calibration_model)) NA_real_ else
    unname(coef(calibration_model)[1])
  calibration_slope <- if (is.null(calibration_model)) NA_real_ else
    unname(coef(calibration_model)[2])
  
  # Calibration-in-the-large：斜率固定为1。
  calibration_in_large <- tryCatch(
    unname(coef(glm(true ~ 1, family = binomial(), offset = lp))[1]),
    error = function(e) NA_real_
  )
  
  metrics <- data.table::data.table(
    N = length(true),
    Events = sum(true == 1L),
    Threshold = threshold,
    AUC = as.numeric(pROC::auc(roc_obj)),
    AUC_CI_Lower = ci_auc[1],
    AUC_CI_Upper = ci_auc[3],
    AUPRC = as.numeric(pr_obj$auc.integral),
    Brier = mean((true - prob)^2),
    Calibration_Intercept = calibration_intercept,
    Calibration_Slope = calibration_slope,
    Calibration_in_the_Large = calibration_in_large,
    Accuracy = unname(cm$overall["Accuracy"]),
    Sensitivity = unname(cm$byClass["Sensitivity"]),
    Specificity = unname(cm$byClass["Specificity"]),
    PPV = unname(cm$byClass["Precision"]),
    NPV = unname(cm$byClass["Neg Pred Value"]),
    F1 = unname(cm$byClass["F1"])
  )
  
  list(metrics = metrics, roc = roc_obj)
}

###############################################################################
# 6. Read SHAP Top-15 feature list and authoritative prediction
###############################################################################

read_shap_top15 <- function(outcome_col) {
  outcome_clean <- clean_outcome_name(outcome_col)
  path <- file.path(
    shap_importance_root, outcome_clean, "SHAP_Importance_Top15.csv"
  )
  
  if (!file.exists(path)) stop("找不到SHAP Top-15清单：", path)
  tab <- data.table::fread(path, check.names = FALSE)
  if (!"Feature" %in% names(tab) || nrow(tab) < 15L) {
    stop("SHAP Top-15文件格式不正确：", path)
  }
  
  features <- as.character(tab$Feature[1:15])
  if (anyDuplicated(features) || !all(features %in% candidate_features)) {
    stop(outcome_col, "的SHAP Top-15含重复或未知变量。")
  }
  
  out <- data.table::data.table(
    Outcome = display_outcome_name(outcome_col),
    Selection_Method = "SHAP Top-15",
    Rank = seq_len(15L),
    Feature = features
  )
  list(features = features, table = out, source = path)
}

read_authoritative_shap_prediction <- function(outcome_col, expected_labels) {
  outcome_clean <- clean_outcome_name(outcome_col)
  path <- file.path(
    shap_prediction_dir, paste0(outcome_clean, "Top15_Prediction.csv")
  )
  
  if (!file.exists(path)) stop("找不到当前SHAP Top-15预测：", path)
  pred <- standardize_prediction(data.table::fread(path), "SHAP Top-15预测")
  verify_prediction_alignment(pred, expected_labels, "SHAP Top-15预测")
  
  list(prediction = pred, source = path)
}

###############################################################################
# 7. Gain Top-15 from the complete 78-feature XGBoost model
###############################################################################

select_gain_top15 <- function(outcome_col) {
  outcome_clean <- clean_outcome_name(outcome_col)
  path <- file.path(
    full_xgb_dir, paste0(outcome_clean, "Feature_Importance.csv")
  )
  
  if (!file.exists(path)) stop("找不到完整模型Gain结果：", path)
  tab <- data.table::fread(path, check.names = FALSE)
  if (!all(c("Feature", "Gain") %in% names(tab))) {
    stop("Gain文件缺少Feature或Gain列：", path)
  }
  
  tab[, Gain := as.numeric(Gain)]
  if (anyNA(tab$Gain)) stop("Gain列含NA或非数值：", path)
  data.table::setorder(tab, -Gain, Feature)
  
  features <- as.character(tab$Feature[1:15])
  if (length(features) != 15L || anyDuplicated(features) ||
      !all(features %in% candidate_features)) {
    stop(outcome_col, "的Gain Top-15含重复、缺失或未知变量。")
  }
  
  out <- data.table::data.table(
    Outcome = display_outcome_name(outcome_col),
    Selection_Method = "Gain Top-15",
    Rank = seq_len(15L),
    Feature = features,
    Selection_Score = tab$Gain[1:15]
  )
  
  list(features = features, table = out, source = path)
}

###############################################################################
# 8. LASSO Top-15 selected using the training set only
###############################################################################

select_lasso_top15 <- function(x_train, y_train, feature_names, outcome_col) {
  set.seed(GLOBAL_SEED)
  
  # 固定、分层5折，仅用于LASSO正则化路径评价。
  folds <- caret::createFolds(
    factor(y_train, levels = c(0, 1)),
    k = 5,
    list = TRUE,
    returnTrain = FALSE
  )
  foldid <- integer(length(y_train))
  for (k in seq_along(folds)) foldid[folds[[k]]] <- k
  if (any(foldid == 0L)) stop(outcome_col, "的LASSO foldid生成失败。")
  
  cvfit <- glmnet::cv.glmnet(
    x = x_train,
    y = y_train,
    family = "binomial",
    alpha = 1,
    standardize = TRUE,
    intercept = TRUE,
    foldid = foldid,
    type.measure = "auc",
    keep = TRUE
  )
  
  # 在CV路径上寻找非零变量数最接近15且不少于15的lambda；
  # 若并列，选择CV AUC最高者。随后按标准化系数绝对值取前15。
  path_df <- cvfit$glmnet.fit$df
  eligible <- which(path_df >= 15L)
  if (!length(eligible)) {
    stop(outcome_col, "的LASSO路径没有产生至少15个非零变量。")
  }
  minimum_excess <- min(path_df[eligible] - 15L)
  candidates <- eligible[path_df[eligible] - 15L == minimum_excess]
  best_idx <- candidates[which.max(cvfit$cvm[candidates])]
  selected_lambda <- cvfit$lambda[best_idx]
  
  beta <- as.matrix(coef(cvfit, s = selected_lambda))
  beta <- beta[rownames(beta) != "(Intercept)", , drop = FALSE]
  
  feature_sd <- apply(x_train, 2, stats::sd)
  coef_table <- data.table::data.table(
    Feature = rownames(beta),
    Coefficient = as.numeric(beta[, 1])
  )
  coef_table[, Feature_SD := feature_sd[Feature]]
  coef_table[, Standardized_Coefficient := Coefficient * Feature_SD]
  coef_table[, Abs_Standardized_Coefficient := abs(Standardized_Coefficient)]
  coef_table <- coef_table[Coefficient != 0]
  data.table::setorder(coef_table, -Abs_Standardized_Coefficient, Feature)
  
  if (nrow(coef_table) < 15L) {
    stop(outcome_col, "最终LASSO lambda下不足15个非零变量。")
  }
  
  selected <- coef_table[1:15]
  features <- selected$Feature
  
  out <- data.table::data.table(
    Outcome = display_outcome_name(outcome_col),
    Selection_Method = "LASSO Top-15",
    Rank = seq_len(15L),
    Feature = features,
    Selection_Score = selected$Abs_Standardized_Coefficient,
    Standardized_Coefficient = selected$Standardized_Coefficient,
    Original_Scale_Coefficient = selected$Coefficient
  )
  
  info <- data.table::data.table(
    Outcome = display_outcome_name(outcome_col),
    Lambda_Min = cvfit$lambda.min,
    Lambda_1SE = cvfit$lambda.1se,
    Selected_Lambda = selected_lambda,
    Nonzero_at_Selected_Lambda = nrow(coef_table),
    CV_AUC_at_Selected_Lambda = cvfit$cvm[best_idx],
    Final_Number_of_Features = 15L,
    CV_Folds = 5L,
    Seed = GLOBAL_SEED,
    Selection_Rule = paste0(
      "Closest number of nonzero variables to 15 (>=15) on the CV path; ",
      "ties resolved by highest CV AUC; Top-15 ranked by absolute ",
      "standardized coefficient"
    )
  )
  
  list(
    features = features,
    table = out,
    all_coefficients = coef_table,
    info = info,
    cvfit = cvfit
  )
}

###############################################################################
# 9. Independently tuned XGBoost: identical framework for Gain and LASSO
###############################################################################

train_top15_xgboost <- function(
    train_data, test_data, outcome_col, selected_features, method_label
) {
  if (length(selected_features) != 15L || anyDuplicated(selected_features)) {
    stop(method_label, "必须恰好包含15个不重复特征。")
  }
  
  x_train <- data.matrix(train_data[, ..selected_features])
  x_test  <- data.matrix(test_data[, ..selected_features])
  y_train <- as.integer(train_data[[outcome_col]])
  y_test  <- as.integer(test_data[[outcome_col]])
  
  if (anyNA(x_train) || anyNA(x_test) ||
      any(!is.finite(x_train)) || any(!is.finite(x_test))) {
    stop(outcome_col, " / ", method_label, "的XGBoost矩阵含NA/NaN/Inf。")
  }
  
  pos_num <- sum(y_train == 1L)
  neg_num <- sum(y_train == 0L)
  if (pos_num == 0L || neg_num == 0L) stop("训练集结局只有一个类别。")
  scale_pos_weight <- neg_num / pos_num
  
  task_df <- data.frame(x_train, check.names = FALSE)
  task_df[[outcome_col]] <- factor(
    y_train, levels = c(0, 1), labels = c("0", "1")
  )
  
  task <- mlr3::TaskClassif$new(
    id = paste0(clean_outcome_name(outcome_col), gsub("[^A-Za-z0-9]", "_", method_label)),
    backend = task_df,
    target = outcome_col,
    positive = "1"
  )
  
  learner <- mlr3::lrn(
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
  
  search_space <- paradox::ps(
    eta = paradox::p_dbl(0.02, 0.06),
    max_depth = paradox::p_int(3, 5),
    min_child_weight = paradox::p_dbl(6, 15),
    subsample = paradox::p_dbl(0.65, 0.80),
    colsample_bytree = paradox::p_dbl(0.65, 0.80),
    gamma = paradox::p_dbl(1, 5),
    lambda = paradox::p_dbl(3, 10),
    alpha = paradox::p_dbl(1, 6)
  )
  
  tuner <- mlr3tuning::auto_tuner(
    tuner = mlr3tuning::tnr("random_search"),
    learner = learner,
    resampling = mlr3::rsmp("cv", folds = 5),
    measure = mlr3::msr("classif.auc"),
    search_space = search_space,
    terminator = mlr3tuning::trm("evals", n_evals = 50)
  )
  
  # 每个候选Top-15模型均从相同种子独立调参，保证可复现和规则一致。
  set.seed(GLOBAL_SEED)
  start_time <- Sys.time()
  tuner$train(task)
  runtime_minutes <- as.numeric(
    difftime(Sys.time(), start_time, units = "mins")
  )
  
  best_params <- tuner$archive$best()$x_domain[[1]]
  final_params <- list(
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
    scale_pos_weight = scale_pos_weight,
    nthread = 4,
    seed = GLOBAL_SEED
  )
  
  set.seed(GLOBAL_SEED)
  model <- xgboost::xgb.train(
    params = final_params,
    data = xgboost::xgb.DMatrix(x_train, label = y_train),
    nrounds = 200,
    verbose = 0
  )
  
  prob <- predict(model, xgboost::xgb.DMatrix(x_test, label = y_test))
  prediction <- data.table::data.table(
    Row_Index = seq_along(y_test),
    True_Label = y_test,
    Predicted_Prob = as.numeric(prob),
    Predicted_Class = ifelse(prob > OPERATING_THRESHOLD, 1L, 0L)
  )
  
  tuning_history <- as.data.frame(tuner$archive$data)
  tuning_history <- tuning_history[
    , vapply(tuning_history, function(x) !is.list(x), logical(1)),
    drop = FALSE
  ]
  
  params_table <- data.table::data.table(
    Parameter = names(final_params),
    Value = vapply(final_params, as.character, character(1))
  )
  
  run_info <- data.table::data.table(
    Outcome = display_outcome_name(outcome_col),
    Selection_Method = method_label,
    Train_Rows = nrow(x_train),
    Test_Rows = nrow(x_test),
    Predictors = ncol(x_train),
    CV_Folds = 5L,
    Random_Search_Evaluations = 50L,
    Nrounds = 200L,
    Scale_Pos_Weight = scale_pos_weight,
    Operating_Threshold = OPERATING_THRESHOLD,
    Seed = GLOBAL_SEED,
    Runtime_Minutes = runtime_minutes
  )
  
  list(
    model = model,
    prediction = prediction,
    best_params = final_params,
    params_table = params_table,
    tuning_history = tuning_history,
    run_info = run_info
  )
}

###############################################################################
# 10. Paired DeLong helper
###############################################################################

paired_delong <- function(outcome, reference_name, reference_pred,
                          comparator_name, comparator_pred) {
  if (!identical(reference_pred$True_Label, comparator_pred$True_Label)) {
    stop("DeLong比较的真实标签或测试集顺序不一致。")
  }
  
  roc_ref <- pROC::roc(
    reference_pred$True_Label, reference_pred$Predicted_Prob,
    levels = c(0, 1), direction = "<", quiet = TRUE
  )
  roc_cmp <- pROC::roc(
    comparator_pred$True_Label, comparator_pred$Predicted_Prob,
    levels = c(0, 1), direction = "<", quiet = TRUE
  )
  test <- pROC::roc.test(roc_ref, roc_cmp, method = "delong", paired = TRUE)
  
  data.table::data.table(
    Outcome = outcome,
    Reference_Model = reference_name,
    Comparator_Model = comparator_name,
    Reference_AUC = as.numeric(pROC::auc(roc_ref)),
    Comparator_AUC = as.numeric(pROC::auc(roc_cmp)),
    Delta_AUC_Reference_Minus_Comparator =
      as.numeric(pROC::auc(roc_ref)) - as.numeric(pROC::auc(roc_cmp)),
    Paired_DeLong_P = as.numeric(test$p.value)
  )
}

###############################################################################
# 11. Main analysis loop
###############################################################################

performance_list <- list()
delong_list <- list()
feature_list <- list()
overlap_list <- list()
run_info_list <- list()
provenance_list <- list()
complete_results <- list()

for (outcome_col in outcomes) {
  outcome_display <- display_outcome_name(outcome_col)
  outcome_clean <- clean_outcome_name(outcome_col)
  
  cat("\n============================================================\n")
  cat("Outcome:", outcome_display, "\n")
  cat("============================================================\n")
  
  y_train <- as.integer(train_all[[outcome_col]])
  y_test  <- as.integer(test_all[[outcome_col]])
  if (!all(y_train %in% c(0L, 1L)) || !all(y_test %in% c(0L, 1L))) {
    stop(outcome_col, "不是严格0/1编码。")
  }
  
  # (1) SHAP Top-15 特征清单与对应的测试集预测。
  shap_selection <- read_shap_top15(outcome_col)
  shap_source <- read_authoritative_shap_prediction(outcome_col, y_test)
  shap_pred <- shap_source$prediction
  
  # (2) Gain Top-15 特征清单。
  gain_selection <- select_gain_top15(outcome_col)
  
  # (3) LASSO Top-15 特征清单（训练集内完成）。
  x_train_full <- data.matrix(train_all[, ..candidate_features])
  lasso_selection <- select_lasso_top15(
    x_train = x_train_full,
    y_train = y_train,
    feature_names = candidate_features,
    outcome_col = outcome_col
  )
  
  # D. 使用完全相同的主分析XGBoost框架独立训练Gain和LASSO模型。
  gain_fit <- train_top15_xgboost(
    train_all, test_all, outcome_col,
    gain_selection$features, "Gain Top-15"
  )
  lasso_fit <- train_top15_xgboost(
    train_all, test_all, outcome_col,
    lasso_selection$features, "LASSO Top-15"
  )
  
  verify_prediction_alignment(gain_fit$prediction, y_test, "Gain Top-15预测")
  verify_prediction_alignment(lasso_fit$prediction, y_test, "LASSO Top-15预测")
  
  # E. 三种模型统一重新计算评价指标。
  shap_eval  <- calculate_metrics(shap_pred$True_Label, shap_pred$Predicted_Prob)
  gain_eval  <- calculate_metrics(gain_fit$prediction$True_Label,
                                  gain_fit$prediction$Predicted_Prob)
  lasso_eval <- calculate_metrics(lasso_fit$prediction$True_Label,
                                  lasso_fit$prediction$Predicted_Prob)
  
  performance <- data.table::rbindlist(list(
    cbind(data.table::data.table(
      Outcome = outcome_display, Selection_Method = "SHAP Top-15"
    ), shap_eval$metrics),
    cbind(data.table::data.table(
      Outcome = outcome_display, Selection_Method = "Gain Top-15"
    ), gain_eval$metrics),
    cbind(data.table::data.table(
      Outcome = outcome_display, Selection_Method = "LASSO Top-15"
    ), lasso_eval$metrics)
  ), use.names = TRUE)
  
  # 防止再次误读旧SHAP预测：输出并检查当前AUC。
  expected_shap_auc <- c(
    "Outcome_Disease Non-Progression" = 0.910505133152552,
    "Outcome_Liver Cirrhosis" = 0.881864789970840,
    "Outcome_Hepatocellular Carcinoma" = 0.889750310438627
  )
  observed_shap_auc <- performance[
    Selection_Method == "SHAP Top-15", AUC
  ]
  if (abs(observed_shap_auc - expected_shap_auc[[outcome_col]]) > 1e-10) {
    stop(
      outcome_col,
      "的SHAP AUC与论文当前权威预测不一致。实际=",
      format(observed_shap_auc, digits = 15),
      "；预期=", format(expected_shap_auc[[outcome_col]], digits = 15)
    )
  }
  
  # F. 三组两两配对DeLong；主要比较均以SHAP为参考。
  delong <- data.table::rbindlist(list(
    paired_delong(
      outcome_display, "SHAP Top-15", shap_pred,
      "Gain Top-15", gain_fit$prediction
    ),
    paired_delong(
      outcome_display, "SHAP Top-15", shap_pred,
      "LASSO Top-15", lasso_fit$prediction
    ),
    paired_delong(
      outcome_display, "Gain Top-15", gain_fit$prediction,
      "LASSO Top-15", lasso_fit$prediction
    )
  ))
  
  # G. 与SHAP Top-15的特征重叠。
  overlap <- data.table::data.table(
    Outcome = outcome_display,
    Selection_Method = c("SHAP Top-15", "Gain Top-15", "LASSO Top-15"),
    Overlap_with_SHAP_n = c(
      15L,
      length(intersect(shap_selection$features, gain_selection$features)),
      length(intersect(shap_selection$features, lasso_selection$features))
    )
  )
  overlap[, Overlap_with_SHAP_Percent := Overlap_with_SHAP_n / 15 * 100]
  
  # H. 保存全部独立结果。
  feature_tables <- data.table::rbindlist(list(
    shap_selection$table[, Selection_Score := NA_real_],
    gain_selection$table[, .(
      Outcome, Selection_Method, Rank, Feature, Selection_Score
    )],
    lasso_selection$table[, .(
      Outcome, Selection_Method, Rank, Feature, Selection_Score
    )]
  ), use.names = TRUE, fill = TRUE)
  
  data.table::fwrite(
    feature_tables,
    file.path(save_dir, "Feature_Lists",
              paste0(outcome_clean, "Three_Methods_Top15_Features.csv"))
  )
  data.table::fwrite(
    lasso_selection$all_coefficients,
    file.path(save_dir, "Feature_Lists",
              paste0(outcome_clean, "LASSO_All_Nonzero_Coefficients.csv"))
  )
  data.table::fwrite(
    lasso_selection$info,
    file.path(save_dir, "Checks",
              paste0(outcome_clean, "LASSO_Selection_Info.csv"))
  )
  
  saveRDS(
    lasso_selection$cvfit,
    file.path(save_dir, "Models", paste0(outcome_clean, "LASSO_CV.rds"))
  )
  saveRDS(
    gain_fit$model,
    file.path(save_dir, "Models", paste0(outcome_clean, "Gain_Top15_XGBoost.rds"))
  )
  saveRDS(
    lasso_fit$model,
    file.path(save_dir, "Models", paste0(outcome_clean, "LASSO_Top15_XGBoost.rds"))
  )
  
  data.table::fwrite(
    shap_pred,
    file.path(save_dir, "Predictions", paste0(outcome_clean, "SHAP_Top15_Prediction.csv"))
  )
  data.table::fwrite(
    gain_fit$prediction,
    file.path(save_dir, "Predictions", paste0(outcome_clean, "Gain_Top15_Prediction.csv"))
  )
  data.table::fwrite(
    lasso_fit$prediction,
    file.path(save_dir, "Predictions", paste0(outcome_clean, "LASSO_Top15_Prediction.csv"))
  )
  
  data.table::fwrite(
    performance,
    file.path(save_dir, "Performance", paste0(outcome_clean, "Three_Methods_Performance.csv"))
  )
  data.table::fwrite(
    delong,
    file.path(save_dir, "DeLong", paste0(outcome_clean, "Three_Paired_DeLong_Tests.csv"))
  )
  data.table::fwrite(
    gain_fit$tuning_history,
    file.path(save_dir, "Tuning_History", paste0(outcome_clean, "Gain_Tuning_History.csv"))
  )
  data.table::fwrite(
    lasso_fit$tuning_history,
    file.path(save_dir, "Tuning_History", paste0(outcome_clean, "LASSO_Tuning_History.csv"))
  )
  data.table::fwrite(
    gain_fit$params_table,
    file.path(save_dir, "Best_Parameters", paste0(outcome_clean, "Gain_Best_Parameters.csv"))
  )
  data.table::fwrite(
    lasso_fit$params_table,
    file.path(save_dir, "Best_Parameters", paste0(outcome_clean, "LASSO_Best_Parameters.csv"))
  )
  
  provenance <- data.table::data.table(
    Outcome = outcome_display,
    Data_Train = train_path,
    Data_Test = test_path,
    SHAP_Feature_Source = shap_selection$source,
    SHAP_Prediction_Source = shap_source$source,
    Gain_Importance_Source = gain_selection$source
  )
  
  performance_list[[outcome_col]] <- performance
  delong_list[[outcome_col]] <- delong
  feature_list[[outcome_col]] <- feature_tables
  overlap_list[[outcome_col]] <- overlap
  run_info_list[[outcome_col]] <- data.table::rbindlist(list(
    gain_fit$run_info, lasso_fit$run_info
  ))
  provenance_list[[outcome_col]] <- provenance
  
  complete_results[[outcome_col]] <- list(
    performance = performance,
    delong = delong,
    overlap = overlap,
    shap_features = shap_selection$features,
    gain_features = gain_selection$features,
    lasso_features = lasso_selection$features,
    gain_model = gain_fit$model,
    lasso_model = lasso_fit$model,
    shap_prediction = shap_pred,
    gain_prediction = gain_fit$prediction,
    lasso_prediction = lasso_fit$prediction
  )
  
  cat("SHAP AUC :", sprintf("%.4f", performance[1, AUC]), "\n")
  cat("Gain AUC :", sprintf("%.4f", performance[2, AUC]), "\n")
  cat("LASSO AUC:", sprintf("%.4f", performance[3, AUC]), "\n")
}

###############################################################################
# 12. Combined outputs and summary tables
###############################################################################

performance_all <- data.table::rbindlist(performance_list, use.names = TRUE)
delong_all <- data.table::rbindlist(delong_list, use.names = TRUE)
features_all <- data.table::rbindlist(feature_list, use.names = TRUE, fill = TRUE)
overlap_all <- data.table::rbindlist(overlap_list, use.names = TRUE)
run_info_all <- data.table::rbindlist(run_info_list, use.names = TRUE)
provenance_all <- data.table::rbindlist(provenance_list, use.names = TRUE)

data.table::fwrite(
  performance_all,
  file.path(save_dir, "All_Three_Methods_Performance_FULL.csv")
)
data.table::fwrite(
  delong_all,
  file.path(save_dir, "All_Three_Methods_Paired_DeLong.csv")
)
data.table::fwrite(
  features_all,
  file.path(save_dir, "All_Three_Methods_Top15_Features.csv")
)
data.table::fwrite(
  overlap_all,
  file.path(save_dir, "All_Three_Methods_Feature_Overlap.csv")
)
data.table::fwrite(
  run_info_all,
  file.path(save_dir, "Checks", "All_Model_Run_Info.csv")
)
data.table::fwrite(
  provenance_all,
  file.path(save_dir, "Checks", "Authoritative_Input_Provenance.csv")
)

# 仅以SHAP为参考提取Gain和LASSO比较结果，生成补充表。
delong_vs_shap <- delong_all[Reference_Model == "SHAP Top-15"]

paper_table <- merge(
  performance_all,
  overlap_all,
  by = c("Outcome", "Selection_Method"),
  all.x = TRUE,
  sort = FALSE
)

paper_table[, `:=`(
  Delta_AUC_SHAP_minus_Method = NA_real_,
  Paired_DeLong_P_vs_SHAP = NA_real_
)]

for (i in seq_len(nrow(delong_vs_shap))) {
  idx <- paper_table$Outcome == delong_vs_shap$Outcome[i] &
    paper_table$Selection_Method == delong_vs_shap$Comparator_Model[i]
  paper_table[idx, Delta_AUC_SHAP_minus_Method :=
                delong_vs_shap$Reference_AUC[i] - AUC]
  paper_table[idx, Paired_DeLong_P_vs_SHAP :=
                delong_vs_shap$Paired_DeLong_P[i]]
}

paper_table[, AUC_95CI := sprintf(
  "%.4f (%.4f-%.4f)", AUC, AUC_CI_Lower, AUC_CI_Upper
)]
paper_table[, Feature_Overlap_with_SHAP := sprintf(
  "%d/15 (%.1f%%)", Overlap_with_SHAP_n, Overlap_with_SHAP_Percent
)]

combined_table <- paper_table[, .(
  Outcome,
  Feature_Selection_Method = Selection_Method,
  Feature_Overlap_with_SHAP,
  AUC_95CI,
  Accuracy,
  Sensitivity,
  Specificity,
  PPV,
  NPV,
  F1_Score = F1,
  Delta_AUC_SHAP_minus_Method,
  Paired_DeLong_P_vs_SHAP
)]

data.table::fwrite(
  combined_table,
  file.path(save_dir, "Supplementary_Table_SHAP_Gain_LASSO_Top15.csv")
)

saveRDS(
  list(
    results = complete_results,
    performance = performance_all,
    delong = delong_all,
    features = features_all,
    overlap = overlap_all,
    summary_table = combined_table,
    provenance = provenance_all,
    seed = GLOBAL_SEED,
    threshold = OPERATING_THRESHOLD
  ),
  file.path(save_dir, "SHAP_Gain_LASSO_Top15_Complete_Results.rds")
)

capture.output(
  sessionInfo(),
  file = file.path(save_dir, "Checks", "R_Session_Info.txt")
)

cat("\n============================================================\n")
cat("全部完成：SHAP、Gain和LASSO Top-15统一比较。\n")
cat("输出目录：", save_dir, "\n")
cat("补充表：Supplementary_Table_SHAP_Gain_LASSO_Top15.csv\n")
cat("SHAP预测来源已锁定并通过当前论文AUC校验。\n")
cat("旧结果未被覆盖。\n")
cat("============================================================\n")


#######################################################################
# B-2. Random Forest Permutation importance Top-15
#
# 每个结局的流程：
#   1. 仅用训练集训练 Random Forest；
#   2. 按 Permutation importance 选择 Top-15 特征；
#   3. 用该 Top-15 按相同框架独立调参并训练 XGBoost；
#   4. 在同一测试集上与 SHAP、Gain、LASSO Top-15 及 Full 模型做配对 DeLong 检验。
#######################################################################

rm(list = ls())
gc()

#######################################################################
# 1. 软件包与参数
#######################################################################

required_packages <- c(
  "data.table", "readxl", "ranger", "pROC", "xgboost", "caret",
  "mlr3", "mlr3learners", "mlr3tuning", "paradox", "future"
)
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop("请先安装R包：", paste(missing_packages, collapse = ", "))
}

suppressPackageStartupMessages({
  library(data.table)
  library(readxl)
  library(ranger)
  library(pROC)
  library(xgboost)
  library(caret)
  library(mlr3)
  library(mlr3learners)
  library(mlr3tuning)
  library(paradox)
  library(future)
})

future::plan("sequential")
GLOBAL_SEED <- 123L
OPERATING_THRESHOLD <- 0.50
RF_NUM_TREES <- 1000L
set.seed(GLOBAL_SEED)

#######################################################################
# 2. 路径
#######################################################################

train_path <- "J:/rtools45/train_preprocessed.xlsx"
test_path  <- "J:/rtools45/test_preprocessed.xlsx"

combined_dir <- "J:/rtools45/SHAP_Gain_LASSO_Top15_Combined_Comparison"
reference_prediction_dir <- file.path(combined_dir, "Predictions")
full_prediction_dir <- "J:/rtools45/XGBoost_Results"

output_dir <- "J:/rtools45/RF_Permutation_Top15_XGBoost_Exploratory"
subdirs <- c(
  "RF_Models", "RF_Importance", "Feature_Lists", "XGBoost_Models",
  "Predictions", "Performance", "DeLong", "Best_Parameters",
  "Tuning_History", "Checks"
)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
invisible(lapply(file.path(output_dir, subdirs), dir.create,
                 recursive = TRUE, showWarnings = FALSE))

if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("找不到训练或测试数据。")
}

#######################################################################
# 3. 数据和结局
#######################################################################

train_data <- as.data.table(read_excel(train_path))
test_data  <- as.data.table(read_excel(test_path))

# 保持与原独立调参脚本相同的运行顺序
outcomes <- c(
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma",
  "Outcome_Disease Non-Progression"
)
publication_order <- c(
  "Outcome_Disease Non-Progression",
  "Outcome_Liver Cirrhosis",
  "Outcome_Hepatocellular Carcinoma"
)

if (!all(outcomes %in% names(train_data)) || !all(outcomes %in% names(test_data))) {
  stop("输入数据缺少预设结局列。")
}

candidate_features <- setdiff(names(train_data), outcomes)
if (!identical(candidate_features, setdiff(names(test_data), outcomes))) {
  stop("训练集与测试集的候选特征或顺序不一致。")
}
if (length(candidate_features) < 15L) stop("候选特征少于15个。")

clean_outcome <- function(x) gsub("[^A-Za-z0-9]", "_", x)

#######################################################################
# 4. RF Permutation importance Top-15
#######################################################################

select_rf_top15 <- function(outcome, seed_offset = 0L) {
  x <- as.data.frame(train_data[, ..candidate_features], check.names = FALSE)
  if (anyNA(x)) stop(outcome, " RF输入中存在NA。")
  y <- as.integer(train_data[[outcome]])
  if (!all(y %in% c(0L, 1L))) stop(outcome, " 结局不是0/1。")
  
  rf_df <- x
  rf_df$.target <- factor(y, levels = c(0, 1), labels = c("0", "1"))
  n0 <- sum(y == 0L)
  n1 <- sum(y == 1L)
  class_weights <- c("0" = length(y) / (2 * n0),
                     "1" = length(y) / (2 * n1))
  rf_mtry <- max(1L, floor(sqrt(length(candidate_features))))
  
  set.seed(GLOBAL_SEED + seed_offset)
  rf_fit <- ranger(
    dependent.variable.name = ".target",
    data = rf_df,
    probability = TRUE,
    num.trees = RF_NUM_TREES,
    mtry = rf_mtry,
    min.node.size = 5,
    replace = TRUE,
    sample.fraction = 1,
    importance = "permutation",
    scale.permutation.importance = TRUE,
    class.weights = class_weights,
    num.threads = 4,
    seed = GLOBAL_SEED + seed_offset,
    respect.unordered.factors = "order",
    verbose = FALSE
  )
  
  imp <- data.table(
    Feature = names(rf_fit$variable.importance),
    Permutation_Importance = as.numeric(rf_fit$variable.importance)
  )
  setorder(imp, -Permutation_Importance, Feature)
  imp[, Rank := seq_len(.N)]
  setcolorder(imp, c("Rank", "Feature", "Permutation_Importance"))
  top15 <- imp$Feature[1:15]
  if (length(top15) != 15L || anyNA(top15) || anyDuplicated(top15)) {
    stop(outcome, " RF Top-15选择失败。")
  }
  
  list(
    model = rf_fit,
    importance = imp,
    top15 = top15,
    mtry = rf_mtry,
    class_weights = class_weights,
    oob_brier = rf_fit$prediction.error
  )
}

#######################################################################
# 5. 性能计算
#######################################################################

calculate_metrics <- function(true, prob) {
  true <- as.integer(true)
  pred <- as.integer(prob > OPERATING_THRESHOLD)
  cm <- confusionMatrix(
    factor(pred, levels = c(0, 1)),
    factor(true, levels = c(0, 1)),
    positive = "1"
  )
  roc_obj <- roc(true, prob, levels = c(0, 1), direction = "<", quiet = TRUE)
  ci_obj <- ci.auc(roc_obj, method = "delong")
  sensitivity <- unname(cm$byClass["Sensitivity"])
  precision <- unname(cm$byClass["Pos Pred Value"])
  f1 <- if ((sensitivity + precision) == 0) NA_real_ else {
    2 * sensitivity * precision / (sensitivity + precision)
  }
  data.table(
    AUC = as.numeric(auc(roc_obj)),
    CI_Lower = as.numeric(ci_obj[1]),
    CI_Upper = as.numeric(ci_obj[3]),
    Accuracy = unname(cm$overall["Accuracy"]),
    Sensitivity = sensitivity,
    Specificity = unname(cm$byClass["Specificity"]),
    PPV = precision,
    NPV = unname(cm$byClass["Neg Pred Value"]),
    F1 = f1,
    Threshold = OPERATING_THRESHOLD
  )
}

#######################################################################
# 6. 与原模型相同的XGBoost独立调参函数
#######################################################################

train_top15_xgboost <- function(outcome, top15) {
  if (!all(top15 %in% candidate_features)) stop(outcome, " Top-15特征无效。")
  x_train <- data.matrix(train_data[, ..top15])
  x_test  <- data.matrix(test_data[, ..top15])
  y_train <- as.integer(train_data[[outcome]])
  y_test  <- as.integer(test_data[[outcome]])
  if (anyNA(x_train) || anyNA(x_test)) stop(outcome, " XGBoost矩阵存在NA。")
  
  pos_num <- sum(y_train == 1L)
  neg_num <- sum(y_train == 0L)
  scale_pos_weight <- neg_num / pos_num
  
  task_df <- as.data.frame(x_train, check.names = FALSE)
  task_df[[outcome]] <- factor(y_train, levels = c(0, 1), labels = c("0", "1"))
  task <- TaskClassif$new(
    id = paste0(clean_outcome(outcome), "_RF_Top15"),
    backend = task_df,
    target = outcome,
    positive = "1"
  )
  
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
  
  search_space <- ps(
    eta = p_dbl(0.02, 0.06),
    max_depth = p_int(3, 5),
    min_child_weight = p_dbl(6, 15),
    subsample = p_dbl(0.65, 0.80),
    colsample_bytree = p_dbl(0.65, 0.80),
    gamma = p_dbl(1, 5),
    lambda = p_dbl(3, 10),
    alpha = p_dbl(1, 6)
  )
  
  at <- auto_tuner(
    tuner = tnr("random_search"),
    learner = learner,
    resampling = rsmp("cv", folds = 5),
    measure = msr("classif.auc"),
    search_space = search_space,
    terminator = trm("evals", n_evals = 50)
  )
  
  at$train(task)
  best <- at$archive$best()$x_domain[[1]]
  params <- list(
    booster = "gbtree",
    objective = "binary:logistic",
    eval_metric = "auc",
    eta = best$eta,
    max_depth = best$max_depth,
    min_child_weight = best$min_child_weight,
    subsample = best$subsample,
    colsample_bytree = best$colsample_bytree,
    gamma = best$gamma,
    lambda = best$lambda,
    alpha = best$alpha,
    scale_pos_weight = scale_pos_weight,
    seed = GLOBAL_SEED
  )
  
  dtrain <- xgb.DMatrix(x_train, label = y_train, feature_names = top15)
  dtest  <- xgb.DMatrix(x_test, label = y_test, feature_names = top15)
  model <- xgb.train(params, dtrain, nrounds = 200, verbose = 0)
  prob <- predict(model, dtest)
  
  list(
    model = model,
    prediction = data.table(
      Row_ID = seq_along(y_test),
      True_Label = y_test,
      Predicted_Prob = prob,
      Predicted_Class = as.integer(prob > OPERATING_THRESHOLD)
    ),
    performance = calculate_metrics(y_test, prob),
    importance = as.data.table(xgb.importance(top15, model = model)),
    best_params = as.data.table(best),
    tuning_history = as.data.table(at$archive$data),
    scale_pos_weight = scale_pos_weight
  )
}

#######################################################################
# 7. 安全保存和配对DeLong
#######################################################################

make_fwrite_safe <- function(x) {
  x <- as.data.table(copy(x))
  list_cols <- names(x)[vapply(x, is.list, logical(1))]
  for (nm in list_cols) {
    x[, (nm) := vapply(
      get(nm),
      function(v) paste(capture.output(dput(v)), collapse = ""),
      character(1)
    )]
  }
  x
}

read_reference_prediction <- function(path, expected_label, name) {
  if (!file.exists(path)) stop(name, "预测文件不存在：", path)
  z <- fread(path)
  if (!all(c("True_Label", "Predicted_Prob") %in% names(z))) {
    stop(name, "预测文件缺少True_Label或Predicted_Prob。")
  }
  label <- as.integer(z$True_Label)
  prob <- as.numeric(z$Predicted_Prob)
  if (!identical(label, as.integer(expected_label))) {
    stop(name, "的测试标签、样本数或行序与RF Top-15模型不一致。")
  }
  if (anyNA(prob) || any(!is.finite(prob))) stop(name, "预测概率无效。")
  list(label = label, prob = prob)
}

paired_delong <- function(true, rf_prob, ref_prob, comparator) {
  roc_rf <- roc(true, rf_prob, levels = c(0, 1), direction = "<", quiet = TRUE)
  roc_ref <- roc(true, ref_prob, levels = c(0, 1), direction = "<", quiet = TRUE)
  tst <- roc.test(roc_rf, roc_ref, paired = TRUE, method = "delong")
  data.table(
    Comparator = comparator,
    RF_Top15_AUC = as.numeric(auc(roc_rf)),
    Comparator_AUC = as.numeric(auc(roc_ref)),
    Delta_AUC_RF_Minus_Comparator = as.numeric(auc(roc_rf) - auc(roc_ref)),
    DeLong_Z = as.numeric(tst$statistic),
    P_Value = as.numeric(tst$p.value)
  )
}

#######################################################################
# 8. 三个结局
#######################################################################

all_performance <- list()
all_delong <- list()
all_features <- list()
all_overlap <- list()

for (i in seq_along(outcomes)) {
  outcome <- outcomes[i]
  oid <- clean_outcome(outcome)
  cat("\n============================================================\n")
  cat("Outcome:", outcome, "\n")
  cat("Step 1: Random Forest permutation importance\n")
  
  rf <- select_rf_top15(outcome, seed_offset = i - 1L)
  saveRDS(rf$model, file.path(output_dir, "RF_Models",
                              paste0(oid, "_RF_Selection_Model.rds")))
  fwrite(rf$importance, file.path(output_dir, "RF_Importance",
                                  paste0(oid, "_RF_Permutation_Importance.csv")))
  
  rf_top15_table <- data.table(
    Outcome = outcome,
    Rank = seq_len(15L),
    Feature = rf$top15,
    Permutation_Importance = rf$importance$Permutation_Importance[1:15]
  )
  fwrite(rf_top15_table, file.path(output_dir, "Feature_Lists",
                                   paste0(oid, "_RF_Top15.csv")))
  
  cat("Step 2: RF Top-15 XGBoost independent tuning\n")
  xgb <- train_top15_xgboost(outcome, rf$top15)
  saveRDS(xgb$model, file.path(output_dir, "XGBoost_Models",
                               paste0(oid, "_RF_Top15_XGBoost.rds")))
  fwrite(xgb$prediction, file.path(output_dir, "Predictions",
                                   paste0(oid, "_RF_Top15_Prediction.csv")))
  fwrite(xgb$importance, file.path(output_dir, "RF_Importance",
                                   paste0(oid, "_RF_Top15_XGBoost_Gain.csv")))
  fwrite(xgb$best_params, file.path(output_dir, "Best_Parameters",
                                    paste0(oid, "_RF_Top15_Best_Parameters.csv")))
  fwrite(make_fwrite_safe(xgb$tuning_history),
         file.path(output_dir, "Tuning_History",
                   paste0(oid, "_RF_Top15_Tuning_History.csv")))
  
  perf <- copy(xgb$performance)
  perf[, `:=`(
    Outcome = outcome,
    Selection_Method = "Random Forest permutation importance Top-15",
    Number_of_Features = 15L,
    RF_Num_Trees = RF_NUM_TREES,
    RF_mtry = rf$mtry,
    RF_OOB_Brier = rf$oob_brier
  )]
  setcolorder(perf, c("Outcome", "Selection_Method", "Number_of_Features"))
  fwrite(perf, file.path(output_dir, "Performance",
                         paste0(oid, "_RF_Top15_Performance.csv")))
  
  ref_paths <- c(
    "SHAP Top-15" = file.path(reference_prediction_dir,
                              paste0(oid, "_SHAP_Top15_Prediction.csv")),
    "Gain Top-15" = file.path(reference_prediction_dir,
                              paste0(oid, "_Gain_Top15_Prediction.csv")),
    "LASSO Top-15" = file.path(reference_prediction_dir,
                               paste0(oid, "_LASSO_Top15_Prediction.csv")),
    "Full-feature XGBoost" = file.path(full_prediction_dir,
                                       paste0(oid, "_prediction.csv"))
  )
  y_test <- xgb$prediction$True_Label
  rf_prob <- xgb$prediction$Predicted_Prob
  refs <- lapply(names(ref_paths), function(nm) {
    read_reference_prediction(ref_paths[[nm]], y_test, nm)
  })
  names(refs) <- names(ref_paths)
  
  dl <- rbindlist(lapply(names(refs), function(nm) {
    paired_delong(y_test, rf_prob, refs[[nm]]$prob, nm)
  }))
  dl[, `:=`(Outcome = outcome,
            New_Model = "RF permutation Top-15 -> XGBoost")]
  setcolorder(dl, c("Outcome", "New_Model", "Comparator"))
  fwrite(dl, file.path(output_dir, "DeLong",
                       paste0(oid, "_RF_Top15_Paired_DeLong.csv")))
  
  # 与SHAP/Gain/LASSO的特征重叠数
  existing_feature_file <- file.path(
    combined_dir, "Feature_Lists",
    paste0(oid, "_Three_Methods_Top15_Features.csv")
  )
  if (!file.exists(existing_feature_file)) stop("现有三方法特征表不存在。")
  ef <- fread(existing_feature_file)
  method_col <- intersect(c("Method", "Selection_Method"), names(ef))[1]
  feature_col <- intersect(c("Feature", "Selected_Feature"), names(ef))[1]
  if (is.na(method_col) || is.na(feature_col)) {
    stop("既有特征表缺少方法或特征列：", existing_feature_file)
  }
  overlap <- rbindlist(lapply(unique(ef[[method_col]]), function(method) {
    old_features <- as.character(ef[get(method_col) == method][[feature_col]])
    data.table(
      Outcome = outcome,
      Comparator_Selection_Method = method,
      Shared_Features_n = length(intersect(rf$top15, old_features)),
      Shared_Features_Percent = 100 * length(intersect(rf$top15, old_features)) / 15
    )
  }))
  
  audit <- data.table(
    Outcome = outcome,
    Candidate_Features = length(candidate_features),
    RF_Num_Trees = RF_NUM_TREES,
    RF_mtry = rf$mtry,
    RF_Min_Node_Size = 5L,
    RF_Importance = "Scaled permutation importance",
    XGBoost_CV_Folds = 5L,
    XGBoost_Random_Search_Evaluations = 50L,
    XGBoost_Nrounds = 200L,
    Operating_Threshold = OPERATING_THRESHOLD,
    Test_Records = length(y_test),
    Reference_Labels_Identical = TRUE
  )
  fwrite(audit, file.path(output_dir, "Checks", paste0(oid, "_Run_Audit.csv")))
  
  all_performance[[outcome]] <- perf
  all_delong[[outcome]] <- dl
  all_features[[outcome]] <- rf_top15_table
  all_overlap[[outcome]] <- overlap
}

#######################################################################
# 9. 总表
#######################################################################

sort_outcomes <- function(dt) {
  dt[, .Outcome_Order := match(Outcome, publication_order)]
  setorder(dt, .Outcome_Order)
  dt[, .Outcome_Order := NULL]
  dt
}

performance_summary <- sort_outcomes(rbindlist(all_performance, fill = TRUE))
delong_summary <- sort_outcomes(rbindlist(all_delong, fill = TRUE))
feature_summary <- sort_outcomes(rbindlist(all_features, fill = TRUE))
overlap_summary <- sort_outcomes(rbindlist(all_overlap, fill = TRUE))

fwrite(performance_summary,
       file.path(output_dir, "All_RF_Top15_XGBoost_Performance.csv"))
fwrite(delong_summary,
       file.path(output_dir, "All_RF_Top15_Paired_DeLong.csv"))
fwrite(feature_summary,
       file.path(output_dir, "All_RF_Top15_Features.csv"))
fwrite(overlap_summary,
       file.path(output_dir, "All_RF_Feature_Overlap.csv"))

run_info <- data.table(
  Item = c(
    "Analysis", "Global seed", "RF trees", "RF importance",
    "XGBoost CV folds", "XGBoost random-search evaluations",
    "XGBoost rounds", "Operating threshold", "Training records",
    "Test records", "Output directory"
  ),
  Value = c(
    "Random Forest permutation Top-15 followed by XGBoost",
    GLOBAL_SEED, RF_NUM_TREES, "Scaled permutation importance",
    5, 50, 200, OPERATING_THRESHOLD, nrow(train_data), nrow(test_data),
    output_dir
  )
)
fwrite(run_info, file.path(output_dir, "Checks", "Run_Information.csv"))
capture.output(sessionInfo(), file = file.path(output_dir, "Checks", "R_Session_Info.txt"))

cat("\n============================================================\n")
cat("三个结局的 RF Permutation Top-15 分析完成。\n")
cat("结果目录：", output_dir, "\n")

cat("============================================================\n")






#########################################################################
#                  C
##########################################################################
###########################################################################
# C. 模型内部稳健性
#
# 以患者为单位的 Bootstrap（B = 2000），在内部测试集上评估
# 三个模型的 Calibration 与 DCA 性能。
###########################################################################


######################################################################
# COMPLETE VALIDATION: ORIGINAL + REDUCED + INTERACTION MODELS
# Patient-level bootstrap evaluation in the held-out internal test set
#
# 重要说明：
# 1. 本脚本不会重新训练模型，仅使用既有测试集预测概率。
# 2. Bootstrap以患者为抽样单位；每次保留抽中患者的全部住院记录。
# 3. 同一结局的三个模型使用相同的患者级Bootstrap样本。
# 4. 该分析估计测试集AUC的抽样不确定性，不是模型乐观偏倚校正。
######################################################################

rm(list = ls())
gc()

suppressPackageStartupMessages({
  library(data.table)
  library(dplyr)
  library(ggplot2)
  library(pROC)
  library(patchwork)
  library(grid)
  library(gridExtra)
})

GLOBAL_SEED <- 123L
N_BOOT <- 2000L
set.seed(GLOBAL_SEED)

######################################################################
# 1. FILE PATHS
######################################################################

pred_dir <- paste0(
  "J:/rtools45/",
  "手工4乘积交互Model_Comparison_Independent_Tuning/Prediction_Files"
)

test_with_id_path <- "J:/rtools45/test_preprocessed_with_id.rds"

original_files <- list(
  Outcome_Liver_Cirrhosis_ =
    "J:/rtools45/XGBoost_Results/Outcome_Liver_Cirrhosis_prediction.csv",
  Outcome_Hepatocellular_Carcinoma_ =
    "J:/rtools45/XGBoost_Results/Outcome_Hepatocellular_Carcinoma_prediction.csv",
  Outcome_Disease_Non_Progression_ =
    "J:/rtools45/XGBoost_Results/Outcome_Disease_Non_Progression_prediction.csv"
)

top15_files <- list(
  Outcome_Liver_Cirrhosis_ =
    file.path(pred_dir, "Outcome_Liver_Cirrhosis_Top15_Prediction.csv"),
  Outcome_Hepatocellular_Carcinoma_ =
    file.path(pred_dir, "Outcome_Hepatocellular_Carcinoma_Top15_Prediction.csv"),
  Outcome_Disease_Non_Progression_ =
    file.path(pred_dir, "Outcome_Disease_Non_Progression_Top15_Prediction.csv")
)

interaction_files <- list(
  Outcome_Liver_Cirrhosis_ =
    file.path(pred_dir, "Outcome_Liver_Cirrhosis_Interaction_Prediction.csv"),
  Outcome_Hepatocellular_Carcinoma_ =
    file.path(pred_dir, "Outcome_Hepatocellular_Carcinoma_Interaction_Prediction.csv"),
  Outcome_Disease_Non_Progression_ =
    file.path(pred_dir, "Outcome_Disease_Non_Progression_Interaction_Prediction.csv")
)

all_models <- list(
  Original = list(
    files = original_files,
    label = "Original",
    color = "#E41A1C"
  ),
  Reduced = list(
    files = top15_files,
    label = "Reduced",
    color = "#377EB8"
  ),
  Interaction = list(
    files = interaction_files,
    label = "Interaction",
    color = "#4DAF4A"
  )
)

outcome_order <- c(
  "Outcome_Disease_Non_Progression_",
  "Outcome_Liver_Cirrhosis_",
  "Outcome_Hepatocellular_Carcinoma_"
)

outdir <- "J:/rtools45/Validation_Analysis_Patient_Level_Bootstrap"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

######################################################################
# 2. INPUT CHECKS AND PATIENT IDs
######################################################################

all_input_files <- c(
  test_with_id_path,
  unlist(original_files, use.names = FALSE),
  unlist(top15_files, use.names = FALSE),
  unlist(interaction_files, use.names = FALSE)
)

missing_files <- all_input_files[!file.exists(all_input_files)]

if (length(missing_files) > 0L) {
  stop(
    "以下输入文件不存在：\n",
    paste(missing_files, collapse = "\n")
  )
}

test_with_id <- as.data.table(readRDS(test_with_id_path))

if (!("id" %in% names(test_with_id))) {
  stop("test_preprocessed_with_id.rds中没有id列。")
}

patient_id <- as.character(test_with_id$id)

if (length(patient_id) != 10874L) {
  stop("测试集记录数不是10874；实际为：", length(patient_id))
}

if (uniqueN(patient_id) != 6600L) {
  stop("测试集患者数不是6600；实际为：", uniqueN(patient_id))
}

if (anyNA(patient_id) || any(trimws(patient_id) == "")) {
  stop("测试集患者ID中存在缺失值或空值。")
}

cat("============================================================\n")
cat("PATIENT-LEVEL TEST-SET VALIDATION\n")
cat("Test records:", length(patient_id), "\n")
cat("Test patients:", uniqueN(patient_id), "\n")
cat("Bootstrap replicates:", N_BOOT, "\n")
cat("Output:", outdir, "\n")
cat("============================================================\n")

######################################################################
# 3. UTILITIES
######################################################################

clean_outcome <- function(x) {
  x <- gsub("Outcome_", "", x)
  x <- gsub("_", " ", x)
  x <- gsub("Disease Non Progression", "Disease Non-Progression", x)
  x
}

theme_tj <- function(base_size = 12) {
  theme_bw(base_size = base_size) +
    theme(
      plot.title = element_text(
        face = "bold", hjust = 0.5, size = base_size + 1
      ),
      panel.grid = element_blank(),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.text = element_text(size = base_size - 2)
    )
}

read_and_check_prediction <- function(path, model_label, outcome_name) {
  pred <- fread(path)
  
  required_cols <- c("True_Label", "Predicted_Prob")
  missing_cols <- setdiff(required_cols, names(pred))
  
  if (length(missing_cols) > 0L) {
    stop(
      model_label, " - ", outcome_name,
      "预测文件缺少列：", paste(missing_cols, collapse = ", ")
    )
  }
  
  if (nrow(pred) != length(patient_id)) {
    stop(
      model_label, " - ", outcome_name,
      "预测文件行数为", nrow(pred),
      "，患者ID数量为", length(patient_id), "。"
    )
  }
  
  pred[, Patient_ID := patient_id]
  pred[, True_Label := as.numeric(True_Label)]
  pred[, Predicted_Prob := as.numeric(Predicted_Prob)]
  
  if (!all(pred$True_Label %in% c(0, 1))) {
    stop(model_label, " - ", outcome_name, "的结局标签不是0/1。")
  }
  
  if (anyNA(pred$Predicted_Prob) ||
      any(!is.finite(pred$Predicted_Prob)) ||
      any(pred$Predicted_Prob < 0 | pred$Predicted_Prob > 1)) {
    stop(model_label, " - ", outcome_name, "的预测概率不合法。")
  }
  
  pred
}

patient_bootstrap_auc <- function(y, p, ids, outcome_name) {
  patient_rows <- split(seq_along(ids), ids)
  unique_patient_ids <- names(patient_rows)
  n_patients <- length(unique_patient_ids)
  
  outcome_seed <- GLOBAL_SEED + match(outcome_name, outcome_order)
  set.seed(outcome_seed)
  
  boot_auc <- rep(NA_real_, N_BOOT)
  
  for (i in seq_len(N_BOOT)) {
    sampled_ids <- sample(
      unique_patient_ids,
      size = n_patients,
      replace = TRUE
    )
    
    # 重复抽中的患者，其全部住院记录也重复进入Bootstrap样本。
    idx <- unlist(patient_rows[sampled_ids], use.names = FALSE)
    y_boot <- y[idx]
    p_boot <- p[idx]
    
    if (length(unique(y_boot)) < 2L) {
      next
    }
    
    boot_auc[i] <- as.numeric(
      pROC::auc(
        pROC::roc(
          response = y_boot,
          predictor = p_boot,
          levels = c(0, 1),
          direction = "<",
          quiet = TRUE
        )
      )
    )
  }
  
  valid_auc <- boot_auc[is.finite(boot_auc)]
  
  if (length(valid_auc) < 0.95 * N_BOOT) {
    warning(
      clean_outcome(outcome_name),
      "有效患者级Bootstrap次数不足95%。"
    )
  }
  
  list(
    auc = valid_auc,
    mean = mean(valid_auc),
    ci = as.numeric(quantile(valid_auc, c(0.025, 0.975))),
    sd = sd(valid_auc),
    valid_replicates = length(valid_auc),
    number_of_patients = n_patients
  )
}

######################################################################
# 4. MAIN ANALYSIS
######################################################################

summary_table <- data.frame()
bootstrap_table <- data.frame()

cal_store <- list()
dca_store <- list()
boot_store <- list()
test_auc_store <- list()

reference_labels <- list()

for (model_name in names(all_models)) {
  cal_store[[model_name]] <- list()
  dca_store[[model_name]] <- list()
  boot_store[[model_name]] <- list()
  test_auc_store[[model_name]] <- list()
  
  files <- all_models[[model_name]]$files
  model_label <- all_models[[model_name]]$label
  
  for (outcome_name in outcome_order) {
    pred <- read_and_check_prediction(
      files[[outcome_name]], model_label, outcome_name
    )
    
    y <- pred$True_Label
    p <- pred$Predicted_Prob
    
    if (is.null(reference_labels[[outcome_name]])) {
      reference_labels[[outcome_name]] <- y
    } else if (!identical(reference_labels[[outcome_name]], y)) {
      stop(
        outcome_name,
        "的Original、Reduced和Interaction预测标签或行顺序不一致。"
      )
    }
    
    roc_obj <- pROC::roc(
      response = y,
      predictor = p,
      levels = c(0, 1),
      direction = "<",
      quiet = TRUE
    )
    
    auc_value <- as.numeric(pROC::auc(roc_obj))
    brier <- mean((p - y)^2)
    
    eps <- 1e-10
    p_clipped <- pmin(pmax(p, eps), 1 - eps)
    logit_p <- qlogis(p_clipped)
    
    calibration_model <- glm(y ~ logit_p, family = binomial())
    calibration_intercept <- unname(coef(calibration_model)[1])
    calibration_slope <- unname(coef(calibration_model)[2])
    
    citl_model <- glm(
      y ~ 1,
      offset = logit_p,
      family = binomial()
    )
    calibration_in_the_large <- unname(coef(citl_model)[1])
    
    boot <- patient_bootstrap_auc(
      y = y,
      p = p,
      ids = pred$Patient_ID,
      outcome_name = outcome_name
    )
    
    summary_table <- rbind(
      summary_table,
      data.frame(
        Model = model_label,
        Outcome = clean_outcome(outcome_name),
        Test_AUC = round(auc_value, 4),
        Patient_Bootstrap_CI_Lower = round(boot$ci[1], 4),
        Patient_Bootstrap_CI_Upper = round(boot$ci[2], 4),
        Brier = round(brier, 4),
        Calibration_Intercept = round(calibration_intercept, 4),
        Calibration_Slope = round(calibration_slope, 4),
        Calibration_in_the_Large = round(calibration_in_the_large, 4),
        stringsAsFactors = FALSE
      )
    )
    
    bootstrap_table <- rbind(
      bootstrap_table,
      data.frame(
        Model = model_label,
        Outcome = clean_outcome(outcome_name),
        Test_AUC = round(auc_value, 4),
        Patient_Bootstrap_Mean_AUC = round(boot$mean, 4),
        CI_Lower = round(boot$ci[1], 4),
        CI_Upper = round(boot$ci[2], 4),
        Bootstrap_SD = round(boot$sd, 4),
        Mean_Minus_Test_AUC = round(boot$mean - auc_value, 4),
        Bootstrap_Replicates = N_BOOT,
        Valid_Bootstrap_Replicates = boot$valid_replicates,
        Number_of_Patients = boot$number_of_patients,
        stringsAsFactors = FALSE
      )
    )
    
    cal_df <- data.frame(Observed = y, Predicted = p)
    cal_df$Decile <- dplyr::ntile(cal_df$Predicted, 10)
    cal_decile <- cal_df %>%
      group_by(Decile) %>%
      summarise(
        Predicted = mean(Predicted),
        Observed = mean(Observed),
        .groups = "drop"
      ) %>%
      mutate(Model = model_label)
    
    thresholds <- seq(0.01, 0.50, 0.01)
    dca_res <- data.frame()
    
    for (threshold in thresholds) {
      predicted_positive <- p > threshold
      tp <- sum(predicted_positive & y == 1)
      fp <- sum(predicted_positive & y == 0)
      n <- length(y)
      net_benefit <- tp / n - fp / n * threshold / (1 - threshold)
      
      dca_res <- rbind(
        dca_res,
        data.frame(
          Threshold = threshold,
          NetBenefit = net_benefit,
          Model = model_label
        )
      )
    }
    
    cal_store[[model_name]][[outcome_name]] <- cal_decile
    dca_store[[model_name]][[outcome_name]] <- dca_res
    boot_store[[model_name]][[outcome_name]] <- data.frame(
      AUC = boot$auc,
      Model = model_label
    )
    test_auc_store[[model_name]][[outcome_name]] <- auc_value
    
    cat(sprintf(
      "[%s] %s: AUC=%.4f, patient-bootstrap 95%% CI %.4f-%.4f, Brier=%.4f\n",
      model_label,
      clean_outcome(outcome_name),
      auc_value,
      boot$ci[1],
      boot$ci[2],
      brier
    ))
  }
}

######################################################################
# 5. TREAT-ALL AND TREAT-NONE DCA REFERENCES
######################################################################

dca_ref_store <- list()

for (outcome_name in outcome_order) {
  pred <- read_and_check_prediction(
    all_models$Original$files[[outcome_name]],
    "Original",
    outcome_name
  )
  
  y <- pred$True_Label
  prevalence <- mean(y)
  thresholds <- seq(0.01, 0.50, 0.01)
  
  dca_ref_store[[outcome_name]] <- rbind(
    data.frame(
      Threshold = thresholds,
      NetBenefit = prevalence -
        (1 - prevalence) * thresholds / (1 - thresholds),
      Model = "Treat All"
    ),
    data.frame(
      Threshold = thresholds,
      NetBenefit = 0,
      Model = "Treat None"
    )
  )
}

######################################################################
# 6. TOTAL FIGURE: 3 VALIDATION VIEWS x 3 OUTCOMES
######################################################################

tag_letters <- letters[1:9]
cal_panels <- list()
dca_panels <- list()
boot_panels <- list()

model_colors <- c(
  Original = "#E41A1C",
  Reduced = "#377EB8",
  Interaction = "#4DAF4A"
)

for (i in seq_along(outcome_order)) {
  outcome_name <- outcome_order[i]
  outcome_label <- clean_outcome(outcome_name)
  
  cal_all <- bind_rows(
    cal_store$Original[[outcome_name]],
    cal_store$Reduced[[outcome_name]],
    cal_store$Interaction[[outcome_name]]
  )
  
  cal_panels[[i]] <-
    ggplot(cal_all, aes(Predicted, Observed, color = Model)) +
    geom_point(size = 4) +
    geom_line(linewidth = 1.4) +
    geom_abline(
      slope = 1, intercept = 0, linetype = 2, color = "grey50"
    ) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_color_manual(values = model_colors) +
    labs(
      title = outcome_label,
      x = "Predicted",
      y = "Observed",
      tag = tag_letters[i]
    ) +
    theme_tj(base_size = 22)
  
  dca_all <- bind_rows(
    dca_store$Original[[outcome_name]],
    dca_store$Reduced[[outcome_name]],
    dca_store$Interaction[[outcome_name]],
    dca_ref_store[[outcome_name]]
  )
  
  dca_panels[[i]] <-
    ggplot(
      dca_all,
      aes(Threshold, NetBenefit, color = Model, linetype = Model)
    ) +
    geom_line(linewidth = 1.4) +
    coord_cartesian(xlim = c(0, 0.5)) +
    scale_color_manual(values = c(
      model_colors,
      "Treat All" = "black",
      "Treat None" = "grey40"
    )) +
    scale_linetype_manual(values = c(
      "Original" = "solid",
      "Reduced" = "solid",
      "Interaction" = "solid",
      "Treat All" = "dashed",
      "Treat None" = "dotted"
    )) +
    labs(
      title = outcome_label,
      x = "Threshold",
      y = "Net Benefit",
      tag = tag_letters[i + 3]
    ) +
    theme_tj(base_size = 22) +
    guides(color = guide_legend(nrow = 2))
  
  boot_all <- bind_rows(
    boot_store$Original[[outcome_name]],
    boot_store$Reduced[[outcome_name]],
    boot_store$Interaction[[outcome_name]]
  )
  
  test_auc_lines <- data.frame(
    Model = c("Original", "Reduced", "Interaction"),
    AUC = c(
      test_auc_store$Original[[outcome_name]],
      test_auc_store$Reduced[[outcome_name]],
      test_auc_store$Interaction[[outcome_name]]
    )
  )
  
  boot_panels[[i]] <-
    ggplot(boot_all, aes(AUC, fill = Model)) +
    geom_density(alpha = 0.35) +
    geom_vline(
      data = test_auc_lines,
      aes(xintercept = AUC, color = Model),
      linewidth = 1.4
    ) +
    scale_fill_manual(values = model_colors) +
    scale_color_manual(values = model_colors) +
    labs(
      title = outcome_label,
      x = "Patient-level bootstrap AUC",
      y = "Density",
      tag = tag_letters[i + 6]
    ) +
    theme_tj(base_size = 22) +
    theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    )
}

row_cal <- grid::textGrob(
  "Calibration",
  rot = 90,
  gp = grid::gpar(fontface = "bold", fontsize = 22)
)
row_dca <- grid::textGrob(
  "Decision Curve",
  rot = 90,
  gp = grid::gpar(fontface = "bold", fontsize = 22)
)
row_boot <- grid::textGrob(
  "Patient-level Bootstrap AUC",
  rot = 90,
  gp = grid::gpar(fontface = "bold", fontsize = 22)
)

total_figure <- (
  wrap_elements(row_cal) +
    cal_panels[[1]] + cal_panels[[2]] + cal_panels[[3]] +
    wrap_elements(row_dca) +
    dca_panels[[1]] + dca_panels[[2]] + dca_panels[[3]] +
    wrap_elements(row_boot) +
    boot_panels[[1]] + boot_panels[[2]] + boot_panels[[3]] +
    plot_layout(
      nrow = 3,
      byrow = TRUE,
      widths = c(0.5, 1, 1, 1)
    )
) &
  theme(
    plot.tag.position = c(0.06, 0.96),
    plot.tag = element_text(face = "bold", size = 24)
  )

ggsave(
  file.path(outdir, "Figure_Validation_Total_Patient_Level.tiff"),
  total_figure,
  width = 22,
  height = 20,
  dpi = 600,
  compression = "lzw"
)
ggsave(
  file.path(outdir, "Figure_Validation_Total_Patient_Level.pdf"),
  total_figure,
  width = 22,
  height = 20
)

######################################################################
# 7. INDIVIDUAL 3 x 3 FIGURE FOR EACH MODEL
######################################################################

for (model_name in names(all_models)) {
  model_label <- all_models[[model_name]]$label
  model_color <- all_models[[model_name]]$color
  panels <- list()
  panel_index <- 1L
  
  for (outcome_name in outcome_order) {
    outcome_label <- clean_outcome(outcome_name)
    
    cal_df <- cal_store[[model_name]][[outcome_name]]
    p_cal <-
      ggplot(cal_df, aes(Predicted, Observed)) +
      geom_point(size = 3, color = model_color) +
      geom_line(linewidth = 1, color = model_color) +
      geom_abline(
        slope = 1, intercept = 0, linetype = 2, color = "grey50"
      ) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      labs(
        title = outcome_label,
        x = "Predicted",
        y = "Observed"
      ) +
      theme_tj(18)
    
    dca_all <- bind_rows(
      dca_store[[model_name]][[outcome_name]],
      dca_ref_store[[outcome_name]]
    )
    
    p_dca <-
      ggplot(
        dca_all,
        aes(Threshold, NetBenefit, color = Model, linetype = Model)
      ) +
      geom_line(linewidth = 1) +
      coord_cartesian(xlim = c(0, 0.5)) +
      scale_color_manual(values = c(
        setNames(model_color, model_label),
        "Treat All" = "grey50",
        "Treat None" = "grey70"
      )) +
      scale_linetype_manual(values = c(
        setNames("solid", model_label),
        "Treat All" = "dashed",
        "Treat None" = "dotted"
      )) +
      labs(
        title = outcome_label,
        x = "Threshold",
        y = "Net Benefit"
      ) +
      theme_tj(18)
    
    boot_df <- boot_store[[model_name]][[outcome_name]]
    test_auc <- test_auc_store[[model_name]][[outcome_name]]
    
    p_boot <-
      ggplot(boot_df, aes(AUC)) +
      geom_density(fill = model_color, alpha = 0.6) +
      geom_vline(
        xintercept = test_auc,
        color = "red",
        linetype = 2,
        linewidth = 1
      ) +
      labs(
        title = outcome_label,
        x = "Patient-level bootstrap AUC",
        y = "Density"
      ) +
      theme_tj(18) +
      theme(
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank()
      )
    
    panels[[panel_index]] <- p_cal
    panels[[panel_index + 1L]] <- p_dca
    panels[[panel_index + 2L]] <- p_boot
    panel_index <- panel_index + 3L
  }
  
  panels[[1]] <- panels[[1]] + labs(tag = "a")
  panels[[2]] <- panels[[2]] + labs(tag = "b")
  panels[[3]] <- panels[[3]] + labs(tag = "c")
  
  big_figure <- (
    panels[[1]] + panels[[2]] + panels[[3]] +
      panels[[4]] + panels[[5]] + panels[[6]] +
      panels[[7]] + panels[[8]] + panels[[9]] +
      plot_layout(nrow = 3, byrow = TRUE)
  ) &
    theme(plot.tag.position = c(0.02, 0.98))
  
  suffix <- gsub(" ", "_", model_label)
  
  ggsave(
    file.path(
      outdir,
      paste0("Figure_Validation_3x3_", suffix, "_Patient_Level.tiff")
    ),
    big_figure,
    width = 18,
    height = 15,
    dpi = 600,
    compression = "lzw"
  )
  ggsave(
    file.path(
      outdir,
      paste0("Figure_Validation_3x3_", suffix, "_Patient_Level.pdf")
    ),
    big_figure,
    width = 18,
    height = 15
  )
}

######################################################################
# 8. EXPORT TABLES AND COMPLETE RESULTS
######################################################################

fwrite(
  as.data.table(summary_table),
  file.path(
    outdir,
    "Supplementary_Table_Calibration_AllModels_Patient_Level.csv"
  )
)

fwrite(
  as.data.table(bootstrap_table),
  file.path(
    outdir,
    "Supplementary_Table_Patient_Level_Bootstrap_AUC_AllModels.csv"
  )
)

saveRDS(
  list(
    summary_table = summary_table,
    bootstrap_table = bootstrap_table,
    calibration_data = cal_store,
    dca_data = dca_store,
    patient_bootstrap_auc = boot_store,
    test_auc = test_auc_store,
    bootstrap_replicates = N_BOOT,
    number_of_test_records = length(patient_id),
    number_of_test_patients = uniqueN(patient_id),
    seed = GLOBAL_SEED
  ),
  file.path(outdir, "Patient_Level_Validation_Complete_Results.rds")
)

cat("\n============================================================\n")
cat("VALIDATION COMPLETED\n")
cat("Models:", paste(names(all_models), collapse = ", "), "\n")
cat("Bootstrap unit: patient\n")
cat("Model refitting during test bootstrap: no\n")
cat("Output directory:", outdir, "\n")
cat("============================================================\n")


