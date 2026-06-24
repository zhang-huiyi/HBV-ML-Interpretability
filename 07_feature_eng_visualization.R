################################################################
# SCI特征工程完整可视化（最终完整版）
# 功能：
#   1. 特征数量对比图
#   2. 特征重要性热图（SHAP Top15）
#   3. 特征重叠分析
#   4. 交互特征对展示
#   5. 交互特征强度分析（Gain）
#   6. 交互特征分布热图
#   7. 特征缩减比例图
#   8. 合并大图（abcd标记）
################################################################
rm(list = ls())
gc()
cat("\n", rep("=", 60), "\n", sep = "")
cat("SCI Feature Engineering Complete Visualization\n")
cat(rep("=", 60), "\n", sep = "")

# 加载必要的包
library(ggplot2)
library(data.table)
library(dplyr)
library(tidyr)
library(patchwork)

################################################################
# 1. 辅助函数
################################################################

# 结局名称清理
clean_outcome_name <- function(outcome_name) {
  display_name <- gsub("^Outcome_", "", outcome_name)
  display_name <- gsub("_", " ", display_name)
  display_name <- tools::toTitleCase(tolower(display_name))
  return(display_name)
}

# 交互特征显示名称
generate_interaction_display_name <- function(feature1, feature2) {
  return(paste(feature1, "×", feature2))
}

# 交互对定义
get_interaction_pairs_info <- function() {
  pairs <- list(
    "Outcome_Disease Non-Progression" = list(
      c("Absolute Lymphocyte Count", "Platelet Count"),
      c("Number of Hospitalizations", "Platelet Count"),
      c("Plateletcrit", "Alpha-fetoprotein (AFP)"),
      c("C-Reactive Protein (CRP)", "Alpha-fetoprotein (AFP)")
    ),
    "Outcome_Liver Cirrhosis" = list(
      c("Plateletcrit", "Platelet Count"),
      c("Absolute Lymphocyte Count", "Platelet Count"),
      c("Admission Interval", "Alpha-fetoprotein (AFP)"),
      c("Cholinesterase", "Plateletcrit")
    ),
    "Outcome_Hepatocellular Carcinoma" = list(
      c("Direct Bilirubin", "Alpha-fetoprotein (AFP)"),
      c("Number of Hospitalizations", "C-Reactive Protein (CRP)"),
      c("Age", "C-Reactive Protein (CRP)"),
      c("C-Reactive Protein (CRP)", "Alpha-fetoprotein (AFP)")
    )
  )
  return(pairs)
}

################################################################
# 2. 数据加载
################################################################

load_all_data <- function() {
  
  feature_dir <- "J:/rtools45/手工4乘积交互Feature_Engineering"
  comparison_dir <- "J:/rtools45/手工4乘积交互Model_Comparison_Independent_Tuning"
  
  outcome_map <- c(
    "Outcome_Disease_Non_Progression_" = "Outcome_Disease Non-Progression",
    "Outcome_Liver_Cirrhosis_" = "Outcome_Liver Cirrhosis",
    "Outcome_Hepatocellular_Carcinoma_" = "Outcome_Hepatocellular Carcinoma"
  )
  
  top_features_per_outcome <- list()
  reduced_dataset_list <- list()
  interaction_dataset_list <- list()
  interaction_results <- list()
  
  interaction_pairs_info <- get_interaction_pairs_info()
  
  for (outcome_clean in names(outcome_map)) {
    
    outcome_name <- outcome_map[outcome_clean]
    cat("\n加载:", outcome_name, "\n")
    
    # 1. SHAP Top15
    shap_file <- file.path("J:/rtools45/SHAP_Results_TOPJOURNAL", 
                           outcome_clean, 
                           "SHAP_Importance_Top15.csv")
    if (file.exists(shap_file)) {
      shap_imp <- fread(shap_file)
      top_features_per_outcome[[outcome_name]] <- shap_imp$Feature[1:15]
      cat("  ✓ Top15特征:", length(top_features_per_outcome[[outcome_name]]), "\n")
    } else {
      cat("  ❌ SHAP文件不存在\n")
      next
    }
    
    # 2. Reduced数据
    reduced_train_file <- file.path(feature_dir, "Reduced_Model_Data", 
                                    paste0(outcome_clean, "_Reduced_Train.csv"))
    if (file.exists(reduced_train_file)) {
      reduced_train <- fread(reduced_train_file)
      outcome_col <- grep("outcome|Outcome", names(reduced_train), 
                          value = TRUE, ignore.case = TRUE)[1]
      if (is.na(outcome_col) || length(outcome_col) == 0) {
        outcome_col <- outcome_name
      }
      feature_cols <- setdiff(names(reduced_train), outcome_col)
      reduced_dataset_list[[outcome_name]] <- list(
        feature_count = length(feature_cols),
        features_used = feature_cols
      )
      cat("  ✓ Reduced:", length(feature_cols), "特征\n")
    }
    
    # 3. Interaction数据
    interaction_train_file <- file.path(feature_dir, "Interaction_Model_Data", 
                                        paste0(outcome_clean, "_Interaction_Train.csv"))
    if (file.exists(interaction_train_file)) {
      interaction_train <- fread(interaction_train_file)
      outcome_col_int <- grep("outcome|Outcome", names(interaction_train), 
                              value = TRUE, ignore.case = TRUE)[1]
      if (is.na(outcome_col_int) || length(outcome_col_int) == 0) {
        outcome_col_int <- outcome_name
      }
      feature_cols_int <- setdiff(names(interaction_train), outcome_col_int)
      interaction_feats <- grep("^INT_PROD_", feature_cols_int, value = TRUE)
      
      # 交互对显示名称
      interaction_pairs_display <- list()
      if (outcome_name %in% names(interaction_pairs_info)) {
        pairs <- interaction_pairs_info[[outcome_name]]
        for (i in 1:length(pairs)) {
          pair <- pairs[[i]]
          clean1 <- gsub("[^A-Za-z0-9]", "_", pair[1])
          clean2 <- gsub("[^A-Za-z0-9]", "_", pair[2])
          int_feat_name <- paste0("INT_PROD_", clean1, "_", clean2)
          display_name <- generate_interaction_display_name(pair[1], pair[2])
          
          interaction_pairs_display[[i]] <- list(
            feature1 = pair[1],
            feature2 = pair[2],
            interaction_feature = int_feat_name,
            display_name = display_name
          )
        }
      }
      
      interaction_dataset_list[[outcome_name]] <- list(
        feature_count = length(feature_cols_int),
        features_used = feature_cols_int,
        interaction_features = interaction_feats,
        interaction_pairs_display = interaction_pairs_display
      )
      cat("  ✓ Interaction:", length(feature_cols_int), "特征 (", length(interaction_feats), "交互)\n")
    }
    
    # 4. 模型结果（含Gain）
    comparison_file <- file.path(comparison_dir, "SCI_Model_Comparison_Results.rds")
    if (file.exists(comparison_file)) {
      all_comp <- readRDS(comparison_file)
      if (outcome_clean %in% names(all_comp)) {
        comp <- all_comp[[outcome_clean]]
        
        # 读取交互模型Gain
        int_gain_file <- file.path(comparison_dir, paste0(outcome_clean, "Interaction_Gain.csv"))
        if (file.exists(int_gain_file)) {
          int_gain <- fread(int_gain_file)
        } else if (!is.null(comp$interaction_gain)) {
          int_gain <- comp$interaction_gain
        } else {
          int_gain <- data.frame(Feature = character(), Gain = numeric())
        }
        
        interaction_results[[outcome_name]] <- list(
          metrics = comp$interaction_metrics$metrics,
          gain = int_gain
        )
        cat("  ✓ 模型Gain加载\n")
      }
    }
  }
  
  return(list(
    top_features_per_outcome = top_features_per_outcome,
    reduced_dataset_list = reduced_dataset_list,
    interaction_dataset_list = interaction_dataset_list,
    interaction_results = interaction_results
  ))
}

################################################################
# 3. 图1: 特征数量对比图
################################################################

plot_feature_count_comparison <- function(top_features_per_outcome, 
                                          reduced_dataset_list, 
                                          interaction_dataset_list) {
  cat("\n生成图1: 特征数量对比\n")
  
  feature_data <- data.frame()
  outcome_order <- names(top_features_per_outcome)
  
  for (outcome_name in outcome_order) {
    feature_data <- rbind(feature_data,
                          data.frame(Outcome = outcome_name, 
                                     Model = "Original", 
                                     Count = 78),      # 原始78个特征
                          data.frame(Outcome = outcome_name, 
                                     Model = "Reduced", 
                                     Count = 15),      # SHAP筛选后15个特征
                          data.frame(Outcome = outcome_name, 
                                     Model = "Interaction", 
                                     Count = 19))      # 15个原始 + 4个交互 = 19个特征
  }
  
  # 设置因子顺序，保证x轴和填充颜色顺序一致
  feature_data$Outcome <- factor(feature_data$Outcome, levels = outcome_order)
  feature_data$Model <- factor(feature_data$Model, 
                               levels = c("Original", "Reduced", "Interaction"))
  feature_data$Outcome_Display <- factor(
    sapply(as.character(feature_data$Outcome), clean_outcome_name),
    levels = sapply(outcome_order, clean_outcome_name))
  
  p <- ggplot(feature_data, aes(x = Outcome_Display, y = Count, fill = Model)) +
    # 分组柱状图：position_dodge控制组内柱子间距，width控制柱子宽度
    geom_bar(stat = "identity", position = position_dodge(width = 0.7),  # 柱子并列间距0.7
             width = 0.65,        # 每根柱子宽度0.65
             alpha = 0.7,        # 透明度0.88（轻微透明增加层次感）
             color = "white",     # 柱子边框白色
             size = 0.3) +        # 边框线粗细0.3mm
    # 柱顶数字标签：显示具体特征数量
    geom_text(aes(label = Count), 
              position = position_dodge(width = 0.7),  # 与柱子对齐
              vjust = -0.5,        # 垂直偏移-0.5（向上偏移半行）
              size = 6,          # 文字大小4.5mm
              fontface = "bold") + # 加粗
    # 添加"+4 interactions"标注，说明19特征的来源（15+4）
    annotate("text", x = 1, y = 24, label = "+4", 
             color = "#2E8B57",    # 深绿色文字
             size = 7,           # 文字大小3.5mm
             fontface = "italic") + # 斜体
    annotate("text", x = 2, y = 24, label = "+4", 
             color = "#2E8B57", size = 7, fontface = "italic") +
    annotate("text", x = 3, y = 24, label = "+4", 
             color = "#2E8B57", size = 7, fontface = "italic") +
    # 标题和轴标签
    labs(title = "Feature Count Comparison",                                    # 主标题
         subtitle = " ",  # 副标题说明构成
         x = "",                                                                 # x轴标签为空（已有结局名）
         y = "Number of Features") +                                             # y轴标签
    # 手动指定填充颜色：原始=粉色，Top15=蓝色，+交互=绿色
    scale_fill_manual(values = c("Original" = "#FF9999",      # 浅红色（原始78）
                                 "Reduced" = "#66C2FF",    # 浅蓝色（15个特征）
                                 "Interaction" = "#A1D99B"),  # 浅绿色（15+4=19）
                      labels = c("Original",              # 图例标签1
                                 "Reduced",           # 图例标签2
                                 "Interaction")) + # 图例标签3
    # y轴范围：从0到82，给78留足空间
    coord_cartesian(ylim = c(0, 82)) +
    # 使用简约主题
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字基础大小16pt
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 17, face = "bold"),  # x轴刻度：水平居中，16pt，加粗
      axis.text.y = element_text(size = 24),                               # y轴刻度：16pt
      plot.title = element_text(face = "bold", size = 24, hjust = 0.5,
                                margin = margin(b = 2)),                  # 标题：加粗20pt，居中，底部边距15pt（增加标题与图形间距）
      plot.subtitle = element_text(size = 1, hjust = 0.5, color = "gray40",
                                   margin = margin(b = 0)),               # 副标题：底部边距10pt
      plot.title.position = "plot",                                        # 标题相对于整个绘图区域对齐（非面板）
      axis.title.y = element_text(face = "bold", size = 20,
                                  margin = margin(r = 2)),                # y轴标题：加粗18pt，右侧边距10pt（增加与轴刻度间距）
      legend.title = element_blank(),                                      # 图例标题为空
      legend.text = element_text(size = 15),                               # 图例文字15pt
      legend.position = "top",                                             # 图例位置在顶部
      legend.margin = margin(b = 2),                                       # 图例底部边距8pt（增加图例与图形间距）
      panel.grid.major.x = element_blank(),                                # 移除x轴主网格线
      panel.grid.minor = element_blank(),                                  # 移除所有次网格线
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2)                # 整体页面边距：上15，右20，下10，左15（单位pt）
    )
  
  return(p)
}

################################################################
# 4. 图2: 特征缩减比例图
################################################################

plot_feature_reduction_ratio <- function(top_features_per_outcome, 
                                         reduced_dataset_list, 
                                         interaction_dataset_list) {
  cat("\n生成图2: 特征缩减比例\n")
  
  ratio_data <- data.frame()
  outcome_order <- names(top_features_per_outcome)
  
  for (outcome_name in outcome_order) {
    reduced_count <- length(reduced_dataset_list[[outcome_name]]$features_used)        # Top15特征数
    interaction_count <- length(interaction_dataset_list[[outcome_name]]$features_used) # 交互模型特征数
    
    reduced_ratio <- round((78 - reduced_count) / 78 * 100, 1)          # 缩减百分比（正值表示减少）
    interaction_change <- round((interaction_count - 78) / 78 * 100, 1) # 变化百分比（负值表示减少）
    
    ratio_data <- rbind(ratio_data,
                        data.frame(Outcome = outcome_name, 
                                   Type = "Reduced",        # 类型1：SHAP筛选
                                   Ratio = reduced_ratio,      # 缩减比例（正值）
                                   Label = paste0("-", reduced_ratio, "%")),  # 标签格式：-XX%
                        data.frame(Outcome = outcome_name, 
                                   Type = "Interaction",       # 类型2：加交互后
                                   Ratio = interaction_change, # 变化比例（负值）
                                   Label = paste0(interaction_change, "%")))  # 标签格式：-XX%
  }
  
  ratio_data$Outcome <- factor(ratio_data$Outcome, levels = outcome_order)
  ratio_data$Type <- factor(ratio_data$Type, levels = c("Reduced", "Interaction"))
  ratio_data$Outcome_Display <- factor(
    sapply(as.character(ratio_data$Outcome), clean_outcome_name),
    levels = sapply(outcome_order, clean_outcome_name))
  
  p <- ggplot(ratio_data, aes(x = Outcome_Display, y = Ratio, fill = Type)) +
    # 分组柱状图
    geom_bar(stat = "identity", position = position_dodge(width = 0.7), 
             width = 0.6,         # 柱子宽度0.6
             alpha = 0.85,        # 透明度0.85
             color = "white",     # 白色边框
             size = 0.3) +        # 边框线宽0.3mm
    # y=0参考线：区分正负变化
    geom_hline(yintercept = 0, linetype = "dashed",  # 虚线
               color = "gray40",  # 深灰色
               size = 0.6) +      # 线宽0.6mm
    # 数值标签：正值向下偏移，负值向上偏移避免重叠
    geom_text(aes(label = Label), 
              position = position_dodge(width = 0.7), 
              vjust = ifelse(ratio_data$Ratio >= 0, -0.5, 1.5),  # 正值vjust=-0.5在上方，负值vjust=1.5在下方
              size = 4.5,          # 文字大小4.5mm
              fontface = "bold") + # 加粗
    # 标题和轴标签
    labs(title = "Feature Count Change Ratio",                           # 主标题
         subtitle = " ",  # 副标题
         x = "",                                                          # x轴为空
         y = "Change Ratio (%)") +                                        # y轴标签
    # 颜色设置：蓝色=SHAP缩减，绿色=+交互变化
    scale_fill_manual(values = c("Reduced" = "#66C2FF",      # 浅蓝色
                                 "Interaction" = "#A1D99B")) +  # 浅绿色
    
    scale_y_continuous(expand = expansion(mult = c(0.12, 0.12)))+
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字13pt
      axis.text.x = element_text(angle = 0, hjust = 0.5, size = 17, face = "bold"),  # x轴刻度加粗
      axis.text.y = element_text(size = 18),                               # y轴刻度10pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5),    # 标题加粗居中
     
      axis.title.y = element_text(face = "bold", size = 20),               # y轴标题加粗
      legend.title = element_blank(),                                      # 隐藏图例标题
      legend.text = element_text(size = 15),                                # 图例文字9pt
      legend.position = "top",  
      legend.margin = margin(b = 2), # 图例在顶部
      panel.grid.major.x = element_blank() ,                                # 移除x轴主网格
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2)   )
  
  return(p)
}

################################################################
# 5. 图3: SHAP Top15特征重要性热图
################################################################

plot_shap_heatmap <- function(top_features_per_outcome) {
  cat("\n生成图3: SHAP特征重要性热图\n")
  
  heatmap_data <- data.frame()
  outcome_order <- names(top_features_per_outcome)
  
  for (outcome_name in outcome_order) {
    features <- top_features_per_outcome[[outcome_name]]
    for (i in 1:length(features)) {
      heatmap_data <- rbind(heatmap_data,
                            data.frame(Outcome = outcome_name,
                                       Feature = features[i],
                                       Rank = i,                # 排名（1-15）
                                       Importance = 16 - i,     # 重要性分数（排名1=15分，排名15=1分）
                                       stringsAsFactors = FALSE))
    }
  }
  
  heatmap_data$Outcome_Display <- factor(
    sapply(as.character(heatmap_data$Outcome), clean_outcome_name),
    levels = sapply(outcome_order, clean_outcome_name))
  
  p <- ggplot(heatmap_data, 
              aes(x = Outcome_Display, y = reorder(Feature, -Rank), fill = Importance)) +
    # 热图方块
    geom_tile(color = "white",   # 方块间白色分隔线
              size = 0.5) +      # 分隔线宽度0.5mm
    # 颜色渐变：
    scale_fill_gradient(low = "#FFD700",        # 低值=金黄色（排名15）
                        high = "#e31a1c",       # 高值=深红色（排名1）
                        name = "Importance Rank",  # 图例标题（去掉\n换行）
                        breaks = c(1, 8, 15),    # 图例刻度：低/中/高
                        labels = c("Low", "Medium", "High")) +  # 图例标签
    # 标题和轴标签
    labs(title = "SHAP Top 15 Feature Importance Heatmap",
         x = "",                # x轴为空
         y = "Feature") +       # y轴标签
    # 使用guides控制图例水平排列
    guides(fill = guide_colorbar(
      direction = "horizontal",          # 颜色条水平方向
      title.position = "top",            # 标题在颜色条上方
      title.hjust = 0.5,                 # 标题水平居中
      barwidth = unit(8, "cm"),          # 颜色条宽度8厘米（水平时控制长度）
      barheight = unit(0.8, "cm"),       # 颜色条高度0.8厘米（水平时控制厚度）
      label.position = "bottom",         # 刻度标签在颜色条下方
      label.hjust = 0.5                  # 刻度标签水平居中
    )) +
    scale_x_discrete(expand = expansion(add = c(0.1, 1.5))) +
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字16pt
      axis.text.x = element_text(angle = 20, hjust = 0.5, size = 16, face = "bold"),  # x轴刻度倾斜15度加粗
      axis.text.y = element_text(size = 22),                                # y轴刻度15pt
      axis.title.y = element_text(size = 25, face = "bold"),               # y轴标题加粗17pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5,
                                margin = margin(b = 2)),                  # 标题加粗居中，底部边距15pt
      plot.title.position = "plot", 
      legend.position = "top",                                             # 图例在顶部
      legend.title = element_text(size = 15, face = "bold"),               # 图例标题12pt加粗
      legend.text = element_text(size = 14),                               # 图例文字11pt
      legend.margin = margin(b = 2),                                      # 图例底部边距10pt
      panel.grid = element_blank() ,
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2) # 移除所有网格线
    )
  
  return(p)
}

################################################################
# 6. 图4: 特征重叠分析
################################################################

plot_feature_overlap <- function(top_features_per_outcome) {
  cat("\n生成图4: 特征重叠分析\n")
  
  overlap_data <- data.frame()
  all_features <- unique(unlist(top_features_per_outcome))  # 获取所有Top15中出现的唯一特征
  
  for (feature in all_features) {
    count <- sum(sapply(top_features_per_outcome, function(x) feature %in% x))  # 统计特征在几个结局中出现
    overlap_data <- rbind(overlap_data,
                          data.frame(Feature = feature, Outcome_Count = count))
  }
  
  overlap_data <- overlap_data %>% 
    arrange(desc(Outcome_Count)) %>%  # 按出现次数降序排列
    head(20)                           # 只展示前20个特征
  
  # 颜色分组：1/2/3个结局共享
  overlap_data$Color_Group <- factor(overlap_data$Outcome_Count,
                                     levels = c(1, 2, 3),
                                     labels = c("1", "2", "3"))
  
  p <- ggplot(overlap_data, 
              aes(x = reorder(Feature, Outcome_Count), y = Outcome_Count, fill = Color_Group)) +
    # 横向柱状图
    geom_bar(stat = "identity", alpha = 0.85, width = 0.7) +  # 透明度0.85，宽度0.7
    coord_flip() +  # 翻转坐标轴，变成横向柱状图
    # 颜色：金黄=1个结局，橙色=2个，红色=3个
    scale_fill_manual(values = c("1" = "#FFD700",      # 金黄色
                                 "2" = "#FF8C00",     # 橙色
                                 "3" = "#e31a1c")   # 红色
                      ) +                 # 图例标题
    # 标题和轴标签
    labs(title = "Feature Overlap Across Outcomes",
         subtitle = " ",
         x = "",                          # x轴为空（特征名已在y轴）
         y = "Number of Outcomes") +      # y轴标签
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字13pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5),    # 标题加粗居中
      plot.subtitle = element_text(size = 1, hjust = 0.5, color = "gray40"),  # 副标题灰色
      plot.title.position = "plot", 
      axis.text.y = element_text(size = 20),                               # y轴（特征名）10pt
      axis.text.x = element_text(size = 20),                               # x轴（计数）10pt
      axis.title.x = element_text(face = "bold", size = 24),               # x轴标题加粗
      legend.title = element_text(size = 15, face = "bold"),               # 图例标题加粗
      legend.text = element_text(size = 14),                                # 图例文字9pt
      legend.position = "top"  ,
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2) # 图例在顶部
    )
  
  return(p)
}

################################################################
# 7. 图5: 交互特征对展示
################################################################

plot_interaction_pairs <- function(interaction_dataset_list) {
  cat("\n生成图5: 交互特征对展示\n")
  
  interaction_details <- data.frame()
  outcome_order <- names(interaction_dataset_list)
  
  for (outcome_name in outcome_order) {
    pairs_display <- interaction_dataset_list[[outcome_name]]$interaction_pairs_display
    for (pair_info in pairs_display) {
      interaction_details <- rbind(interaction_details,
                                   data.frame(Outcome = outcome_name,
                                              Display_Name = pair_info$display_name,  # 如：Feature1 × Feature2
                                              stringsAsFactors = FALSE))
    }
  }
  
  interaction_details$Outcome_Display <- factor(
    sapply(as.character(interaction_details$Outcome), clean_outcome_name),
    levels = sapply(outcome_order, clean_outcome_name))
  
  p <- ggplot(interaction_details, 
              aes(x = Outcome_Display, y = Display_Name, fill = Outcome_Display)) +
    # 彩色方块（无文字标签）
    geom_tile(color = "white",    # 白色分隔线
              size = 0.8,         # 分隔线宽度0.8mm
              alpha = 0.85) +     # 透明度0.85
    # 颜色：粉红/蓝/绿 对应三个结局
    scale_fill_manual(values = c("#fd9d9c",   # 浅粉红
                                 "#66C2FF",   # 浅蓝色
                                 "#A1D99B"),  # 浅绿色
                      guide = "none") +       # 隐藏图例（颜色在x轴已有）
    scale_x_discrete(expand = expansion(add = c(0.2, 1.5))) +
    # 标题和轴标签
    labs(title = "Product Interaction Feature Pairs",
         x = "",                          # x轴为空
         y = "Interaction Pair") +        # y轴标签
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字13pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5),    # 标题加粗居中
      plot.title.position = "plot", 
      axis.text.x = element_text(angle=10,size = 15, face = "bold"),                # x轴加粗
      axis.text.y = element_text(size = 19, face = "bold"),                # y轴加粗（交互对名称）
      axis.title.y = element_text(size = 24, face = "bold"),               # y轴标题加粗
      panel.grid = element_blank(),                                        # 移除网格
      panel.border = element_rect(color = "gray80", fill = NA, size = 0.5),
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2) # 浅灰色外边框
    )
  
  return(p)
}

################################################################
# 8. 图6: 交互特征强度（Gain）
################################################################

plot_interaction_strength <- function(interaction_results, interaction_dataset_list) {
  cat("\n生成图6: 交互特征强度分析\n")
  
  strength_data <- data.frame()
  outcome_order <- names(interaction_results)
  
  for (outcome_name in outcome_order) {
    if (!is.null(interaction_results[[outcome_name]]$gain)) {
      gain_data <- interaction_results[[outcome_name]]$gain
      int_gain <- gain_data[grepl("^INT_PROD_", Feature), ]  # 筛选INT_PROD_开头的交互特征
      
      if (nrow(int_gain) > 0) {
        pairs_display <- interaction_dataset_list[[outcome_name]]$interaction_pairs_display
        
        for (i in 1:nrow(int_gain)) {
          feat_name <- int_gain$Feature[i]
          display_name <- feat_name
          for (pair_info in pairs_display) {
            if (pair_info$interaction_feature == feat_name) {
              display_name <- pair_info$display_name  # 替换为可读名称
              break
            }
          }
          
          strength_data <- rbind(strength_data,
                                 data.frame(Outcome = outcome_name,
                                            Display_Name = display_name,
                                            Gain = int_gain$Gain[i],  # XGBoost Gain重要性
                                            stringsAsFactors = FALSE))
        }
      }
    }
  }
  
  if (nrow(strength_data) == 0) {
    cat("  ⚠️ 无交互Gain数据\n")
    return(ggplot() + labs(title = "No Interaction Gain Data") + theme_void())
  }
  
  strength_data$Outcome_Display <- factor(
    sapply(as.character(strength_data$Outcome), clean_outcome_name),
    levels = sapply(outcome_order, clean_outcome_name))
  strength_data <- strength_data %>% arrange(desc(Gain))  # 按Gain降序排列
  
  p <- ggplot(strength_data, 
              aes(x = reorder(Display_Name, Gain), y = Gain, fill = Outcome_Display)) +
    # 横向柱状图
    geom_bar(stat = "identity", alpha = 0.85, width = 0.7, 
             color = "white",    # 白色边框
             size = 0.3) +       # 边框0.3mm
    # 数值标签：显示4位小数Gain值
    geom_text(aes(label = sprintf("%.3f", Gain),           # 格式化为4位小数
                  y = Gain + max(Gain)*0.06),               # 在柱子末端右侧偏移
              size = 4,           # 文字大小3.5mm
              fontface = "bold") +  # 加粗
    coord_flip() +  # 横向显示
    # 颜色：按结局区分
    scale_fill_manual(values = c("#fd9d9c",   # 浅粉红
                                 "#66C2FF",   # 浅蓝色
                                 "#A1D99B"),  # 浅绿色
                      name = "Outcome") +     # 图例标题
    # 标题和轴标签
    labs(title = "Interaction Feature Strength (Gain)",
         x = "",          # x轴为空
         y = "Gain") +    # y轴标签
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字13pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5),    # 标题加粗居中
      plot.title.position = "plot", 
      axis.text.y = element_text(size = 19, face = "bold"),                # y轴加粗（交互对名）
      axis.text.x = element_text(size = 20),                               # x轴10pt
      axis.title.x = element_text(face = "bold", size = 24),               # x轴标题加粗
      legend.title = element_text(size = 15, face = "bold"),               # 图例标题加粗
      legend.text = element_text(size = 14),                                # 图例文字9pt
      legend.position = "top",                                          # 图例在底部
      panel.grid.major.y = element_blank() ,
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2) # 移除y轴主网格
    )
  
  return(p)
}

################################################################
# 9. 图7: 交互特征分布热图
################################################################

plot_interaction_heatmap <- function(interaction_dataset_list, interaction_results) {
  cat("\n生成图7: 交互特征分布热图\n")
  
  all_data <- data.frame()
  outcome_order <- names(interaction_dataset_list)
  
  for (outcome_name in outcome_order) {
    pairs_display <- interaction_dataset_list[[outcome_name]]$interaction_pairs_display
    for (pair_info in pairs_display) {
      gain_value <- 0
      if (!is.null(interaction_results[[outcome_name]]$gain)) {
        gain_data <- interaction_results[[outcome_name]]$gain
        match_row <- gain_data[gain_data$Feature == pair_info$interaction_feature, ]
        if (nrow(match_row) > 0) gain_value <- match_row$Gain[1]  # 提取Gain值
      }
      all_data <- rbind(all_data,
                        data.frame(Outcome = outcome_name,
                                   Display_Name = pair_info$display_name,
                                   Gain = gain_value,
                                   stringsAsFactors = FALSE))
    }
  }
  
  all_data$Outcome_Display <- factor(
    sapply(as.character(all_data$Outcome), clean_outcome_name),
    levels = sapply(outcome_order, clean_outcome_name))
  
  # 创建完整矩阵（包含不存在交互的单元格）
  all_pairs <- unique(all_data$Display_Name)
  complete_matrix <- expand.grid(
    Outcome_Display = levels(all_data$Outcome_Display),
    Display_Name = all_pairs,
    stringsAsFactors = FALSE)
  complete_matrix <- complete_matrix %>%
    left_join(all_data[, c("Outcome_Display", "Display_Name", "Gain")],
              by = c("Outcome_Display", "Display_Name"))
  complete_matrix$Gain[is.na(complete_matrix$Gain)] <- NA  # 不存在交互设为NA
  
  p <- ggplot(complete_matrix, 
              aes(x = Outcome_Display, y = Display_Name, fill = Gain)) +
    # 热图方块
    geom_tile(color = "white",    # 白色分隔线
              size = 0.8) +       # 分隔线0.8mm
    # 方块内显示Gain值（4位小数）
    geom_text(aes(label = ifelse(!is.na(Gain), sprintf("%.3f", Gain), "")),  # NA不显示
              size = 5,          # 文字大小3.5mm
              fontface = "bold") + # 加粗
    # 颜色渐变：浅绿=低Gain，深绿=高Gain，灰色=不存在
    scale_fill_gradient(low = "#c6e8c6",        # 浅绿色（低Gain）
                        high = "#2E8B57",       # 深绿色（高Gain）
                        name = "Gain",          # 图例标题
                        na.value = "#f0f0f0") + # NA值用浅灰色填充
    # 标题和轴标签
    labs(title = "Interaction Feature Distribution Heatmap",
         x = "",                       # x轴为空
         y = "Interaction Pair") +     # y轴标签
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字16pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5,
                                margin = margin(b = 2)),                  # 标题加粗居中，底部边距15pt
      plot.title.position = "plot",                                        # 标题对齐整个绘图区域
      axis.text.x = element_text(angle = 15, size = 20, face = "bold"),   # x轴刻度倾斜20度加粗
      axis.text.y = element_text(size = 20, face = "bold"),               # y轴刻度加粗
      axis.title.y = element_text(size = 24, face = "bold"),              # y轴标题加粗
      legend.position = "top",                                             # 图例在顶部
      legend.direction = "horizontal",                                     # 图例水平平铺
      legend.title = element_text(size = 15, face = "bold"),              # 图例标题12pt加粗
      legend.text = element_text(size = 14),                              # 图例文字11pt
      legend.margin = margin(b = 2),                                      # 图例底部边距10pt
      panel.grid = element_blank()   ,
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2) # 移除网格
    )
  
  return(p)
}

################################################################
# 10. 图8: 特征参与交互网络分析
################################################################

plot_feature_participation <- function(interaction_dataset_list) {
  cat("\n生成图8: 特征参与交互分析\n")
  
  participation_data <- data.frame()
  
  for (outcome_name in names(interaction_dataset_list)) {
    pairs_display <- interaction_dataset_list[[outcome_name]]$interaction_pairs_display
    for (pair_info in pairs_display) {
      # 每个交互对中的两个特征各算一次参与
      participation_data <- rbind(participation_data,
                                  data.frame(Feature = pair_info$feature1,
                                             Outcome = outcome_name,
                                             stringsAsFactors = FALSE),
                                  data.frame(Feature = pair_info$feature2,
                                             Outcome = outcome_name,
                                             stringsAsFactors = FALSE))
    }
  }
  
  if (nrow(participation_data) == 0) {
    return(ggplot() + labs(title = "No Data") + theme_void())
  }
  
  # 统计每个特征参与交互的总次数
  feature_count <- participation_data %>%
    group_by(Feature) %>%
    summarise(Count = n(), .groups = 'drop') %>%
    arrange(desc(Count))
  
  p <- ggplot(head(feature_count, 15),   # 只展示参与最多的前15个特征
              aes(x = reorder(Feature, Count), y = Count, fill = Count)) +
    # 横向柱状图
    geom_bar(stat = "identity", alpha = 0.85, width = 0.7) +
    coord_flip() +  # 横向
    # 颜色渐变：浅绿=少，深绿=多
    scale_fill_gradient(low = "#A1D99B",   # 浅绿色（参与少）
                        high = "#2E8B57",  # 深绿色（参与多）
                        guide = "none") +  # 隐藏图例
    # 数值标签
    geom_text(aes(label = Count), 
              hjust = -0.3,        # 水平偏移-0.3（在柱子外侧）
              size = 6,            # 文字大小4mm
              fontface = "bold") + # 加粗
    # 标题和轴标签
    labs(title = "Feature Participation in Interactions",
         subtitle = " ",
         x = "",                           # x轴为空
         y = "Number of Interactions") +   # y轴标签
    theme_minimal() +
    theme(
      text = element_text(size = 26),                                      # 全局文字13pt
      plot.title = element_text(face = "bold", size = 28, hjust = 0.5),    # 标题加粗居中
      plot.subtitle = element_text(size = 1, hjust = 0.5, color = "gray40"),  # 副标题灰色
      plot.title.position = "plot", 
      axis.text.y = element_text(size = 23),                               # y轴（特征名）10pt
      axis.text.x = element_text(size = 23),                               # x轴10pt
      axis.title.x = element_text(face = "bold", size = 25)  ,
      plot.margin = margin(t = 2, r = 2, b = 2, l = 2) # x轴标题加粗
    )
  
  return(p)
}

################################################################
# 11. 保存所有图片 + 生成合并大图
################################################################

generate_all_figures <- function(data_list) {
  
  output_dir <- "J:/rtools45/feature_reduction_visualization"
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("\n")
  cat(rep("=", 60), "\n")
  cat("开始生成所有可视化图片\n")
  cat(rep("=", 60), "\n")
  
  # 生成所有图片
  p1 <- plot_feature_count_comparison(data_list$top_features_per_outcome,
                                      data_list$reduced_dataset_list,
                                      data_list$interaction_dataset_list)
  
  p2 <- plot_feature_reduction_ratio(data_list$top_features_per_outcome,
                                     data_list$reduced_dataset_list,
                                     data_list$interaction_dataset_list)
  
  p3 <- plot_shap_heatmap(data_list$top_features_per_outcome)
  
  p4 <- plot_feature_overlap(data_list$top_features_per_outcome)
  
  p5 <- plot_interaction_pairs(data_list$interaction_dataset_list)
  
  p6 <- plot_interaction_strength(data_list$interaction_results,
                                  data_list$interaction_dataset_list)
  
  p7 <- plot_interaction_heatmap(data_list$interaction_dataset_list,
                                 data_list$interaction_results)
  
  p8 <- plot_feature_participation(data_list$interaction_dataset_list)
  
  # 单独保存每张图片
  ggsave(file.path(output_dir, "1_Feature_Count_Comparison.png"), 
         p1, width = 13, height = 10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "2_Feature_Reduction_Ratio.png"), 
         p2, width = 13, height = 10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "3_SHAP_Feature_Heatmap.png"), 
         p3, width = 13, height =  10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "4_Feature_Overlap.png"), 
         p4, width = 13, height = 10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "5_Interaction_Pairs.png"), 
         p5, width = 13, height = 10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "6_Interaction_Strength.png"), 
         p6, width = 15, height = 10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "7_Interaction_Heatmap.png"), 
         p7, width = 13, height =  10, dpi = 600, bg = "white")
  
  ggsave(file.path(output_dir, "8_Feature_Participation.png"), 
         p8, width = 13, height = 10, dpi = 600, bg = "white")
  
  
  
  
  
}

################################################################
# 12. 主执行函数
################################################################

run_complete_visualization <- function() {
  cat("\n")
  cat(rep("=", 60), "\n")
  cat("SCI特征工程完整可视化流程\n")
  cat(rep("=", 60), "\n")
  
  # 1. 加载数据
  cat("\n[步骤 1/2] 加载特征工程数据...\n")
  data_list <- load_all_data()
  
  # 2. 生成所有可视化
  cat("\n[步骤 2/2] 生成所有可视化图片...\n")
  figures <- generate_all_figures(data_list)
  
  cat("\n🎉 特征工程可视化完成!\n")
  
  return(list(data = data_list, figures = figures))
}

# 执行
results <- run_complete_visualization()




################################################################
# 生成【唯一一张】合并大图运行后再按执行
################################################################

generate_all_figures <- function(data_list) {
  
  output_dir <- "J:/rtools45/feature_reduction_visualization"
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
  
  cat("\n")
  cat(rep("=", 60), "\n")
  cat("开始生成所有可视化图片（单一总大图版本）\n")
  cat(rep("=", 60), "\n")
  
  # 生成所有图片对象（不变）
  p1 <- plot_feature_count_comparison(data_list$top_features_per_outcome,
                                      data_list$reduced_dataset_list,
                                      data_list$interaction_dataset_list)
  
  p2 <- plot_feature_reduction_ratio(data_list$top_features_per_outcome,
                                     data_list$reduced_dataset_list,
                                     data_list$interaction_dataset_list)
  
  p3 <- plot_shap_heatmap(data_list$top_features_per_outcome)
  
  p4 <- plot_feature_overlap(data_list$top_features_per_outcome)
  
  p5 <- plot_interaction_pairs(data_list$interaction_dataset_list)
  
  p6 <- plot_interaction_strength(data_list$interaction_results,
                                  data_list$interaction_dataset_list)
  
  p7 <- plot_interaction_heatmap(data_list$interaction_dataset_list,
                                 data_list$interaction_results)
  
  p8 <- plot_feature_participation(data_list$interaction_dataset_list)
  
  # =========================================================
  # ⭐ 只生成【一张总大图】
  # 布局要求：
  # - 底部两个热图：p3 + p7
  # =========================================================
  
  ################################################################
  # SCI Main Figure Layout (No Empty Panel)
  ################################################################
  
  ################################################################
  # SCI Main Figure Layout (2 Columns × 4 Rows)
  ################################################################
  
  ################################################################
  # SCI Main Figure Layout (2 Columns × 4 Rows)
  ################################################################
  
  p1 <- p1 + labs(tag = "a")
  p2 <- p2 + labs(tag = "b")
  p3 <- p3 + labs(tag = "h")
  p4 <- p4 + labs(tag = "d")
  p5 <- p5 + labs(tag = "e")
  p6 <- p6 + labs(tag = "f")
  p7 <- p7 + labs(tag = "g")
  p8 <- p8 + labs(tag = "c")
  
  combined_all <-
    (
      p1 | p2
    ) /
    (
      p8 | p4
    ) /
    (
      p5 | p6
    ) /
    (
      p7 | p3
    ) +
    plot_layout(
      heights = c(
        1.2,
        1,   # SHAP Heatmap行
        1,
        1.5    # Interaction Heatmap行
      )
    ) &
    theme(
      plot.tag = element_text(
        face = "bold",
        size = 28
      ),
      plot.tag.position = c(0.02, 0.98)
    )
  # 保存唯一总图（TIFF - 投稿主版本）
  ggsave(
    file.path(
      output_dir,
      "SCI_Feature_Engineering_FULL_ONE_PANEL.tiff"
    ),
    combined_all,
    width = 26,
    height = 34,
    dpi = 800,
    compression = "lzw",
    bg = "white"
  )
  
  # 保存 PNG（预览/汇报用）
  ggsave(
    file.path(
      output_dir,
      "SCI_Feature_Engineering_FULL_ONE_PANEL.png"
    ),
    combined_all,
    width = 26,
    height = 34,
    dpi = 800,
    bg = "white"
  )
  
  # 保存 PDF（矢量版，投稿补充/修改用）
  ggsave(
    file.path(
      output_dir,
      "SCI_Feature_Engineering_FULL_ONE_PANEL.pdf"
    ),
    combined_all,
    width = 26,
    height = 35,
    device = cairo_pdf   # 保证中文/字体不乱
  )
  cat("\n")
  cat(rep("=", 60), "\n")
  cat("✅ 已生成唯一总图（所有内容合并完成）\n")
  cat("📁 输出路径:", output_dir, "\n")
  cat("📄 文件: SCI_Feature_Engineering_FULL_ONE_PANEL.png\n")
  cat(rep("=", 60), "\n")
  
  return(list(
    p1 = p1, p2 = p2, p3 = p3, p4 = p4,
    p5 = p5, p6 = p6, p7 = p7, p8 = p8,
    combined = combined_all
  ))
}

################################################################
# 12. 主执行函数
################################################################

run_complete_visualization <- function() {
  cat("\n")
  cat(rep("=", 60), "\n")
  cat("SCI特征工程完整可视化流程\n")
  cat(rep("=", 60), "\n")
  
  # 1. 加载数据
  cat("\n[步骤 1/2] 加载特征工程数据...\n")
  data_list <- load_all_data()
  
  # 2. 生成所有可视化
  cat("\n[步骤 2/2] 生成所有可视化图片...\n")
  figures <- generate_all_figures(data_list)
  
  cat("\n🎉 特征工程可视化完成!\n")
  
  return(list(data = data_list, figures = figures))
}

# 执行
results <- run_complete_visualization()


















################################################################
# SCI特征工程完整可视化（合并版）
# cowplot::plot_grid(align="none") 整图对齐
################################################################
rm(list=ls());gc()
library(ggplot2);library(data.table);library(dplyr);library(tidyr);library(cowplot)

output_dir <- "J:/rtools45/feature_reduction_visualization"
dir.create(output_dir, showWarnings=FALSE, recursive=TRUE)

################################################################
clean_outcome_name <- function(x){ x<-gsub("^Outcome_","",x); x<-gsub("_"," ",x); tools::toTitleCase(tolower(x)) }
generate_interaction_display_name <- function(f1,f2) paste(f1,"×",f2)

get_interaction_pairs_info <- function(){
  list(
    "Outcome_Disease Non-Progression"=list(
      c("Absolute Lymphocyte Count","Platelet Count"),
      c("Number of Hospitalizations","Platelet Count"),
      c("Plateletcrit","Alpha-fetoprotein (AFP)"),
      c("C-Reactive Protein (CRP)","Alpha-fetoprotein (AFP)")),
    "Outcome_Liver Cirrhosis"=list(
      c("Plateletcrit","Platelet Count"),
      c("Absolute Lymphocyte Count","Platelet Count"),
      c("Admission Interval","Alpha-fetoprotein (AFP)"),
      c("Cholinesterase","Plateletcrit")),
    "Outcome_Hepatocellular Carcinoma"=list(
      c("Direct Bilirubin","Alpha-fetoprotein (AFP)"),
      c("Number of Hospitalizations","C-Reactive Protein (CRP)"),
      c("Age","C-Reactive Protein (CRP)"),
      c("C-Reactive Protein (CRP)","Alpha-fetoprotein (AFP)")))
}

################################################################
load_all_data <- function(){
  feature_dir <- "J:/rtools45/手工4乘积交互Feature_Engineering"
  comparison_dir <- "J:/rtools45/手工4乘积交互Model_Comparison_Independent_Tuning"
  outcome_map <- c("Outcome_Disease_Non_Progression_"="Outcome_Disease Non-Progression",
                   "Outcome_Liver_Cirrhosis_"="Outcome_Liver Cirrhosis",
                   "Outcome_Hepatocellular_Carcinoma_"="Outcome_Hepatocellular Carcinoma")
  top_features_per_outcome <- list()
  reduced_dataset_list <- list()
  interaction_dataset_list <- list()
  interaction_results <- list()
  interaction_pairs_info <- get_interaction_pairs_info()
  for(outcome_clean in names(outcome_map)){
    outcome_name <- outcome_map[outcome_clean]; cat("  loading:",outcome_name,"\n")
    shap_file <- file.path("J:/rtools45/SHAP_Results_TOPJOURNAL",outcome_clean,"SHAP_Importance_Top15.csv")
    if(file.exists(shap_file)){ shap_imp<-fread(shap_file); top_features_per_outcome[[outcome_name]]<-shap_imp$Feature[1:15] }
    reduced_train_file<-file.path(feature_dir,"Reduced_Model_Data",paste0(outcome_clean,"_Reduced_Train.csv"))
    if(file.exists(reduced_train_file)){
      reduced_train<-fread(reduced_train_file)
      oc<-grep("outcome|Outcome",names(reduced_train),value=TRUE,ignore.case=TRUE)[1]; if(is.na(oc)||length(oc)==0) oc<-outcome_name
      reduced_dataset_list[[outcome_name]]<-list(feature_count=length(setdiff(names(reduced_train),oc)),features_used=setdiff(names(reduced_train),oc))
    }
    interaction_train_file<-file.path(feature_dir,"Interaction_Model_Data",paste0(outcome_clean,"_Interaction_Train.csv"))
    if(file.exists(interaction_train_file)){
      interaction_train<-fread(interaction_train_file)
      oc<-grep("outcome|Outcome",names(interaction_train),value=TRUE,ignore.case=TRUE)[1]; if(is.na(oc)||length(oc)==0) oc<-outcome_name
      fci<-setdiff(names(interaction_train),oc); intf<-grep("^INT_PROD_",fci,value=TRUE)
      ipd<-list()
      if(outcome_name%in%names(interaction_pairs_info)){
        pairs<-interaction_pairs_info[[outcome_name]]
        for(i in 1:length(pairs)){
          pr<-pairs[[i]]
          c1<-gsub("[^A-Za-z0-9]","_",pr[1]); c2<-gsub("[^A-Za-z0-9]","_",pr[2])
          ipd[[i]]<-list(feature1=pr[1],feature2=pr[2],
                         interaction_feature=paste0("INT_PROD_",c1,"_",c2),
                         display_name=generate_interaction_display_name(pr[1],pr[2]))
        }
      }
      interaction_dataset_list[[outcome_name]]<-list(feature_count=length(fci),features_used=fci,interaction_features=intf,interaction_pairs_display=ipd)
    }
    comparison_file<-file.path(comparison_dir,"SCI_Model_Comparison_Results.rds")
    if(file.exists(comparison_file)){
      all_comp<-readRDS(comparison_file)
      if(outcome_clean%in%names(all_comp)){
        comp<-all_comp[[outcome_clean]]
        int_gain_file<-file.path(comparison_dir,paste0(outcome_clean,"Interaction_Gain.csv"))
        if(file.exists(int_gain_file)) int_gain<-fread(int_gain_file)
        else if(!is.null(comp$interaction_gain)) int_gain<-comp$interaction_gain
        else int_gain<-data.frame(Feature=character(),Gain=numeric())
        interaction_results[[outcome_name]]<-list(metrics=comp$interaction_metrics$metrics,gain=int_gain)
      }
    }
  }
  list(top_features_per_outcome=top_features_per_outcome,reduced_dataset_list=reduced_dataset_list,
       interaction_dataset_list=interaction_dataset_list,interaction_results=interaction_results)
}

################################################################
# 子图函数（样式保持原样，只调大关键字体）
################################################################

plot_feature_count_comparison <- function(top_features_per_outcome,reduced_dataset_list,interaction_dataset_list){
  outcome_order<-names(top_features_per_outcome)
  fd<-data.frame()
  for(nm in outcome_order) fd<-rbind(fd,
                                     data.frame(Outcome=nm,Model="Original",Count=78),
                                     data.frame(Outcome=nm,Model="Reduced",Count=15),
                                     data.frame(Outcome=nm,Model="Interaction",Count=19))
  fd$Outcome<-factor(fd$Outcome,levels=outcome_order)
  fd$Model<-factor(fd$Model,levels=c("Original","Reduced","Interaction"))
  fd$Outcome_Display<-factor(sapply(as.character(fd$Outcome),clean_outcome_name),
                             levels=sapply(outcome_order,clean_outcome_name))
  ggplot(fd,aes(Outcome_Display,Count,fill=Model))+
    geom_bar(stat="identity",position=position_dodge(width=0.7),width=0.65,alpha=0.7,color="white",size=0.3)+
    geom_text(aes(label=Count),position=position_dodge(width=0.7),vjust=-0.5,size=8.5,fontface="bold")+
    annotate("text",x=1,y=24,label="+4",color="#2E8B57",size=7,fontface="italic")+
    annotate("text",x=2,y=24,label="+4",color="#2E8B57",size=7,fontface="italic")+
    annotate("text",x=3,y=24,label="+4",color="#2E8B57",size=7,fontface="italic")+
    labs(title="Feature Count Comparison",x="",y="Number of Features")+
    scale_fill_manual(values=c("Original"="#FF9999","Reduced"="#66C2FF","Interaction"="#A1D99B"),
                      labels=c("Original","Reduced","Interaction"))+
    coord_cartesian(ylim=c(0,82))+
    theme_minimal()+
    theme(text=element_text(size=26),axis.text.x=element_text(angle=0,hjust=0.5,size=20,face="bold"),
          axis.text.y=element_text(size=26),plot.title=element_text(face="bold",size=32,hjust=0.5,margin=margin(b=2)),
          plot.title.position="plot",axis.title.y=element_text(face="bold",size=28,margin=margin(r=2)),
          legend.title=element_blank(),legend.text=element_text(size=22),legend.position="top",
          legend.margin=margin(b=2),panel.grid.major.x=element_blank(),panel.grid.minor=element_blank(),
          plot.margin=margin(t=4,r=18,b=4,l=4))
}

plot_feature_reduction_ratio <- function(top,red,int){
  outcome_order<-names(top)
  rd<-data.frame()
  for(nm in outcome_order){
    rc<-length(red[[nm]]$features_used); ic<-length(int[[nm]]$features_used)
    rr<-round((78-rc)/78*100,1); ich<-round((ic-78)/78*100,1)
    rd<-rbind(rd,data.frame(Outcome=nm,Type="Reduced",Ratio=rr,Label=paste0("-",rr,"%")),
              data.frame(Outcome=nm,Type="Interaction",Ratio=ich,Label=paste0(ich,"%")))
  }
  rd$Outcome<-factor(rd$Outcome,levels=outcome_order)
  rd$Type<-factor(rd$Type,levels=c("Reduced","Interaction"))
  rd$Outcome_Display<-factor(sapply(as.character(rd$Outcome),clean_outcome_name),
                             levels=sapply(outcome_order,clean_outcome_name))
  ggplot(rd,aes(Outcome_Display,Ratio,fill=Type))+
    geom_bar(stat="identity",position=position_dodge(width=0.7),width=0.6,alpha=0.85,color="white",size=0.3)+
    geom_hline(yintercept=0,linetype="dashed",color="gray40",size=0.6)+
    geom_text(aes(label=Label),position=position_dodge(width=0.7),vjust=ifelse(rd$Ratio>=0,-0.5,1.5),size=8.5,fontface="bold")+
    labs(title="Feature Count Change Ratio",x="",y="Change Ratio (%)")+
    scale_fill_manual(values=c("Reduced"="#66C2FF","Interaction"="#A1D99B"))+
    scale_y_continuous(expand = expansion(mult = c(0.12, 0.12))) +
    theme_minimal()+
    theme(text=element_text(size=26),axis.text.x=element_text(angle=0,hjust=0.5,size=20,face="bold"),
          axis.text.y=element_text(size=22),plot.title=element_text(face="bold",size=32,hjust=0.5),
          axis.title.y=element_text(face="bold",size=28),legend.title=element_blank(),
          legend.text=element_text(size=22),legend.position="top",legend.margin=margin(b=2),
          panel.grid.major.x=element_blank(),plot.margin=margin(t=4,r=18,b=4,l=4))
}

plot_shap_heatmap <- function(top){
  outcome_order<-names(top); hd<-data.frame()
  for(nm in outcome_order){
    feats<-top[[nm]]
    for(i in 1:length(feats)) hd<-rbind(hd,data.frame(Outcome=nm,Feature=feats[i],Rank=i,Importance=16-i,stringsAsFactors=FALSE))
  }
  hd$Outcome_Display<-factor(sapply(as.character(hd$Outcome),clean_outcome_name),
                             levels=sapply(outcome_order,clean_outcome_name))
  ggplot(hd,aes(Outcome_Display,reorder(Feature,-Rank),fill=Importance))+
    geom_tile(color="white",size=0.5)+
    scale_x_discrete(expand=expansion(add=c(0.6,1.5)))+
    scale_fill_gradient(low="#FFD700",high="#e31a1c",name="Importance Rank",breaks=c(1,8,15),labels=c("Low","Medium","High"))+
    coord_cartesian(clip="off")+
    guides(fill=guide_colorbar(direction="horizontal",title.position="top",title.hjust=0.5,
                               barwidth=unit(8,"cm"),barheight=unit(0.8,"cm"),label.position="bottom",label.hjust=0.5))+
    labs(title="SHAP Top 15 Feature Importance Heatmap",x="",y="Feature")+
    theme_minimal()+
    theme(text=element_text(size=26),axis.text.x=element_text(angle=20,hjust=0.5,size=20,face="bold"),
          axis.text.y=element_text(size=24),axis.title.y=element_text(size=32,face="bold"),
          plot.title=element_text(face="bold",size=32,hjust=0.5,margin=margin(b=2)),
          plot.title.position="plot",legend.position="top",legend.title=element_text(size=22,face="bold"),
          legend.text=element_text(size=20),legend.margin=margin(b=2),panel.grid=element_blank(),
          plot.margin=margin(t=4,r=22,b=15,l=4))
}

plot_feature_overlap <- function(top){
  all_feats<-unique(unlist(top)); od<-data.frame()
  for(f in all_feats){ cnt<-sum(sapply(top,function(x) f%in%x)); od<-rbind(od,data.frame(Feature=f,Outcome_Count=cnt)) }
  od<-od%>%arrange(desc(Outcome_Count))%>%head(20)
  od$Color_Group<-factor(od$Outcome_Count,levels=c(1,2,3),labels=c("1","2","3"))
  ggplot(od,aes(reorder(Feature,Outcome_Count),Outcome_Count,fill=Color_Group))+
    geom_bar(stat="identity",alpha=0.85,width=0.7)+coord_flip()+
    scale_fill_manual(values=c("1"="#FFD700","2"="#FF8C00","3"="#e31a1c"))+
    labs(title="Feature Overlap Across Outcomes",x="",y="Number of Outcomes")+
    theme_minimal()+
    theme(text=element_text(size=26),plot.title=element_text(face="bold",size=32,hjust=0.5),
          plot.title.position="plot",axis.text.y=element_text(size=24),axis.text.x=element_text(size=24),
          axis.title.x=element_text(face="bold",size=28),legend.title=element_text(size=22,face="bold"),
          legend.text=element_text(size=20),legend.position="top",plot.margin=margin(t=4,r=18,b=4,l=4))
}

plot_interaction_pairs <- function(int_ds){
  outcome_order<-names(int_ds); id<-data.frame()
  for(nm in outcome_order){
    pd<-int_ds[[nm]]$interaction_pairs_display
    for(pi in pd) id<-rbind(id,data.frame(Outcome=nm,Display_Name=pi$display_name,stringsAsFactors=FALSE))
  }
  id$Outcome_Display<-factor(sapply(as.character(id$Outcome),clean_outcome_name),
                             levels=sapply(outcome_order,clean_outcome_name))
  ggplot(id,aes(Outcome_Display,Display_Name,fill=Outcome_Display))+
    geom_tile(color="white",size=0.8,alpha=0.85)+
    scale_x_discrete(expand=expansion(add=c(0.6,1.5)))+
    scale_fill_manual(values=c("#fd9d9c","#66C2FF","#A1D99B"),guide="none")+
    coord_cartesian(clip="off")+
    labs(title="Product Interaction Feature Pairs",x="",y="Interaction Pair")+
    theme_minimal()+
    theme(text=element_text(size=26),plot.title=element_text(face="bold",size=32,hjust=0.5),
          plot.title.position="plot",axis.text.x=element_text(angle=10,size=22,face="bold"),
          axis.text.y=element_text(size=22,face="bold"),axis.title.y=element_text(size=32,face="bold"),
          panel.grid=element_blank(),panel.border=element_rect(color="gray80",fill=NA,size=0.5),
          plot.margin=margin(t=4,r=18,b=15,l=4))
}

plot_interaction_strength <- function(int_res,int_ds){
  outcome_order<-names(int_res); sd<-data.frame()
  for(nm in outcome_order){
    if(!is.null(int_res[[nm]]$gain)){
      gd<-int_res[[nm]]$gain; ig<-gd[grepl("^INT_PROD_",Feature),]
      if(nrow(ig)>0){
        pd<-int_ds[[nm]]$interaction_pairs_display
        for(i in 1:nrow(ig)){
          fn<-ig$Feature[i]; dn<-fn
          for(pi in pd) if(pi$interaction_feature==fn) dn<-pi$display_name
          sd<-rbind(sd,data.frame(Outcome=nm,Display_Name=dn,Gain=ig$Gain[i],stringsAsFactors=FALSE))
        }
      }
    }
  }
  if(nrow(sd)==0) return(ggplot()+labs(title="No Data")+theme_void())
  sd$Outcome_Display<-factor(sapply(as.character(sd$Outcome),clean_outcome_name),
                             levels=sapply(outcome_order,clean_outcome_name))
  sd<-sd%>%arrange(desc(Gain))
  ggplot(sd,aes(reorder(Display_Name,Gain),Gain,fill=Outcome_Display))+
    geom_bar(stat="identity",alpha=0.85,width=0.7,color="white",size=0.3)+
    coord_flip()+
    scale_y_continuous(expand=expansion(mult=c(0,0.6)))+
    scale_fill_manual(values=c("#fd9d9c","#66C2FF","#A1D99B"),name=NULL)+
    labs(title="Interaction Feature Strength (Gain)",x="",y="Gain")+
    theme_minimal()+
    theme(text=element_text(size=26),plot.title=element_text(face="bold",size=32,hjust=0.5),
          plot.title.position="plot",axis.text.y=element_text(size=22,face="bold"),
          axis.text.x=element_text(size=24),axis.title.x=element_text(face="bold",size=28),
          legend.position=c(0.7,0.02),legend.justification=c(1,0),
          legend.title=element_blank(),legend.text=element_text(size=18),
          legend.background=element_rect(fill=alpha("white",0.5),color=NA),
          panel.grid.major.y=element_blank(),plot.margin=margin(t=4,r=18,b=4,l=4))
}

plot_interaction_heatmap <- function(int_ds,int_res){
  outcome_order<-names(int_ds); ad<-data.frame()
  for(nm in outcome_order){
    pd<-int_ds[[nm]]$interaction_pairs_display
    for(pi in pd){
      gv<-0
      if(!is.null(int_res[[nm]]$gain)){
        gd<-int_res[[nm]]$gain; mr<-gd[gd$Feature==pi$interaction_feature,]
        if(nrow(mr)>0) gv<-mr$Gain[1]
      }
      ad<-rbind(ad,data.frame(Outcome=nm,Display_Name=pi$display_name,Gain=gv,stringsAsFactors=FALSE))
    }
  }
  ad$Outcome_Display<-factor(sapply(as.character(ad$Outcome),clean_outcome_name),
                             levels=sapply(outcome_order,clean_outcome_name))
  ap<-unique(ad$Display_Name)
  cm<-expand.grid(Outcome_Display=levels(ad$Outcome_Display),Display_Name=ap,stringsAsFactors=FALSE)
  cm<-cm%>%left_join(ad[,c("Outcome_Display","Display_Name","Gain")],by=c("Outcome_Display","Display_Name"))
  cm$Gain[is.na(cm$Gain)]<-NA
  ggplot(cm,aes(Outcome_Display,Display_Name,fill=Gain))+
    geom_tile(color="white",size=0.8)+
    geom_text(aes(label=ifelse(!is.na(Gain),sprintf("%.3f",Gain),"")),size=6,fontface="bold")+
    scale_fill_gradient(low="#c6e8c6",high="#2E8B57",name="Gain",na.value="#f0f0f0")+
    coord_cartesian(clip="off")+
    labs(title="Interaction Feature Distribution Heatmap",x="",y="Interaction Pair")+
    theme_minimal()+
    theme(text=element_text(size=26),plot.title=element_text(face="bold",size=32,hjust=0.5,margin=margin(b=2)),
          plot.title.position="plot",axis.text.x=element_text(angle=15,size=24,face="bold"),
          axis.text.y=element_text(size=22,face="bold"),axis.title.y=element_text(size=32,face="bold"),
          legend.position="top",legend.direction="horizontal",legend.title=element_text(size=22,face="bold"),
          legend.text=element_text(size=20),legend.margin=margin(b=2),panel.grid=element_blank(),
          plot.margin=margin(t=4,r=18,b=15,l=4))
}

plot_feature_participation <- function(int_ds){
  outcome_order<-names(int_ds); pd<-data.frame()
  for(nm in outcome_order){
    pds<-int_ds[[nm]]$interaction_pairs_display
    for(pi in pds) pd<-rbind(pd,data.frame(Feature=pi$feature1,Outcome=nm,stringsAsFactors=FALSE),
                             data.frame(Feature=pi$feature2,Outcome=nm,stringsAsFactors=FALSE))
  }
  if(nrow(pd)==0) return(ggplot()+labs(title="No Data")+theme_void())
  fc<-pd%>%group_by(Feature)%>%summarise(Count=n(),.groups="drop")%>%arrange(desc(Count))
  ggplot(head(fc,15),aes(reorder(Feature,Count),Count,fill=Count))+
    geom_bar(stat="identity",alpha=0.85,width=0.7)+coord_flip()+
    scale_fill_gradient(low="#A1D99B",high="#2E8B57",guide="none")+
    geom_text(aes(label=Count),hjust=-0.3,size=7.5,fontface="bold")+
    labs(title="Feature Participation in Interactions",x="",y="Number of Interactions")+
    theme_minimal()+
    theme(text=element_text(size=26),plot.title=element_text(face="bold",size=32,hjust=0.5),
          plot.title.position="plot",axis.text.y=element_text(size=26),axis.text.x=element_text(size=26),
          axis.title.x=element_text(face="bold",size=32),plot.margin=margin(t=4,r=16,b=4,l=4))
}

################################################################
# 主执行
################################################################
cat("===== 加载数据 =====\n")
data_list <- load_all_data()

cat("\n===== 生成子图 =====\n")
p1 <- plot_feature_count_comparison(data_list$top_features_per_outcome,data_list$reduced_dataset_list,data_list$interaction_dataset_list)
p2 <- plot_feature_reduction_ratio(data_list$top_features_per_outcome,data_list$reduced_dataset_list,data_list$interaction_dataset_list)
p3 <- plot_shap_heatmap(data_list$top_features_per_outcome)
p4 <- plot_feature_overlap(data_list$top_features_per_outcome)
p5 <- plot_interaction_pairs(data_list$interaction_dataset_list)
p6 <- plot_interaction_strength(data_list$interaction_results,data_list$interaction_dataset_list)
p7 <- plot_interaction_heatmap(data_list$interaction_dataset_list,data_list$interaction_results)
p8 <- plot_feature_participation(data_list$interaction_dataset_list)

################################################################
# 合并
################################################################
cat("\n===== 拼接 2×4 =====\n")
TAG <- theme(plot.tag.position=c(0.02,0.98), plot.tag=element_text(size=32,face="bold"))

p1 <- p1 + labs(tag="a") + TAG
p2 <- p2 + labs(tag="b") + TAG
p3 <- p3 + labs(tag="h") + TAG
p4 <- p4 + labs(tag="d") + TAG
p5 <- p5 + labs(tag="e") + TAG
p6 <- p6 + labs(tag="f") + TAG
p7 <- p7 + labs(tag="g") + TAG
p8 <- p8 + labs(tag="c") + TAG

# 2列×4行
combined <- cowplot::plot_grid(p1,p2,p8,p4,p5,p6,p7,p3, ncol=2, align="none", rel_heights=c(1,1,1.3,1.6))

base <- file.path(output_dir, "SCI_Feature_Engineering_Combined")
ggsave(paste0(base,".tiff"), combined, width=28, height=42, dpi=600, compression="lzw", bg="white")
ggsave(paste0(base,".pdf"),  combined, width=28, height=42, device="pdf", bg="white")
ggsave(paste0(base,".png"),  combined, width=28, height=42, dpi=600, bg="white")

cat("\nDONE ->", basename(base), "\n")
