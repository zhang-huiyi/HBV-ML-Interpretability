######################################################################
# 00_data_merging.R
# ----------------------------------------------------------------------------
# 用途：读入患者信息、病历号与患者唯一号的对应关系、检验数据，按患者与时点
#       合并，形成后续建模所用的原始宽表，并输出基础统计结果。
# 输出：合并后的 dhbv.xlsx 与 统计结果.xlsx，均写入当前工作目录。
# 运行前提：本脚本为流程第一步，无上游依赖。
# ----------------------------------------------------------------------------
######################################################################

# 读入数据
library(readxl)
患者 <- read.csv(
  "C:/Users/user/Desktop/毕业课题/HBV治愈预测模型/HBV住院数据/患者.csv"
)
病历号对应患者唯一号 <- read_excel("C:/Users/user/Desktop/毕业课题/HBV治愈预测模型/HBV住院数据/病历号对应患者唯一号.xlsx")
检验1 <- read.csv("C:/Users/user/Desktop/毕业课题/HBV治愈预测模型/HBV住院数据/检验.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE, fileEncoding = "GB18030")
head(检验1)
sum(检验1$收费项目名称 == "两对半定量", na.rm = TRUE)
# 读取 CSV 文件（文件路径请改为实际路径）
检验2 <- read.csv("C:/Users/user/Desktop/毕业课题/HBV治愈预测模型/HBV住院数据/检验2.csv", header = TRUE, sep = ",", stringsAsFactors = FALSE)
head(检验2)
sum(检验2$收费项目名称 == "两对半定量", na.rm = TRUE)

##合并行并去除重复
library(tidyverse)
检验 <- bind_rows(检验1, 检验2)
检验 <- distinct(检验)# 去除所有列完全相同的重复行

# 查看前几行数据
head(data)

# 使用 dplyr 包进行操作
library(dplyr)
# 将两列都转换为字符类型
患者$住院号 <- as.character(患者$住院号)
病历号对应患者唯一号$住院号 <- as.character(病历号对应患者唯一号$住院号)

dhbv <- left_join(患者, 病历号对应患者唯一号, by = "住院号")

# 查看结果
head(dhbv)

dhbv <- left_join(患者, 病历号对应患者唯一号, by = "住院号")

# 保留唯一值并统计重复次数
count_unique <- 病历号对应患者唯一号 %>%
  group_by(id) %>% 
  summarise(
    住院次数 = n(), # 统计每个 id 出现的次数
    .groups = "drop" # 取消分组
  )

# 将统计结果合并回原始数据框
df_with_counts <- left_join(病历号对应患者唯一号, count_unique, by = "id")
# 去除重复行，只保留每个 id 的第一行
df_unique_counts <- df_with_counts %>%
  distinct(id, .keep_all = TRUE)


# 查看结果
summary(df_unique_counts$住院次数)

sum(df_unique_counts$住院次数 >= 2)
sum(df_unique_counts$住院次数 >= 3)
sum(df_unique_counts$住院次数 >= 6)

dhbv <- left_join(dhbv, df_unique_counts[, c(2,3)], by = "id")

##匹配信息
colnames(dhbv)
head(dhbv)
head(dhbv$入院时间)
library(lubridate)

# 将入院时间转换为日期时间格式
dhbv$入院时间 <- dmy_hms(dhbv$入院时间)
dhbv$出院时间 <- dmy_hms(dhbv$出院时间)

# 检查入院时间的格式
head(dhbv$入院时间)
数据提取时间段 <- c(min(dhbv$入院时间, na.rm = TRUE), max(dhbv$入院时间, na.rm = TRUE))# 获取数据中的最早入院时间和最晚入院时间
数据提取时间段

# 排序并提取每个住院号的首次入院时间和最后次入院时间
dhbv <- dhbv %>%
  group_by(id) %>%
  mutate(
    首次入院时间 = min(入院时间, na.rm = TRUE),
    最后次入院时间 = max(入院时间, na.rm = TRUE),
    入院时间间隔 = as.numeric(最后次入院时间 - 首次入院时间, units = "days"),
    住院时间间隔 = as.numeric(出院时间 - 入院时间, units = "days")
  )

head(dhbv[, c("首次入院时间", "最后次入院时间", "入院时间间隔")])
colSums(is.na(dhbv[, c("首次入院时间", "最后次入院时间", "入院时间间隔")]))

##合并检验数据
head(dhbv)
head(检验)

library(dplyr)
# 统计病历号的唯一值数量
unique_count <- 检验 %>% 
  summarise(病历号_unique_count = n_distinct(病历号))

unique_count


library(tidyverse)
检验_wide <- 检验 %>%
  pivot_wider(
    id_cols = 病历号,              # 以病历号作为唯一标识
    names_from = 检验项目名称,     # 检验项目名称变为列名
    values_from = 结果,            # 结果列作为新列的值
    values_fn = list(结果 = max)   # 处理同一病历号+检验项目的多个结果，选最大值
  ) %>%
  rename(住院号 = 病历号)
# 查看转换后的数据
head(检验_wide)

#合并检验结果
检验_wide$住院号 <- as.character(检验_wide$住院号)
dhbv <- left_join(dhbv, 检验_wide, by = "住院号")
head(dhbv)

dhbv <- dhbv %>%
  select(住院号, id, 入院时间, 首次入院时间, 最后次入院时间, 入院时间间隔, 住院次数,
         入院病区, 出院病区, 出院时间, 住院时间间隔, everything()) #将id列移到住院列后
head(dhbv[, c("id" ,"首次入院时间", "最后次入院时间", "入院时间间隔", "住院次数","出院时间", "住院时间间隔")])

colnames(dhbv)
str(dhbv)

table(dhbv$性别)   #男 女
table(dhbv$肝硬化) #是
class(dhbv$血小板总数)
table(dhbv$血小板总数)
class(dhbv$丙氨酸氨基转移酶)
table(dhbv$丙氨酸氨基转移酶)
table(dhbv$乙肝表面抗原)
class(dhbv$`HBV-DNA定量`)
class(dhbv$`HBV-DNA定量(实时荧光定量PCR法)`)
table(dhbv$`HBV-DNA定量`)
table(dhbv$`HBV-DNA定量(实时荧光定量PCR法)`)
table(dhbv$`HBV-DNA定性`)
table(dhbv$乙肝表面抗原)
head(dhbv$抗病毒治疗开始日期)
table(dhbv$替诺福韦)
table(dhbv$拉米夫定)
table(dhbv$恩曲他滨)
table(dhbv$阿德福韦)
table(dhbv$替比夫定)
install.packages("writexl")
library(writexl)
write_xlsx(dhbv, "dhbv.xlsx")
########################table1###############################
df <- dhbv[, c("住院号", "id", "入院时间", "首次入院时间", "入院时间间隔", "住院次数", "实际住院天数", "性别", "年龄", 
               "肝硬化", "血小板总数", "丙氨酸氨基转移酶", "乙肝表面抗原", "HBV-DNA定性", 
               "抗病毒治疗开始日期", "替诺福韦", "拉米夫定", "恩曲他滨", "阿德福韦", "替比夫定")]
str(df)
head(df[, c("入院时间","抗病毒治疗开始日期")], 100)
df$抗病毒治疗开始日期 <- as.POSIXct(df$抗病毒治疗开始日期, format = "%Y%m%d%H:%M:%S", tz = "Asia/Shanghai")
library(dplyr)

df <- df %>%
  mutate(
    ART_naive = ifelse(!is.na(抗病毒治疗开始日期) & (入院时间 - 抗病毒治疗开始日期 <= 30) & (入院时间 >= 抗病毒治疗开始日期),
                       1, 0),
    ART_duration = as.numeric(abs(difftime(入院时间, 抗病毒治疗开始日期, units = "days")))
  )
df <- df %>%
  mutate(
    肝硬化 = ifelse(肝硬化 == "是", 1, 0),
    替诺福韦 = ifelse(替诺福韦 == "是", 1, 0),
    拉米夫定 = ifelse(拉米夫定 == "是", 1, 0),
    恩曲他滨 = ifelse(恩曲他滨 == "是", 1, 0),
    阿德福韦 = ifelse(阿德福韦 == "是", 1, 0),
    `HBV-DNA定性`=ifelse( `HBV-DNA定性` == "阴性", 0, 1)
  )

df <- df %>%
  mutate(
    血小板总数 = as.numeric(血小板总数),
    丙氨酸氨基转移酶 = as.numeric(丙氨酸氨基转移酶)
  )

# 假设乙肝表面抗原列名为 "乙肝表面抗原"
df <- df %>%
  mutate(
    # 将 ">250.0" 记为 251
    乙肝表面抗原 = gsub(">250.0", "251", 乙肝表面抗原),
    
    # 转换为 numeric 类型
    乙肝表面抗原 = as.numeric(乙肝表面抗原),
    
    # 根据乙肝表面抗原值生成新列
    乙肝表面抗原分组 = case_when(
      乙肝表面抗原 >= 0.05 & 乙肝表面抗原 <= 100 ~ "0.05–100",
      乙肝表面抗原 > 100 & 乙肝表面抗原 < 250 ~ "100–250",
      乙肝表面抗原 >= 250 ~ "≥250",
      TRUE ~ NA_character_  # 处理其他非预期值
    )
  )
table(df$乙肝表面抗原分组)


library(dplyr)
library(writexl)

# 过滤数据
filtered_df <- df %>%
  filter(入院时间 == 首次入院时间)

calculate_stats <- function(data) {
  # 移除可能存在的隐藏分组
  data <- data %>% ungroup()
  
  # 连续变量统计（采用 Q1/Q3 分位数）
  cont_stats <- data %>%
    summarise(
      across(c(年龄, 入院时间间隔, 住院次数, 血小板总数, 丙氨酸氨基转移酶, ART_duration),
             list(median = ~median(., na.rm = TRUE),
                  q1 = ~quantile(., probs = 0.25, na.rm = TRUE),  # 第一分位数
                  q3 = ~quantile(., probs = 0.75, na.rm = TRUE)), # 第三分位数
             .names = "{.col}_{.fn}") 
    ) %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Value") %>%
    mutate(Value = as.character(round(Value, 2)))  # 转换为字符型统一格式
  
  # 分类变量统计（保持原样）
  cat_stats <- data %>%
    summarise(
      男性 = paste0(sum(性别 == "男", na.rm = TRUE), 
                  " (", round(mean(性别 == "男", na.rm = TRUE)*100, 1), "%)"),
      肝硬化 = paste0(sum(肝硬化 == 1, na.rm = TRUE), 
                   " (", round(mean(肝硬化 == 1, na.rm = TRUE)*100, 1), "%)"),
      HBV_阴性 = paste0(sum(`HBV-DNA定性` == 0, na.rm = TRUE), 
                      " (", round(mean(`HBV-DNA定性` == 0, na.rm = F)*100, 1), "%)"),
      ART_naive = paste0(sum(ART_naive == 1, na.rm = TRUE), 
                         " (", round(mean(ART_naive == 1, na.rm = TRUE)*100, 1), "%)")
    ) %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Value")
  
  # 药物统计（保持原样）
  drug_stats <- data %>%
    summarise(
      替诺福韦 = paste0(sum(替诺福韦 == 1, na.rm = TRUE), 
                    " (", round(mean(替诺福韦 == 1, na.rm = F)*100, 1), "%)"),
      拉米夫定 = paste0(sum(拉米夫定 == 1, na.rm = TRUE), 
                    " (", round(mean(拉米夫定 == 1, na.rm = F)*100, 1), "%)"),
      恩曲他滨 = paste0(sum(恩曲他滨 == 1, na.rm = TRUE), 
                    " (", round(mean(恩曲他滨 == 1, na.rm = F)*100, 1), "%)"),
      阿德福韦 = paste0(sum(阿德福韦 == 1, na.rm = TRUE), 
                    " (", round(mean(阿德福韦 == 1, na.rm = F)*100, 1), "%)"),
      替比夫定 = paste0(sum(替比夫定 == 1, na.rm = TRUE), 
                    " (", round(mean(替比夫定 == 1, na.rm = F)*100, 1), "%)")
    ) %>%
    pivot_longer(everything(), names_to = "Variable", values_to = "Value")
  
  # 乙肝表面抗原分组统计（保持原样）
  hbsag_counts <- data %>%
    count(乙肝表面抗原分组) %>%
    mutate(percent = n / sum(n) * 100,
           Value = paste0(n, " (", round(percent, 1), "%)")) %>%
    select(Variable = 乙肝表面抗原分组, Value)
  
  # 合并所有结果
  bind_rows(
    cont_stats,
    cat_stats,
    drug_stats,
    hbsag_counts
  ) %>% 
    mutate(Type = case_when(
      str_detect(Variable, "_median|_q1|_q3") ~ "连续变量",  # 更新类型标识
      Variable %in% c("男性", "肝硬化", "HBV_阴性", "ART_naive") ~ "人口学特征",
      Variable %in% c("替诺福韦", "拉米夫定", "恩曲他滨", "阿德福韦", "替比夫定") ~ "药物使用",
      TRUE ~ "乙肝表面抗原分组"
    ))
}

results <- calculate_stats(filtered_df)
# 写入Excel（按类型排序）
write_xlsx(results, "统计结果.xlsx")

