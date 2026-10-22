# 05_SHAP_visualization.R
# ----------------------------------------------------------------------------
# 用途：基于 04 输出的 SHAP 结果，生成论文主图与补充图：
#   - Figure 1 ：三结局的 SHAP 重要性、蜂群图、依赖图与交互图整合图
#   - Figure S1：瀑布图（单个样本的贡献分解）
#   - Figure S2：其余关键特征的依赖图
# 运行前提：先运行 01–04 生成各结局的 SHAP 结果；稳定性相关图形另需先运行 10。
# 输出：output_dir 下的 TIFF / PNG / PDF 图形文件。
# 依赖包：data.table, dplyr, ggplot2, stringr, scales, ggbeeswarm, cowplot, gridExtra
# ----------------------------------------------------------------------------

##############################################################################

rm(list = ls()); gc()

pkgs <- c("data.table","dplyr","ggplot2","stringr","scales",
          "ggbeeswarm","cowplot","gridExtra","forcats","tools","tibble")
for(p in pkgs) if(!require(p, character.only=TRUE)){ install.packages(p); library(p, character.only=TRUE) }

base_dir  <- "J:/rtools45/SHAP_Results_TOPJOURNAL"
bootstrap_refit_dir <- "J:/rtools45/Patient_Bootstrap_TreeSHAP_Results_B100_N1000_FINAL"
afp_sensitivity_dir <- "J:/rtools45/AFP_Missingness_Sensitivity_Analysis"
observed_afp_dependence_dir <- file.path(
  afp_sensitivity_dir, "03_Observed_AFP_Only_SHAP_Plots", "Plot_Data"
)
observed_afp_summary_file <- file.path(
  afp_sensitivity_dir, "03_Observed_AFP_Only_SHAP_Plots",
  "Observed_AFP_Only_SHAP_Plot_Summary.csv"
)
observed_afp_interaction_dir <- file.path(
  afp_sensitivity_dir, "04_Observed_AFP_Only_Interaction_Plots", "Plot_Data"
)
interaction_b100_file <- file.path(
  bootstrap_refit_dir,
  "Supplementary_Table6_Top20_Interaction_Reproducibility_B100.csv"
)
output_dir <- file.path(base_dir, "Publication_Figures",
                        "Combined_Figures_FINAL_B100_HCC_GenderAFP_ObservedAFP")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

OUTCOME_DIRS <- c(
  "Outcome_Disease_Non_Progression_"  = "Disease Non-Progression",
  "Outcome_Liver_Cirrhosis_"          = "Liver Cirrhosis",
  "Outcome_Hepatocellular_Carcinoma_" = "Hepatocellular Carcinoma"
)
DNP <- "Disease Non-Progression"; LC <- "Liver Cirrhosis"; HCC <- "Hepatocellular Carcinoma"

##############################################################################
# 工具
##############################################################################
clean_feature_names <- function(x){
  str_replace_all(x, c(
    "High-Sensitivity C-Reactive Protein \\(hs-CRP\\)"="High-Sensitivity C-Reactive Protein",
    "High-Sensitivity C-Reactive Protein \\(hs-CRP"="High-Sensitivity C-Reactive Protein",
    "Alpha-fetoprotein \\(AFP\\)"="Alpha-fetoprotein",
    "C-Reactive Protein \\(CRP\\)"="C-Reactive Protein",
    "Aspartate Aminotransferase \\(AST\\)"="Aspartate Aminotransferase",
    "Alanine Aminotransferase \\(ALT\\)"="Alanine Aminotransferase",
    "Gamma-Glutamyl Transferase \\(GGT\\)"="Gamma-Glutamyl Transferase",
    "γ-glutamyl transferase"="Gamma-Glutamyl Transferase",
    "Hepatitis B e Antibody \\(HBeAb\\)"="Hepatitis B e Antibody",
    "Hepatitis B Core Antibody \\(HBcAb\\)"="Hepatitis B Core Antibody",
    "Indirect bilirubin"="Indirect Bilirubin"))
}
feature_units <- c(
  "Alpha-fetoprotein"="ng/mL","Alanine Aminotransferase"="U/L","Total Bile Acids"="μmol/L",
  "Aspartate Aminotransferase"="U/L","Gamma-Glutamyl Transferase"="U/L",
  "Platelet Count"="×10⁹/L","Plateletcrit"="%","Cholinesterase"="KU/L","Prealbumin"="mg/L",
  "Indirect Bilirubin"="μmol/L","Direct Bilirubin"="μmol/L",
  "Absolute Lymphocyte Count"="×10⁹/L","C-Reactive Protein"="mg/L",
  "High-Sensitivity C-Reactive Protein"="mg/L","Lactate"="mmol/L",
  "Age"="years","Admission Interval"="days","Number of Hospitalizations"="count")
must_log_features <- c("Alpha-fetoprotein","Alanine Aminotransferase","Total Bile Acids")

##############################################################################
# 数据加载
##############################################################################
read_shap_imp   <- function(d){ f<-file.path(base_dir,d,"SHAP_Importance_Top15.csv"); if(!file.exists(f))return(NULL); dt<-fread(f); dt$Feature<-clean_feature_names(dt$Feature); dt%>%arrange(desc(MeanAbsSHAP))%>%slice_head(n=15) }
read_beeswarm   <- function(d){ f<-file.path(base_dir,d,"SHAP_Beeswarm_Data.csv"); if(!file.exists(f))return(NULL); dt<-fread(f)[,.(Feature,SHAP,Feature_Value)]; dt$Feature<-clean_feature_names(dt$Feature); dt }
read_inter_arr  <- function(d){ a<-file.path(base_dir,d,"TreeSHAP_Interaction_Array_FULL.rds"); r<-file.path(base_dir,d,"Feature_Raw_Value.csv"); if(!file.exists(a)||!file.exists(r))return(NULL); arr<-readRDS(a); raw<-fread(r); names(raw)<-clean_feature_names(names(raw)); dimnames(arr)[[2]]<-clean_feature_names(dimnames(arr)[[2]]); dimnames(arr)[[3]]<-clean_feature_names(dimnames(arr)[[3]]); list(array=arr,raw=raw) }
read_waterfall  <- function(d){ s<-file.path(base_dir,d,"SHAP_Main_Effect.csv"); i<-file.path(base_dir,d,"Waterfall_Sample_Selection.csv"); if(!file.exists(s)||!file.exists(i))return(NULL); sm<-fread(s); si<-fread(i); colnames(sm)<-clean_feature_names(colnames(sm)); if(is.null(rownames(sm))||all(rownames(sm)==""))rownames(sm)<-as.character(si$Sample_ID); list(shap=sm,samples=si) }

##############################################################################
# 单图函数
##############################################################################

mk_importance <- function(data, outcome_name) {
  if(is.null(data)) return(NULL)
  data <- data %>% arrange(MeanAbsSHAP)
  data$Feature <- factor(data$Feature, levels=data$Feature)
  ggplot(data, aes(MeanAbsSHAP, Feature)) +
    geom_col(fill="#66C2FF", width=0.75) +
    geom_text(aes(label=sprintf("%.3f",MeanAbsSHAP)), hjust=-0.1, size=5.5, color="#2058a2") +
    labs(title=outcome_name, x="Mean |SHAP value|", y=NULL) +
    expand_limits(x=max(data$MeanAbsSHAP)*2) +
    scale_x_continuous(limits=c(0,0.8), breaks=seq(0,0.8,0.2),
                       labels=sprintf("%.3f",seq(0,0.8,0.2))) +
    theme_bw(base_size=20) +
    theme(plot.title=element_text(hjust=0.5,size=25,face="bold",margin=margin(b=2)),
          plot.title.position="plot", axis.title.x=element_text(size=20,margin=margin(t=8)),
          axis.text.y=element_text(size=17,color="black",lineheight=0.92),
          axis.text.x=element_text(size=16,color="black"),
          panel.grid.major.y=element_blank(), panel.grid.minor=element_blank(),
          panel.grid.major.x=element_line(color="grey90",linewidth=0.2),
          panel.border=element_rect(color="grey90",linewidth=0.4),
          plot.margin=margin(t=4,r=6,b=4,l=8),
          panel.background=element_rect(fill="#F5F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA))
}

mk_beeswarm <- function(data, outcome_name) {
  if(is.null(data)) return(NULL)
  fo <- data %>% group_by(Feature) %>% summarise(M=mean(abs(SHAP)),.groups="drop") %>% arrange(M)
  data$Feature <- factor(data$Feature, levels=fo$Feature)
  data <- data %>% mutate(
    Display_Value=if_else(Feature %in% must_log_features, log1p(Feature_Value), Feature_Value)
  ) %>% group_by(Feature) %>% mutate(Color_Value=rescale(Display_Value,to=c(-1,1))) %>% ungroup()
  ggplot(data, aes(SHAP, Feature, color=Color_Value)) +
    geom_quasirandom(groupOnX=FALSE, alpha=0.60, size=0.9, width=0.3) +
    geom_vline(xintercept=0, linetype="dashed", linewidth=0.5, color="grey45") +
    scale_color_gradientn(
      colors=c("#66C2FF","#00FF00","#fd8d3c","#fd9d9c","#e31a1c"),
      name="Feature value", breaks=c(-1,1), labels=c("Low","High")
    ) +
    guides(color=guide_colorbar(title.position="top", barheight=unit(2.5,"cm"), barwidth=unit(0.45,"cm"))) +
    labs(title=outcome_name, x="SHAP value", y=NULL) +
    theme_bw() +
    theme(plot.title=element_text(hjust=0.5,size=25,face="bold",margin=margin(b=2)),
          plot.title.position="plot", axis.title.x=element_text(size=20,margin=margin(t=4)),
          axis.text.y=element_text(size=17,color="black",lineheight=0.92),
          axis.text.x=element_text(size=16,color="black"),
          panel.grid.major.y=element_blank(), panel.grid.minor=element_blank(),
          panel.grid.major.x=element_line(color="grey90",linewidth=0.2),
          panel.border=element_rect(color="grey90",linewidth=0.4),
          plot.margin=margin(t=4,r=6,b=4,l=8),
          panel.background=element_rect(fill="#F5F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA),
          legend.position=c(0.8,0.15), legend.background=element_rect(fill=alpha("white",0.1),color=NA),
          legend.title=element_text(size=13,face="bold"), legend.text=element_text(size=11))
}

mk_dependence <- function(data, outcome_name, feature_name) {
  if(is.null(data)) return(NULL)
  pd <- data %>% filter(Feature==feature_name)
  if(nrow(pd)==0) return(NULL)
  logf <- feature_name %in% must_log_features
  pd$Display_Value <- if(logf) log1p(pd$Feature_Value) else pd$Feature_Value
  q1 <- quantile(pd$Display_Value,0.01,na.rm=TRUE); q99 <- quantile(pd$Display_Value,0.99,na.rm=TRUE)
  pd <- pd %>% filter(Display_Value>=q1, Display_Value<=q99)
  u <- feature_units[feature_name]; if(is.na(u)) u <- ""
  if(u!="") {
    if(logf) { x_lab <- paste0(feature_name," (",u,", log1p-transformed)")
    } else { x_lab <- paste0(feature_name," (",u,")") }
  } else {
    x_lab <- if(logf) paste0(feature_name," (log1p-transformed)") else feature_name
  }
  legend_title <- x_lab
  p <- ggplot(pd, aes(Display_Value, SHAP, color=Display_Value)) +
    geom_point(alpha=0.52, size=1.15) +
    geom_smooth(method="loess",se=TRUE,linewidth=1.0,color="#fee395",fill="#fef9b7",alpha=0.32) +
    geom_hline(yintercept=0,linetype="dashed",linewidth=0.6,color="grey45") +
    scale_color_gradientn(
      colors=c("#66C2FF","#fd9d9c","#e31a1c"), name=legend_title,
      breaks=pretty(pd$Display_Value,n=5), labels=scales::label_number(accuracy=0.01),
      guide=guide_colorbar(frame.colour="white",ticks.colour="black",
                           barheight=unit(2.2,"cm"),barwidth=unit(0.38,"cm"))) +
    labs(title=outcome_name, x=x_lab, y="SHAP value") +
    theme_bw(base_size=15) +
    theme(plot.title=element_text(hjust=0.5,size=24,face="bold",lineheight=0.95,margin=margin(b=2)),
          axis.title.x=element_text(size=18,margin=margin(t=4)),
          axis.title.y=element_text(size=18,margin=margin(r=2)),
          axis.text=element_text(size=15,color="black"),
          legend.position=c(0.88,0.22), legend.justification=c(1,0),
          legend.background=element_rect(fill=alpha("white",0.2),color=NA),
          legend.key.height=unit(2.2,"cm"), legend.key.width=unit(0.38,"cm"),
          legend.title=element_text(size=13,face="bold"), legend.text=element_text(size=11),
          panel.grid.minor=element_blank(), panel.grid.major=element_line(color="grey93",linewidth=0.3),
          panel.border=element_rect(color="grey90",linewidth=0.5),
          plot.margin=margin(t=4,r=10,b=4,l=6),
          panel.background=element_rect(fill="#F4F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA))
  if(feature_name=="Admission Interval") {
    xl <- quantile(pd$Display_Value, probs=c(0.01,0.90), na.rm=TRUE)
    p <- p + coord_cartesian(xlim=xl)
  }
  return(p)
}

# AFP图形敏感性展示：仅改变散点图中显示的记录，不重新拟合模型，
# 也不重新计算SHAP或TreeSHAP interaction值。
OBSERVED_AFP_DEPENDENCE_FILES <- c(
  "Disease Non-Progression"  = "Disease_Non_Progression_Observed_AFP_Only_SHAP_Data.csv",
  "Liver Cirrhosis"          = "Liver_Cirrhosis_Observed_AFP_Only_SHAP_Data.csv",
  "Hepatocellular Carcinoma" = "Hepatocellular_Carcinoma_Observed_AFP_Only_SHAP_Data.csv"
)

OBSERVED_AFP_INTERACTION_FILES <- c(
  "Disease Non-Progression|Plateletcrit|Alpha-fetoprotein" =
    "DNP_Plateletcrit_x_AFP_Observed_AFP_Only_Plot_Data.csv",
  "Disease Non-Progression|C-Reactive Protein|Alpha-fetoprotein" =
    "DNP_CRP_x_AFP_Observed_AFP_Only_Plot_Data.csv",
  "Liver Cirrhosis|Admission Interval|Alpha-fetoprotein" =
    "LC_Admission_Interval_x_AFP_Observed_AFP_Only_Plot_Data.csv",
  "Hepatocellular Carcinoma|Direct Bilirubin|Alpha-fetoprotein" =
    "HCC_Direct_Bilirubin_x_AFP_Observed_AFP_Only_Plot_Data.csv",
  "Hepatocellular Carcinoma|Number of Hospitalizations|Alpha-fetoprotein" =
    "HCC_Number_of_Hospitalizations_x_AFP_Observed_AFP_Only_Plot_Data.csv",
  "Hepatocellular Carcinoma|Gender|Alpha-fetoprotein" =
    "HCC_AFP_x_Gender_Observed_AFP_Only_Plot_Data.csv"
)

read_observed_afp_dependence <- function(outcome_name) {
  fn <- unname(OBSERVED_AFP_DEPENDENCE_FILES[outcome_name])
  if(length(fn) != 1L || is.na(fn)) stop("No observed-AFP dependence file mapped for: ", outcome_name)
  f <- file.path(observed_afp_dependence_dir, fn)
  if(!file.exists(f)) stop("Observed-AFP dependence file not found: ", f)
  x <- fread(f)
  required <- c("Outcome", "AFP_missing_indicator", "AFP_original",
                "Primary_model_AFP_SHAP", "AFP_log1p")
  if(!all(required %in% names(x))) stop("Missing required columns in: ", f)
  if(any(x$AFP_missing_indicator != 0L) || anyNA(x$AFP_original)) {
    stop("The observed-AFP dependence file contains missing/imputed AFP records: ", f)
  }
  if(any(x$Outcome != outcome_name)) stop("Outcome mismatch in: ", f)
  x
}

mk_observed_afp_dependence <- function(outcome_name) {
  pd <- read_observed_afp_dependence(outcome_name)
  pd <- pd[is.finite(AFP_log1p) & is.finite(Primary_model_AFP_SHAP)]
  # 所有真实AFP记录均参与LOESS拟合；坐标窗口只限制显示范围，不删除记录。
  qx <- quantile(pd$AFP_log1p, c(0.001, 0.999), na.rm=TRUE)
  if(!file.exists(observed_afp_summary_file)) {
    stop("Observed-AFP summary file not found: ", observed_afp_summary_file)
  }
  afp_summary <- fread(observed_afp_summary_file)
  summary_row <- afp_summary[Outcome == outcome_name]
  if(nrow(summary_row) != 1L ||
     !is.finite(summary_row$Main_analysis_imputation_value_log1p[1])) {
    stop("Invalid primary AFP imputation value in summary for: ", outcome_name)
  }
  imputation_value_log1p <- summary_row$Main_analysis_imputation_value_log1p[1]
  x_lab <- "Alpha-fetoprotein (ng/mL, log1p-transformed)"
  ggplot(pd, aes(AFP_log1p, Primary_model_AFP_SHAP, color=AFP_log1p)) +
    geom_point(alpha=0.52, size=1.15) +
    geom_smooth(method="loess",se=TRUE,linewidth=1.0,
                color="#fee395",fill="#fef9b7",alpha=0.32) +
    geom_hline(yintercept=0,linetype="dashed",linewidth=0.6,color="grey45") +
    geom_vline(xintercept=imputation_value_log1p,linetype="dotted",
               linewidth=0.65,color="#444444") +
    scale_color_gradientn(
      colors=c("#66C2FF","#fd9d9c","#e31a1c"),
      name="Observed AFP\n(log1p-transformed)",
      breaks=pretty(pd$AFP_log1p,n=5), labels=scales::label_number(accuracy=0.01),
      guide=guide_colorbar(frame.colour="white",ticks.colour="black",
                           barheight=unit(2.2,"cm"),barwidth=unit(0.38,"cm"))) +
    labs(title=outcome_name, x=x_lab, y="SHAP value") +
    coord_cartesian(xlim=qx) +
    theme_bw(base_size=15) +
    theme(plot.title=element_text(hjust=0.5,size=24,face="bold",lineheight=0.95,margin=margin(b=2)),
          axis.title.x=element_text(size=18,margin=margin(t=4)),
          axis.title.y=element_text(size=18,margin=margin(r=2)),
          axis.text=element_text(size=15,color="black"),
          legend.position=c(0.88,0.22), legend.justification=c(1,0),
          legend.background=element_rect(fill=alpha("white",0.2),color=NA),
          legend.key.height=unit(2.2,"cm"), legend.key.width=unit(0.38,"cm"),
          legend.title=element_text(size=13,face="bold"), legend.text=element_text(size=11),
          panel.grid.minor=element_blank(), panel.grid.major=element_line(color="grey93",linewidth=0.3),
          panel.border=element_rect(color="grey90",linewidth=0.5),
          plot.margin=margin(t=4,r=10,b=4,l=6),
          panel.background=element_rect(fill="#F4F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA))
}

read_observed_afp_interaction <- function(outcome_name, feat_x, feat_col) {
  key <- paste(outcome_name, feat_x, feat_col, sep="|")
  fn <- unname(OBSERVED_AFP_INTERACTION_FILES[key])
  if(length(fn) != 1L || is.na(fn)) stop("No observed-AFP interaction file mapped for: ", key)
  f <- file.path(observed_afp_interaction_dir, fn)
  if(!file.exists(f)) stop("Observed-AFP interaction file not found: ", f)
  x <- fread(f)
  required <- c("Outcome", "Interaction", "AFP_original", "Display_X",
                "Display_Color", "Primary_Global_MeanAbsInteraction")
  if(!all(required %in% names(x))) stop("Missing required columns in: ", f)
  if(anyNA(x$AFP_original) || any(x$Outcome != outcome_name)) {
    stop("Observed-AFP interaction file failed AFP/outcome validation: ", f)
  }
  strengths <- unique(x$Primary_Global_MeanAbsInteraction[is.finite(x$Primary_Global_MeanAbsInteraction)])
  if(length(strengths) != 1L) stop("Expected one primary global interaction strength in: ", f)
  x
}

mk_observed_afp_interaction <- function(outcome_name, feat_x, feat_col) {
  plot_dt <- read_observed_afp_interaction(outcome_name, feat_x, feat_col)
  mai <- unique(plot_dt$Primary_Global_MeanAbsInteraction)[1]
  sl <- paste0("Primary global mean |Interaction| = ", sprintf("%.4f", mai))

  # AFP × Gender文件按AFP作为横轴、Gender作为分组变量保存。
  if(outcome_name==HCC && feat_x=="Gender" && feat_col=="Alpha-fetoprotein") {
    plot_dt[, Gender_Group:=factor(Display_Color,levels=c(1,2),labels=c("Male","Female"))]
    if(anyNA(plot_dt$Gender_Group)) stop("Unexpected Gender coding in observed-AFP interaction data.")
    return(
      ggplot(plot_dt,aes(Display_X,Interaction,color=Gender_Group)) +
        geom_point(alpha=0.52,size=1.0) +
        geom_smooth(aes(group=Gender_Group,fill=Gender_Group),
                    method="loess",se=TRUE,linewidth=1.1,alpha=0.15) +
        geom_hline(yintercept=0,linetype="dashed",linewidth=0.6,color="grey45") +
        scale_color_manual(values=c("Male"="#ade219","Female"="#e91a1c"),name="Gender") +
        scale_fill_manual(values=c("Male"="#ade219","Female"="#e91a1c"),guide="none") +
        labs(title=outcome_name,subtitle=sl,
             x="Alpha-fetoprotein (ng/mL, log1p-transformed)",
             y="SHAP Interaction Value") +
        theme_bw(base_size=15) +
        theme(plot.title=element_text(hjust=0.5,size=24,face="bold",lineheight=0.95,margin=margin(b=2)),
              plot.subtitle=element_text(hjust=0.5,size=13,face="bold",color="grey30",margin=margin(b=2)),
              axis.title.x=element_text(size=18,margin=margin(t=2)),
              axis.title.y=element_text(size=18,margin=margin(r=2)),
              axis.text=element_text(size=15,color="black"),
              legend.position=c(0.82,0.22),legend.justification=c(1,0),
              legend.background=element_rect(fill=alpha("white",0.2),color=NA),
              legend.title=element_text(size=13,face="bold"),legend.text=element_text(size=11),
              panel.grid.minor=element_blank(),panel.grid.major=element_line(color="grey93",linewidth=0.3),
              panel.border=element_rect(color="grey90",linewidth=0.5),
              plot.margin=margin(t=4,r=10,b=4,l=6),
              panel.background=element_rect(fill="#F4F8FB",color=NA),
              plot.background=element_rect(fill="white",color=NA))
    )
  }

  ux <- feature_units[feat_x]; if(is.na(ux)) ux <- ""
  xl <- if(ux!="") paste0(feat_x," (",ux,")") else feat_x
  cl <- "Observed AFP\n(log1p-transformed)"
  ggplot(plot_dt, aes(Display_X, Interaction, color=Display_Color)) +
    geom_point(alpha=0.52, size=1.1) +
    geom_smooth(method="loess",se=TRUE,linewidth=1.3,color="#71b8ed",fill="white",alpha=0.32) +
    geom_hline(yintercept=0,linetype="dashed",linewidth=0.6,color="grey45") +
    scale_color_gradientn(colors=c("#ade219","#fea766","#e91a1c"), name=cl,
                          breaks=pretty(plot_dt$Display_Color,n=5), labels=label_number(accuracy=0.01),
                          guide=guide_colorbar(frame.colour="white",ticks.colour="black",
                                               barheight=unit(2.2,"cm"),barwidth=unit(0.38,"cm"))) +
    labs(title=outcome_name, subtitle=sl, x=xl, y="SHAP Interaction Value") +
    theme_bw(base_size=15) +
    theme(plot.title=element_text(hjust=0.5,size=24,face="bold",lineheight=0.95,margin=margin(b=2)),
          plot.subtitle=element_text(hjust=0.5,size=13,face="bold",color="grey30",margin=margin(b=2)),
          axis.title.x=element_text(size=18,margin=margin(t=2)),
          axis.title.y=element_text(size=18,margin=margin(r=2)),
          axis.text=element_text(size=15,color="black"),
          legend.position=c(0.96,0.08), legend.justification=c(1,0),
          legend.background=element_rect(fill=alpha("white",0.2),color=NA),
          legend.title=element_text(size=10,face="bold",lineheight=0.95),
          legend.text=element_text(size=9),
          panel.grid.minor=element_blank(), panel.grid.major=element_line(color="grey93",linewidth=0.3),
          panel.border=element_rect(color="grey90",linewidth=0.5),
          plot.margin=margin(t=4,r=10,b=4,l=6),
          panel.background=element_rect(fill="#F4F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA))
}

mk_interaction <- function(inter_data, outcome_name, feat_x, feat_col) {
  if(is.null(inter_data)) return(NULL)
  inter_arr <- inter_data$array; raw_data <- inter_data$raw
  dn <- dimnames(inter_arr)[[2]]
  ix <- which(dn==feat_x); ic <- which(dn==feat_col)
  if(length(ix)==0||length(ic)==0) return(NULL)
  inter_values <- inter_arr[,ix,ic]
  val_x <- raw_data[[feat_x]]; val_col <- raw_data[[feat_col]]
  plot_dt <- data.table(X=val_x,Color=val_col,Interaction=inter_values)
  # 交互强度使用全部记录计算；后续1%–99%筛选只限制可视化范围。
  mai <- mean(abs(inter_values),na.rm=TRUE)
  sl <- paste0("Mean |Interaction| = ",sprintf("%.4f",mai))
  lx <- feat_x%in%must_log_features; lc <- feat_col%in%must_log_features
  if(lx) plot_dt[,DX:=log1p(X)] else plot_dt[,DX:=X]
  if(lc) plot_dt[,DC:=log1p(Color)] else plot_dt[,DC:=Color]
  qx <- quantile(plot_dt$DX,c(0.01,0.99),na.rm=TRUE)
  qc <- quantile(plot_dt$DC,c(0.01,0.99),na.rm=TRUE)
  plot_dt <- plot_dt[DX>=qx[1]&DX<=qx[2]&DC>=qc[1]&DC<=qc[2]]
  ux <- feature_units[feat_x]; if(is.na(ux)) ux <- ""
  uc <- feature_units[feat_col]; if(is.na(uc)) uc <- ""
  if(lx) { xl <- if(ux!="") paste0(feat_x," (",ux,", log1p)") else paste0(feat_x," (log1p)") }
  else   { xl <- if(ux!="") paste0(feat_x," (",ux,")") else feat_x }
  if(lc) { cl <- if(uc!="") paste0(feat_col," (",uc,", log1p)") else paste0(feat_col," (log1p)") }
  else   { cl <- if(uc!="") paste0(feat_col," (",uc,")") else feat_col }

  # Gender × AFP：Gender是二分类变量（1=Male，2=Female）。
  # 保留Male/Female同图叠加形式，仅将配色统一为绿色和红色。
  if(feat_x=="Gender" && feat_col=="Alpha-fetoprotein") {
    plot_dt[,Gender_Group:=factor(X,levels=c(1,2),labels=c("Male","Female"))]
    return(
      ggplot(plot_dt,aes(DC,Interaction,color=Gender_Group)) +
        geom_point(alpha=0.52,size=1.0) +
        geom_smooth(aes(group=Gender_Group,fill=Gender_Group),
                    method="loess",se=TRUE,linewidth=1.1,alpha=0.15) +
        geom_hline(yintercept=0,linetype="dashed",linewidth=0.6,color="grey45") +
        scale_color_manual(values=c("Male"="#ade219","Female"="#e91a1c"),name="Gender") +
        scale_fill_manual(values=c("Male"="#ade219","Female"="#e91a1c"),guide="none") +
        labs(title=outcome_name,subtitle=sl,
             x="Alpha-fetoprotein (ng/mL, log1p-transformed)",
             y="SHAP Interaction Value") +
        theme_bw(base_size=15) +
        theme(plot.title=element_text(hjust=0.5,size=24,face="bold",lineheight=0.95,margin=margin(b=2)),
              plot.subtitle=element_text(hjust=0.5,size=13,face="bold",color="grey30",margin=margin(b=2)),
              axis.title.x=element_text(size=18,margin=margin(t=2)),
              axis.title.y=element_text(size=18,margin=margin(r=2)),
              axis.text=element_text(size=15,color="black"),
              legend.position=c(0.82,0.22),legend.justification=c(1,0),
              legend.background=element_rect(fill=alpha("white",0.2),color=NA),
              legend.title=element_text(size=13,face="bold"),legend.text=element_text(size=11),
              panel.grid.minor=element_blank(),panel.grid.major=element_line(color="grey93",linewidth=0.3),
              panel.border=element_rect(color="grey90",linewidth=0.5),
              plot.margin=margin(t=4,r=10,b=4,l=6),
              panel.background=element_rect(fill="#F4F8FB",color=NA),
              plot.background=element_rect(fill="white",color=NA))
    )
  }
  ggplot(plot_dt, aes(DX, Interaction, color=DC)) +
    geom_point(alpha=0.52, size=1.1) +
    geom_smooth(method="loess",se=TRUE,linewidth=1.3,color="#71b8ed",fill="white",alpha=0.32) +
    geom_hline(yintercept=0,linetype="dashed",linewidth=0.6,color="grey45") +
    scale_color_gradientn(colors=c("#ade219","#fea766","#e91a1c"), name=cl,
                          breaks=pretty(plot_dt$DC,n=5), labels=label_number(accuracy=0.01),
                          guide=guide_colorbar(frame.colour="white",ticks.colour="black",
                                               barheight=unit(2.2,"cm"),barwidth=unit(0.38,"cm"))) +
    labs(title=outcome_name, subtitle=sl, x=xl, y="SHAP Interaction Value") +
    theme_bw(base_size=15) +
    theme(plot.title=element_text(hjust=0.5,size=24,face="bold",lineheight=0.95,margin=margin(b=2)),
          plot.subtitle=element_text(hjust=0.5,size=13,face="bold",color="grey30",margin=margin(b=2)),
          axis.title.x=element_text(size=18,margin=margin(t=2)),
          axis.title.y=element_text(size=18,margin=margin(r=2)),
          axis.text=element_text(size=15,color="black"),
          legend.position=c(0.82,0.22), legend.justification=c(1,0),
          legend.background=element_rect(fill=alpha("white",0.2),color=NA),
          legend.key.height=unit(2.2,"cm"), legend.key.width=unit(0.38,"cm"),
          legend.title=element_text(size=13,face="bold"), legend.text=element_text(size=11),
          panel.grid.minor=element_blank(), panel.grid.major=element_line(color="grey93",linewidth=0.3),
          panel.border=element_rect(color="grey90",linewidth=0.5),
          plot.margin=margin(t=4,r=10,b=4,l=6),
          panel.background=element_rect(fill="#F4F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA))
}

mk_waterfall <- function(wf_data, outcome_name, risk_type, sample_id) {
  if(is.null(wf_data)) return(NULL)
  sm <- wf_data$shap; si <- wf_data$samples
  sv <- as.numeric(sm[sample_id,]); names(sv) <- colnames(sm)
  bv <- sv["BIAS"]; sv <- sv[names(sv)!="BIAS"]
  pdf <- data.frame(Feature=names(sv),SHAP=as.numeric(sv)) %>%
    mutate(absSHAP=abs(SHAP)) %>% arrange(desc(absSHAP)) %>% slice_head(n=10)
  pdf$Feature <- str_replace_all(pdf$Feature, c(
    "Red Blood Cell Distribution Width - Coefficient of Variation \\(RDW-CV\\)
"="RDW-CV"))
  pdf <- pdf %>% mutate(Unit=feature_units[Feature], Unit=ifelse(is.na(Unit),"",Unit),
                        Feature_Label=ifelse(Unit=="",Feature,paste0(Feature,"\n(",Unit,")"))) %>%
    arrange(desc(abs(SHAP))) %>%
    mutate(cum=bv+cumsum(SHAP), start=lag(cum,default=bv), end=cum,
           Direction=ifelse(SHAP>=0,"Positive","Negative"))
  pdf$Feature_Label <- factor(pdf$Feature_Label, levels=rev(pdf$Feature_Label))
  rr <- si[Sample_ID==sample_id]
  pp <- if(nrow(rr)>0&&"Predicted_Probability"%in%names(rr)) round(as.numeric(rr$Predicted_Probability[1]),3) else NA
  ggplot(pdf) +
    geom_rect(aes(xmin=start,xmax=end,ymin=as.numeric(Feature_Label)-0.38,
                  ymax=as.numeric(Feature_Label)+0.38,fill=Direction),alpha=0.95) +
    geom_text(aes(x=end,y=Feature_Label,label=round(SHAP,3)),
              size=4.5, hjust=ifelse(pdf$SHAP>0,-0.1,1.11)) +
    geom_vline(xintercept=bv,linetype="dashed",linewidth=0.6,color="grey45") +
    scale_fill_manual(values=c("Positive"="#2b8cbe","Negative"="skyblue")) +
    scale_x_continuous(expand=expansion(mult=c(0.1,0.1))) +
    labs(title=paste0(outcome_name,"\n",risk_type," Risk Sample"),
         subtitle=if(!is.na(pp)) paste0("Predicted probability: ",pp) else NULL,
         x="SHAP value", y=NULL) +
    theme_bw(base_size=15) +
    theme(plot.title=element_text(hjust=0.5,size=24,face="bold"),
          plot.subtitle=element_text(hjust=0.5,size=18),
          axis.text=element_text(color="black"), legend.position="none",
          plot.margin=margin(t=2,r=10,b=2,l=2),
          panel.background=element_rect(fill="#F4F8FB",color=NA),
          plot.background=element_rect(fill="white",color=NA),
          panel.grid.major.y=element_blank())
}

##############################################################################
# 标签工具
##############################################################################
TAG_THEME <- theme(plot.tag.position = c(0.02, 0.98),
                   plot.tag = element_text(size = 22, face = "bold"))
add_tag <- function(p, tag) {
  if(is.null(p)) return(NULL)
  p + labs(tag = tag) + TAG_THEME
}
pick <- function(lst, key) { if(key %in% names(lst)) lst[[key]] else NULL }

save_figure <- function(fig, name, w, h) {
  base <- file.path(output_dir, name)
  ggsave(paste0(base,".tiff"), fig, width=w, height=h, dpi=600, compression="lzw", bg="white")
  ggsave(paste0(base,".png"),  fig, width=w, height=h, dpi=600, bg="white")
  ggsave(paste0(base,".pdf"),  fig, width=w, height=h, device="pdf", bg="white")
  if(require("svglite", quietly=TRUE)) {
    ggsave(paste0(base,".svg"), fig, width=w, height=h, device="svg", bg="white")
  }
  cat("  ->", basename(base), "\n")
}

##############################################################################
INTER_PAIRS <- list(
  "Disease Non-Progression"=list(
    c("Absolute Lymphocyte Count","Platelet Count"),
    c("Number of Hospitalizations","Platelet Count"),
    c("Plateletcrit","Alpha-fetoprotein"),
    c("C-Reactive Protein","Alpha-fetoprotein")),
  "Liver Cirrhosis"=list(
    c("Plateletcrit","Platelet Count"),
    c("Absolute Lymphocyte Count","Platelet Count"),
    c("Admission Interval","Alpha-fetoprotein"),
    c("Cholinesterase","Plateletcrit")),
  "Hepatocellular Carcinoma"=list(
    c("Direct Bilirubin","Alpha-fetoprotein"),
    c("Number of Hospitalizations","Alpha-fetoprotein"),
    c("Number of Hospitalizations","C-Reactive Protein"),
    c("Gender","Alpha-fetoprotein")))

##############################################################################
# Figure 1: cowplot 整图对齐 — 前3行3列 + 后3行4列
##############################################################################
generate_figure1 <- function() {
  cat("\n===== Figure 1: 6-row (cowplot align='none') =====\n")
  key_dep <- c("Disease Non-Progression"="Platelet Count",
               "Liver Cirrhosis"="Plateletcrit",
               "Hepatocellular Carcinoma"="Alpha-fetoprotein")
  imp <- list(); bee <- list(); dep <- list(); inter_all <- list()
  for(d in names(OUTCOME_DIRS)) {
    nm <- OUTCOME_DIRS[[d]]; cat("  loading:",nm,"\n")
    imp[[nm]]  <- mk_importance(read_shap_imp(d), nm)
    bee[[nm]]  <- mk_beeswarm(read_beeswarm(d), nm)
    dep[[nm]]  <- if(key_dep[nm] == "Alpha-fetoprotein") {
      mk_observed_afp_dependence(nm)
    } else {
      mk_dependence(read_beeswarm(d), nm, key_dep[nm])
    }
    arr <- read_inter_arr(d)
    for(i in seq_along(INTER_PAIRS[[nm]])) {
      pr <- INTER_PAIRS[[nm]][[i]]
      inter_all[[paste0(nm,"_",i)]] <- if("Alpha-fetoprotein" %in% pr) {
        mk_observed_afp_interaction(nm, pr[1], pr[2])
      } else {
        mk_interaction(arr, nm, pr[1], pr[2])
      }
    }
  }
  
  # 前3行 (3列) — cowplot 整图对齐
  top_list <- list(
    add_tag(pick(imp,DNP),"a"), add_tag(pick(imp,LC),NULL), add_tag(pick(imp,HCC),NULL),
    add_tag(pick(bee,DNP),"b"), add_tag(pick(bee,LC),NULL), add_tag(pick(bee,HCC),NULL),
    add_tag(pick(dep,DNP),"c"), add_tag(pick(dep,LC),NULL), add_tag(pick(dep,HCC),NULL)
  )
  top_list <- top_list[!sapply(top_list, is.null)]
  top_grid <- cowplot::plot_grid(plotlist = top_list, ncol = 3, align = "none")
  
  # 后3行 (4列)
  dnp_int <- lapply(1:4, function(i) pick(inter_all,paste0(DNP,"_",i)))
  lc_int  <- lapply(1:4, function(i) pick(inter_all,paste0(LC,"_",i)))
  hcc_int <- lapply(1:4, function(i) pick(inter_all,paste0(HCC,"_",i)))
  bot_list <- c(
    add_tag(dnp_int[[1]],"d"), add_tag(dnp_int[[2]],NULL), add_tag(dnp_int[[3]],NULL), add_tag(dnp_int[[4]],NULL),
    add_tag(lc_int[[1]], "e"), add_tag(lc_int[[2]], NULL), add_tag(lc_int[[3]], NULL), add_tag(lc_int[[4]], NULL),
    add_tag(hcc_int[[1]],"f"), add_tag(hcc_int[[2]],NULL), add_tag(hcc_int[[3]],NULL), add_tag(hcc_int[[4]],NULL)
  )
  bot_list <- bot_list[!sapply(bot_list, is.null)]
  bot_grid <- cowplot::plot_grid(plotlist = bot_list, ncol = 4, align = "none")
  
  # 上下叠加：cowplot 自动匹配宽度
  fig <- cowplot::plot_grid(top_grid, bot_grid, ncol = 1, align = "v", axis = "lr",
                            rel_heights = c(3, 3))
  save_figure(fig, "Figure1_Combined_6Row", 20, 34)
  fig
}
##############################################################################
# Figure S1: 瀑布图
##############################################################################
generate_figureS1 <- function() {
  cat("\n===== Figure S1: Waterfall =====\n")
  risks <- c("Low","Median","High")
  all <- list()
  for(d in names(OUTCOME_DIRS)) {
    nm <- OUTCOME_DIRS[[d]]; cat("  ",nm,"\n")
    wf <- read_waterfall(d)
    if(is.null(wf)) next
    for(r in risks) {
      rr <- wf$samples %>% filter(str_detect(Sample_Type,r)) %>% slice_head(n=1)
      if(nrow(rr)>0) all[[paste0(nm,"_",r)]] <- mk_waterfall(wf,nm,r,rr$Sample_ID[1])
    }
  }
  ordered <- list()
  for(r in risks) for(nm in c(DNP,LC,HCC)) {
    k <- paste0(nm,"_",r); if(k %in% names(all)) ordered[[k]] <- all[[k]]
  }
  tags <- letters[1:length(ordered)]
  plotlist <- list()
  for(i in seq_along(ordered)) plotlist[[i]] <- add_tag(ordered[[i]], tags[i])
  fig <- cowplot::plot_grid(plotlist = plotlist, ncol = 3, align = "none")
  save_figure(fig, "FigureS1_Waterfall", 22, 18)
  fig
}

##############################################################################
# Figure S2: 剩余依赖图
##############################################################################
generate_figureS2 <- function() {
  cat("\n===== Figure S2: Dependence Remaining =====\n")
  secondary <- list(
    "Disease Non-Progression"  = c("Alpha-fetoprotein","Plateletcrit","Number of Hospitalizations"),
    "Liver Cirrhosis"          = c("Platelet Count","Cholinesterase","Alpha-fetoprotein"),
    "Hepatocellular Carcinoma" = c("Number of Hospitalizations","C-Reactive Protein","Admission Interval")
  )
  all <- list()
  for(d in names(OUTCOME_DIRS)) {
    nm <- OUTCOME_DIRS[[d]]; cat("  ",nm,"\n")
    bee <- read_beeswarm(d)
    for(f in secondary[[nm]]) {
      p <- if(f == "Alpha-fetoprotein") {
        mk_observed_afp_dependence(nm)
      } else {
        mk_dependence(bee, nm, f)
      }
      if(!is.null(p)) all[[paste0(nm,"_",f)]] <- p
    }
  }
  ordered <- list()
  for(nm in c(DNP,LC,HCC)) for(f in secondary[[nm]]) {
    k <- paste0(nm,"_",f); if(k %in% names(all)) ordered[[k]] <- all[[k]]
  }
  plotlist <- list()
  plotlist[[1]] <- add_tag(ordered[[1]], "a"); for(i in 2:3) plotlist[[i]] <- add_tag(ordered[[i]], NULL)
  plotlist[[4]] <- add_tag(ordered[[4]], "b"); for(i in 5:6) plotlist[[i]] <- add_tag(ordered[[i]], NULL)
  plotlist[[7]] <- add_tag(ordered[[7]], "c"); for(i in 8:9) plotlist[[i]] <- add_tag(ordered[[i]], NULL)
  fig <- cowplot::plot_grid(plotlist = plotlist, ncol = 3, align = "none")
  save_figure(fig, "FigureS2_Dependence_Remaining", 19, 17)
  fig
}


tryCatch(generate_figure1(),  error=function(e) cat("FIG1 ERROR:",e$message,"\n"))
tryCatch(generate_figureS1(), error=function(e) cat("FIG S1 ERROR:",e$message,"\n"))
tryCatch(generate_figureS2(), error=function(e) cat("FIG S2 ERROR:",e$message,"\n"))

fs <- list.files(output_dir, pattern="\\.(tiff|png|pdf|svg)$", full.names=TRUE)
cat("\nDone.",length(fs),"files\n")
for(f in fs) cat("  ",basename(f),"\n")
##############################################################################
# END
##############################################################################


