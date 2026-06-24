###############################################################
# COMPLETE VALIDATION: ORIGINAL + TOP15 + INTERACTION MODELS
# Total Figure: 3 Outcomes × 3 Validation Views (Cal/DCA/Boot)
#   Each panel overlays 3 models for direct comparison
###############################################################

library(data.table)
library(dplyr)
library(ggplot2)
library(pROC)
library(patchwork)
library(grid)
library(gridExtra)

set.seed(123)

# ============================================================
# 1. FILE PATHS
# ============================================================
pred_dir <- "J:/rtools45/手工4乘积交互Model_Comparison_Independent_Tuning/Prediction_Files"

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
  Original    = list(files = original_files,    label = "Original",    color = "#E41A1C"),
  Reduced     = list(files = top15_files,       label = "Reduced",     color = "#377EB8"),
  Interaction = list(files = interaction_files, label = "Interaction", color = "#4DAF4A")
)

outcome_order <- c("Outcome_Disease_Non_Progression_",
                   "Outcome_Liver_Cirrhosis_",
                   "Outcome_Hepatocellular_Carcinoma_")

outdir <- "J:/rtools45/Validation_Analysis"
if(!dir.exists(outdir)) dir.create(outdir, recursive = TRUE)

# ============================================================
# 2. UTILITIES
# ============================================================
clean_outcome <- function(x){
  x <- gsub("Outcome_", "", x)
  x <- gsub("_", " ", x)
  x <- gsub("Disease Non Progression", "Disease Non-Progression", x)
  x
}

theme_tj <- function(base_size = 12){
  theme_bw(base_size = base_size) +
    theme(
      plot.title   = element_text(face = "bold", hjust = .5, size = base_size + 1),
      panel.grid   = element_blank(),
      legend.title = element_blank(),
      legend.position = "bottom",
      legend.text  = element_text(size = base_size - 2)
    )
}

# ============================================================
# 3. MAIN LOOP: Store ALL intermediate data
# ============================================================
summary_table   <- data.frame()
bootstrap_table <- data.frame()

# Nested storage: cal[[model]][[outcome]], dca[[model]][[outcome]], etc.
cal_store   <- list()
dca_store   <- list()
boot_store  <- list()
app_auc_store <- list()

for(model_name in names(all_models)){
  files        <- all_models[[model_name]]$files
  model_label  <- all_models[[model_name]]$label
  
  for(outcome_name in outcome_order){
    
    pred <- fread(files[[outcome_name]])
    y <- pred$True_Label
    p <- pred$Predicted_Prob
    
    # ----- Calibration metrics -----
    roc_obj   <- roc(y, p, quiet = TRUE)
    auc_value <- as.numeric(auc(roc_obj))
    ci_auc    <- ci.auc(roc_obj)
    brier     <- mean((p - y)^2)
    
    eps     <- 1e-10
    logit_p <- log((p + eps) / (1 - p + eps))
    cal_model <- glm(y ~ logit_p, family = binomial())
    intercept  <- coef(cal_model)[1]
    slope      <- coef(cal_model)[2]
    
    # ----- Bootstrap (B = 2000) -----
    n_boot   <- 2000
    boot_auc <- numeric(n_boot)
    for(i in seq_len(n_boot)){
      idx <- sample(seq_along(y), replace = TRUE)
      boot_auc[i] <- as.numeric(auc(roc(y[idx], p[idx], quiet = TRUE)))
    }
    boot_mean  <- mean(boot_auc)
    boot_ci    <- quantile(boot_auc, c(.025, .975))
    boot_sd    <- sd(boot_auc)
    optimism   <- auc_value - boot_mean
    
    # ----- Summary table -----
    summary_table <- rbind(summary_table, data.frame(
      Model = model_label, Outcome = clean_outcome(outcome_name),
      AUC = round(auc_value, 4), AUC_Lower = round(ci_auc[1], 4),
      AUC_Upper = round(ci_auc[3], 4), Brier = round(brier, 4),
      Intercept = round(intercept, 4), Slope = round(slope, 4),
      stringsAsFactors = FALSE
    ))
    bootstrap_table <- rbind(bootstrap_table, data.frame(
      Model = model_label, Outcome = clean_outcome(outcome_name),
      Apparent_AUC = round(auc_value, 4), Bootstrap_AUC = round(boot_mean, 4),
      CI_Lower = round(boot_ci[1], 4), CI_Upper = round(boot_ci[2], 4),
      Bootstrap_SD = round(boot_sd, 4), Optimism = round(optimism, 4),
      stringsAsFactors = FALSE
    ))
    
    # ----- Store calibration decile data -----
    cal_df <- data.frame(Observed = y, Predicted = p)
    cal_df$Decile <- ntile(cal_df$Predicted, 10)
    cal_decile <- cal_df %>%
      group_by(Decile) %>%
      summarise(Predicted = mean(Predicted), Observed = mean(Observed), .groups = "drop") %>%
      mutate(Model = model_label)
    cal_store[[model_name]][[outcome_name]] <- cal_decile
    
    # ----- Store DCA data (model only) -----
    thresholds <- seq(.01, .50, .01)
    prevalence <- mean(y)
    dca_res <- data.frame()
    for(t in thresholds){
      pred_pos <- p > t
      TP <- sum(pred_pos & y == 1)
      FP <- sum(pred_pos & y == 0)
      N  <- length(y)
      model_nb <- TP / N - FP / N * t / (1 - t)
      dca_res <- rbind(dca_res, data.frame(Threshold = t, NetBenefit = model_nb, Model = model_label))
    }
    dca_store[[model_name]][[outcome_name]] <- dca_res
    
    # ----- Store bootstrap AUC -----
    boot_store[[model_name]][[outcome_name]] <- data.frame(AUC = boot_auc, Model = model_label)
    app_auc_store[[model_name]][[outcome_name]] <- auc_value
    
    cat(sprintf("  [%s] %s: AUC=%.4f, Brier=%.4f\n",
                model_label, clean_outcome(outcome_name), auc_value, brier))
  }
}

# ============================================================
# 4. BUILD TOTAL FIGURE
#    3 rows (Calibration, DCA, Bootstrap) × 3 columns (NP, LC, HCC)
#    Each panel overlays 3 models
# ============================================================

# Collect DCA "Treat All" / "Treat None" from Original model
dca_ref_store <- list()
for(outcome_name in outcome_order){
  pred <- fread(all_models$Original$files[[outcome_name]])
  y <- pred$True_Label
  prevalence <- mean(y)
  dca_ref <- data.frame()
  for(t in seq(.01, .50, .01)){
    all_nb <- prevalence - (1 - prevalence) * t / (1 - t)
    dca_ref <- rbind(dca_ref, data.frame(Threshold = t, NetBenefit = all_nb, Model = "Treat All"))
  }
  dca_ref <- rbind(dca_ref, data.frame(Threshold = seq(.01, .50, .01), NetBenefit = 0, Model = "Treat None"))
  dca_ref_store[[outcome_name]] <- dca_ref
}

# Build panels column-major: Row1=Cal(NP,LC,HCC), Row2=DCA(NP,LC,HCC), Row3=Boot(NP,LC,HCC)
panel_list <- list()
tag_letters <- letters[1:9]

cal_panels  <- list()
dca_panels  <- list()
boot_panels <- list()

for(i in seq_along(outcome_order)){
  outcome_name <- outcome_order[i]
  oc_clean <- clean_outcome(outcome_name)
  tag_cal  <- tag_letters[i]
  tag_dca  <- tag_letters[i + 3]
  tag_boot <- tag_letters[i + 6]
  
  # ---- Panel: Calibration ----
  cal_all <- bind_rows(
    cal_store$Original[[outcome_name]],
    cal_store$Reduced[[outcome_name]],
    cal_store$Interaction[[outcome_name]]
  )
  p_cal <- ggplot(cal_all, aes(Predicted, Observed, color = Model)) +
    geom_point(size = 4) +
    geom_line(linewidth = 1.4) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey50") +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    scale_color_manual(values = c("Original" = "#E41A1C", "Reduced" = "#377EB8", "Interaction" = "#4DAF4A")) +
    labs(title = oc_clean, x = "Predicted", y = "Observed", tag = tag_cal) +
    theme_tj(base_size = 22)
  cal_panels[[i]] <- p_cal
  
  # ---- Panel: DCA ----
  dca_all <- bind_rows(
    dca_store$Original[[outcome_name]],
    dca_store$Reduced[[outcome_name]],
    dca_store$Interaction[[outcome_name]],
    dca_ref_store[[outcome_name]]
  )
  p_dca <- ggplot(dca_all, aes(Threshold, NetBenefit, color = Model, linetype = Model)) +
    geom_line(linewidth = 1.4) +
    coord_cartesian(xlim = c(0, .5)) +
    scale_color_manual(values = c(
      "Original" = "#E41A1C", "Reduced" = "#377EB8", "Interaction" = "#4DAF4A",
      "Treat All" = "black", "Treat None" = "grey40"
    )) +
    scale_linetype_manual(values = c(
      "Original" = "solid", "Reduced" = "solid", "Interaction" = "solid",
      "Treat All" = "dashed", "Treat None" = "dotted"
    )) +
    labs(title = oc_clean, x = "Threshold", y = "Net Benefit", tag = tag_dca) +
    theme_tj(base_size = 22) +
    guides(color = guide_legend(nrow = 2))
  dca_panels[[i]] <- p_dca
  
  # ---- Panel: Bootstrap ----
  boot_all <- bind_rows(
    boot_store$Original[[outcome_name]],
    boot_store$Reduced[[outcome_name]],
    boot_store$Interaction[[outcome_name]]
  )
  app_lines <- data.frame(
    Model = c("Original", "Reduced", "Interaction"),
    AUC   = c(app_auc_store$Original[[outcome_name]],
              app_auc_store$Reduced[[outcome_name]],
              app_auc_store$Interaction[[outcome_name]])
  )
  p_boot <- ggplot(boot_all, aes(AUC, fill = Model)) +
    geom_density(alpha = 0.35) +
    geom_vline(data = app_lines, aes(xintercept = AUC, color = Model), linewidth = 1.4) +
    scale_fill_manual(values = c("Original" = "#E41A1C", "Reduced" = "skyblue", "Interaction" = "#4DAF4A")) +
    scale_color_manual(values = c("Original" = "#E41A1C", "Reduced" = "skyblue", "Interaction" = "#4DAF4A")) +
    labs(title = oc_clean, x = "Bootstrap AUC", y = "Density", tag = tag_boot) +
    theme_tj(base_size = 22) +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
  boot_panels[[i]] <- p_boot
}

# ---- Row labels (left side) ----
row_cal  <- grid::textGrob("Calibration",        rot = 90, gp = gpar(fontface = "bold", fontsize = 22))
row_dca  <- grid::textGrob("Decision Curve",      rot = 90, gp = gpar(fontface = "bold", fontsize = 22))
row_boot <- grid::textGrob("Bootstrap Stability", rot = 90, gp = gpar(fontface = "bold", fontsize = 22))

total_figure <- (
  wrap_elements(row_cal) +
    cal_panels[[1]]  + cal_panels[[2]]  + cal_panels[[3]] +
    wrap_elements(row_dca) +
    dca_panels[[1]]  + dca_panels[[2]]  + dca_panels[[3]] +
    wrap_elements(row_boot) +
    boot_panels[[1]] + boot_panels[[2]] + boot_panels[[3]] +
    plot_layout(nrow = 3, byrow = TRUE,
                widths = c(0.5, 1, 1, 1))
) &
  theme(plot.tag.position = c(.06, .96),
        plot.tag = element_text(face = "bold", size = 24))

ggsave(file.path(outdir, "Figure_Validation_Total.tiff"),
       total_figure, width = 22, height = 20, dpi = 600, compression = "lzw")
ggsave(file.path(outdir, "Figure_Validation_Total.pdf"),
       total_figure, width = 22, height = 20)

cat("\nSaved: Figure_Validation_Total.tiff\n")
cat("Saved: Figure_Validation_Total.pdf\n")

# ============================================================
# 4.5 INDIVIDUAL 3×3 PER MODEL
#     Rows: NP/LC/HCC, Columns: Calibration/DCA/Bootstrap
# ============================================================

for(model_name in names(all_models)){
  
  model_label <- all_models[[model_name]]$label
  model_color <- all_models[[model_name]]$color
  
  panels <- list()
  idx <- 1
  
  for(outcome_name in outcome_order){
    oc_clean <- clean_outcome(outcome_name)
    
    # Calibration
    cal_df <- cal_store[[model_name]][[outcome_name]]
    p_cal <- ggplot(cal_df, aes(Predicted, Observed)) +
      geom_point(size = 3, color = model_color) +
      geom_line(linewidth = 1, color = model_color) +
      geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey50") +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      labs(title = oc_clean, x = "Predicted", y = "Observed") +
      theme_tj(18)
    
    # DCA
    dca_model <- dca_store[[model_name]][[outcome_name]]
    dca_ref   <- dca_ref_store[[outcome_name]]
    dca_all <- bind_rows(dca_model, dca_ref)
    p_dca <- ggplot(dca_all, aes(Threshold, NetBenefit, color = Model, linetype = Model)) +
      geom_line(linewidth = 1) +
      coord_cartesian(xlim = c(0, .5)) +
      scale_color_manual(values = c(
        setNames(model_color, model_label),
        "Treat All" = "grey50", "Treat None" = "grey70"
      )) +
      scale_linetype_manual(values = c(
        setNames("solid", model_label),
        "Treat All" = "dashed", "Treat None" = "dotted"
      )) +
      labs(title = oc_clean, x = "Threshold", y = "Net Benefit") +
      theme_tj(18)
    
    # Bootstrap
    boot_df <- boot_store[[model_name]][[outcome_name]]
    app_auc <- app_auc_store[[model_name]][[outcome_name]]
    p_boot <- ggplot(boot_df, aes(AUC)) +
      geom_density(fill = model_color, alpha = 0.6) +
      geom_vline(xintercept = app_auc, color = "red", linetype = 2, linewidth = 1) +
      labs(title = oc_clean, x = "Bootstrap AUC", y = "Density") +
      theme_tj(18) +
      theme(axis.text.y = element_blank(), axis.ticks.y = element_blank())
    
    panels[[idx]]     <- p_cal
    panels[[idx + 1]] <- p_dca
    panels[[idx + 2]] <- p_boot
    idx <- idx + 3
  }
  
  # Add panel tags
  panels[[1]] <- panels[[1]] + labs(tag = "a")
  panels[[2]] <- panels[[2]] + labs(tag = "b")
  panels[[3]] <- panels[[3]] + labs(tag = "c")
  
  big_figure <- (
    panels[[1]] + panels[[2]] + panels[[3]] +
      panels[[4]] + panels[[5]] + panels[[6]] +
      panels[[7]] + panels[[8]] + panels[[9]] +
      plot_layout(nrow = 3, byrow = TRUE)
  ) &
    theme(plot.tag.position = c(.02, .98))
  
  suffix <- gsub(" ", "_", model_label)
  ggsave(file.path(outdir, paste0("Figure_Validation_3x3_", suffix, ".tiff")),
         big_figure, width = 18, height = 15, dpi = 600, compression = "lzw")
  ggsave(file.path(outdir, paste0("Figure_Validation_3x3_", suffix, ".pdf")),
         big_figure, width = 18, height = 15)
  
  cat(sprintf("Saved: Figure_Validation_3x3_%s.tiff/pdf\n", suffix))
}

# ============================================================
# 5. EXPORT TABLES
# ============================================================
write.csv(summary_table,
          file.path(outdir, "Supplementary_Table_Calibration_AllModels.csv"),
          row.names = FALSE)
write.csv(bootstrap_table,
          file.path(outdir, "Supplementary_Table_Bootstrap_AllModels.csv"),
          row.names = FALSE)

cat("\n=== VALIDATION COMPLETED ===\n")
cat("Models validated:", paste(names(all_models), collapse = ", "), "\n")
cat("Output directory:", outdir, "\n")

