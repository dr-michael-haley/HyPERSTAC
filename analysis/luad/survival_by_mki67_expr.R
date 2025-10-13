library(MCPcounter)
library(tidyverse)
library(survminer)
library(survival)

tcga_survival <- read_tsv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD.survival.tsv")

tcga_tpm <- read_csv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD_star_tpm_symbols.csv") %>% 
  column_to_rownames(var = "gene")

shared_data_patients <- intersect(colnames(tcga_tpm), tcga_survival$`_PATIENT`)

tcga_tpm_filtered <- tcga_tpm %>% 
  select(all_of(shared_data_patients))

tcga_survival_filtered <- tcga_survival %>% 
  filter(`_PATIENT` %in% shared_data_patients) %>% 
  select(-sample) %>% 
  distinct(.keep_all = TRUE) %>% 
  mutate(OS.time.mo = OS.time / 365 * 12) %>% 
  rename(samples = `_PATIENT`)

tcga_tpm_filtered_t <- t(tcga_tpm_filtered) %>% 
  as.data.frame()

###

median_mki67 <- median(tcga_tpm_filtered_t$MKI67)

subset_tpm <- tcga_tpm_filtered_t %>% 
  select(c("MKI67")) %>% 
  tibble::rownames_to_column("samples") %>% 
  mutate(high_mki67 = case_when(MKI67 > median_mki67 ~ 1,
                                .default = 0)) %>% 
  left_join(tcga_survival_filtered, join_by(samples == samples))

km_fit <- survfit(Surv(OS.time.mo, OS) ~ high_mki67, data = subset_tpm)
surv_diff <- survdiff(Surv(OS.time.mo, OS) ~ high_mki67, data = subset_tpm)

ggsurvplot(km_fit, 
           pval = paste0("p = ", sprintf("%.4f", surv_diff$pvalue)),
           pvalue.format = TRUE,
           xlim = c(0, 60),
           break.x.by = 10,
           legend = "top",
           legend.title = "MKI67 expression",
           #legend.labs = c("High", "Low"),
           data = subset_tpm) + 
  labs(x = "Time (months)", y = "Survival probability (overall survival)")

###

cell_counts <- MCPcounter.estimate(tcga_tpm_filtered, featuresType = "HUGO_symbols") %>% 
  t() %>% 
  as.data.frame() %>% 
  filter(rownames(.) %in% shared_data_patients)
  # mutate(all_T = `T cells` + `CD8 T cells`)

median_t_cell <- median(cell_counts$`T cell`)

subset_cell_counts <- cell_counts %>% 
  tibble::rownames_to_column("samples") %>% 
  select(c("samples", "T cells")) %>% 
  mutate(high_t = case_when(`T cells` > median_t_cell ~ 1,
                            .default = 0))

joint_mki67_t_cell <- subset_tpm %>% 
  left_join(subset_cell_counts, join_by(samples == samples))

# KM for all cases split by T cell count
km_fit <- survfit(Surv(OS.time.mo, OS) ~ high_t, data = joint_mki67_t_cell)
surv_diff <- survdiff(Surv(OS.time.mo, OS) ~ high_t, data = joint_mki67_t_cell)

ggsurvplot(km_fit, 
           pval = paste0("p = ", sprintf("%.4f", surv_diff$pvalue)),
           pvalue.format = TRUE,
           xlim = c(0, 60),
           break.x.by = 10,
           legend = "top",
           legend.title = "T cell",
           title = "All cases",
           #legend.labs = c("High", "Low"),
           data = joint_mki67_t_cell) + 
  labs(x = "Time (months)", y = "Survival probability (overall survival)")

# KM for low Ki67 split by T cell count
ki67_low_df <- joint_mki67_t_cell %>% 
  filter(high_mki67 == 0)

km_fit <- survfit(Surv(OS.time.mo, OS) ~ high_t, data = ki67_low_df)
surv_diff <- survdiff(Surv(OS.time.mo, OS) ~ high_t, data = ki67_low_df)

ggsurvplot(km_fit, 
           pval = paste0("p = ", sprintf("%.4f", surv_diff$pvalue)),
           pvalue.format = TRUE,
           xlim = c(0, 60),
           break.x.by = 10,
           legend = "top",
           legend.title = "T cell",
           title = "Low Ki67",
           legend.labs = c("Low", "High"),
           data = ki67_low_df) + 
  labs(x = "Time (months)", y = "Survival probability (overall survival)")

# KM for high Ki67 split by T cell count
ki67_high_df <- joint_mki67_t_cell %>% 
  filter(high_mki67 == 1)

km_fit <- survfit(Surv(OS.time.mo, OS) ~ high_t, data = ki67_high_df)
surv_diff <- survdiff(Surv(OS.time.mo, OS) ~ high_t, data = ki67_high_df)

ggsurvplot(km_fit, 
           pval = paste0("p = ", sprintf("%.4f", surv_diff$pvalue)),
           pvalue.format = TRUE,
           xlim = c(0, 60),
           break.x.by = 10,
           legend = "top",
           legend.title = "T cell",
           title = "High Ki67",
           legend.labs = c("Low", "High"),
           data = ki67_high_df) + 
  labs(x = "Time (months)", y = "Survival probability (overall survival)")

