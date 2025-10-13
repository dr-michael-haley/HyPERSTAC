library(tidyverse)
library(ggpubr)
library(survminer)
library(survival)
library(glmnet)
library(ComplexHeatmap)

# Read data & Cox PH ------------------------------------------------------

component_data <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/lattice_nmf_ncomponent_12.csv') %>% 
  column_to_rownames("...1")

core_data <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/tma_cores_cases.csv') %>% 
  column_to_rownames("...1")

# Fit a model -- is it the same as the lifelines implementation?
cox_fit <- coxph(
  Surv(os_event_data, os_event_ind) ~ .,
  data = component_data %>% select(!samples))

# Broadly similar -- component 6 not quite stat. significant here
ggforest(cox_fit,
         data = component_data)


# Component 11 ------------------------------------------------------------

# What is the distribution of Comp. 11 -- long right skew
ggplot(data = component_data, aes(x = Component_11)) + 
  geom_histogram()

## Find a cut point on a single variable
comp_11_cut <- surv_cutpoint(component_data,
                             time = "os_event_data",
                             event = "os_event_ind",
                             variables = "Component_11")

summary(comp_11_cut)

## Classify data points in the df based on the cut identified above
comp11_classified_df <- surv_categorize(comp_11_cut) # high n=102, low n=804
km_fit <- survfit(Surv(os_event_data, os_event_ind) ~ Component_11, data = comp11_classified_df)
surv_diff <- survdiff(Surv(os_event_data, os_event_ind) ~ Component_11, data = comp11_classified_df)
ggsurvplot(km_fit, 
           pval = paste0("p = ", sprintf("%.4f", surv_diff$pvalue)),
           pvalue.format = TRUE,
           xlim = c(0, 60),
           break.x.by = 10,
           legend = "top",
           legend.title = "Component 11",
           legend.labs = c("High", "Low"),
           data = comp11_classified_df) + 
  labs(x = "Time (months)", y = "Survival probability (Overall survival)")

comp11_classified_df$samples = component_data$samples

###

library(scales)
library(ggrepel)
library(DESeq2)
library(fgsea)
library(pheatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')

load("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/LUADChohort.RData")

raw_counts <- LUADCohort$rawExpr

hallmarks <- loadDB("h.all.v2024.1.Hs.symbols.gmt")
go_bp <- loadDB("GO_Biological_Process_2023.txt")
kegg <- loadDB("KEGG_2021_Human.txt")

metadata <- LUADCohort$Metadata %>% 
  mutate(samples = sprintf("ACA_%04s", `Case Number`)) %>% 
  right_join(comp11_classified_df, join_by(samples == samples)) %>% 
  select(c("...1", "Component_11", "samples")) %>% 
  na.omit() %>% 
  column_to_rownames(., "...1")

filtered_raw_counts <- as.data.frame(raw_counts) %>% 
  select(rownames(metadata))

metadata <- metadata[colnames(filtered_raw_counts), ]

# gene_vars <- apply(raw_counts, 1, var)
# variance_threshold <- quantile(gene_vars, 0.25)
# raw_counts_filtered_genes <- filtered_raw_counts[gene_vars > variance_threshold, ] %>% 
#   round(., 0)

raw_counts_filtered_genes <- round(filtered_raw_counts, 0) %>%
  subset(., apply(filtered_raw_counts, 1, mean) >= 1) %>% # Subset genes which have mean expr >= 1
  na.omit(.)

raw_counts_matrix <- as.matrix(raw_counts_filtered_genes)
metadata$Component_11 <- factor(metadata$Component_11)

dds <- DESeqDataSetFromMatrix(countData = raw_counts_matrix, colData = metadata, design = ~ Component_11)
dds <- DESeq(dds)  

resdata <- data.frame(round(counts(dds, normalized = TRUE), 2))
colnames(resdata) <- colnames(raw_counts_matrix)

de <- data.frame(results(dds, c("Component_11", "high", "low"))) %>% 
  arrange(padj)

de$mlog10p <- -log(de$padj, 10)

sig_up <- de %>% 
  filter(log2FoldChange > 1 & padj < 0.01)

sig_down <- de %>% 
  filter(log2FoldChange < -1 & padj < 0.01)

ggplot(de, aes(x = log2FoldChange, y = mlog10p)) +
  geom_point(aes(color = "Not Significant")) +
  geom_point(data = sig_up, aes(color = "Up in Component 11 High")) +
  geom_point(data = sig_down, aes(color = "Down in Component 11 High")) +
  scale_color_manual(values = c(
    "Not Significant" = "grey",
    "Up in Component 11 High" = "#E15759",
    "Down in Component 11 High" = "#4E79A7"
  ), name = "") +
  geom_text_repel(data = sig_up %>% slice_head(n = 10), aes(label = row.names(sig_up %>% slice_head(n = 10))), max.overlaps = 40) + 
  geom_text_repel(data = sig_down %>% slice_head(n = 10), aes(label = row.names(sig_down %>% slice_head(n = 10))), max.overlaps = 40) + 
  labs(x = 'L2FC', y = '-log10p', title = 'DE Component 11') +
  theme_bw() + 
  theme(panel.grid = element_blank())

# Rank genes for GSEA (L2FC)

ranked_genes <- de$log2FoldChange
names(ranked_genes) <- row.names(de)
ranked_genes <- sort(ranked_genes, decreasing = TRUE)

# Hallmarks

gsea_results_hallmarks <- fgsea(
  pathways = hallmarks,
  stats = ranked_genes,      
  minSize = 5,
  maxSize = length(ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_hallmarks, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = sub("^HALLMARK_", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 11 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# KEGG

gsea_results_kegg <- fgsea(
  pathways = kegg,
  stats = ranked_genes,      
  minSize = 5,
  maxSize = length(ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_kegg, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 11 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# GO BP

gsea_results_gobp <- fgsea(
  pathways = go_bp,
  stats = ranked_genes,      
  minSize = 5,
  maxSize = length(ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_gobp, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = gsub("\\(.*?\\)", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 11 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())


# Component 6 -------------------------------------------------------------

# What is the distribution of Comp. 6 -- long right skew
ggplot(data = component_data, aes(x = Component_6)) + 
  geom_histogram()

## Find a cut point on a single variable
comp_6_cut <- surv_cutpoint(component_data,
                             time = "os_event_data",
                             event = "os_event_ind",
                             variables = "Component_6")

summary(comp_6_cut)

## Classify data points in the df based on the cut identified above
comp6_classified_df <- surv_categorize(comp_6_cut) # high n=102, low n=804
km_fit <- survfit(Surv(os_event_data, os_event_ind) ~ Component_6, data = comp6_classified_df)
surv_diff <- survdiff(Surv(os_event_data, os_event_ind) ~ Component_6, data = comp6_classified_df)
ggsurvplot(km_fit, 
           pval = paste0("p = ", sprintf("%.4f", surv_diff$pvalue)),
           pvalue.format = TRUE,
           xlim = c(0, 60),
           break.x.by = 10,
           legend = "top",
           legend.title = "Component 6",
           legend.labs = c("High", "Low"),
           data = comp6_classified_df) + 
  labs(x = "Time (months)", y = "Survival probability (Overall survival)")

comp6_classified_df$samples <- component_data$samples

###

metadata <- LUADCohort$Metadata %>% 
  mutate(samples = sprintf("ACA_%04s", `Case Number`)) %>% 
  right_join(comp6_classified_df, join_by(samples == samples)) %>% 
  select(c("...1", "Component_6", "samples")) %>% 
  na.omit() %>% 
  column_to_rownames(., "...1")

filtered_raw_counts <- as.data.frame(raw_counts) %>% 
  select(rownames(metadata))

metadata <- metadata[colnames(filtered_raw_counts), ]

# gene_vars <- apply(raw_counts, 1, var)
# variance_threshold <- quantile(gene_vars, 0.25)
# raw_counts_filtered_genes <- filtered_raw_counts[gene_vars > variance_threshold, ] %>% 
#   round(., 0)

raw_counts_filtered_genes <- round(filtered_raw_counts, 0) %>%
  subset(., apply(filtered_raw_counts, 1, mean) >= 1) %>% # Subset genes which have mean expr >= 1
  na.omit(.)

raw_counts_matrix <- as.matrix(raw_counts_filtered_genes)
metadata$Component_6 <- factor(metadata$Component_6)

dds <- DESeqDataSetFromMatrix(countData = raw_counts_matrix, colData = metadata, design = ~ Component_6)
dds <- DESeq(dds,
             BPPARAM = MulticoreParam(workers = detectCores() - 1))  

resdata <- data.frame(round(counts(dds, normalized = TRUE), 2))
colnames(resdata) <- colnames(raw_counts_matrix)

de <- data.frame(results(dds, c("Component_6", "high", "low"))) %>% 
  arrange(padj)

de$mlog10p <- -log(de$padj, 10)

sig_up <- de %>% 
  filter(log2FoldChange > 1 & padj < 0.01)

sig_down <- de %>% 
  filter(log2FoldChange < -1 & padj < 0.01)

ggplot(de, aes(x = log2FoldChange, y = mlog10p)) +
  geom_point(aes(color = "Not Significant")) +
  geom_point(data = sig_up, aes(color = "Up in Component 6 High")) +
  geom_point(data = sig_down, aes(color = "Down in Component 6 High")) +
  scale_color_manual(values = c(
    "Not Significant" = "grey",
    "Up in Component 6 High" = "#E15759",
    "Down in Component 6 High" = "#4E79A7"
  ), name = "") +
  geom_text_repel(data = sig_up %>% slice_head(n = 10), aes(label = row.names(sig_up %>% slice_head(n = 10))), max.overlaps = 40) + 
  geom_text_repel(data = sig_down %>% slice_head(n = 10), aes(label = row.names(sig_down %>% slice_head(n = 10))), max.overlaps = 40) + 
  labs(x = 'L2FC', y = '-log10p', title = 'DE Component 6') +
  theme_bw() + 
  theme(panel.grid = element_blank())

# Rank genes for GSEA (L2FC)

ranked_genes <- de$log2FoldChange
names(ranked_genes) <- row.names(de)
ranked_genes <- sort(ranked_genes, decreasing = TRUE)

# Hallmarks

gsea_results_hallmarks <- fgsea(
  pathways = hallmarks,
  stats = ranked_genes,      
  minSize = 5,
  maxSize = length(ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_hallmarks, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = sub("^HALLMARK_", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 6 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# KEGG

gsea_results_kegg <- fgsea(
  pathways = kegg,
  stats = ranked_genes,      
  minSize = 5,
  maxSize = length(ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_kegg, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 6 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# GO BP

gsea_results_gobp <- fgsea(
  pathways = go_bp,
  stats = ranked_genes,      
  minSize = 5,
  maxSize = length(ranked_genes) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_gobp, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = gsub("\\(.*?\\)", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 6 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# Rank genes for GSEA (SNR)

norm_counts <- counts(dds, normalized = TRUE)
conds <- colData(dds)$Component_6
snr_values <- apply(norm_counts, 1, function(counts_gene) {
  high <- counts_gene[conds == "high"]
  low <- counts_gene[conds == "low"]
  
  (mean(high) - mean(low)) / (sd(high) + sd(low))
})

snr_ranked <- sort(snr_values, decreasing = TRUE)
head(snr_ranked)

# Hallmarks

gsea_results_hallmarks <- fgsea(
  pathways = hallmarks,
  stats = snr_ranked,      
  minSize = 5,
  maxSize = length(snr_ranked) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_hallmarks, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = sub("^HALLMARK_", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 6 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# KEGG

gsea_results_kegg <- fgsea(
  pathways = kegg,
  stats = snr_ranked,      
  minSize = 5,
  maxSize = length(snr_ranked) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_kegg, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 6 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# GO BP

gsea_results_gobp <- fgsea(
  pathways = go_bp,
  stats = snr_ranked,      
  minSize = 5,
  maxSize = length(snr_ranked) - 1,
  BPPARAM = MulticoreParam(workers = detectCores() - 1)
)

top_pathways <- filter(gsea_results_gobp, padj < 0.01) %>%
  arrange(desc(abs(NES))) %>%
  mutate(pathway = gsub("\\(.*?\\)", "", pathway)) %>% 
  mutate(pathway = str_wrap(pathway, width = 50)) %>% 
  slice_head(n = 20)

top_pathways$mlog10p <- -log10(top_pathways$padj)

ggplot(top_pathways, aes(x = reorder(pathway, NES), y = NES, fill = mlog10p)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Up in Component 6 High", x = "Pathways", y = "Normalised Enrichment Score") +
  scale_fill_gradient(high = "#E15759", low = "gray", name = "-log10(p)", limits = c(2, 10), oob=squish) +
  theme_bw() + 
  theme(panel.grid = element_blank())

# ssGSEA across components ------------------------------------------------

component_data_split <- read_csv('/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/lattice_nmf_ncomponent_12_split_train_valid.csv') %>% 
  select(-...1)

metadata <- LUADCohort$Metadata %>% 
  mutate(samples = sprintf("ACA_%04s", `Case Number`))

metadata <- LUADCohort$Metadata %>% 
  mutate(samples = sprintf("ACA_%04s", `Case Number`)) %>% 
  rownames_to_column(var = "name") %>% 
  # right_join(component_data, join_by(samples == samples)) %>% 
  right_join(component_data_split, join_by(samples == samples)) %>% 
  # select(c("...1", "samples", "Component_1", "Component_2", "Component_3", "Component_4", "Component_5", "Component_6", "Component_7", "Component_8", "Component_9", "Component_10", "Component_11", "Component_12")) %>% 
  select(c("samples", "name", "Component_1", "Component_2", "Component_3", "Component_4", "Component_5", "Component_6", "Component_7", "Component_8", "Component_9", "Component_10", "Component_11", "Component_12")) %>% 
  na.omit() %>% 
  column_to_rownames(var = "name")

hallmarks_ssgsea <- read_csv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Temposeq/final_figures/Data/HALLMARKS_ssGSEA_LATTICeA_n774.csv") %>% 
  column_to_rownames("...1") %>% 
  select(rownames(metadata))

metadata <- metadata[colnames(hallmarks_ssgsea), ]

ha <- HeatmapAnnotation(df = metadata %>% select(-samples))

Heatmap(as.matrix(hallmarks_ssgsea) %>% pheatmap:::scale_rows(),
        show_column_names = FALSE,
        top_annotation = ha)
