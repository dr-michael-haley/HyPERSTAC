library(tidyverse)
library(ggrepel)
library(DESeq2)
library(GSVA)
library(circlize)
library(ComplexHeatmap)
library(pheatmap)
library(BiocParallel)
library(parallel)
library('exCITingpath')

tcga_survival <- read_tsv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD.survival.tsv")

tcga_tpm <- read_csv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD_star_tpm_symbols.csv") %>% 
  column_to_rownames(var = "gene")

shared_data_patients <- intersect(colnames(tcga_tpm), tcga_survival$`_PATIENT`)

# Filter genes with less then 1 TPM in less than 20% of cases
thr <- log2(1 + 1)
prev <- 0.2

tcga_tpm_filtered <- tcga_tpm %>% 
  select(all_of(shared_data_patients)) %>% 
  tibble::rownames_to_column("gene") %>%
  mutate(prop_expressed = {
    mat <- data.matrix(select(., -gene))     
    rowMeans(mat >= thr, na.rm = TRUE)       
  }) %>%
  filter(prop_expressed >= prev) %>%
  select(-prop_expressed) %>%
  tibble::column_to_rownames("gene")

###

msigdb_hallmark <- loadDB("h.all.v2024.1.Hs.symbols.gmt")

#ssgsea_param <- ssgseaParam(expr = as.matrix(tcga_tpm_filtered),
#                            geneSets = msigdb_hallmark)

ssgsea_results <- gsva(as.matrix(tcga_tpm_filtered),
                       gset.idx.list = msigdb_hallmark,
                       method = "ssgsea",
                       BPPARAM = MulticoreParam(workers = detectCores() - 1))

col_colors <- colorRamp2(c(-2, 0, 2), c("#4E79A7", "white", "#E15759"))

Heatmap(
  ssgsea_results %>% pheatmap:::scale_rows(),
  show_column_names = FALSE,
  col = col_colors
)

write.csv(ssgsea_results, file = "/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD_ssGSEA_Hallmarks.csv")