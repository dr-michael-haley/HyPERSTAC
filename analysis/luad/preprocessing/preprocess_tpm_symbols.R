library(tidyverse)
library(stringr)
library(biomaRt)
library(data.table)

tcga_tpm <- read_tsv("/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD.star_tpm.tsv")

ids <- str_remove(tcga_tpm$Ensembl_ID, "\\..*")

mart <- useEnsembl("ensembl", dataset = "hsapiens_gene_ensembl")

map <- getBM(attributes = c("ensembl_gene_id", "hgnc_symbol"),
             filters    = "ensembl_gene_id",
             values     = ids,
             mart       = mart)

symbol_vec <- map$hgnc_symbol[match(ids, map$ensembl_gene_id)]

tcga_tpm_sym <- tcga_tpm %>% 
  mutate(gene = symbol_vec) %>% 
  filter(gene != "" & !is.na(gene)) %>% 
  group_by(gene) %>% 
  summarise(across(where(is.numeric), sum))

tcga_tpm_sym_t <- tcga_tpm_sym %>% 
  dplyr:::select(-gene) %>% 
  t(.) %>% 
  as.data.frame(.)

colnames(tcga_tpm_sym_t) <- tcga_tpm_sym$gene

tcga_tpm_sym_t <- tcga_tpm_sym_t %>% 
  rownames_to_column(var = "samples")

tcga_tpm_sym_t$samples <- substr(tcga_tpm_sym_t$samples, 1, 12)

tcga_tpm_sym_t <- tcga_tpm_sym_t %>% 
  group_by(samples) %>% 
  summarise(across(where(is.numeric), mean))

tcga_tpm_sym_tt <- tcga_tpm_sym_t %>% 
  column_to_rownames(var = "samples") %>% 
  t(.) %>% 
  as.data.frame(.)

write_csv(tcga_tpm_sym_tt %>% rownames_to_column(var = "gene"), file = "/Users/Kai/Library/CloudStorage/OneDrive-UniversityofGlasgow/Documents/Collaborations/mIF SSL/TCGA Proliferation Immune/TCGA-LUAD_star_tpm_symbols.csv")