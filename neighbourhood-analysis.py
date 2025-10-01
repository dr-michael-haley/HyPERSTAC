import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import squidpy as sq
import cellcharter as cc
from tqdm.auto import tqdm
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
import os
from lifelines import CoxPHFitter

from survival_analysis_functions import preprocess_inputs

def reformat_name(x, csv):
    split = x.split('_')
    tma = split[0][-1]
    row = split[1].split('-')[0]
    col = split[1].split('-')[1]
    return csv.loc[(csv['TMA']==int(tma))&(csv['Row']==row)&(csv['Column']==int(col))]['PatientID'].values[0]

def match_core_to_ACA(core_name, core_df):
    match_idx = core_df[core_df['core'] == core_name].index
    try:
        aca_number = core_df.loc[match_idx, 'patient_ID'].values[0]

        if len(aca_number) > 8:
            aca_number = aca_number[-8:]
        return aca_number
    
    except IndexError:
        return 'None'

csv_path = "/path/to/tma_cores_cases.csv"
cores_df = pd.read_csv(csv_path, index_col=0)
    
overall_adata = ad.read_h5ad("/path/to/data.h5ad")
    
bioclavis_cores = overall_adata.obs['Core_ID'].unique()
bioclavis_core_dict = dict()

for core in bioclavis_cores:
    bioclavis_core_dict[core] = match_core_to_ACA(core, cores_df)
    
overall_adata = overall_adata[~overall_adata.obs['leiden'].isna()]
overall_adata.obs['samples'] = overall_adata.obs['Core_ID'].map(bioclavis_core_dict)
overall_adata.obsm['spatial']=overall_adata.obs[['CellX','CellY']]
pdl1_df = pd.read_csv("ki67_pdl1_df.csv", index_col=0)
pdl1_pos = pdl1_df[pdl1_df['PDL_positive']==1].samples#.apply(lambda x: int(x.split('_')[1]))
pdl1_neg = pdl1_df[pdl1_df['PDL_positive']==0].samples#.apply(lambda x: int(x.split('_')[1]))
overall_adata.obs = pd.merge(overall_adata.obs, pdl1_df[['samples','PDL_positive']], on='samples', how='outer')

fold_train_data, fold_test_data, n_desired_clusters = preprocess_inputs(csv_path="/path/to/survival.csv",
                                                                      adata_path="/path/to/adata.h5ad",
                                                                      fold=0
                                                                     )

pdl1_cluster_data = pd.concat([fold_train_data,fold_test_data])

pdl1_adata = overall_adata[overall_adata.obs['PDL_positive']==1.0]
pdl1_adata = ad.AnnData(X=pdl1_adata.X,
                        obs=pdl1_adata.obs,
                        var=pdl1_adata.var,
                       )
pdl1_adata.obsm['spatial']=pdl1_adata.obs[['CellX','CellY']]
pdl1_adata.obs['cluster_26_high'] = '0'
pdl1_adata.obs['cluster_26_high'][pdl1_adata.obs.samples.isin(pdl1_cluster_data[pdl1_cluster_data['26']>pdl1_cluster_data['26'].median()].samples)] = '1'
pdl1_adata.obs['cluster_26_high'] = pdl1_adata.obs['cluster_26_high'].astype('category')

pdl1_adata.uns['spatial'] = {'library_id': pdl1_adata.obs['Core_ID']}
sq.gr.spatial_neighbors(pdl1_adata, library_key='Core_ID')

cc.gr.diff_nhood_enrichment(
    pdl1_adata,
    cluster_key='cell_lineage',
    condition_key='cluster_26_high',
    condition_groups=['1','0'],
    library_key='Core_ID',
    pvalues=True,
    n_jobs=15,
    n_perms=1000
)

cc.pl.diff_nhood_enrichment(
    pdl1_adata,
    cluster_key='cell_lineage',
    condition_key='cluster_26_high',
    condition_groups=['1','0'],
    annotate=True,
    figsize=(6,6),
    significance=0.05,
    fontsize=12,
)
