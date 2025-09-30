print("Running")
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
os.environ['SCIPY_ARRAY_API'] = "1"
import numpy as np
import glob
import h5py
from tqdm.auto import tqdm
import scanpy as sc
import anndata as ad
import rapids_singlecell as rsc
import pandas as pd

def read_features(path):
    with h5py.File(path, "r") as f:
        return f.get('features')[:]
    
def read_coords(path):
    with h5py.File(path, "r") as f:
        return f.get('coords')[:]
    
def read_paths(path):
    with h5py.File(path, "r") as f:
        return f.get('paths')[:]

def get_patient_id(x):
    patient_label = x.split('/')[-1].split('_')[1]
    return patient_label

pd1_df = pd.read_csv("ki67_pdl1_df.csv", index_col=0)

pd1_pos = pd1_df[pd1_df['PDL_positive']==1].samples

reps = ["/path/to/data/224px-2-fold/"]

NUM_FOLDS = 2
for k, v in subset_patient_dict.items():
    for reps_name, rep_dir in tqdm(reps.items()):
        for FOLD in range(NUM_FOLDS):
            save_path = f"/path/to/save/data/fold-{FOLD}"
            rep_path = os.path.join(rep_dir,f"siamese_unprivileged_multiplex_k_{FOLD}/feats_h5/")

            _, adata_valid = create_hdf5_for_subset(save_path=save_path, rep_path=rep_path, ids=list(v.str[-4:].astype(int)), num_folds=NUM_FOLDS, fold=FOLD)
