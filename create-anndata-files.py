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

def create_hdf5(save_path, rep_path, num_folds=2, fold=0, scanpy_package=rsc, resolution=1):
    paths = glob.glob(os.path.join(rep_path,'*'))
    
    # Extracts patient IDs to make sure two slides from the same patient are assigned to the same fold
    case_numbers = []
    for path in paths:
        patient_label = int(get_patient_id(path))
        case_numbers.append(patient_label)
    case_numbers = sorted(list(set(case_numbers)))
    
    train_cases = [case_id for case_id in case_numbers if case_id%NUM_FOLDS != fold]
    valid_cases = [case_id for case_id in case_numbers if case_id%NUM_FOLDS == fold]

    train_paths = [path for path in paths if int(get_patient_id(path)) in train_cases]
    valid_paths = [path for path in paths if int(get_patient_id(path)) in valid_cases]

    train_features = np.concatenate([read_features(i) for i in train_paths if os.path.exists(i)], axis=0)
    valid_features = np.concatenate([read_features(i) for i in valid_paths if os.path.exists(i)], axis=0)

    train_patch_paths = np.concatenate([read_paths(i) for i in train_paths if os.path.exists(i)], axis=0)
    valid_patch_paths = np.concatenate([read_paths(i) for i in valid_paths if os.path.exists(i)], axis=0)
    
    adata = ad.AnnData(train_features)
    scanpy_package.pp.neighbors(adata)
    scanpy_package.tl.umap(adata)
    scanpy_package.tl.leiden(adata, resolution=resolution)
    cluster_labels_old = np.array(sorted(list(adata.obs['leiden'].unique().astype(int)))).astype(str)
    cluster_labels_new = np.arange(len(cluster_labels_old))
    map_dict = {cluster_labels_old[i]: cluster_labels_new[i] for i in range(len(cluster_labels_old))}
    adata.obs['leiden'] = list(map(lambda x: str(map_dict[x]), adata.obs['leiden']))
    adata.obs['patch_path'] = train_patch_paths
    adata_valid = ad.AnnData(valid_features)
    adata_valid.obs['patch_path'] = valid_patch_paths
    adata.obs['sample_id'] = adata.obs['patch_path'].astype(str).apply(lambda x: x.split('/')[-2])
    adata.obs['samples'] = adata.obs['sample_id']
    adata_valid.obs['sample_id'] = adata_valid.obs['patch_path'].astype(str).apply(lambda x: x.split('/')[-2])
    adata_valid.obs['samples'] = adata_valid.obs['sample_id']
    scanpy_package.pp.neighbors(adata_valid)
    scanpy_package.tl.umap(adata_valid)
    sc.tl.ingest(adata_valid, adata, obs='leiden', embedding_method='umap')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    adata.write(os.path.join(save_path, "adata-train.hdf5"))
    adata_valid.write(os.path.join(save_path, "adata-valid.hdf5"))
    return adata, adata_valid


reps = ["/path/to/representations/224px-2-fold/"]

NUM_FOLDS = 2
for rep_dir in tqdm(reps):
    for FOLD in range(NUM_FOLDS):
        save_path = f"/path/to/save/data/fold-{FOLD}"
        rep_path = os.path.join(rep_dir,f"siamese_unprivileged_multiplex_k_{FOLD}/feats_h5/")

        _, adata_valid = create_hdf5(save_path=save_path, rep_path=rep_path, num_folds=NUM_FOLDS, fold=FOLD)
