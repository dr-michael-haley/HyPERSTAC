print("Running")
import os
import numpy as np
import glob
import h5py
import pickle
from tqdm.auto import tqdm
import anndata as ad
import pandas as pd
from sksurv.util import Surv
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.compare import compare_survival
from sksurv.linear_model import CoxPHSurvivalAnalysis
from lifelines import CoxPHFitter
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_sample_weight
from sksurv.metrics import cumulative_dynamic_auc
import seaborn as sns
from contextlib import redirect_stdout
import copy

def save_outputs(save_dir, risk_groups_test, cph):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir,"model_summary.txt"), 'w') as f:
        with redirect_stdout(f):
            cph.print_summary(style='ascii' )
            
    with open(os.path.join(save_dir,"model.pickle"), 'wb') as f:
        pickle.dump(cph, f)
    
    risk_groups_test['low'].to_csv(os.path.join(save_dir,"low_risk_group.csv"))
    risk_groups_test['high'].to_csv(os.path.join(save_dir,"high_risk_group.csv"))

def reformat_name(x, csv):
    split = x.split('_')
    tma = split[0][-1]
    row = split[1].split('-')[0]
    col = split[1].split('-')[1]
    return csv.loc[(csv['TMA']==int(tma))&(csv['Row']==row)&(csv['Column']==int(col))]['PatientID'].values[0]

def load_and_truncate_csv(path, drop=None, keep_ids=None, truncate_lim=None):
    if isinstance(path, pd.DataFrame):
        csv = path.copy()
    else:
        csv = pd.read_csv(path, index_col=0)
    if truncate_lim is not None:
        csv['os_event_ind'][csv['os_event_data']>truncate_lim] = 0
        csv['os_event_data'][csv['os_event_data']>truncate_lim] = truncate_lim
    if keep_ids is not None:
        csv = csv[csv["samples"].isin(keep_ids)]
    if drop is not None:
        csv = csv.drop(columns=drop)
    return csv

def map_stage(x):
    if x=="I":
        return 0
    elif x=="II":
        return 1
    elif x=="III":
        return 2
    elif x=="IV":
        return 2
    else:
        return np.nan

def map_mean(slide_data, i):
    out = slide_data[slide_data.obs['leiden']==i].X.mean(axis=0)
    out = np.where(np.isnan(out), 0, out)
    return out

def map_std(slide_data, i):
    out = slide_data[slide_data.obs['leiden']==i].X.std(axis=0)
    out = np.where(np.isnan(out), 0, out)
    return out

def map_slide_values(sample_id, adata, n_clusters=None):
    if n_clusters is None:
        n_clusters = adata.obs['leiden'].unique()
    slide_data = adata[adata.obs['sample_id']==sample_id]
    means = np.array([map_mean(slide_data, i) for i in list(range(n_clusters))])
    stds = np.array([map_std(slide_data, i) for i in list(range(n_clusters))])
    return np.concatenate([means.reshape(-1), stds.reshape(-1)])
    
def generate_inputs(adata, csv, panel_name='latticeA', label_col=['os_event_data','os_event_ind'], n_desired_clusters=None, use_full_representations=False, low_var_threshold=0.00, low_var_columns=None, drop_infrequent_columns_threshold=None):
    if panel_name=='icgc':
        adata.obs['sample_id'] = adata.obs['sample_id'].map(lambda x: reformat_name(x, csv))
    n_clusters = len(adata.obs['leiden'].unique())
    if n_desired_clusters is None:
        n_desired_clusters = n_clusters
    mappings = adata.obs[['leiden','sample_id']]
    bins = np.arange(n_clusters+1)
    counts = pd.crosstab(mappings['sample_id'], pd.cut(mappings['leiden'].astype(int), bins, right=False))
    counts.columns = np.arange(len(counts.columns))
    counts = counts.div(counts.sum(axis=1), axis=0)
    counts = counts.reset_index()
    missing_cols = np.arange(len(counts.columns)-1, n_desired_clusters)
    counts[missing_cols] = 0.0
    if panel_name=='latticeA':
        counts['samples'] = counts['sample_id'].str[:8]
    elif panel_name=='icgc':
        counts['samples'] = counts['sample_id'].str[:9]
    else:
        counts['samples'] = counts['sample_id']

    data = csv[csv['samples'].isin(counts['samples'])].merge(counts[counts['samples'].isin(csv['samples'])], on='samples', how='left')
    data.columns = data.columns.astype(str)
    data = data[~data.isna().any(axis=1)]
    if low_var_columns is None:
        data_std = data.std(axis=0, numeric_only=True)
        low_var_columns = data_std.index[data_std<low_var_threshold]
        if drop_infrequent_columns_threshold is not None:
            infrequent_columns = data._get_numeric_data().columns[(data._get_numeric_data()>0).sum(axis=0, numeric_only=True)<drop_infrequent_columns_threshold]
            low_var_columns = list(low_var_columns)+list(infrequent_columns)
        for col in label_col:
            if col in low_var_columns:
                low_var_columns.remove(col)
        data = data.drop(columns=low_var_columns)
    else:
        for col in label_col:
            if col in low_var_columns:
                low_var_columns.remove(col)
        data = data.drop(columns=low_var_columns, errors='ignore')
    if 'stage' in data.columns:
        data['stage'] = data['stage'].apply(map_stage)
    
    if use_full_representations:
        slide_values = pd.DataFrame.from_dict({sample_id: map_slide_values(sample_id, adata, n_clusters=n_desired_clusters) for sample_id in tqdm(data.sample_id.values)}, orient='index')
        slide_values.index.name = 'sample_id'
        print(slide_values.shape)
        if low_var_columns is None:
            slide_values_std = slide_values.std(axis=0)
            low_var_columns = slide_values_std.index[slide_values_std<low_var_threshold]
            slide_values = slide_values.drop(columns=low_var_columns)
        else:
            slide_values = slide_values.drop(columns=low_var_columns)
        data = pd.merge(data, slide_values, on='sample_id')
    
    data = data[~data.isna().any(axis=1)]
    return data, low_var_columns

def aggregate_multiple_inputs(adata_paths, csv, panel_name='latticeA', label_col=['os_event_data','os_event_ind'], n_desired_clusters=None, use_full_representations=False, low_var_threshold=0.01, dropped_low_var_columns=None, shared_columns=['samples','age', 'sex', 'stage', 'os_event_ind', 'os_event_data'], drop_infrequent_columns_threshold=None):
    if n_desired_clusters is None:
        n_desired_clusters = [None]*len(adata_paths)
    adata_list = [load_single_adata_of_multiple(path, csv, panel_name=panel_name, label_col=label_col, n_desired_clusters=n_desired_clusters[i], suffix=f"_{i}", use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, low_var_columns=dropped_low_var_columns, shared_columns=shared_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold) for i, path in enumerate(adata_paths)]
    df = adata_list[0][0]
    for adata, _, _ in adata_list[1:]:
        df = pd.merge(df.groupby('samples', as_index=False).mean(), adata.groupby('samples', as_index=False).mean(), on=shared_columns)
    return df, [i for _, i, _ in adata_list], [i for _, _, sublist in adata_list if sublist is not None for i in sublist]

def load_single_adata_of_multiple(adata_path, csv, panel_name='latticeA', label_col=['os_event_data','os_event_ind'], n_desired_clusters=None, suffix='_m', use_full_representations=False, low_var_threshold=0.01, low_var_columns=None, shared_columns=['samples','age', 'sex', 'stage', 'os_event_ind', 'os_event_data'], drop_infrequent_columns_threshold=None):
    adata = ad.read_h5ad(adata_path)
    fold_data, dropped_low_var_columns = generate_inputs(adata, csv, panel_name=panel_name, label_col=label_col, n_desired_clusters=n_desired_clusters, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, low_var_columns=low_var_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)
    if panel_name=='latticeA':
        fold_data.columns = [c+suffix for c in fold_data.columns]
        dropped_low_var_columns = [c+suffix for c in dropped_low_var_columns]
    elif panel_name=='gri':
        fold_data.columns = [c+suffix if c!='samples' else c for c in fold_data.columns]
        dropped_low_var_columns = [c+suffix if c!='samples' else c for c in dropped_low_var_columns]
    elif panel_name=='icgc':
        fold_data.columns = [c+suffix if c not in drop_columns else c for c in fold_data.columns]
        dropped_low_var_columns = [c+suffix if c not in drop_columns else c for c in dropped_low_var_columns]
    fold_data = fold_data.drop(columns=['sample_id'+suffix]).rename(columns={key+suffix: key for key in shared_columns})
    return fold_data, len(adata.obs['leiden'].unique()), dropped_low_var_columns

def load_single_adata(adata_path, csv, panel_name='latticeA', label_col=['os_event_data','os_event_ind'], n_desired_clusters=None, use_full_representations=False, low_var_threshold=0.01, dropped_low_var_columns=None, drop_infrequent_columns_threshold=None):
    adata = ad.read_h5ad(adata_path)
    fold_data, dropped_low_var_columns = generate_inputs(adata, csv, panel_name=panel_name, label_col=label_col, n_desired_clusters=n_desired_clusters, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, low_var_columns=dropped_low_var_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)
    fold_data = fold_data.groupby('samples', as_index=False).mean(numeric_only=True)
    return fold_data, len(adata.obs['leiden'].unique()), dropped_low_var_columns
    
def train_cox_ph_model(fold_train_data, fold_test_data, fold=None, duration_col='os_event_data', event_col='os_event_ind', drop_columns=[], penalty=0.1, fitter=CoxPHFitter, plot_variables=True, select_variables=True, plot_auc=True):    
    dropped_columns = drop_columns.copy()
    n_desired_clusters = len(fold_train_data.drop(columns=dropped_columns, errors='ignore').columns)
    if select_variables:
        for column in fold_train_data.drop(columns=dropped_columns, errors='ignore').columns:
            if column not in [duration_col, event_col]:
                input_data = fold_train_data.drop(columns=dropped_columns, errors='ignore')[[column,duration_col,event_col]]
                test_scores = []
                try:
                    for i in range(2):
                        column_model = fitter(penalizer=penalty)
                        column_model.fit(input_data[i::2], duration_col=duration_col, event_col=event_col)
                        score_test = column_model.score(input_data[1-i::2], scoring_method='concordance_index')
                        test_scores.append(score_test)
                    if np.mean(score_test)<=0.5:
                        dropped_columns.append(column)
                except:
                    dropped_columns.append(column)
    
    cph = fitter(penalizer=penalty)
    cph.fit(fold_train_data.drop(columns=dropped_columns, errors='ignore'), duration_col=duration_col, event_col=event_col, robust=True)

    train_c_index = cph.score(fold_train_data.drop(columns=dropped_columns, errors='ignore'), scoring_method='concordance_index')
    test_c_index = cph.score(fold_test_data.drop(columns=list(set(fold_test_data.columns) & set(dropped_columns)), errors='ignore'), scoring_method='concordance_index')
    print(f"Fold {fold} train c index: ", train_c_index)
    print(f"Fold {fold} test c index: ", test_c_index)
    
    train_predictions = cph.predict_partial_hazard(fold_train_data.drop(columns=list(set(fold_train_data.columns) & set(dropped_columns)), errors='ignore'))
    test_predictions = cph.predict_partial_hazard(fold_test_data.drop(columns=list(set(fold_test_data.columns) & set(dropped_columns)), errors='ignore'))
    train_predictions.index = fold_train_data['samples']
    test_predictions.index = fold_test_data['samples']
    
    if plot_variables:
        cph.plot()
        plt.show()
        plt.clf()
    if plot_auc:
        test_hazard = cph.predict_log_partial_hazard(fold_test_data)
        train_arr = Surv.from_arrays(event=fold_train_data['os_event_ind'], time=fold_train_data['os_event_data'])
        test_arr = Surv.from_arrays(event=fold_test_data['os_event_ind'], time=fold_test_data['os_event_data'])
        plot_cumulative_dynamic_auc(train_arr, test_arr, test_hazard, "Train AUC")
        plt.show()
        plt.clf()
        
    return cph, fold_train_data, fold_test_data, dropped_columns, (train_c_index, test_c_index), (train_predictions, test_predictions)

def find_high_low_risk_groups(cph, fold_train_data, fold_test_data, drop=None):    
    median_lifetimes_train = cph.predict_expectation(fold_train_data.drop(columns=drop, errors='ignore'))
    median_lifetimes_test = cph.predict_expectation(fold_test_data.drop(columns=list(set(fold_test_data.columns) & set(drop)), errors='ignore'))
    median = median_lifetimes_train.median()

    low_risk_ids_train = median_lifetimes_train>=median
    high_risk_ids_train = median_lifetimes_train<median
    low_risk_ids_test = median_lifetimes_test>=median
    high_risk_ids_test = median_lifetimes_test<median
    
    train_ids = {'low': low_risk_ids_train, 'high': high_risk_ids_train}
    test_ids  = {'low': low_risk_ids_test, 'high': high_risk_ids_test}
    
    return train_ids, test_ids

def preprocess_inputs(csv_path, adata_path, fold, panel_name='latticeA', label_col='os_event_data', stratify_stage=False, use_full_representations=False, low_var_threshold=0.01, keep_ids=None, truncate_lim=None, shared_columns=['samples','age', 'sex', 'stage', 'os_event_ind', 'os_event_data'], drop_infrequent_columns_threshold=None):
    csv = load_and_truncate_csv(csv_path, drop=None, keep_ids=keep_ids, truncate_lim=truncate_lim)
    if isinstance(stratify_stage, str):
        csv = csv[csv['stage']==stratify_stage]
    elif isinstance(stratify_stage, list):
        csv = csv[csv['stage'].isin(stratify_stage)]

    if isinstance(adata_path, str):
        adata_train_path = os.path.join(adata_path, f"fold-{fold}/adata-train.hdf5")
        adata_valid_path = os.path.join(adata_path, f"fold-{fold}/adata-valid.hdf5")
    elif isinstance(adata_path, list):
        adata_train_path = [os.path.join(path, f"fold-{fold}/adata-train.hdf5") for path in adata_path]
        adata_valid_path = [os.path.join(path, f"fold-{fold}/adata-valid.hdf5") for path in adata_path]
    
    if isinstance(adata_train_path, str):
        assert isinstance(adata_valid_path, str)
        fold_train_data, n_desired_clusters, dropped_low_var_columns = load_single_adata(adata_train_path, csv, panel_name=panel_name, label_col=label_col, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, dropped_low_var_columns=None, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)
        fold_test_data, _, _ = load_single_adata(adata_valid_path, csv, panel_name=panel_name, label_col=label_col, n_desired_clusters=n_desired_clusters, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, dropped_low_var_columns=dropped_low_var_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)
    elif isinstance(adata_train_path, list):
        fold_train_data, n_desired_clusters, dropped_low_var_columns = aggregate_multiple_inputs(adata_train_path, csv, panel_name=panel_name, label_col=label_col, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, dropped_low_var_columns=None, shared_columns=shared_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)
        fold_test_data, _, _ = aggregate_multiple_inputs(adata_valid_path, csv, panel_name=panel_name, label_col=label_col, n_desired_clusters=n_desired_clusters, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, dropped_low_var_columns=dropped_low_var_columns, shared_columns=shared_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)
        n_desired_clusters = sum(n_desired_clusters)
        
    return fold_train_data, fold_test_data, n_desired_clusters

def find_aggregate_risk_groups(csv_path, adata_path, panel_name='latticeA', drop_columns=None, stratify_stage=False, num_folds=2, risk_group_keys=['low','high'], penalty=0.1, fitter=CoxPHFitter, use_full_representations=False, low_var_threshold=0.01, keep_ids=None, truncate_lim=1825, shared_columns=['samples','age', 'sex', 'stage', 'os_event_ind', 'os_event_data'], drop_infrequent_columns_threshold=None, plot_variables=False, select_variables=True, plot_auc=False):
    train_risk_data = {key: [] for key in risk_group_keys}
    test_risk_data = {key: [] for key in risk_group_keys}
    
    c_indices = []
    predictions = []
    for fold in range(num_folds):
        fold_train_data, fold_test_data, n_desired_clusters = preprocess_inputs(csv_path, adata_path, fold=fold, panel_name=panel_name, stratify_stage=stratify_stage, use_full_representations=use_full_representations, low_var_threshold=low_var_threshold, keep_ids=keep_ids, truncate_lim=truncate_lim, shared_columns=shared_columns, drop_infrequent_columns_threshold=drop_infrequent_columns_threshold)

        cph, fold_train_data, fold_test_data, drop, c_index, preds = train_cox_ph_model(fold_train_data, fold_test_data, fold=fold, drop_columns=drop_columns, penalty=penalty, fitter=fitter, plot_variables=plot_variables, select_variables=select_variables, plot_auc=plot_auc)

        train_risk_ids, test_risk_ids = find_high_low_risk_groups(cph, fold_train_data, fold_test_data, drop=drop)

        for i, ids in train_risk_ids.items():
            train_risk_data[i].append(fold_train_data[ids])
        for i, ids in test_risk_ids.items():
            test_risk_data[i].append(fold_test_data[ids])
        
        c_indices.append(c_index)
        predictions.append(preds)
        if fold==0:
            cph_out = copy.deepcopy(cph)

    risk_groups_train = {key: pd.concat(group, axis=0) for key, group in train_risk_data.items()}
    risk_groups_test = {key: pd.concat(group, axis=0) for key, group in test_risk_data.items()}
    
    return risk_groups_train, risk_groups_test, c_indices, predictions, cph_out
