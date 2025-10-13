import numpy as np
import os
import pandas as pd
from skbio.stats.composition import clr, multiplicative_replacement
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import time

def generate_frequency_vector(complete_df, matching_field, groupby, leiden_clusters, transform=True, meta_field=None, min_tiles=1, min_perc=0):
    if min_perc >= 1:
        raise ValueError('Minimum percentage should be in the range 0-1')
    
    lr_data = list()
    lr_label = list()
    lr_samples = list()

    for sample in pd.unique(complete_df[matching_field].unique()):
        samples_df = complete_df[complete_df[matching_field] == sample]
        samples_df = samples_df[samples_df[groupby].isin(leiden_clusters)]

        num_tiles = samples_df.shape[0]
        if num_tiles < min_tiles:
            print(f'Sample: {sample} - {num_tiles} tiles. Skipping')
            continue
        
        # samples_features = [0]*len(leiden_clusters)
        samples_features = dict()
        for clust_id in leiden_clusters:
            samples_features[clust_id] = 0

        clusters_slide, clusters_counts = np.unique(samples_df[groupby], return_counts=True)
        for clust_id, count in zip(clusters_slide, clusters_counts):
            # samples_features[int(clust_id)] = count

            if (count / num_tiles) > min_perc:
                samples_features[clust_id] = count
            else:
                samples_features[clust_id] = 0

        # samples_features = np.array(samples_features, dtype=np.float64)
        samples_features = np.fromiter(samples_features.values(), dtype=np.float64)
        samples_features = np.array(samples_features) / np.sum(samples_features)
        if transform:
            samples_features = multiplicative_replacement(np.reshape(samples_features, (1,-1)))
            samples_features = clr(np.reshape(samples_features, (1,-1)))

        lr_samples.append(sample)
        lr_data.append(samples_features)

        try:
            samples_label = samples_df[meta_field].values[0]
            lr_label.append(samples_label)
        except:
            continue
            
    sample_rep_df = pd.DataFrame(data=lr_data, columns=leiden_clusters)
    sample_rep_df[matching_field] = lr_samples

    if len(lr_label) > 0:
        sample_rep_df[meta_field] = lr_label
        lr_label = np.stack(lr_label)
        lr_data = np.stack(lr_data)
    
        return lr_data, lr_label, sample_rep_df

    else:
        return sample_rep_df
    
def match_core_to_ACA(core_name, core_df):
    match_idx = core_df[core_df['core'] == core_name].index
    try:
        aca_number = core_df.loc[match_idx, 'patient_ID'].values[0]

        if len(aca_number) > 8:
            aca_number = aca_number[-8:]
        return aca_number
    
    except IndexError:
        return 'None'