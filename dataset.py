import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import json
import logging
from pathlib import Path

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder




def load_metaflow_myeloma_data(parquet_path, rs=23, stratify=False):
    with open('/home/data/myeloma_diagnosis.json', 'r') as f:
            p_states = json.load(f)

    black_list = [1, 2, 4, 7, 8]
    # black_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 47, 50] # patients with no CD56 marker
    # black_list = [7,8, 15, 17, 23, 24, 34, 50] # singular patients in Metaflow analysis
    p_ids = np.array(list(p_states.keys()), dtype=int)
    labels = np.array(list(p_states.values()))
    is_mgus_or_diagmm = np.isin(labels, ['MGUS', 'SMM', 'Diag MM'])
    notin_black_list = ~np.isin(p_ids, black_list, assume_unique=True)
    p_ids = p_ids[notin_black_list & is_mgus_or_diagmm]

    cX = None
    X = []
    Y = (labels[notin_black_list & is_mgus_or_diagmm] == 'Diag MM').astype(int)
    Z = []
    
    #Set logging config
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Selected patients : {p_ids}")

    for i, p_id in enumerate(p_ids):
        # logging.info(f"Loading patient {p_id}")
        data_p = pd.read_parquet(parquet_path+f'p{p_id}', engine='pyarrow')

        Z_i = data_p.label.to_numpy()
        # label_encoder = LabelEncoder()
        # Z_i = label_encoder.fit_transform(Z_i)
        data_p.drop(columns=['label'], inplace=True)

        chans = ['FSC-A', 'SSC-A', 'CD45 V500-A', 'CD19 PerCP-Cy5.5-A', 'CD56 BV786-A', 'CD38 PE-Cy7-A', 'CD138 V450-A', 'H2 GFP-A']
        data_p = data_p[chans]
        data_p.iloc[:,2:] = data_p.iloc[:,2:] / data_p.iloc[:,2:].max(axis=0)
        data_p.iloc[:,2:] = 10**5 * data_p.iloc[:,2:]
        data_p.iloc[:,2:] = np.arcsinh(data_p.iloc[:,2:]/150)
        
        m_neg = 30000
        m_pos = 30000
        m = len(data_p)
        if not stratify:
            X_i, _, Z_i, _ = train_test_split(data_p.to_numpy(), Z_i,
                                      train_size=min(m_neg,m-2) if Y[i]==0 else min(m_pos,m-2), random_state=rs)
        else:
            k = len(np.unique(Z_i))
            X_i, _, Z_i, _ = train_test_split(data_p.to_numpy(), Z_i, 
                                                train_size=min(m_neg,m-k) if Y[i]==0 else min(m_pos,m-k), stratify=Z_i, random_state=rs)
        cX = X_i if cX is None else np.r_[cX, X_i]
        X.append(X_i)
        Z.append(Z_i)
    
    scaler = MinMaxScaler().fit(cX)
    X = [scaler.transform(X_i) for X_i in X]
    return X, Y, Z


def load_metaflow_myeloma_rep_data(parquet_path, rs=23, diagmm_p_id=39):
    with open('/home/data/myeloma_diagnosis.json', 'r') as f:
                p_states = json.load(f)
    chans = ['FSC-A', 'SSC-A', 'CD45 V500-A', 'CD19 PerCP-Cy5.5-A', 'CD56 BV786-A', 'CD38 PE-Cy7-A', 'CD138 V450-A', 'H2 GFP-A']

    
    black_list = [1, 2, 4, 7, 8]
    # black_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 47, 50] # patients with no CD56 marker
    # black_list = [7,8, 15, 17, 23, 24, 34, 50] # singular patients in Metaflow analysis
    p_ids = np.array(list(p_states.keys()), dtype=int)
    labels = np.array(list(p_states.values()))
    is_mgus_or_diagmm = np.isin(labels, ['MGUS', 'SMM'])
    notin_black_list = ~np.isin(p_ids, black_list, assume_unique=True)
    p_ids = p_ids[notin_black_list & is_mgus_or_diagmm]

    cX = None
    X = []
    Y = (labels[notin_black_list & is_mgus_or_diagmm] == 'Diag MM').astype(int)
    Z = []
    
    #Set logging config
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Selected patients : {p_ids}, {diagmm_p_id}")

    for i, p_id in enumerate(p_ids):
        data_p = pd.read_parquet(parquet_path+f'p{p_id}', engine='pyarrow')

        Z_i = data_p.label.to_numpy()
        data_p.drop(columns=['label'], inplace=True)

        data_p = data_p[chans]
        data_p.iloc[:,2:] = data_p.iloc[:,2:] / data_p.iloc[:,2:].max(axis=0)
        data_p.iloc[:,2:] = 10**5 * data_p.iloc[:,2:]
        data_p.iloc[:,2:] = np.arcsinh(data_p.iloc[:,2:]/150)
        
        m_neg = 30000
        m_pos = 30000
        m = len(data_p)
        X_i, _, Z_i, _ = train_test_split(data_p.to_numpy(), Z_i,
                                      train_size=min(m_neg,m-2) if Y[i]==0 else min(m_pos,m-2), random_state=rs)
        cX = X_i if cX is None else np.r_[cX, X_i]
        X.append(X_i)
        Z.append(Z_i)

    data_p = pd.read_parquet(parquet_path+f'p{diagmm_p_id}', engine='pyarrow')
    cluster_p = data_p.label.to_numpy()
    data_p.drop(columns=['label'], inplace=True)
    data_p = data_p[chans]
    data_p.iloc[:,2:] = data_p.iloc[:,2:] / data_p.iloc[:,2:].max(axis=0)
    data_p.iloc[:,2:] = 10**5 * data_p.iloc[:,2:]
    data_p.iloc[:,2:] = np.arcsinh(data_p.iloc[:,2:]/150)
    cX = np.r_[cX, data_p.to_numpy()]
    m_pos = 3000
    m = len(data_p)
    n_folds = m // m_pos
    if n_folds < 2:
        X_i = data_p.to_numpy()
        X.append(X_i)
        Z.append(Z_i)
        p_ids = np.append(p_ids, [diagmm_p_id])
        Y = np.append(Y, [1])
    else:
        kf = KFold(n_splits=n_folds)
        for _, test_index in kf.split(data_p):
            X_i = data_p.to_numpy()[test_index]
            Z_i = cluster_p[test_index]
            X.append(X_i)
            Z.append(Z_i)
        p_ids = np.append(p_ids, [diagmm_p_id] * n_folds)
        Y = np.append(Y, [1] * n_folds)
    
    scaler = MinMaxScaler().fit(cX)
    X = [scaler.transform(X_i) for X_i in X]
    return X, Y, Z



def load_flowcap_aml_data(path, rs=23, stratify=False):
    data = pd.read_parquet(path+'/FlowCAP-AML_Panel6-2.parquet')
    with open(path+('/FlowCAP-AML_status.json')) as f:
        p_states = json.load(f)
    p_ids = np.array(list(p_states), dtype=int)

    cX = data.iloc[:, :7].values
    scaler = MinMaxScaler().fit(cX)

    X = []
    Y = np.zeros(len(p_ids), dtype=int)
    Z = []

    for i, p_id in enumerate(p_ids):
        Y[i] = (p_states[str(p_id)] == 'aml')
        tmp = data[data.p_id==p_id].drop(columns=['p_id'])
        preds = tmp.tomato_cluster.values
        k = preds.max() + 1
        m = len(tmp)
        m_neg = 3000
        m_pos = 5000
        if (Y[i]==0 and m <= m_neg) or (Y[i]==1 and m <= m_pos):
            X_i = tmp.iloc[:, :7].values
            Z.append(preds)
        else:
            if not stratify:
                X_i, _, Z_i, _ = train_test_split(tmp.iloc[:, :7].to_numpy(), preds,
                                                train_size=m_neg if Y[i]==0 else m_pos, random_state=rs)
            else:
                _, counts = np.unique_counts(preds)
                clusters_with_one_event = np.flatnonzero(counts == 1)
                all_inds = np.arange(m)
                alone_inds = all_inds[np.isin(preds, clusters_with_one_event)]
                other_inds = np.setdiff1d(all_inds, alone_inds, assume_unique=True)
                X_i, _, Z_i, _ = train_test_split(tmp.iloc[other_inds, :7].to_numpy(), preds[other_inds], 
                                                train_size=min(m_neg,m-k) if Y[i]==0 else min(m_pos,m-k), stratify=preds[other_inds], random_state=rs)
                X_i = np.r_[X_i, tmp.iloc[alone_inds, :7].to_numpy()]
                Z_i = np.r_[Z_i, preds[alone_inds]]
            Z.append(Z_i)
        X.append(scaler.transform(X_i))
    
    return X, Y, Z


def correct_repeat_values(x):
    sorted_values = np.sort(np.unique(x))
    if len(sorted_values) == len(x):
        return x
    else:
        diffs = np.ediff1d(sorted_values)
        z = np.min(diffs)
        return x - (z/2) + z*np.random.uniform(size=len(x))