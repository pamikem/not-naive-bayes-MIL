import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




def load_flowcap_aml_data(path, rs=23):
    data = pd.read_parquet(path+'/FlowCAP-AML_Panel6-2.parquet')
    with open(path+('/FlowCAP-AML_status.json')) as f:
        p_states = json.load(f)
    p_ids = np.array(list(p_states), dtype=int)

    cX = data.iloc[:, :7].values
    scaler = MinMaxScaler().fit(cX)

    # p_states_vals = np.array(list(p_states.values()))
    # aml_p_ids = p_ids[p_states_vals == "aml"]
    # non_aml_p_ids = p_ids[p_states_vals != "aml"]
    # p_ids_sub, _ = train_test_split(non_aml_p_ids, train_size=75, random_state=42)
    # p_ids_sub = np.append(p_ids_sub, aml_p_ids)
    # np.random.shuffle(p_ids_sub)

    X = []
    Y = np.zeros(len(p_ids), dtype=int)
    Z = []

    for i, p_id in enumerate(p_ids):
        Y[i] = (p_states[str(p_id)] == 'aml')
        tmp = data[data.p_id==p_id].drop(columns=['p_id'])
        preds = tmp.tomato_cluster.values
        k = preds.max() + 1
        m = len(tmp)
        m_neg = 2000
        m_pos = 2000
        if (Y[i]==0 and m <= m_neg) or (Y[i]==1 and m <= m_pos):
            X_i = tmp.iloc[:, :7].values
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