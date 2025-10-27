import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split




def load_flowcap_aml_data(path):
    data = pd.read_parquet(path+'/FlowCAP-AML_Panel6.parquet')
    with open(path+('/FlowCAP-AML_status.json')) as f:
        p_states = json.load(f)
    p_ids = np.array(list(p_states), dtype=int)

    cX = data.iloc[:, :7].values
    scaler = MinMaxScaler().fit(cX)
    # cX = scaler.transform(cX)

    p_states_vals = np.array(list(p_states.values()))
    aml_p_ids = p_ids[p_states_vals == "aml"]
    non_aml_p_ids = p_ids[p_states_vals != "aml"]
    p_ids_sub, _ = train_test_split(non_aml_p_ids, train_size=75, random_state=42)
    p_ids_sub = np.append(p_ids_sub, aml_p_ids)
    np.random.shuffle(p_ids_sub)

    X = []
    Y = np.zeros(len(p_ids_sub), dtype=int)

    for i, p_id in enumerate(p_ids_sub):
        Y[i] = (p_states[str(p_id)] == 'aml')
        tmp = data[data.p_id==p_id].drop(columns=['p_id'])
        m = len(tmp)
        m_neg =  300
        m_pos = 300
        # X_i = tmp.iloc[:, :7].values
        if (Y[i]==0 and m <= m_neg) or (Y[i]==1 and m <= m_pos):
            X_i = tmp.iloc[:, :7].values
        else:
            X_i, _ = train_test_split(tmp.iloc[:, :7].values, train_size=min(m_neg,m-10) if Y[i]==0 else min(m_pos,m-10), random_state=23)
        X.append(scaler.transform(X_i))
    
    return cX, X, Y