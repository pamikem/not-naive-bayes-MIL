
import os
import sys
sys.path.append(os.path.abspath('/home/not-naive-bayes-MIL'))

import numpy as np
import pandas as pd
import json
import logging
import time

from gudhi.point_cloud.knn import KNearestNeighbors
from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.clustering.tomato import Tomato
from sklearn.preprocessing import MinMaxScaler

from dataset import correct_repeat_values
from utils import minimize_graph_connections





def compute_clustering_per_aml_patient():
    # Load data
    absolute_path = '/home/data'
    data = pd.read_parquet(absolute_path+'/FlowCAP-AML_Panel6.parquet')
    with open(absolute_path+('/FlowCAP-AML_status.json')) as f:
        p_states = json.load(f)
    p_ids = np.array(list(p_states), dtype=int)
    # p_states_vals = np.array(list(p_states.values()))
    data['tomato_cluster'] = -1

    # Set logging config
    logging.basicConfig(level=logging.INFO)
    opt_knn_graph_compute_times = []


    for p_id in p_ids:
        logging.info(f"Processing patient {p_id}...")
        tmp = data[data.p_id==p_id].drop(columns=['p_id', 'tomato_cluster'])
        X = tmp.to_numpy()
        del tmp
        n, d = X.shape

        # Preprocessing
        for j in range(d):
            X[:,j] = correct_repeat_values(X[:,j])
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        # Compute kNN graph
        k_max = 50
        knn_estimator = KNearestNeighbors(
            k=k_max,
            metric='euclidean',
            implementation='ckdtree',
            n_jobs=2
        )
        neighbors = knn_estimator.fit_transform(X)

        # Compute DTM weights
        k_dtm = 50

        dtm_estimator = DistanceToMeasure(
            k=k_dtm,
            q=2,
            d=X.shape[1],
            metric='neighbors'
        )
        weights = np.log(dtm_estimator.fit_transform(neighbors[:,:k_dtm]))

        # Optimize kNN graph
        t0 = time.time()
        ngbs_opt = minimize_graph_connections(neighbors, weights)
        opt_knn_graph_compute_times.append(time.time() - t0)
        del neighbors

        # Clustering
        tm = Tomato(
            graph_type='manual',
            density_type='manual',
            n_clusters = None
        ).fit(ngbs_opt, weights=weights)
        persistances = np.sort(tm.diagram_[:,0] - tm.diagram_[:,1])[::-1]
        arg_tau = np.argmax(persistances[:-1] - persistances[1:])
        arg_tau = 8 if arg_tau != 8 else arg_tau
        tau = (persistances[arg_tau] + persistances[arg_tau + 1]) / 2
        tm.merge_threshold_ = tau

        # Save cluster labels
        data.loc[data.p_id==p_id, 'tomato_cluster'] = tm.labels_
        data.iloc[data.p_id==p_id,:d] = scaler.inverse_transform(X)

    # Save all data with cluster labels
    data.to_parquet(absolute_path+'/FlowCAP-AML_Panel6-2.parquet', index=False)
    logging.info(f"DONE \n Estimated time to minimize knn graph connections : {np.mean(opt_knn_graph_compute_times)} s +/- {np.std(opt_knn_graph_compute_times)} s")





if __name__ == "__main__":
    compute_clustering_per_aml_patient()