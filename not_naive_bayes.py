import numpy as np
import pandas as pd
from pykeops.numpy import Genred
from sklearn.model_selection import KFold, train_test_split
from gudhi.point_cloud.knn import KNearestNeighbors
from gudhi.point_cloud.dtm import DistanceToMeasure
from gudhi.clustering.tomato import Tomato

import copy
import logging




class KernelBayesMIL:
    """
    Kernel Bayes Multiple Instance Learning.

    Unline classical Naive Bayes, mutual independence between features is not assumed.
    Instead, features are partitioned into blocks, and independence is assumed between blocks only.
    The density is estimated on each block using Kernel Density Estimation with Gaussian kernels.
    """

    def __init__(self, features_partition, train_grid_size, eval_size, random_state=None, label_prior='unif', adapt_bw_select=False, **params):
        self.__ft_partition = features_partition
        # self.__n_features = sum([len(block) for block in features_partition])
        self.__train_grid_size = train_grid_size
        self.__eval_size = eval_size
        self.__random_state = random_state
        self.__label_prior = label_prior
        self.__adapt_bw_select = adapt_bw_select
        self.__params = copy.copy(params)

    
    def fit(self, X, Y):
        """Fit model on the data"""
        # Split positive and negative bags' instances
        cX0, cX1 = None, None
        class0_bag_ids, class1_bag_ids = None, None
        for i in range(len(X)):
            tmp = X[i]
            bag_ids = np.full(len(tmp), i)
            if Y[i]==0:
                cX0 = tmp.copy() if cX0 is None else np.r_[cX0, tmp]
                class0_bag_ids = bag_ids if class0_bag_ids is None else np.r_[class0_bag_ids, bag_ids]
            else:
                cX1 = tmp.copy() if cX1 is None else np.r_[cX1, tmp]
                class1_bag_ids = bag_ids if class1_bag_ids is None else np.r_[class1_bag_ids, bag_ids]
        
        # Set grid points for KDE
        self.grid_ = {0:None, 1:None}
        for k, cX, class_bag_ids in zip([0,1], [cX0, cX1], [class0_bag_ids, class1_bag_ids]):
            self.grid_[k], _ = train_test_split(cX, train_size=min(self.__train_grid_size,len(cX)-len(X)), random_state=self.__random_state, 
                                                    stratify=class_bag_ids)
        
        del cX0, cX1
        # Set logging config
        logging.basicConfig(level=logging.INFO)
        # Fit Model for each block in the features partition
        self.train_scores_per_block_ = {0:{}, 1:{}}
        if self.__adapt_bw_select:
            for k in range(2):
                cluster_ids = compute_clustering(self.grid_[k], self.__params['n_clusters'])
                self.train_scores_per_block_[k]['cluster_ids'] = cluster_ids
        
        for j, block in enumerate(self.__ft_partition):
            logging.info(f"Fitting model for feature block {j+1} ...")
            for k in range(2):
                if not self.__adapt_bw_select:
                    kde_param_res = CVKDE(self.grid_[k][:,block], **self.__params)
                    self.train_scores_per_block_[k][tuple(block)] = kde_param_res
                else:
                    cluster_ids = self.train_scores_per_block_[k]['cluster_ids']
                    kde_res = Adaptive_KDE(self.grid_[k][:,block], cluster_ids, **self.__params)
                    self.train_scores_per_block_[k][tuple(block) ] = kde_res

        
        # Bag labels proportion
        self.alpha_ = 0.5 if self.__label_prior == 'unif' else np.mean(Y)
        return self
    
    
    def score(self, X):
        # Set logging config
        logging.basicConfig(level=logging.WARNING)

        scores = np.empty((len(X), 2))
        for i in range(len(X)):
            _, X_i = train_test_split(X[i], test_size=min(self.__eval_size,len(X[i])-2), random_state=self.__random_state)
            for k in range(2):
                full_loglikelihood = 0
                continu = True
                s = 0
                while continu and s<len(self.__ft_partition):
                    block = self.__ft_partition[s]
                    if not self.__adapt_bw_select:
                        block_kde_params = self.train_scores_per_block_[k][tuple(block)]
                        lls = compute_kde(block_kde_params['bandwidth'],
                                        self.grid_[k][:, block],
                                        X_i[:, block]
                                        )
                    else:
                        cluster_ids = self.train_scores_per_block_[k]['cluster_ids']
                        bw_per_cluster_id = self.train_scores_per_block_[k][tuple(block)]['bandwidth']
                        hs = pd.Series(cluster_ids).map(bw_per_cluster_id).to_numpy()
                        lls = compute_adaptive_kde(hs,
                                        self.grid_[k][:, block],
                                        X_i[:, block]
                                )
                    if np.all(lls.mask):
                        logging.warning(f"All log likelihoods are invalid for bag {i}, class {k}, block {block}.")
                        full_loglikelihood = -10**6
                        continu = False
                    else:
                        full_loglikelihood += np.sum(lls)
                    s += 1
                scores[i, k] = full_loglikelihood
        
        return scores
                    

    def predict(self, X):
        scores = self.score(X)
        criterion_vals = np.log(self.alpha_) - np.log(1 - self.alpha_) + scores[:,1] - scores[:,0]
        preds = (criterion_vals > 0).astype(int)
        return preds
    

    def get_instances_contrib(self, X_i):
        contribs = None

        for k in range(2):
            full_loglikelihoods = np.zeros(len(X_i))
            for block in self.__ft_partition:
                if not self.__adapt_bw_select:
                    block_kde_params = self.train_scores_per_block_[k][tuple(block)]
                    lls = compute_kde(block_kde_params['bandwidth'],
                                        self.grid_[k][:, block],
                                        X_i[:, block]
                                        )
                else:
                    cluster_ids = self.train_scores_per_block_[k]['cluster_ids']
                    bw_per_cluster_id = self.train_scores_per_block_[k][tuple(block)]['bandwidth']
                    hs = pd.Series(cluster_ids).map(bw_per_cluster_id).to_numpy()
                    lls = compute_adaptive_kde(hs,
                                    self.grid_[k][:, block],
                                    X_i[:, block]
                            )
                full_loglikelihoods += lls
            full_loglikelihoods[np.isnan(full_loglikelihoods)] = -np.inf
            contribs = full_loglikelihoods if contribs is None else full_loglikelihoods - contribs

        return contribs
    

    def get_instances_scores(self, X_i):
        scores = np.empty((len(X_i), 2))

        for k in range(2):
            full_loglikelihoods = np.zeros(len(X_i))
            for block in self.__ft_partition:
                if not self.__adapt_bw_select:
                    block_kde_params = self.train_scores_per_block_[k][tuple(block)]
                    lls = compute_kde(block_kde_params['bandwidth'],
                                        self.grid_[k][:, block],
                                        X_i[:, block]
                                        )
                else:
                    cluster_ids = self.train_scores_per_block_[k]['cluster_ids']
                    bw_per_cluster_id = self.train_scores_per_block_[k][tuple(block)]['bandwidth']
                    hs = pd.Series(cluster_ids).map(bw_per_cluster_id).to_numpy()
                    lls = compute_adaptive_kde(hs,
                                    self.grid_[k][:, block],
                                    X_i[:, block]
                            )
                full_loglikelihoods += lls
            full_loglikelihoods[np.isnan(full_loglikelihoods)] = -np.inf
            scores[:, k] = full_loglikelihoods

        return scores
    

    def get_params(self, deep=True):
        params = {
            "features_partition": self.__ft_partition,
            "train_grid_size": self.__train_grid_size,
            "eval_size": self.__eval_size,
            "random_state": self.__random_state,
            "label_prior": self.__label_prior,
            "adapt_bw_select": self.__adapt_bw_select,
        }
        if deep:
            # include nested parameters
            params.update(self.__params)
        return copy.deepcopy(params)
    

    def set_params(self, **new_params):
        for key, value in new_params.items():
            if key in [
                "features_partition", "train_grid_size", "eval_size",
                "random_state", "label_prior", "adapt_bw_select"
            ]:
                setattr(self, f"_KernelBayesMIL__{key}", value)
        return self


    

### ToMATo clustering
def compute_clustering(X, n_clusters):
    # Compute kNN graph
    # n = len(X)
    knn_estimator = KNearestNeighbors(
        k=50,
        return_distance=True,
        metric='euclidean',
        implementation='ckdtree',
        n_jobs=1
    )
    neighbors, distances = knn_estimator.fit_transform(X)

    # Compute DTM weights
    dtm_estimator = DistanceToMeasure(
        k=50,
        q=2,
        d=X.shape[1],
        metric='neighbors'
    )
    weights = np.log(dtm_estimator.fit_transform(distances))

    # Clustering
    cluster_ids = Tomato(
        graph_type='manual',
        density_type='manual',
        n_clusters = n_clusters
    ).fit_predict(neighbors, weights=weights)

    return cluster_ids

                

### Gaussian KDE estimators
def gaussian_kde_keops(grid_points):
        d = grid_points.shape[1]
        my_conv = Genred(
            "Exp(- SqNorm2(x - y))", ["x = Vi({})".format(d), "y = Vj({})".format(d)], reduction_op="Sum", axis=0
        )
        return my_conv


def compute_kde(bandwidth, grid_points, eval_points):
    N, d = grid_points.shape
    kernel = gaussian_kde_keops(grid_points)
    C = np.sqrt(0.5) / bandwidth
    a = kernel(
        C * np.ascontiguousarray(grid_points), C * np.ascontiguousarray(eval_points), backend="auto"
    ).transpose()[0]
    res = np.ma.masked_invalid(np.log(a) - np.log(N * (bandwidth**d) * np.power(2 * np.pi, d / 2)))
    return res

def compute_adaptive_kde(bandwidths, grid_points, eval_points):
    N, d = grid_points.shape
    kernel = gaussian_kde_keops(grid_points)
    unique_bws = np.unique(bandwidths)
    res = None
    for i, bw in enumerate(unique_bws):
        C = np.sqrt(0.5) / bw
        a = kernel(
            C * np.ascontiguousarray(grid_points[bandwidths==bw]), C * np.ascontiguousarray(eval_points), backend="auto"
        ).transpose()[0]
        if i==0:
            res = np.ma.masked_invalid(np.log(a) - np.log((bw**d)))
        else:
            res += np.ma.masked_invalid(np.log(a) - np.log((bw**d)))
    res += -np.log(N * np.power(2 * np.pi, d / 2))
    return res


def CVKDE(W, **params):
    hs = params["hs"]
    if "n_fold" in params:
        n_fold = params["n_fold"]
    else:
        n_fold = 3

    scores = {h: [] for h in hs}

    kf = KFold(n_splits=n_fold)
    for train_index, test_index in kf.split(W):
        W_train = np.ascontiguousarray(W[train_index, :])
        W_test = np.ascontiguousarray(W[test_index, :])
        n, d = W_train.shape
        kernel = gaussian_kde_keops(W_train)

        for h in hs:
            C = np.sqrt(0.5) / h
            a = kernel(C * W_train, C * W_test, backend="auto").transpose()[0]
            # ll = np.ma.masked_invalid(np.log(a) - (np.log(n) + d*np.log(h) + (d / 2) * np.log(2 * np.pi)))
            ll = np.ma.masked_invalid(np.log(a) - np.log(n * (h**d) * np.power(2 * np.pi, d / 2)))
            if np.all(ll.mask):
                scores[h].append(-10**6)
            else:
                scores[h].append(np.mean(ll))

    mean_scores = [np.mean(scores[h]) for h in hs]
    k = np.argmax(mean_scores)

    return {"bandwidth": hs[k], "log_likelihood":mean_scores[k]}


def Adaptive_KDE(W, grid_cluster_ids, **params):
    """Kernel Density Estimation with adaptive bandwith selection"""
    n_fold = params["n_fold"] if "n_fold" in params else 3
    n_clusters = grid_cluster_ids.max() + 1
    m = len(W)
    n_bws = params['n_bandwidths']

    knn_estimator = KNearestNeighbors(
        k=m,
        return_index=False,
        return_distance=True,
        metric='euclidean',
        implementation='ckdtree',
        n_jobs=1
    )
    grid_dists = knn_estimator.fit_transform(W)
    
    res = {"bandwidth":{}, "log_likelihood":None}
    # bandwith selection : the std of distances is used as reference
    hs = np.empty((n_clusters,n_bws))
    for cluster_id in range(n_clusters):
        sigma_dists = np.std(grid_dists[grid_cluster_ids==cluster_id])
        hs[cluster_id] = np.logspace(-1, 0, n_bws) * sigma_dists
        # hs[cluster_id] = np.logspace(-3, 0, n_bws)
    
    scores = {j: [] for j in range(n_bws)}
    kf = KFold(n_splits=n_fold)
    for train_index, test_index in kf.split(W):
        W_train = np.ascontiguousarray(W[train_index, :])
        W_test = np.ascontiguousarray(W[test_index, :])
        n, d = W_train.shape
        kernel = gaussian_kde_keops(W_train)

        for j in range(n_bws):
            for cluster_id in range(n_clusters):
                bw = hs[cluster_id,j]
                is_in_cluster = (grid_cluster_ids[train_index]==cluster_id)
                n_k = np.sum(is_in_cluster)
                pi_k = n_k / n
                C = np.sqrt(0.5) / bw
                a = kernel(C * W_train[is_in_cluster], C * W_test, backend="auto").transpose()[0]
                if cluster_id==0:
                    ll = pi_k * np.ma.masked_invalid(np.log(a) - np.log(n_k * (bw**d) * np.power(2 * np.pi, d / 2)))
                else:
                    ll += pi_k * np.ma.masked_invalid(np.log(a) - np.log(n_k * (bw**d) * np.power(2 * np.pi, d / 2)))
            if np.all(ll.mask):
                scores[j].append(-10**6)
            else:
                scores[j].append(np.mean(ll))

    mean_scores = [np.mean(scores[j]) for j in range(n_bws)]
    j_max = np.argmax(mean_scores)
    res['log_likelihood'] = mean_scores[j_max]
    res['bandwidth'] = {cluster_id : hs[cluster_id,j_max] for cluster_id in range(n_clusters)}
    return res




    




