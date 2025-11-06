import numpy as np
from pykeops.numpy import Genred
from sklearn.model_selection import KFold, train_test_split

import copy
import logging




class KernelBayesMIL:
    """
    Kernel Bayes Multiple Instance Learning.

    Unline classical Naive Bayes, mutual independence between features is not assumed.
    Instead, features are partitioned into blocks, and independence is assumed between blocks only.
    The density is estimated on each block using Kernel Density Estimation with Gaussian kernels.
    """

    def __init__(self, features_partition, train_grid_size, eval_size, random_state=None, label_prior='unif', **params):
        self.__ft_partition = features_partition
        # self.__n_features = sum([len(block) for block in features_partition])
        self.__train_grid_size = train_grid_size
        self.__eval_size = eval_size
        self.__random_state = random_state
        self.__label_prior = label_prior
        self.__params = copy.copy(params)

    
    def fit(self, X, Y, Z=None):
        """Fit model on the data"""
        # Split positive and negative bags' instances
        cX0, cX1 = None, None
        class0_bag_ids, class1_bag_ids = None, None
        cZ0, cZ1 = None, None
        for i in range(len(X)):
            tmp = X[i]
            bag_ids = np.full(len(tmp), i)
            if Y[i]==0:
                cX0 = tmp.copy() if cX0 is None else np.r_[cX0, tmp]
                class0_bag_ids = bag_ids if class0_bag_ids is None else np.r_[class0_bag_ids, bag_ids]
                if Z is not None:
                    cZ0 = Z[i] if cZ0 is None else np.r_[cZ0, Z[i]]
            else:
                cX1 = tmp.copy() if cX1 is None else np.r_[cX1, tmp]
                class1_bag_ids = bag_ids if class1_bag_ids is None else np.r_[class1_bag_ids, bag_ids]
                if Z is not None:
                    cZ1 = Z[i] if cZ1 is None else np.r_[cZ1, Z[i]]
        
        # Set grid points for KDE
        self.grid_ = {0:None, 1:None}
        for k, cX, cZ, class_bag_ids in zip([0,1], [cX0, cX1], [cZ0, cZ1], [class0_bag_ids, class1_bag_ids]):
            if Z is not None:
                m = len(cZ)
                cZ = 100 * class_bag_ids + cZ
                _, counts = np.unique_counts(cZ)
                clusters_with_one_event = np.flatnonzero(counts == 1)
                all_inds = np.arange(len(cZ))
                alone_inds = all_inds[np.isin(cZ, clusters_with_one_event)]
                other_inds = np.setdiff1d(all_inds, alone_inds, assume_unique=True)
                self.grid_[k], _ = train_test_split(cX[other_inds], train_size=min(self.__train_grid_size,m-len(counts)), random_state=self.__random_state, 
                                                        stratify=cZ[other_inds])
                self.grid_[k] = np.r_[self.grid_[k], cX[alone_inds]]
            else:
                self.grid_[k], _ = train_test_split(cX, train_size=min(self.__train_grid_size,len(cX)-len(X)), random_state=self.__random_state, 
                                                        stratify=class_bag_ids)
        
        del cX0, cX1
        del cZ0, cZ1
        # Set logging config
        logging.basicConfig(level=logging.INFO)
        # Fit Model for each block in the features partition
        self.train_scores_per_block_ = {0:{}, 1:{}}
        for j, block in enumerate(self.__ft_partition):
            logging.info(f"Fitting model for feature block {j+1} ...")
            for k in range(2):
                kde_param_res = CVKDE(self.grid_[k][:,block], **self.__params)
                self.train_scores_per_block_[k][tuple(block)] = kde_param_res
        
        # Bag labels proportion
        self.alpha_ = 0.5 if self.__label_prior == 'unif' else np.mean(Y)
        return self
    
    
    def score(self, X, Z=None):
        # Set logging config
        logging.basicConfig(level=logging.WARNING)

        scores = np.empty((len(X), 2))
        for i in range(len(X)):
            if Z is None:
                _, X_i = train_test_split(X[i], test_size=min(self.__eval_size,len(X[i])-2), random_state=self.__random_state)
            else:
                m = len(X[i])
                _, counts = np.unique_counts(Z[i])
                clusters_with_one_event = np.flatnonzero(counts == 1)
                all_inds = np.arange(m)
                alone_inds = all_inds[np.isin(Z[i], clusters_with_one_event)]
                other_inds = np.setdiff1d(all_inds, alone_inds, assume_unique=True)
                _, X_i = train_test_split(X[i][other_inds], test_size=min(self.__eval_size,m-len(counts)), random_state=self.__random_state, 
                                                        stratify=X[i][other_inds])
                X_i = np.r_[X_i, X[i][alone_inds]]
            for k in range(2):
                full_loglikelihood = 0
                for block in self.__ft_partition:
                    block_kde_params = self.train_scores_per_block_[k][tuple(block)]
                    lls = compute_kde(block_kde_params['bandwidth'],
                                      self.grid_[0][:, block] if k==0 else self.grid_[1][:, block],
                                      X_i[:, block]
                                      )
                    if np.all(lls.mask):
                        logging.warning(f"All log likelihoods are invalid for bag {i}, class {k}, block {block}.")
                        full_loglikelihood += -10000
                    else:
                        full_loglikelihood += np.sum(lls)
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
                block_kde_params = self.train_scores_per_block_[k][tuple(block)]
                lls = compute_kde(block_kde_params['bandwidth'],
                                    self.grid_[0][:, block] if k==0 else self.grid_[1][:, block],
                                    X_i[:, block]
                                    )
                full_loglikelihoods += lls
            contribs = full_loglikelihoods if contribs is None else full_loglikelihoods - contribs

        return contribs
    

    def get_instances_scores(self, X_i):
        scores = np.empty((len(X_i), 2))

        for k in range(2):
            full_loglikelihoods = np.zeros(len(X_i))
            for block in self.__ft_partition:
                block_kde_params = self.train_scores_per_block_[k][tuple(block)]
                lls = compute_kde(block_kde_params['bandwidth'],
                                    self.grid_[0][:, block] if k==0 else self.grid_[1][:, block],
                                    X_i[:, block]
                                    )
                full_loglikelihoods += lls
            scores[:, k] = full_loglikelihoods

        return scores
                


    

    

                

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
    return np.ma.masked_invalid(np.log(a) - np.log(N * (bandwidth**d) * np.power(2 * np.pi, d / 2)))


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
            ll = np.ma.masked_invalid(np.log(a) - np.log(n * (h**d) * np.power(2 * np.pi, d / 2)))
            scores[h].append(np.mean(ll))

    mean_scores = [np.mean(scores[h]) for h in hs]
    k = np.argmax(mean_scores)

    return {"bandwidth": hs[k], "log_likelihood":mean_scores[k]}




