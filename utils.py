import numpy as np
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal




def mil_contamined_gaussian(n_bags=30, pos_bag_rate=0.5, n_instances_per_bag=100, loc_eps_sigma=1, scale_eps_sigma=1, seed=42):
    n_pos_bags = int(n_bags * pos_bag_rate)
    w_tau = 0.5
    # class_sep_tau = 2.0
    X = []
    Y = np.zeros(n_bags, dtype=int)
    Z = np.zeros((n_bags, n_instances_per_bag), dtype=int)
    d = 8

    for i in range(n_bags - n_pos_bags):
        w_pos = np.random.rand(1) * w_tau
        delta_class = np.random.rand(1) * 2.0
        X_i, Z_i = make_classification(n_samples=n_instances_per_bag, n_features=d, n_informative=4, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, 
                           class_sep=delta_class[0], weights=[1 - w_pos[0], w_pos[0]],
                           hypercube=True, flip_y=0.0,
                           random_state=seed)
        d_eps = np.random.randn(1,d) * loc_eps_sigma
        s_eps = np.abs(np.random.randn(1) * scale_eps_sigma)
        X_i[Z_i==0] = eigenv_pertubation(X_i[Z_i==0], eps=s_eps[0])
        X_i[Z_i==1] = eigenv_pertubation(X_i[Z_i==1], eps=s_eps[0])
        X_i = X_i + d_eps
        # X_i, _, Z_i, _ = train_test_split(X_i, Z_i, train_size=n_instances_per_bag, stratify=Z_i, random_state=seed)
        X.append(X_i)
        Y[i] = 0
        Z[i] = Z_i.copy()
    
    for i in range(n_bags - n_pos_bags, n_bags):
        w_pos = np.random.rand(1) * w_tau + (1 - w_tau)
        delta_class = np.random.rand(1) * 2.0 + 2.0
        X_i, Z_i = make_classification(n_samples=n_instances_per_bag, n_features=d, n_informative=4, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, 
                           class_sep=delta_class[0], weights=[1 - w_pos[0], w_pos[0]],
                           hypercube=True, flip_y=0.0,
                           random_state=seed)
        d_eps = np.random.randn(1, d) * loc_eps_sigma
        s_eps = np.abs(np.random.randn(1) * scale_eps_sigma)
        X_i[Z_i==0] = eigenv_pertubation(X_i[Z_i==0], eps=s_eps[0])
        X_i[Z_i==1] = eigenv_pertubation(X_i[Z_i==1], eps=s_eps[0])
        X_i = X_i + d_eps
        # X_i, _, Z_i, _ = train_test_split(X_i, Z_i, train_size=n_instances_per_bag, stratify=Z_i, random_state=seed)
        X.append(X_i)
        Y[i] = 1
        Z[i] = Z_i.copy()

    inds = np.arange(len(X))
    np.random.shuffle(inds)
    X = [X[i] for i in inds]
    Y = Y[inds]
    Z = [Z[i] for i in inds]
    
    return X, Y, Z


def eigenv_pertubation(X, eps=1e-3):
    d = X.shape[1]
    mu = X.mean(axis=0)
    sigma = np.cov((X - mu).T)
    # U = scipy.linalg.cholesky(sigma)
    # W = scipy.linalg.inv(U)
    sigma_p = sigma + eps * np.eye(d)
    # L = scipy.linalg.cholesky(sigma_p, lower=True)
    # return (X - mu) @ W.T @ L.T + mu
    return multivariate_normal.rvs(mean=mu, cov=sigma_p, size=len(X))