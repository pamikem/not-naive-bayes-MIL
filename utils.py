import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from scipy.stats import multivariate_normal
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import seaborn as sns




### Toy data generation
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




### Neighborhood graph optimization
def minimize_graph_connections(neighbors, weights):
    N, k = neighbors.shape
    k_min = int(np.log(N)) + 1
    row = np.ravel(np.outer(np.arange(N), np.ones(k)))
    col = np.ravel(neighbors)
    adj_matrix = coo_matrix((np.ones(N * k), (row, col)), shape=(N, N))
    n_cc, _ = connected_components(adj_matrix, directed=False)

    if n_cc != 1:
        return neighbors
    else:
        max_w_neighbors = weights[neighbors[:, 1:]].max(axis=1)
        peak_inds = np.flatnonzero(max_w_neighbors < weights)

        def _is_peak_notin_ngbs(x):
            return not np.any(np.isin(peak_inds, x, assume_unique=True))

        inds = np.flatnonzero(pd.Series(neighbors.tolist()).apply(_is_peak_notin_ngbs).to_numpy())
        kt = k - 1
        continu = True
        ngbs = neighbors.copy()
        while (kt >= k_min) and continu:
            ngbs[inds, kt] = inds
            col = np.ravel(ngbs)
            adj_matrix = coo_matrix((np.ones(N * k), (row, col)), shape=(N, N))
            n_cc, _ = connected_components(adj_matrix, directed=False)
            continu = n_cc == 1
            kt -= 1
        ngbs[inds, k_min] = neighbors[inds, k_min].copy() if continu else neighbors[inds, kt + 1].copy()
        ngbs_series = pd.Series(ngbs.tolist())

        def _drop_repeated_ngbs(x):
            tab = np.array(x)
            return tab[tab != tab[0]].tolist()

        ngbs = ngbs_series.apply(_drop_repeated_ngbs).to_list()
        return ngbs





### Dataviz utils
def radar_plot(model, X_i, markers, top_contrib_instances, tau, advers=False, color='blue'):
    d = X_i.shape[1]
    retained_vars = np.arange(d)
    Y_scores = model.score([X_i])
    criterion_val = np.log(model.alpha_) - np.log(1 - model.alpha_) + Y_scores[0,1] - Y_scores[0,0]

    top_advers_instances = np.setdiff1d(np.arange(len(X_i)), top_contrib_instances)
    if not advers:
        x_profil_stats = X_i[top_contrib_instances][:,retained_vars].mean(axis=0).tolist()
    else:
        x_profil_stats = X_i[top_advers_instances][:,retained_vars].mean(axis=0).tolist()
    x_avg_stats = X_i[:, retained_vars].mean(axis=0).tolist()

     # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
    x_profil_stats += x_profil_stats[:1]
    x_avg_stats += x_avg_stats[:1]

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [j / float(d) * 2 * np.pi for j in range(d)]
    angles += angles[:1]

    # Initialise the spider plot
    _, ax = plt.subplots(1, 1, figsize=(6, 5), subplot_kw={'projection': 'polar'})

    # If you want the first axis to be on top:
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], markers[retained_vars], color='black', size=8)
    
    # Draw ylabels
    n_bins = 5
    bins = np.round(np.linspace(0, 1, n_bins), 2)
    ax.set_rlabel_position(0)
    plt.yticks(bins[1:-1], bins[1:-1].astype(str), color="grey", size=7)
    plt.ylim(0,1)

    # Plot data
    ax.plot(angles, x_profil_stats, linewidth=1, linestyle='solid', c=color)
    ax.fill(angles, x_profil_stats, c='blue' if not advers else 'goldenrod', alpha=0.1)

    ax.plot(angles, x_avg_stats, linewidth=1, linestyle='--', c='black')
    ax.fill(angles, x_avg_stats, 'black', alpha=0.1)

    belief_score = round(np.abs(criterion_val - tau),1)
    if not advers:
        ratio_top_instances = round(100 * (len(top_contrib_instances) / len(X_i)),1)
        ax.set_title(f"Belief score : {belief_score}| N top instances : {len(top_contrib_instances)} ({ratio_top_instances}%)")
    else:
        ratio_top_instances = round(100 * (len(top_advers_instances) / len(X_i)),1)
        ax.set_title(f"Belief score : {belief_score}| N adv instances : {len(top_advers_instances)} ({ratio_top_instances}%)")
    plt.show()


def ridgeline_plot(X, markers, fname, p_ids, p_labels=[], annot_colors=None, height_ratio=0.6, show_mean=False, show_median=False):
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    fig, axs = plt.subplots(nrows=len(p_ids), ncols=1, figsize=(7,height_ratio*len(p_ids)))
    axs = axs.flatten()

    if len(p_labels)>0 and annot_colors==None:
        annot_colors = {}
        uniq_annots = np.unique(p_labels)
        tab_colors = sns.color_palette('hls', n_colors=len(uniq_annots))
        for annot, c in zip(uniq_annots, tab_colors):
            annot_colors[annot] = c

    xmin, xmax = np.inf, -np.inf

    for i, X_i in enumerate(X):
        data = pd.DataFrame(X_i, columns=markers)
        sns.kdeplot(data=data, x=fname, 
                    bw_adjust=1.0, clip_on=False,
                    color='blue' if len(p_labels)==0 else annot_colors[p_labels[i]], fill=True,
                    linewidth=1.2,
                    ax=axs[i]
        )
        # Add white contours
        # sns.kdeplot(data=data, x=fname, 
        #             bw_adjust=1.0, clip_on=False,
        #             color='white', linewidth=1.5,
        #             ax=axs[i]
        # )
        axs[i].axhline(y=0, lw=1.2, clip_on=False, color='blue' if len(p_labels)==0 else annot_colors[p_labels[i]])
        xmin = min(xmin,data[fname].min())
        xmax = max(xmax, data[fname].max())
        if show_mean:
            axs[i].axvline(data[fname].mean(), color='black', linestyle='--', label='Mean')
        if show_median:
            axs[i].axvline(data[fname].median(), color='red', label='Median')

    fig.subplots_adjust(hspace=-0.3)

    for i, ax in enumerate(axs):
        ax.set_xlim(xmin, xmax)
        ax.text(xmin, 0.02, str(p_ids[i]),
                fontweight='bold', fontsize=12,
                color='grey'
        )
        if i<len(axs)-1:
            ax.set_axis_off()
        else:
            sns.despine(ax=ax, bottom=True, left=True)
            plt.setp(ax.get_xticklabels(), fontsize=12, fontweight='bold')
            ax.set_xlabel(fname, fontweight='bold', fontsize=12)
            ax.set_ylabel('')
            ax.tick_params(left = False, right = False , labelleft = False) 
    
    # fig.suptitle('Density comparion across patients',
    #            fontsize=14,
    #            fontweight=14)
    if show_mean or show_median:
        axs[0].legend(loc='upper right', facecolor='white')
    plt.show()