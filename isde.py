import copy
import itertools
import logging
from ast import literal_eval

import numpy as np
from pulp import LpMaximize, LpProblem, LpVariable, lpSum
from pulp.apis import PULP_CBC_CMD
from pykeops.numpy import Genred
from sklearn.model_selection import KFold, train_test_split


class ISDE:
    def __init__(
        self, grid_size, eval_size, max_block_size=3, min_block_size=2, n_partitions=1, random_state=None, **params
    ):
        """Independence Structure Density Estimator

        Computes the best partition of features based on independence hypothesis.

        Args:
            grid_size (int): size of the training set for density estimation
            eval_size (int): size of testing set
            multidimensional_estimator (obj): multivariate density estimator. KDE for instance.
            max_block_size (int, optional): max size for a block
            in features partition. Defaults to 3.
            min_block_size (int, optional): Defaults to 2.
            params (dict) : keyword arguments for multidimensional estimator.
        """
        self.__grid_size = grid_size
        self.__eval_size = eval_size
        self.__max_block_size = max_block_size
        self.__min_block_size = min_block_size
        self.__n_partitions = n_partitions
        self.__random_state = random_state
        self.__params = copy.deepcopy(params)

    def fit(self, X, y=None):
        """Fit ISDE model on the data.

        Args:
            X (numpy.array): Data points.

        Returns:
            self: self instance
        """
        if not hasattr(self, "scores_by_subsets_"):
            self.scores_by_subsets_ = {}
        d = X.shape[1]
        W, Z = train_test_split(
            X, train_size=self.__grid_size, test_size=self.__eval_size, random_state=self.__random_state, stratify=y
        )
        logging.basicConfig(level=logging.INFO)

        # Compute scores by subsets
        logging.info("Compute scores by subsets")
        for i in range(self.__min_block_size, self.__max_block_size + 1):
            print("Computing estimators for subsets of size {}...".format(i))
            for S in itertools.combinations(range(d), i):
                if S not in self.scores_by_subsets_:
                    f, f_params = CVKDE(W[:, S], **self.__params)
                    lls = np.ma.masked_invalid(f.score_samples(grid_points=W[:, S], eval_points=Z[:, S]))
                    ll = -10000 if np.all(lls.mask) else np.mean(lls)
                    self.scores_by_subsets_[S] = {"log_likelihood": float(ll), "params": f_params}
        self.grid_ = W
        # Compute optimal partition
        logging.info("Compute best partitions")
        self.best_partitions_ = []
        for _ in range(self.__n_partitions):
            self.__find_optimal_partition()

        return self

    def __find_optimal_partition(self):  # noqa C901
        # Create Graph
        weights = {}
        edges = []
        vertices = []

        for s in self.scores_by_subsets_.keys():
            for i in s:
                if i not in vertices:
                    vertices.append(i)

            edges.append(s)
            weights[s] = self.scores_by_subsets_[s]["log_likelihood"]

        # Create model and variables
        model = LpProblem(name="Best_partition", sense=LpMaximize)
        xs = []

        for e in edges:
            # Replace ' ' by '' to avoid extras '_'
            xs.append(LpVariable(name=str(e).replace(" ", ""), lowBound=0, upBound=1, cat="Integer"))

        # Cost function
        objective = lpSum([weights[e] * xs[i] for (i, e) in enumerate(edges)])
        model += objective

        # Constraints
        A = np.zeros(shape=(len(vertices), len(edges)))
        for i, e in enumerate(edges):
            for v in e:
                A[v, i] = 1

        for i, _ in enumerate(vertices):
            model += lpSum([A[i, j] * xs[j] for j in range(len(edges))]) == 1

        # exclude
        if len(self.best_partitions_) > 0:
            xs_name = [list(literal_eval(i.name)) for i in xs]
            for p_exclude in self.best_partitions_:
                model += lpSum([xs[xs_name.index(s)] for s in p_exclude]) <= len(p_exclude) - 1

        # Solve
        model.solve(PULP_CBC_CMD(msg=False))

        output_dict = {var.name: var.value() for var in model.variables()}
        out_partition = []
        for o in output_dict.keys():
            if output_dict[o] != 0:
                out_partition.append(list(literal_eval(o.replace("_", " "))))

        self.best_partitions_.append(out_partition)

    def score(self, X):
        log_density = np.zeros((len(X), self.__n_partitions))

        for i, part in enumerate(self.best_partitions_):
            weights = np.zeros(len(X))
            for S in part:
                h_opt = self.scores_by_subsets_[tuple(S)]["params"]["bandwidth"]
                weights += GaussianKDE(bandwidth=h_opt).score_samples(grid_points=self.grid_[:, S], eval_points=X[:, S])
            log_density[:, i] = weights

        return log_density

    def get_params(self):
        return {k: v for k, v in self.__params}

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self


# Gaussian KDE estimators


def __gaussian_kde_keops(grid_points):
    d = grid_points.shape[1]
    my_conv = Genred(
        "Exp(- SqNorm2(x - y))", ["x = Vi({})".format(d), "y = Vj({})".format(d)], reduction_op="Sum", axis=0
    )
    return my_conv


gaussian_kde = __gaussian_kde_keops


class GaussianKDE:
    def __init__(self, bandwidth):
        self.bandwidth = bandwidth

    def score_samples(self, grid_points, eval_points):
        """Return log-likelihood evaluation"""
        N, d = grid_points.shape
        kernel = gaussian_kde(grid_points)
        C = np.sqrt(0.5) / self.bandwidth
        a = kernel(
            C * np.ascontiguousarray(grid_points), C * np.ascontiguousarray(eval_points), backend="auto"
        ).transpose()[0]
        return np.log(a) - np.log(N * (self.bandwidth**d) * np.power(2 * np.pi, d / 2))


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
        kernel = gaussian_kde(W_train)

        for h in hs:
            C = np.sqrt(0.5) / h
            a = kernel(C * W_train, C * W_test, backend="auto").transpose()[0]
            ll = np.ma.masked_invalid(np.log(a) - np.log(n * (h**d) * np.power(2 * np.pi, d / 2)))
            scores[h].append(np.mean(ll))

    mean_scores = [np.mean(scores[h]) for h in hs]
    h_opt = hs[np.argmax(mean_scores)]

    kde = GaussianKDE(bandwidth=h_opt)

    return kde, {"bandwidth": h_opt}
