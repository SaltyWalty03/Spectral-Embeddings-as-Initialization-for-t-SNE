import numpy as np
from sklearn.decomposition import PCA
#from UMAP.spectral_custom import spectral_layout
from UMAP.spectral import spectral_layout
#from UMAP.layouts_custom import optimize_layout_euclidean
from UMAP.layouts import optimize_layout_euclidean
from UMAP.utils import (ts, fast_knn_indices)
from pynndescent import NNDescent
from scipy.optimize import curve_fit
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors


SMOOTH_K_TOLERANCE = 1e-5
MIN_K_DIST_SCALE = 1e-3
NPY_INFINITY = np.inf


def noisy_scale_coords(Y, random_state, max_coord=10.0, noise=1e-4):
    Y = Y.astype(np.float32)
    Y = Y - Y.mean(axis=0, keepdims=True)
    m = np.abs(Y).max()
    if m > 0:
        Y *= (max_coord / m)
    if random_state is not None:
        Y += noise * random_state.normal(size=Y.shape).astype(np.float32)
    return Y

def pij_from_knn(knn_indices, knn_dists, topk=None, kernel="cauchy", sigma=None):
    n, k_all = knn_indices.shape
    if topk is None or topk > k_all:
        topk = k_all

    idx = knn_indices[:, :topk]
    dst = knn_dists[:, :topk]

    mask = idx >= 0
    rows = np.repeat(np.arange(n), topk)[mask.ravel()]
    cols = idx.ravel()[mask.ravel()]
    d = dst.ravel()[mask.ravel()]
    if sigma is None:
        sigma = np.mean(np.max(dst, axis=1))
    vals = np.exp(-(d) / (2.0 * sigma**2))

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = 0.5 * (W + W.T)
    W.setdiag(0.0); W.eliminate_zeros()
    return W

def spectral_from_W(W: sp.csr_matrix, n_components: int):
    """
    Laplacian Eigenmaps on sparse adjacency W:
    (D - W) v = λ D v, drop all near-zero eigenvectors.
    """
    d = np.asarray(W.sum(axis=1)).ravel()
    d_safe = np.where(d > 0, d, 1e-12)
    D = sp.diags(d_safe, format="csr")
    L = D - W

    k = min(n_components + 5, W.shape[0] - 1)
    vals, vecs = eigsh(L, M=D, k=k, sigma=0.0, which="LM", tol=1e-4)
    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]
    drop = (vals < 1e-10).sum()
    return vecs[:, drop:drop + n_components].astype(np.float32)



def find_ab_params(spread, min_dist):
    """Fit a, b params for the differentiable curve used in lower
    dimensional fuzzy simplicial complex construction. We want the
    smooth curve (from a pre-defined family with simple gradient) that
    best matches an offset exponential decay.
    """

    def curve(x, a, b):
        return 1.0 / (1.0 + a * x ** (2 * b))

    xv = np.linspace(0, spread * 3, 300)
    yv = np.zeros(xv.shape)
    yv[xv < min_dist] = 1.0
    yv[xv >= min_dist] = np.exp(-(xv[xv >= min_dist] - min_dist) / spread)
    params, covar = curve_fit(curve, xv, yv)
    return params[0], params[1]

def make_epochs_per_sample(weights, n_epochs):
    """Given a set of weights and number of epochs generate the number of
    epochs per sample for each weight.

    Parameters
    ----------
    weights: array of shape (n_1_simplices)
        The weights of how much we wish to sample each 1-simplex.

    n_epochs: int
        The total number of epochs we want to train for.

    Returns
    -------
    An array of number of epochs per sample, one for each 1-simplex.
    """
    result = -1.0 * np.ones(weights.shape[0], dtype=np.float64)
    n_samples = n_epochs * (weights / weights.max())
    result[n_samples > 0] = float(n_epochs) / np.float64(n_samples[n_samples > 0])
    return result

def smooth_knn_dist(distances, k, n_iter=64, local_connectivity=1.0, bandwidth=1.0):
    """Compute a continuous version of the distance to the kth nearest
    neighbor. That is, this is similar to knn-distance but allows continuous
    k values rather than requiring an integral k. In essence we are simply
    computing the distance such that the cardinality of fuzzy set we generate
    is k.

    Parameters
    ----------
    distances: array of shape (n_samples, n_neighbors)
        Distances to nearest neighbors for each sample. Each row should be a
        sorted list of distances to a given samples nearest neighbors.

    k: float
        The number of nearest neighbors to approximate for.

    n_iter: int (optional, default 64)
        We need to binary search for the correct distance value. This is the
        max number of iterations to use in such a search.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    bandwidth: float (optional, default 1)
        The target bandwidth of the kernel, larger values will produce
        larger return values.

    Returns
    -------
    knn_dist: array of shape (n_samples,)
        The distance to kth nearest neighbor, as suitably approximated.

    nn_dist: array of shape (n_samples,)
        The distance to the 1st nearest neighbor for each point.
    """
    target = np.log2(k) * bandwidth
    rho = np.zeros(distances.shape[0], dtype=np.float32)
    result = np.zeros(distances.shape[0], dtype=np.float32)

    mean_distances = np.mean(distances)

    for i in range(distances.shape[0]):
        lo = 0.0
        hi = NPY_INFINITY
        mid = 1.0

        # TODO: This is very inefficient, but will do for now. FIXME
        ith_distances = distances[i]
        non_zero_dists = ith_distances[ith_distances > 0.0]
        if non_zero_dists.shape[0] >= local_connectivity:
            index = int(np.floor(local_connectivity))
            interpolation = local_connectivity - index
            if index > 0:
                rho[i] = non_zero_dists[index - 1]
                if interpolation > SMOOTH_K_TOLERANCE:
                    rho[i] += interpolation * (
                        non_zero_dists[index] - non_zero_dists[index - 1]
                    )
            else:
                rho[i] = interpolation * non_zero_dists[0]
        elif non_zero_dists.shape[0] > 0:
            rho[i] = np.max(non_zero_dists)

        for n in range(n_iter):

            psum = 0.0
            for j in range(1, distances.shape[1]):
                d = distances[i, j] - rho[i]
                if d > 0:
                    psum += np.exp(-(d / mid))
                else:
                    psum += 1.0

            if np.fabs(psum - target) < SMOOTH_K_TOLERANCE:
                break

            if psum > target:
                hi = mid
                mid = (lo + hi) / 2.0
            else:
                lo = mid
                if hi == NPY_INFINITY:
                    mid *= 2
                else:
                    mid = (lo + hi) / 2.0

        result[i] = mid

        # TODO: This is very inefficient, but will do for now. FIXME
        if rho[i] > 0.0:
            mean_ith_distances = np.mean(ith_distances)
            if result[i] < MIN_K_DIST_SCALE * mean_ith_distances:
                result[i] = MIN_K_DIST_SCALE * mean_ith_distances
        else:
            if result[i] < MIN_K_DIST_SCALE * mean_distances:
                result[i] = MIN_K_DIST_SCALE * mean_distances

    return result, rho

def compute_membership_strengths(
    knn_indices,
    knn_dists,
    sigmas,
    rhos,
    similarity='exp',
    return_dists=False,
    bipartite=False,
):
    """Construct the membership strength data for the 1-skeleton of each local
    fuzzy simplicial set -- this is formed as a sparse matrix where each row is
    a local fuzzy simplicial set, with a membership strength for the
    1-simplex to each other data point.

    Parameters
    ----------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    sigmas: array of shape(n_samples)
        The normalization factor derived from the metric tensor approximation.

    rhos: array of shape(n_samples)
        The local connectivity adjustment.

    return_dists: bool (optional, default False)
        Whether to return the pairwise distance associated with each edge.

    bipartite: bool (optional, default False)
        Does the nearest neighbour set represent a bipartite graph? That is, are the
        nearest neighbour indices from the same point set as the row indices?

    Returns
    -------
    rows: array of shape (n_samples * n_neighbors)
        Row data for the resulting sparse matrix (coo format)

    cols: array of shape (n_samples * n_neighbors)
        Column data for the resulting sparse matrix (coo format)

    vals: array of shape (n_samples * n_neighbors)
        Entries for the resulting sparse matrix (coo format)

    dists: array of shape (n_samples * n_neighbors)
        Distance associated with each entry in the resulting sparse matrix
    """
    n_samples = knn_indices.shape[0]
    n_neighbors = knn_indices.shape[1]

    rows = np.zeros(knn_indices.size, dtype=np.int32)
    cols = np.zeros(knn_indices.size, dtype=np.int32)
    vals = np.zeros(knn_indices.size, dtype=np.float32)
    if return_dists:
        dists = np.zeros(knn_indices.size, dtype=np.float32)
    else:
        dists = None

    for i in range(n_samples):
        for j in range(n_neighbors):
            if knn_indices[i, j] == -1:
                continue  # We didn't get the full knn for i
            # If applied to an adjacency matrix points shouldn't be similar to themselves.
            # If applied to an incidence matrix (or bipartite) then the row and column indices are different.
            if (bipartite == False) & (knn_indices[i, j] == i):
                val = 0.0
            elif knn_dists[i, j] - rhos[i] <= 0.0 or sigmas[i] == 0.0:
                val = 1.0
            else:
                x = np.clip((knn_dists[i, j] - rhos[i]) / sigmas[i])

                if similarity == 'exp':
                    val = np.exp(-x)
                elif similarity == 'exp_squared':
                    val = np.exp(-x ** 2)
                elif similarity == 'reciprocal':
                    val = 1.0 / (1.0 + x)
                else:
                    raise ValueError("Unknown similarity function")
                    
            rows[i * n_neighbors + j] = i
            cols[i * n_neighbors + j] = knn_indices[i, j]
            vals[i * n_neighbors + j] = val
            if return_dists:
                dists[i * n_neighbors + j] = knn_dists[i, j]

    return rows, cols, vals, dists

def fuzzy_simplicial_set(
    X,
    n_neighbors,
    random_state,
    metric,
    metric_kwds={},
    knn_indices=None,
    knn_dists=None,
    angular=False,
    set_op_mix_ratio=1.0,
    local_connectivity=1.0,
    apply_set_operations=True,
    verbose=False,
    return_dists=None,
    similarity='exp',
):
    """Given a set of data X, a neighborhood size, and a measure of distance
    compute the fuzzy simplicial set (here represented as a fuzzy graph in
    the form of a sparse matrix) associated to the data. This is done by
    locally approximating geodesic distance at each point, creating a fuzzy
    simplicial set for each such point, and then combining all the local
    fuzzy simplicial sets into a global one via a fuzzy union.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The data to be modelled as a fuzzy simplicial set.

    n_neighbors: int
        The number of neighbors to use to approximate geodesic distance.
        Larger numbers induce more global estimates of the manifold that can
        miss finer detail, while smaller values will focus on fine manifold
        structure to the detriment of the larger picture.

    random_state: numpy RandomState or equivalent
        A state capable being used as a numpy random state.

    metric: string or function (optional, default 'euclidean')
        The metric to use to compute distances in high dimensional space.
        If a string is passed it must match a valid predefined metric. If
        a general metric is required a function that takes two 1d arrays and
        returns a float can be provided. For performance purposes it is
        required that this be a numba jit'd function. Valid string metrics
        include:

        * euclidean (or l2)
        * manhattan (or l1)
        * cityblock
        * braycurtis
        * canberra
        * chebyshev
        * correlation
        * cosine
        * dice
        * hamming
        * jaccard
        * kulsinski
        * ll_dirichlet
        * mahalanobis
        * matching
        * minkowski
        * rogerstanimoto
        * russellrao
        * seuclidean
        * sokalmichener
        * sokalsneath
        * sqeuclidean
        * yule
        * wminkowski

        Metrics that take arguments (such as minkowski, mahalanobis etc.)
        can have arguments passed via the metric_kwds dictionary. At this
        time care must be taken and dictionary elements must be ordered
        appropriately; this will hopefully be fixed in the future.

    metric_kwds: dict (optional, default {})
        Arguments to pass on to the metric, such as the ``p`` value for
        Minkowski distance.

    knn_indices: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the indices of the k-nearest neighbors as a row for
        each data point.

    knn_dists: array of shape (n_samples, n_neighbors) (optional)
        If the k-nearest neighbors of each point has already been calculated
        you can pass them in here to save computation time. This should be
        an array with the distances of the k-nearest neighbors as a row for
        each data point.

    angular: bool (optional, default False)
        Whether to use angular/cosine distance for the random projection
        forest for seeding NN-descent to determine approximate nearest
        neighbors.

    set_op_mix_ratio: float (optional, default 1.0)
        Interpolate between (fuzzy) union and intersection as the set operation
        used to combine local fuzzy simplicial sets to obtain a global fuzzy
        simplicial sets. Both fuzzy set operations use the product t-norm.
        The value of this parameter should be between 0.0 and 1.0; a value of
        1.0 will use a pure fuzzy union, while 0.0 will use a pure fuzzy
        intersection.

    local_connectivity: int (optional, default 1)
        The local connectivity required -- i.e. the number of nearest
        neighbors that should be assumed to be connected at a local level.
        The higher this value the more connected the manifold becomes
        locally. In practice this should be not more than the local intrinsic
        dimension of the manifold.

    verbose: bool (optional, default False)
        Whether to report information on the current progress of the algorithm.

    return_dists: bool or None (optional, default None)
        Whether to return the pairwise distance associated with each edge.

    Returns
    -------
    fuzzy_simplicial_set: coo_matrix
        A fuzzy simplicial set represented as a sparse matrix. The (i,
        j) entry of the matrix represents the membership strength of the
        1-simplex between the ith and jth sample points.
    """
    if knn_indices is None or knn_dists is None:
        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            n_neighbors,
            metric,
            metric_kwds,
            angular,
            random_state,
            verbose=verbose,
        )

    knn_dists = knn_dists.astype(np.float32)

    sigmas, rhos = smooth_knn_dist(
        knn_dists,
        float(n_neighbors),
        local_connectivity=float(local_connectivity),
    )

    rows, cols, vals, dists = compute_membership_strengths(
        knn_indices, knn_dists, sigmas, rhos, return_dists=return_dists, similarity=similarity
    )

    result = sp.coo_matrix(
        (vals, (rows, cols)), shape=(X.shape[0], X.shape[0])
    )
    result.eliminate_zeros()

    if apply_set_operations:
        transpose = result.transpose()

        prod_matrix = result.multiply(transpose)

        result = (
            set_op_mix_ratio * (result + transpose - prod_matrix)
            + (1.0 - set_op_mix_ratio) * prod_matrix
        )

    result.eliminate_zeros()

    if return_dists is None:
        return result, sigmas, rhos
    else:
        if return_dists:
            dmat = sp.coo_matrix(
                (dists, (rows, cols)), shape=(X.shape[0], X.shape[0])
            )

            dists = dmat.maximum(dmat.transpose()).todok()
        else:
            dists = None

        return result, sigmas, rhos, dists

def nearest_neighbors(
    X,
    n_neighbors,
    metric,
    metric_kwds,
    angular,
    random_state,
    low_memory=True,
    use_pynndescent=True,
    n_jobs=-1,
    verbose=False,
):
    """Compute the ``n_neighbors`` nearest points for each data point in ``X``
    under ``metric``. This may be exact, but more likely is approximated via
    nearest neighbor descent.

    Parameters
    ----------
    X: array of shape (n_samples, n_features)
        The input data to compute the k-neighbor graph of.

    n_neighbors: int
        The number of nearest neighbors to compute for each sample in ``X``.

    metric: string or callable
        The metric to use for the computation.

    metric_kwds: dict
        Any arguments to pass to the metric computation function.

    angular: bool
        Whether to use angular rp trees in NN approximation.

    random_state: np.random state
        The random state to use for approximate NN computations.

    low_memory: bool (optional, default True)
        Whether to pursue lower memory NNdescent.

    verbose: bool (optional, default False)
        Whether to print status data during the computation.

    Returns
    -------
    knn_indices: array of shape (n_samples, n_neighbors)
        The indices on the ``n_neighbors`` closest points in the dataset.

    knn_dists: array of shape (n_samples, n_neighbors)
        The distances to the ``n_neighbors`` closest points in the dataset.

    rp_forest: list of trees
        The random projection forest used for searching (if used, None otherwise).
    """
    if verbose:
        print(ts(), "Finding Nearest Neighbors")

    if metric == "precomputed":
        # Note that this does not support sparse distance matrices yet ...
        # Compute indices of n nearest neighbors
        knn_indices = fast_knn_indices(X, n_neighbors)
        # knn_indices = np.argsort(X)[:, :n_neighbors]
        # Compute the nearest neighbor distances
        #   (equivalent to np.sort(X)[:,:n_neighbors])
        knn_dists = X[np.arange(X.shape[0])[:, None], knn_indices].copy()
        # Prune any nearest neighbours that are infinite distance apart.
        disconnected_index = knn_dists == np.inf
        knn_indices[disconnected_index] = -1

        knn_search_index = None
    else:
        n_trees = min(64, 5 + int(round((X.shape[0]) ** 0.5 / 20.0)))
        n_iters = max(5, int(round(np.log2(X.shape[0]))))

        knn_search_index = NNDescent(
            X,
            n_neighbors=n_neighbors,
            metric=metric, # <-- alter distance here
            metric_kwds=metric_kwds,
            random_state=random_state,
            n_trees=n_trees,
            n_iters=n_iters,
            max_candidates=60,
            low_memory=low_memory,
            n_jobs=n_jobs,
            verbose=verbose,
            compressed=False,
        )
        knn_indices, knn_dists = knn_search_index.neighbor_graph

    if verbose:
        print(ts(), "Finished Nearest Neighbor Search")
    return knn_indices, knn_dists, knn_search_index


class UMAP_scratch:
    """
    A simplified UMAP implementation with custom nearest neighbor search
    and optional spectral initialization.
    """
    def __init__(
        self,
        n_neighbors=15,
        n_components=2,
        metric="euclidean",
        n_epochs=200,
        learning_rate=1.0,
        init="spectral",
        min_dist=0.1,
        spread=1.0,
        random_state=None,
        verbose=False,
        similarity='exp',
        init_topk=None,
    ):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.metric = metric
        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.init = init
        self.min_dist = min_dist
        self.spread = spread
        self.random_state = np.random.RandomState(random_state)
        self.verbose = verbose
        self.similarity=similarity
        self.init_topk = init_topk

    def _initialize_embedding(self, X, graph):
        if self.init == "random":
            return self.random_state.uniform(
                low=-10.0, high=10.0, size=(X.shape[0], self.n_components)
            ).astype(np.float32)
        elif self.init == "pca":
            return PCA(n_components=self.n_components).fit_transform(X).astype(np.float32)
        elif self.init == "spectral":
            return spectral_layout(X, graph, self.n_components, self.random_state)
        elif self.init == "prof_spectral":
            knn_indices, knn_dists, _ = nearest_neighbors(
                X,
                n_neighbors=self.n_neighbors,
                metric=self.metric,
                metric_kwds=getattr(self, "metric_kwds", {}),
                angular=False,
                random_state=self.random_state,
                low_memory=True,
                use_pynndescent=True,
                n_jobs=-1,
                verbose=self.verbose,
            )

            topk = getattr(self, "init_topk", self.n_neighbors)
            Wp = pij_from_knn(knn_indices, knn_dists, topk=topk, kernel="cauchy")

            Y0 = spectral_from_W(Wp, self.n_components)
            return noisy_scale_coords(Y0, self.random_state)
        else:
            raise ValueError("Invalid init option")

    def fit(self, X):
        if self.verbose:
            print(ts(), "Finding nearest neighbors")

        knn_indices, knn_dists, _ = nearest_neighbors(
            X,
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            metric_kwds=None,
            random_state=self.random_state,
            angular=False,
            verbose=self.verbose,
        )

        if self.verbose:
            print(ts(), "Computing fuzzy simplicial set")

        graph, sigmas, rhos = fuzzy_simplicial_set(
            X,
            n_neighbors=self.n_neighbors,
            random_state=self.random_state,
            metric=self.metric,
            knn_indices=knn_indices,
            knn_dists=knn_dists,
            angular=False,
            set_op_mix_ratio=1.0,
            local_connectivity=1.0,
            verbose=self.verbose,
            similarity=self.similarity,
        )

        graph = graph.tocoo()

        if self.verbose:
            print(ts(), "Initializing embedding")

        self._a, self._b = find_ab_params(self.spread, self.min_dist)
        
        self.embedding_ = self._initialize_embedding(X, graph).astype(np.float32)
        self.embedding_ = np.ascontiguousarray(self.embedding_, dtype=np.float32)

        graph_data = graph.data.astype(np.float32)
        graph_row = graph.row.astype(np.int32)
        graph_col = graph.col.astype(np.int32)


        epochs_per_sample = make_epochs_per_sample(graph_data, self.n_epochs).astype(np.float32)
        rng_state = self.random_state.randint(0, np.iinfo(np.int32).max, 3).astype(np.int64)

        self.embedding_ = optimize_layout_euclidean(
            self.embedding_,
            self.embedding_,
            graph_row,
            graph_col,
            self.n_epochs,
            X.shape[0],
            epochs_per_sample,
            self._a,
            self._b,
            rng_state,
            gamma=1.0,
            initial_alpha=self.learning_rate,
            negative_sample_rate=5,
            parallel=False,
            verbose=self.verbose,
            )


        return self

    def fit_transform(self, X):
        return self.fit(X).embedding_
