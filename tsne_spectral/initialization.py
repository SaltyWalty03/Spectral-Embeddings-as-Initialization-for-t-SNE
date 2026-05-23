"""
tsne_spectral/initialization.py

Initialization strategies for t-SNE embeddings.

Three strategies are implemented and compared in the thesis:

  PCA        — standard linear projection onto top principal components.
  spectral   — eigenvectors of D(p) - P, motivated by the quadratic
               approximation of the SNE objective (the core thesis result).
  laplacian  — Laplacian Eigenmaps (Belkin & Niyogi, 2003), using the
               normalised or unnormalised graph Laplacian of the kNN graph.
  random     — isotropic Gaussian noise (baseline).
"""
from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors

from .affinities import get_original_pairwise_affinities, get_symmetric_p_ij


def spectral_init(
    X: np.ndarray,
    n_dimensions: int = 2,
    k: int = 10,
    delta: float | None = None,
) -> np.ndarray:
    """
    Initialise t-SNE from the non-trivial eigenvectors of L = D(p) - P.

    This is the core spectral initialisation of the thesis.  The SNE
    objective's quadratic approximation around the origin has minimisers
    equal to the eigenvectors of D(p) - P, so this start places the
    optimiser at the theoretically-motivated energy minimum.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data (PCA-reduced recommended).
    n_dimensions : int
        Target embedding dimensionality.
    k : int
        Nearest-neighbour count used when building the affinity graph.
    delta : float or None
        Scale factor for the initial embedding.  Defaults to 1/n^2.

    Returns
    -------
    np.ndarray of shape (n, n_dimensions)
        Initial embedding coordinates.
    """
    n = X.shape[0]
    if delta is None:
        delta = 1.0 / (n ** 2)

    p_ij = get_original_pairwise_affinities(X, 30)
    p_ij_symmetric = get_symmetric_p_ij(p_ij)

    W = p_ij_symmetric.copy()
    np.fill_diagonal(W, 0)
    D = np.diag(W.sum(axis=1))
    L = D - W

    _, eigvecs = np.linalg.eigh(L)
    s = n_dimensions
    scale = np.sqrt((n - 1) * delta / (2 * s))
    return scale * eigvecs[:, 1:n_dimensions + 1]


def laplacian_eigenmaps_init(
    X: np.ndarray,
    n_dimensions: int = 2,
    k: int = 10,
    sigma: float | None = None,
    normalized: bool = True,
) -> np.ndarray:
    """
    Initialise t-SNE via Laplacian Eigenmaps (Belkin & Niyogi, 2003).

    Constructs a k-NN graph, optionally applies a heat kernel, builds
    the (optionally normalised) graph Laplacian, and returns the
    non-trivial eigenvectors.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data.
    n_dimensions : int
        Target embedding dimensionality.
    k : int
        Number of nearest neighbours.
    sigma : float or None
        Heat-kernel bandwidth.  If None, uses binary (0/1) weights as
        described in Belkin & Niyogi Section 3.1.
    normalized : bool
        If True, uses symmetric normalised Laplacian
        L = I - D^{-1/2} W D^{-1/2}.
        If False, uses unnormalised Laplacian L = D - W.

    Returns
    -------
    np.ndarray of shape (n, n_dimensions)
        Initial embedding from the non-trivial eigenvectors of L.
    """
    n = X.shape[0]

    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    distances, indices = nbrs.kneighbors()
    distances, indices = distances[:, 1:], indices[:, 1:]

    rows = np.repeat(np.arange(n), k)
    cols = indices.ravel()

    if sigma is None:
        vals = np.ones(len(rows))
    else:
        vals = np.exp(-distances.ravel() ** 2 / sigma)

    W = csr_matrix((vals, (rows, cols)), shape=(n, n))
    W = W.maximum(W.T)           # union of neighbourhoods (not average)

    degrees = np.array(W.sum(axis=1)).ravel()

    if normalized:
        inv_sqrt_deg = np.zeros_like(degrees)
        nonzero = degrees > 0
        inv_sqrt_deg[nonzero] = 1.0 / np.sqrt(degrees[nonzero])
        D_inv_sqrt = csr_matrix(np.diag(inv_sqrt_deg))
        L = csr_matrix(np.eye(n)) - D_inv_sqrt @ W @ D_inv_sqrt
    else:
        D_mat = csr_matrix(np.diag(degrees))
        L = D_mat - W

    _, vecs = eigsh(L, k=n_dimensions + 1, which="SM")
    return vecs[:, 1:n_dimensions + 1]


def initialization(
    X: np.ndarray,
    n_dimensions: int = 2,
    initialization: str = "PCA",
    **kwargs,
) -> np.ndarray:
    """
    Unified dispatcher for t-SNE initialisation strategies.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data.
    n_dimensions : int
        Number of output dimensions.
    initialization : str
        One of ``"random"``, ``"PCA"``, ``"spectral"``, ``"laplacian"``.
    **kwargs
        Passed through to the chosen initialiser:

        - ``k`` (int): nearest-neighbour count for spectral/laplacian.
        - ``delta`` (float): scale factor for spectral.
        - ``sigma`` (float): heat-kernel bandwidth for laplacian.
        - ``normalized`` (bool): Laplacian normalisation flag.

    Returns
    -------
    np.ndarray of shape (n, n_dimensions)
        Initial embedding coordinates.

    Raises
    ------
    ValueError
        If ``initialization`` is not one of the four supported methods.
    """
    if initialization == "random":
        return np.random.normal(loc=0, scale=1e-4, size=(len(X), n_dimensions))

    elif initialization == "PCA":
        X_centered = X - X.mean(axis=0)
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        return X_centered @ Vt.T[:, :n_dimensions]

    elif initialization == "spectral":
        return spectral_init(
            X,
            n_dimensions=n_dimensions,
            k=kwargs.get("k", 10),
            delta=kwargs.get("delta", None),
        )

    elif initialization == "laplacian":
        return laplacian_eigenmaps_init(
            X,
            n_dimensions=n_dimensions,
            k=kwargs.get("k", 10),
            sigma=kwargs.get("sigma", None),
            normalized=kwargs.get("normalized", True),
        )

    else:
        raise ValueError(
            f"Unknown initialization '{initialization}'. "
            "Choose from 'random', 'PCA', 'spectral', 'laplacian'."
        )
