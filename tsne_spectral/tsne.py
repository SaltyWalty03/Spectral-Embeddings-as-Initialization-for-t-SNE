"""
tsne_spectral/tsne.py

Core t-SNE gradient-descent optimisation loop.
"""
from __future__ import annotations

import numpy as np

from .affinities import get_original_pairwise_affinities, get_symmetric_p_ij
from .initialization import initialization as _init_dispatch


def get_low_dimensional_affinities(Y: np.ndarray) -> np.ndarray:
    """
    Compute the low-dimensional Student-t affinities q_{ij}.

    Uses the unnormalised Cauchy kernel (1 + ||y_i - y_j||^2)^{-1},
    zeroes the diagonal, then normalises over all pairs so the matrix
    sums to one.

    Parameters
    ----------
    Y : np.ndarray of shape (n, d_low)
        Current low-dimensional embedding.

    Returns
    -------
    np.ndarray of shape (n, n)
        Normalised low-dimensional affinity matrix with all entries
        clipped away from zero.
    """
    n = len(Y)
    q_ij = np.zeros(shape=(n, n))

    for i in range(n):
        diff = Y[i] - Y
        norm = np.linalg.norm(diff, axis=1)
        q_ij[i, :] = (1 + norm ** 2) ** (-1)

    np.fill_diagonal(q_ij, 0)
    q_ij = q_ij / q_ij.sum()

    epsilon = np.nextafter(0, 1)
    q_ij = np.maximum(q_ij, epsilon)
    return q_ij


def get_gradient(
    p_ij: np.ndarray,
    q_ij: np.ndarray,
    Y: np.ndarray,
) -> np.ndarray:
    """
    Compute the gradient of the KL-divergence C = KL(P || Q) w.r.t. Y.

    Implements Equation 5 of van der Maaten & Hinton (2008):

        dC/dy_i = 4 * sum_j (p_ij - q_ij)(1 + ||y_i - y_j||^2)^{-1} (y_i - y_j)

    Parameters
    ----------
    p_ij : np.ndarray of shape (n, n)
        High-dimensional joint affinities (possibly early-exaggerated).
    q_ij : np.ndarray of shape (n, n)
        Low-dimensional Student-t affinities.
    Y : np.ndarray of shape (n, d_low)
        Current embedding.

    Returns
    -------
    np.ndarray of shape (n, d_low)
        Gradient of the cost w.r.t. each embedded point.
    """
    n = len(p_ij)
    gradient = np.zeros(shape=(n, Y.shape[1]))

    for i in range(n):
        diff = Y[i] - Y
        A = np.array([(p_ij[i, :] - q_ij[i, :])])
        B = np.array([(1 + np.linalg.norm(diff, axis=1) ** 2) ** (-1)])
        gradient[i] = 4 * np.sum((A * B).T * diff, axis=0)

    return gradient


def tsne_eigen(
    X: np.ndarray,
    perplexity: int = 30,
    T: int = 1000,
    eta: float = 500.0,
    early_exaggeration: float = 12.0,
    initialization_method: str = "PCA",
    n_dimensions: int = 2,
    **kwargs,
) -> np.ndarray:
    """
    Run t-SNE with a selectable initialisation strategy.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data (PCA-reduced to ~30-40 dimensions recommended).
    perplexity : int
        Perplexity for the high-dimensional affinities.  Typical: 5-50.
    T : int
        Total gradient-descent iterations.  Set ``T=3`` to obtain only
        the initialisation without any optimisation.
    eta : float
        Learning rate (step size).
    early_exaggeration : float
        Multiplier applied to P during the first 250 iterations to
        encourage well-separated cluster formation.
    initialization_method : str
        One of ``"random"``, ``"PCA"``, ``"spectral"``, ``"laplacian"``.
    n_dimensions : int
        Dimensionality of the output embedding.
    **kwargs
        Forwarded to the chosen initialiser (e.g. ``k``, ``delta``,
        ``sigma``, ``normalized``).

    Returns
    -------
    np.ndarray of shape (n, n_dimensions)
        Final low-dimensional embedding Y.
    """
    n = len(X)

    p_ij = get_original_pairwise_affinities(X, perplexity)
    p_ij_symmetric = get_symmetric_p_ij(p_ij)

    Y = np.zeros(shape=(T, n, n_dimensions))
    Y[0] = np.zeros(shape=(n, n_dimensions))
    Y[1] = _init_dispatch(
        X, n_dimensions=n_dimensions,
        initialization=initialization_method, **kwargs
    )

    print("Optimizing Low Dimensional Embedding....")
    for t in range(1, T - 1):
        exag = early_exaggeration if t < 250 else 1.0

        q_ij = get_low_dimensional_affinities(Y[t])
        gradient = get_gradient(exag * p_ij_symmetric, q_ij, Y[t])
        Y[t + 1] = Y[t] - eta * gradient   # momentum term omitted (alpha=0)

        if t % 50 == 0 or t == 1:
            cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / q_ij))
            print(f"Iteration {t}: Cost = {cost:.6f}")

    final_q = get_low_dimensional_affinities(Y[-1])
    final_cost = np.sum(p_ij_symmetric * np.log(p_ij_symmetric / final_q))
    print(f"Completed. Final cost = {final_cost:.6f}")
    return Y[-1]
