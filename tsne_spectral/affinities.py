"""
tsne_spectral/affinities.py

High-dimensional affinity computations for t-SNE.
"""
from __future__ import annotations

import numpy as np


def grid_search(diff_i: np.ndarray, i: int, perplexity: float) -> float:
    """
    Find the bandwidth sigma_i whose Gaussian entropy matches the target perplexity.

    Searches over 200 linearly-spaced candidate sigmas in
    [0.01 * std(||diff||), 5 * std(||diff||)] and returns the one
    minimising |log(perplexity) - H * log(2)|.

    Parameters
    ----------
    diff_i : np.ndarray of shape (n, d)
        Row-wise differences X[i] - X for all j.
    i : int
        Index of the reference point; self-similarity is zeroed out.
    perplexity : float
        Target perplexity Perp = 2^H.

    Returns
    -------
    float
        Optimal bandwidth sigma_i.
    """
    result = np.inf
    norm = np.linalg.norm(diff_i, axis=1)
    std_norm = np.std(norm)

    for sigma_search in np.linspace(0.01 * std_norm, 5 * std_norm, 200):
        p = np.exp(-(norm ** 2) / (2 * sigma_search ** 2))
        p[i] = 0
        epsilon = np.nextafter(0, 1)
        p_new = np.maximum(p / (np.sum(p) + epsilon), epsilon)
        H = -np.sum(p_new * np.log2(p_new))
        if np.abs(np.log(perplexity) - H * np.log(2)) < np.abs(result):
            result = np.log(perplexity) - H * np.log(2)
            sigma = sigma_search

    return sigma


def get_original_pairwise_affinities(
    X: np.ndarray,
    perplexity: int = 10,
) -> np.ndarray:
    """
    Compute the asymmetric high-dimensional affinity matrix p_{j|i}.

    Each row is a Gaussian conditional distribution over neighbours of
    point i with bandwidth sigma_i chosen so that the row perplexity
    equals the target.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Input data (PCA-reduced recommended).
    perplexity : int
        Target perplexity. Typical values: 5-50.

    Returns
    -------
    np.ndarray of shape (n, n)
        Row-normalised asymmetric affinity matrix with zeros on the
        diagonal and all entries clipped away from zero.
    """
    n = len(X)
    print("Computing Pairwise Affinities....")
    p_ij = np.zeros(shape=(n, n))

    for i in range(n):
        diff = X[i] - X
        sigma_i = grid_search(diff, i, perplexity)
        norm = np.linalg.norm(diff, axis=1)
        p_ij[i, :] = np.exp(-(norm ** 2) / (2 * sigma_i ** 2))
        np.fill_diagonal(p_ij, 0)
        p_ij[i, :] = p_ij[i, :] / np.sum(p_ij[i, :])

    epsilon = np.nextafter(0, 1)
    p_ij = np.maximum(p_ij, epsilon)
    print("Completed Pairwise Affinities Matrix.\n")
    return p_ij


def get_symmetric_p_ij(p_ij: np.ndarray) -> np.ndarray:
    """
    Symmetrise the affinity matrix and normalise by 2n.

    Implements p_{ij} = (p_{j|i} + p_{i|j}) / (2n), the joint
    probability used by t-SNE (van der Maaten & Hinton, 2008).

    Parameters
    ----------
    p_ij : np.ndarray of shape (n, n)
        Asymmetric affinity matrix from ``get_original_pairwise_affinities``.

    Returns
    -------
    np.ndarray of shape (n, n)
        Symmetric, jointly-normalised affinity matrix with all entries
        clipped away from zero.
    """
    print("Computing Symmetric p_ij matrix....")
    n = len(p_ij)
    p_ij_symmetric = np.zeros(shape=(n, n))

    for i in range(n):
        for j in range(n):
            p_ij_symmetric[i, j] = (p_ij[i, j] + p_ij[j, i]) / (2 * n)

    epsilon = np.nextafter(0, 1)
    p_ij_symmetric = np.maximum(p_ij_symmetric, epsilon)
    print("Completed Symmetric p_ij Matrix.\n")
    return p_ij_symmetric
