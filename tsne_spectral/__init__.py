"""
tsne_spectral
=============

A from-scratch implementation of t-SNE with spectral and Laplacian Eigenmaps
initialisation strategies, developed for a Northwestern University senior
mathematics thesis: *Spectral Embeddings as Initialization for t-SNE*.

The key theoretical result is that the SNE objective admits a quadratic
energy approximation E(Y) = tr(Y^T L Y), where L = D(p) - P is the graph
Laplacian of the high-dimensional affinity matrix.  The minimisers of this
quadratic are the non-trivial eigenvectors of L, motivating the spectral
initialisation strategy implemented here.

Public API
----------
tsne_eigen                        Main t-SNE function
initialization                    Initialisation dispatcher
spectral_init                     Spectral init via D(p)-P eigenvectors
laplacian_eigenmaps_init          Laplacian Eigenmaps initialisation
get_original_pairwise_affinities  High-dim Gaussian affinity matrix
get_symmetric_p_ij                Symmetrised joint affinity matrix p_{ij}
get_low_dimensional_affinities    Low-dim Student-t affinity matrix q_{ij}
get_gradient                      KL-divergence gradient w.r.t. Y
plot_embedding                    2-D scatter-plot utility
load_mnist                        MNIST loader (keras, PCA-reduced)
load_coil20                       COIL-20 loader (PNG directory)
load_olivetti                     Olivetti Faces loader (sklearn)
preprocess                        Standardise + PCA pipeline
"""

from .affinities import (
    grid_search,
    get_original_pairwise_affinities,
    get_symmetric_p_ij,
)
from .initialization import (
    spectral_init,
    laplacian_eigenmaps_init,
    initialization,
)
from .tsne import (
    get_low_dimensional_affinities,
    get_gradient,
    tsne_eigen,
)
from .utils import (
    preprocess,
    load_mnist,
    load_coil20,
    load_olivetti,
    plot_embedding,
)

__all__ = [
    "grid_search",
    "get_original_pairwise_affinities",
    "get_symmetric_p_ij",
    "spectral_init",
    "laplacian_eigenmaps_init",
    "initialization",
    "get_low_dimensional_affinities",
    "get_gradient",
    "tsne_eigen",
    "preprocess",
    "load_mnist",
    "load_coil20",
    "load_olivetti",
    "plot_embedding",
]
