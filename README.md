# Spectral Embeddings as Initialization for t-SNE

**Senior Mathematics Thesis — Northwestern University, Department of Mathematics**

## Abstract

This thesis investigates spectral methods as principled initialization strategies for t-Distributed Stochastic Neighbor Embedding (t-SNE). We prove that the SNE objective admits a quadratic energy approximation in the small-scale limit, and that the minimizers of this quadratic form are the eigenvectors of the matrix D(p) − P, where P is the symmetric affinity matrix and D(p) is its diagonal degree matrix. This theoretical result motivates initializing t-SNE from these eigenvectors — termed *spectral initialization* — as an alternative to the standard PCA or random initialization. We benchmark three strategies (PCA, Spectral, and Laplacian Eigenmaps) across six datasets: MNIST, COIL-20, Olivetti Faces, PBMC3K single-cell RNA, Credit Card fraud, and CIFAR-10 (ResNet-18 embeddings), measuring KL divergence, visual embedding quality, and runtime.

## Mathematical Overview

The central theoretical contribution connects the SNE cost function to spectral graph theory. In the small-scale limit (as the embedding collapses toward the origin), the KL-divergence objective reduces to the quadratic energy

$$E(Y) = \operatorname{tr}(Y^\top L\, Y), \qquad L = D(p) - P$$

where $P$ is the symmetric t-SNE affinity matrix and $D(p)$ is its diagonal row-sum matrix. This is the Rayleigh quotient for the graph Laplacian $L$, whose minimum over orthonormal $Y$ is achieved by the matrix whose columns are the eigenvectors corresponding to the $k$ smallest non-zero eigenvalues of $L$. Initializing t-SNE from these eigenvectors places the optimizer at the exact minimizer of the local quadratic approximation of the true objective — a theoretically grounded warm start that empirically reduces KL divergence and improves cluster separation relative to random initialization.

## Repository Structure

| File / Directory | Purpose |
|---|---|
| `tsne_spectral/` | Shared Python module — all core algorithm code |
| `Thesis_code.ipynb` | **Section 5** — Core demonstration: spectral vs. PCA vs. Laplacian on MNIST |
| `Gradient.ipynb` | **Section 6** — Comparative analysis across MNIST, COIL-20, Olivetti Faces |
| `More_data.ipynb` | **Section 7** — Extended experiments and UMAP comparison |
| `MNIST TSNE.ipynb` | **Section 4** — Gradient flow analysis and ODE dynamics |
| `Example 4,1.ipynb` | **Section 4.1** — Theoretical example: 3-point collapse under t-SNE gradient flow |
| `T-SNE.ipynb` | **Section 2** — Background: t-SNE algorithm review |
| `SD_tSNE.ipynb` | **Appendix B** — Supervised Dissimilarity t-SNE variant |
| `t-SNE_scratch.ipynb` | **Appendix A** — Experimental variants (tangent distance, single-scale, Daniel's kernel) |

### Module layout

```
tsne_spectral/
    __init__.py         Public API re-exports
    affinities.py       grid_search, get_original_pairwise_affinities, get_symmetric_p_ij
    initialization.py   spectral_init, laplacian_eigenmaps_init, initialization dispatcher
    tsne.py             get_low_dimensional_affinities, get_gradient, tsne_eigen
    utils.py            load_mnist, load_coil20, load_olivetti, preprocess, plot_embedding
```

## Installation

```bash
pip install -r requirements.txt
```

Python 3.9+ is required. Experiments on COIL-20, PBMC3K, and CIFAR-10 embeddings benefit from a machine with ≥ 16 GB RAM. All heavy t-SNE runs (T ≥ 500, n ≥ 1000) may take several minutes on CPU.

## Usage

```python
from tsne_spectral import tsne_eigen, load_mnist, plot_embedding

# Load MNIST — 1000 samples, standardised and PCA-reduced to 30 dims
X, y = load_mnist(n_samples=1000, pca_components=30)

# Run t-SNE with spectral initialisation (1000 gradient-descent steps)
Y = tsne_eigen(X, perplexity=30, T=1000, eta=500,
               initialization_method="spectral")

plot_embedding(Y, y, title="MNIST — Spectral Initialisation")
```

To compare all four initialisation methods and visualise only the starting points (no gradient descent), set `T=3`:

```python
for method in ["random", "PCA", "spectral", "laplacian"]:
    Y_init = tsne_eigen(X, initialization_method=method, T=3)
    plot_embedding(Y_init, y, title=f"Initialisation: {method}")
```

To load other datasets:

```python
from tsne_spectral import load_coil20, load_olivetti

X_coil, y_coil = load_coil20("coil-20-proc/", pca_components=40)
X_oli,  y_oli  = load_olivetti(pca_components=40)
```
