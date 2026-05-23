"""
tsne_spectral/utils.py

Data-loading helpers, shared preprocessing pipeline, and plotting utilities.
"""
from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(X: np.ndarray, pca_components: int = 30) -> np.ndarray:
    """
    Standard preprocessing pipeline: z-score standardisation then PCA.

    Parameters
    ----------
    X : np.ndarray of shape (n, d)
        Raw feature matrix.
    pca_components : int
        Number of principal components to retain.

    Returns
    -------
    np.ndarray of shape (n, pca_components)
        Preprocessed data.
    """
    X_scaled = StandardScaler().fit_transform(X)
    return PCA(n_components=pca_components).fit_transform(X_scaled)


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_mnist(
    n_samples: int = 1000,
    pca_components: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and preprocess MNIST digit images.

    Parameters
    ----------
    n_samples : int
        Number of training images (drawn from the first n_samples).
    pca_components : int
        PCA dimensionality after standardisation.

    Returns
    -------
    X : np.ndarray of shape (n_samples, pca_components)
    y : np.ndarray of shape (n_samples,)
        Integer class labels 0-9.
    """
    from tensorflow.keras.datasets import mnist as _mnist
    (x_train, y_train), _ = _mnist.load_data()
    x_train = x_train[:n_samples]
    y_train = y_train[:n_samples]
    X_flat = x_train.reshape(n_samples, -1).astype(np.float32)
    return preprocess(X_flat, pca_components), y_train


def load_coil20(
    data_dir: str,
    pca_components: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the COIL-20 dataset from a directory of PNG images.

    Expects filenames of the form ``obj<class>__<frame>.png``, which is
    the standard COIL-20 distribution layout.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the PNG files.
    pca_components : int
        PCA dimensionality after standardisation.

    Returns
    -------
    X : np.ndarray of shape (n_images, pca_components)
    y : np.ndarray of shape (n_images,)
        Integer class labels 1-20.
    """
    from PIL import Image

    images, labels = [], []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".png"):
            continue
        cls = int(fname.split("obj")[1].split("__")[0])
        img = np.array(
            Image.open(os.path.join(data_dir, fname)).convert("L"),
            dtype=np.float32,
        ).ravel()
        images.append(img)
        labels.append(cls)

    X = np.stack(images)
    y = np.array(labels)
    return preprocess(X, pca_components), y


def load_olivetti(pca_components: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Olivetti Faces dataset via scikit-learn.

    Parameters
    ----------
    pca_components : int
        PCA dimensionality after standardisation.

    Returns
    -------
    X : np.ndarray of shape (400, pca_components)
    y : np.ndarray of shape (400,)
        Integer subject IDs 0-39.
    """
    from sklearn.datasets import fetch_olivetti_faces

    data = fetch_olivetti_faces(shuffle=False)
    return preprocess(data.data, pca_components), data.target


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_embedding(
    Y: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Embedding",
    figsize: Tuple[int, int] = (8, 6),
    s: int = 10,
    alpha: float = 0.8,
    cmap: str = "tab10",
    save_path: str | None = None,
) -> None:
    """
    Scatter-plot a 2-D embedding coloured by class label.

    Parameters
    ----------
    Y : np.ndarray of shape (n, 2)
        2-D embedding coordinates.
    labels : np.ndarray of shape (n,)
        Integer class labels used for colouring.
    title : str
        Plot title.
    figsize : tuple of int
        Figure size in inches (width, height).
    s : int
        Marker size.
    alpha : float
        Marker opacity in [0, 1].
    cmap : str
        Matplotlib colormap name.
    save_path : str or None
        If provided, save the figure to this path instead of displaying.
    """
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(Y[:, 0], Y[:, 1], c=labels, cmap=cmap, s=s, alpha=alpha)
    plt.colorbar(scatter, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
