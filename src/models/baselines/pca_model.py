"""
Baseline 1: PCA Shape Model.

Flattens vertex positions → PCA → Gaussian in latent space.
Justifies using deep learning at all.
"""

import numpy as np
from sklearn.decomposition import PCA


class PCAShapeModel:
    def __init__(self, n_components: int = 50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.mean_shape: np.ndarray | None = None
        self.latent_std: np.ndarray | None = None

    def fit(self, shapes: np.ndarray):
        """
        Args:
            shapes: (N, V, 3) array of registered mesh vertex positions.
        """
        N, V, _ = shapes.shape
        flat = shapes.reshape(N, -1)  # (N, V*3)
        self.mean_shape = flat.mean(0)
        self.pca.fit(flat)
        codes = self.pca.transform(flat)
        self.latent_std = codes.std(0)

    def encode(self, shape: np.ndarray) -> np.ndarray:
        """shape: (V, 3) → latent (n_components,)"""
        flat = shape.reshape(1, -1) - self.mean_shape
        return self.pca.transform(flat)[0]

    def decode(self, code: np.ndarray, V: int) -> np.ndarray:
        """code: (n_components,) → shape (V, 3)"""
        flat = self.pca.inverse_transform(code[None]) + self.mean_shape
        return flat.reshape(V, 3)

    def reconstruct(self, shape: np.ndarray) -> np.ndarray:
        code = self.encode(shape)
        return self.decode(code, shape.shape[0])

    def sample(self, n: int, V: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample n shapes from fitted Gaussian. Returns (n, V, 3)."""
        if rng is None:
            rng = np.random.default_rng()
        codes = rng.standard_normal((n, self.n_components)) * self.latent_std
        return np.stack([self.decode(c, V) for c in codes])

    def explained_variance_ratio(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_
