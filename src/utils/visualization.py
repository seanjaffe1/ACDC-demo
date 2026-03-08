"""Visualisation utilities: mesh rendering, latent space plots."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def plot_mesh(vertices: np.ndarray, faces: np.ndarray, title: str = "", ax=None, color: str = "steelblue", alpha: float = 0.6):
    """Plot a 3D mesh using matplotlib."""
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    if ax is None:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")

    tris = vertices[faces]
    poly = Poly3DCollection(tris, alpha=alpha, facecolor=color, edgecolor="none")
    ax.add_collection3d(poly)

    lims = np.array([vertices.min(0), vertices.max(0)])
    for dim, setter in enumerate([ax.set_xlim, ax.set_ylim, ax.set_zlim]):
        setter(lims[0, dim], lims[1, dim])

    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    return ax


def plot_latent_space(latents: np.ndarray, labels: np.ndarray, label_names: list[str] | None = None,
                       method: str = "umap", out_path: str | None = None):
    """2D embedding of latent codes coloured by pathology class."""
    if method == "umap":
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding = reducer.fit_transform(latents)
        except ImportError:
            method = "tsne"

    if method == "tsne":
        from sklearn.manifold import TSNE
        embedding = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1)).fit_transform(latents)

    fig, ax = plt.subplots(figsize=(8, 6))
    classes = np.unique(labels)
    cmap = plt.cm.get_cmap("tab10", len(classes))
    for i, cls in enumerate(classes):
        mask = labels == cls
        name = label_names[cls] if label_names else str(cls)
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[cmap(i)], label=name, s=30, alpha=0.8)

    ax.legend()
    ax.set_title(f"Latent space ({method.upper()})")
    ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_reconstruction_comparison(real_verts: np.ndarray, pred_verts: np.ndarray,
                                    faces: np.ndarray, out_path: str | None = None):
    """Side-by-side real vs. reconstructed mesh."""
    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    plot_mesh(real_verts, faces, title="Ground truth", ax=ax1, color="steelblue")
    plot_mesh(pred_verts, faces, title="Reconstruction", ax=ax2, color="coral")
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metric_table(results: dict, out_path: str | None = None):
    """Print and optionally save a comparison table of all model metrics."""
    import pandas as pd
    rows = []
    for model_name, metrics in results.items():
        row = {"model": model_name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index("model")
    print(df.to_string())
    if out_path:
        df.to_csv(out_path)
    return df
