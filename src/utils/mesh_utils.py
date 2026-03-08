"""Mesh utility functions: smoothing, normals, template helpers."""

import numpy as np
import trimesh


def laplacian_smooth(vertices: np.ndarray, faces: np.ndarray, iterations: int = 10, lam: float = 0.5) -> np.ndarray:
    """In-place Laplacian smoothing. Returns smoothed vertices."""
    mesh = trimesh.Trimesh(vertices=vertices.copy(), faces=faces, process=False)
    trimesh.smoothing.filter_laplacian(mesh, lamb=lam, iterations=iterations)
    return mesh.vertices


def compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / (norms + 1e-8)


def normalise_to_unit_sphere(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Centre and scale to unit sphere. Returns (normalised, centroid, scale)."""
    centroid = vertices.mean(0)
    v = vertices - centroid
    scale = np.linalg.norm(v, axis=-1).max()
    return v / scale, centroid, float(scale)


def denormalise(vertices: np.ndarray, centroid: np.ndarray, scale: float) -> np.ndarray:
    return vertices * scale + centroid


def mesh_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return float(0.5 * np.linalg.norm(cross, axis=-1).sum())
