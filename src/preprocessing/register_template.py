"""
Register all subject meshes to a template mesh via Procrustes + ICP.
Normalizes translation to centroid.

Usage:
    python -m src.preprocessing.register_template \
        --mesh_dir data/meshes \
        --out_dir data/registered \
        --template_id patient001
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree


def procrustes_align(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Kabsch algorithm: find rotation R and translation t that minimises
    ||target - (source @ R.T + t)||_F.
    Returns (R, t, scale).
    """
    mu_s = source.mean(0)
    mu_t = target.mean(0)
    src_c = source - mu_s
    tgt_c = target - mu_t

    # Scale to unit size
    scale = np.sqrt((tgt_c ** 2).sum() / (src_c ** 2).sum())
    src_c = src_c * scale

    H = src_c.T @ tgt_c
    U, _, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    t = mu_t - (src_c @ R.T).mean(0)  # approx
    t = mu_t - mu_s @ R.T
    return R, t, scale


def icp(source: np.ndarray, target: np.ndarray, max_iter: int = 50, tol: float = 1e-5) -> np.ndarray:
    """
    Iterative Closest Point — returns aligned source vertices.
    """
    src = source.copy()
    tree = cKDTree(target)
    prev_err = np.inf

    for _ in range(max_iter):
        _, idx = tree.query(src)
        matched = target[idx]
        R, t, _ = procrustes_align(src, matched)
        src = src @ R.T + t
        err = np.mean(np.linalg.norm(src - matched, axis=1))
        if abs(prev_err - err) < tol:
            break
        prev_err = err
    return src


def register_mesh(source_mesh: trimesh.Trimesh, template_verts: np.ndarray) -> trimesh.Trimesh:
    """Align source_mesh vertices to template_verts, return new mesh."""
    src_verts = source_mesh.vertices.copy()

    # Coarse Procrustes
    R, t, scale = procrustes_align(src_verts, template_verts)
    src_verts = src_verts @ R.T + t

    # Fine ICP
    src_verts = icp(src_verts, template_verts)

    # Normalise: subtract centroid
    src_verts -= src_verts.mean(0)

    return trimesh.Trimesh(vertices=src_verts, faces=source_mesh.faces, process=False)


def select_template(mesh_dir: Path, template_id: str, frame: str = "ed", struct: str = "lv") -> Path:
    return mesh_dir / template_id / f"{frame}_{struct}.ply"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_dir", default="data/meshes")
    parser.add_argument("--out_dir", default="data/registered")
    parser.add_argument("--template_id", default=None, help="Subject ID to use as template. Defaults to first subject.")
    parser.add_argument("--frame", default="ed")
    args = parser.parse_args()

    mesh_dir = Path(args.mesh_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    subject_dirs = sorted([d for d in mesh_dir.iterdir() if d.is_dir()])
    template_id = args.template_id or subject_dirs[0].name

    templates = {}
    for struct in ["lv", "rv", "myo"]:
        tmpl_path = mesh_dir / template_id / f"{args.frame}_{struct}.ply"
        tmpl_mesh = trimesh.load(str(tmpl_path), process=False)
        # Normalise template centroid
        tmpl_mesh.vertices -= tmpl_mesh.vertices.mean(0)
        templates[struct] = tmpl_mesh.vertices
        # Save template itself
        tmpl_out = out_dir / template_id
        tmpl_out.mkdir(parents=True, exist_ok=True)
        tmpl_mesh.export(str(tmpl_out / f"{args.frame}_{struct}.ply"))

    for subj_dir in subject_dirs:
        subj_id = subj_dir.name
        subj_out = out_dir / subj_id
        subj_out.mkdir(parents=True, exist_ok=True)

        for frame in ["ed", "es"]:
            for struct in ["lv", "rv", "myo"]:
                src_path = subj_dir / f"{frame}_{struct}.ply"
                out_path = subj_out / f"{frame}_{struct}.ply"
                if not src_path.exists() or out_path.exists():
                    continue
                src_mesh = trimesh.load(str(src_path), process=False)
                reg_mesh = register_mesh(src_mesh, templates[struct])
                reg_mesh.export(str(out_path))
                print(f"  {subj_id} {frame} {struct} → {out_path.name}")


if __name__ == "__main__":
    main()
