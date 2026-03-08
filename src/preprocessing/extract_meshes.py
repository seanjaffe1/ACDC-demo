"""
Marching cubes mesh extraction from ACDC segmentation masks.

Usage:
    python -m src.preprocessing.extract_meshes \
        --data_dir data/raw \
        --out_dir data/meshes \
        --target_vertices 2500
"""

import argparse
from pathlib import Path

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from src.utils.acdc_loader import ACDCDataset, LABEL_RV, LABEL_MYO, LABEL_LV


STRUCTURES = {
    "lv": LABEL_LV,
    "rv": LABEL_RV,
    "myo": LABEL_MYO,
}


def extract_mesh(seg: np.ndarray, label: int, spacing: np.ndarray) -> trimesh.Trimesh:
    """Run marching cubes on a binary mask and return a Trimesh."""
    binary = (seg == label).astype(np.float32)
    if binary.sum() == 0:
        return None
    verts, faces, normals, _ = marching_cubes(binary, level=0.5, spacing=spacing)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    return mesh


def smooth_mesh(mesh: trimesh.Trimesh, iterations: int = 10) -> trimesh.Trimesh:
    """Laplacian smoothing."""
    trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)
    return mesh


def decimate_mesh(mesh: trimesh.Trimesh, target_faces: int) -> trimesh.Trimesh:
    """Decimate to approximately target_faces faces."""
    if len(mesh.faces) <= target_faces:
        return mesh
    ratio = target_faces / len(mesh.faces)
    mesh = mesh.simplify_quadric_decimation(int(len(mesh.faces) * ratio))
    return mesh


def process_subject(subject, out_dir: Path, target_vertices: int = 2500, smooth_iter: int = 10):
    subj_out = out_dir / subject.subject_id
    subj_out.mkdir(parents=True, exist_ok=True)

    for frame_name, load_fn in [("ed", subject.load_ed), ("es", subject.load_es)]:
        (_, _), (seg, spacing) = load_fn()
        for struct_name, label in STRUCTURES.items():
            out_path = subj_out / f"{frame_name}_{struct_name}.ply"
            if out_path.exists():
                continue
            mesh = extract_mesh(seg, label, spacing)
            if mesh is None:
                print(f"  [WARN] {subject.subject_id} {frame_name} {struct_name}: empty mask")
                continue
            mesh = smooth_mesh(mesh, iterations=smooth_iter)
            # target_faces ≈ 2 * target_vertices (Euler: F ≈ 2V for closed mesh)
            mesh = decimate_mesh(mesh, target_faces=2 * target_vertices)
            mesh.export(str(out_path))
            print(f"  {subject.subject_id} {frame_name} {struct_name}: {len(mesh.vertices)} verts → {out_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/meshes")
    parser.add_argument("--target_vertices", type=int, default=2500)
    parser.add_argument("--smooth_iter", type=int, default=10)
    args = parser.parse_args()

    dataset = ACDCDataset(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, subject in enumerate(dataset.subjects):
        print(f"[{i+1}/{len(dataset)}] {subject.subject_id} ({subject.pathology})")
        process_subject(subject, out_dir, args.target_vertices, args.smooth_iter)


if __name__ == "__main__":
    main()
