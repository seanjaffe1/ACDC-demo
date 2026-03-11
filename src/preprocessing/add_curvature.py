"""
Augment existing PyG graph files with per-vertex curvature node features.

Loads registered mesh PLY files, computes principal curvatures via libigl,
and saves new graph objects to a separate directory.  The original graphs
in data/graphs/ are left untouched (backward compatible).

Output graphs have x = (V, 5) tensor: [k1, k2, H=(k1+k2)/2, K=k1*k2, chi].
chi = sign((pd1 x pd2).n) is the handedness pseudoscalar (SO(3)-invariant,
O(3)-sensitive). Principal direction vectors pd1/pd2 are stored as
data.principal_dir1 / data.principal_dir2 (V, 3) equivariant attributes.

Usage:
    python -m src.preprocessing.add_curvature \
        --graph_dir data/graphs \
        --reg_dir data/registered \
        --out_dir data/graphs_curvature
"""

import argparse
import json
from pathlib import Path

import igl
import numpy as np
import torch
import trimesh


def compute_curvature_features(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute (V, 5) curvature feature matrix: [k1, k2, H, K, chi].

    k1  = max principal curvature (most positive / least negative)
    k2  = min principal curvature
    H   = mean curvature = (k1 + k2) / 2
    K   = Gaussian curvature = k1 * k2
    chi = handedness pseudoscalar = sign((pd1 x pd2) . n)
          +1 if the principal frame is right-handed, -1 if left-handed.
          SO(3)-invariant but O(3)-sensitive: flips sign under reflection.

    Also returns principal direction vectors pd1 (V, 3) and pd2 (V, 3)
    as separate equivariant quantities (they co-rotate with the mesh).

    Bad vertices and NaN/Inf values are zeroed out.
    """
    V = mesh.vertices.astype(np.float64)
    F = mesh.faces.astype(np.int64)
    pd1, pd2, pv1, pv2, bad = igl.principal_curvature(V, F)
    H = (pv1 + pv2) / 2.0
    K = pv1 * pv2

    normals = np.array(mesh.vertex_normals, dtype=np.float64)
    chi = np.sign(np.sum(np.cross(pd1, pd2) * normals, axis=1, keepdims=True))
    chi[chi == 0] = 1.0  # assign consistent handedness at flat points

    feats = np.concatenate(
        [np.stack([pv1, pv2, H, K], axis=1), chi], axis=1
    ).astype(np.float32)
    if len(bad) > 0:
        feats[bad] = 0.0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    pd1 = np.nan_to_num(pd1.astype(np.float32), nan=0.0)
    pd2 = np.nan_to_num(pd2.astype(np.float32), nan=0.0)
    if len(bad) > 0:
        pd1[bad] = 0.0
        pd2[bad] = 0.0

    return feats, pd1, pd2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph_dir", default="data/graphs")
    parser.add_argument("--reg_dir", default="data/registered")
    parser.add_argument("--out_dir", default="data/graphs_curvature")
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    reg_dir = Path(args.reg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(graph_dir / "manifest.json") as f:
        manifest = json.load(f)

    new_manifest = []
    n_ok = n_skip = n_fail = 0

    for entry in manifest:
        fname = entry["file"]
        subj_id = entry["subject_id"]
        frame = entry["frame"]
        struct = entry["struct"]

        mesh_path = reg_dir / subj_id / f"{frame}_{struct}.ply"
        if not mesh_path.exists():
            print(f"  SKIP (no mesh): {mesh_path}")
            n_skip += 1
            continue

        try:
            mesh = trimesh.load(str(mesh_path), process=False)
            feats, pd1, pd2 = compute_curvature_features(mesh)

            # Load existing graph and attach curvature features
            data = torch.load(graph_dir / fname, weights_only=False)
            data.x = torch.tensor(feats, dtype=torch.float32)            # (V, 5)
            data.principal_dir1 = torch.tensor(pd1, dtype=torch.float32)  # (V, 3)
            data.principal_dir2 = torch.tensor(pd2, dtype=torch.float32)  # (V, 3)

            torch.save(data, out_dir / fname)
            new_entry = dict(entry)
            new_entry["in_node_dim"] = feats.shape[1]   # 5: [k1, k2, H, K, chi]
            new_manifest.append(new_entry)
            n_ok += 1
            if n_ok % 50 == 0:
                print(f"  Processed {n_ok}/{len(manifest)} ...")
        except Exception as e:
            print(f"  FAIL: {fname} — {e}")
            n_fail += 1

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(new_manifest, f, indent=2)

    print(f"\nDone: {n_ok} augmented, {n_skip} skipped, {n_fail} failed")
    print(f"Graphs saved to {out_dir}")


if __name__ == "__main__":
    main()
