"""
Build PyTorch Geometric graph objects from registered meshes.

Each graph stores:
  - pos  : (V, 3) float32 vertex positions
  - edge_index : (2, E) k-NN edges
  - face : (3, F) mesh faces (for surface area / Dice evaluation)
  - y    : int pathology class label
  - subject_id : str
  - x    : (V, 4) optional curvature node features [k1, k2, H, K]
             where k1/k2 are principal curvatures, H = mean curvature,
             K = Gaussian curvature. All are SE(3)-invariant scalars.
             Only present when --curvature flag is passed.

Usage:
    python -m src.preprocessing.build_graphs \
        --reg_dir data/registered \
        --data_dir data/raw \
        --out_dir data/graphs \
        --k 8 [--curvature]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import trimesh
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph

from src.utils.acdc_loader import ACDCDataset, PATHOLOGY_CLASSES


def compute_curvature_features(mesh: trimesh.Trimesh) -> np.ndarray | None:
    """
    Compute per-vertex curvature features using IGL principal curvature.

    Returns a (V, 4) float32 array with columns:
        [k1, k2, H=(k1+k2)/2, K=k1*k2]
    where k1 >= k2 are the principal curvatures.

    All four quantities are SE(3)-invariant scalars — they are purely
    intrinsic surface properties and transform trivially (unchanged) under
    rigid body motions.  This means appending them as node features
    preserves the SE(3)-equivariance of the EGNN.

    Returns None if igl is unavailable (falls back to constant features).
    """
    try:
        import igl  # optional dependency
    except ImportError:
        return None

    V = mesh.vertices.astype(np.float64)
    F = mesh.faces.astype(np.int64)
    pd1, pd2, pv1, pv2, bad = igl.principal_curvature(V, F)

    # pv1 = max curvature (k1), pv2 = min curvature (k2)
    H = (pv1 + pv2) / 2.0   # mean curvature
    K = pv1 * pv2             # Gaussian curvature

    feats = np.stack([pv1, pv2, H, K], axis=1).astype(np.float32)

    # Replace any NaNs / Infs at bad vertices with zero (safe fallback)
    if len(bad) > 0:
        feats[bad] = 0.0
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)

    return feats


def mesh_to_graph(
    mesh: trimesh.Trimesh,
    k: int,
    y: int,
    subject_id: str,
    frame: str,
    struct: str,
    use_curvature: bool = False,
) -> Data:
    pos = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces.T, dtype=torch.long)  # (3, F)
    edge_index = knn_graph(pos, k=k, loop=False)

    x = None
    if use_curvature:
        curv = compute_curvature_features(mesh)
        if curv is not None:
            x = torch.tensor(curv, dtype=torch.float32)  # (V, 4)
        else:
            # Fallback: constant node features (same as baseline)
            x = torch.ones(pos.shape[0], 1, dtype=torch.float32)

    return Data(
        pos=pos,
        edge_index=edge_index,
        face=faces,
        x=x,
        y=torch.tensor(y, dtype=torch.long),
        subject_id=subject_id,
        frame=frame,
        struct=struct,
        num_nodes=pos.shape[0],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reg_dir", default="data/registered")
    parser.add_argument("--data_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/graphs")
    parser.add_argument("--k", type=int, default=8, help="k for k-NN graph")
    parser.add_argument(
        "--curvature", action="store_true",
        help="Compute and store per-vertex curvature features [k1, k2, H, K] as node features x. "
             "Requires libigl (`pip install libigl`). Output graphs will have in_node_dim=4."
    )
    args = parser.parse_args()

    reg_dir = Path(args.reg_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.curvature:
        try:
            import igl  # noqa: F401
            print("Curvature features enabled (libigl found).")
        except ImportError:
            print("WARNING: --curvature requested but libigl not installed. Falling back to constant features.")

    dataset = ACDCDataset(args.data_dir)
    pathology_map = {s.subject_id: s.pathology_label for s in dataset.subjects}

    manifest = []
    for subj_dir in sorted(reg_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        subj_id = subj_dir.name
        y = pathology_map.get(subj_id, 0)

        for frame in ["ed", "es"]:
            for struct in ["lv", "rv", "myo"]:
                mesh_path = subj_dir / f"{frame}_{struct}.ply"
                if not mesh_path.exists():
                    continue
                mesh = trimesh.load(str(mesh_path), process=False)
                graph = mesh_to_graph(
                    mesh, k=args.k, y=y, subject_id=subj_id,
                    frame=frame, struct=struct, use_curvature=args.curvature,
                )

                out_name = f"{subj_id}_{frame}_{struct}.pt"
                torch.save(graph, out_dir / out_name)
                in_node_dim = graph.x.shape[1] if graph.x is not None else 0
                manifest.append({
                    "file": out_name, "subject_id": subj_id, "frame": frame,
                    "struct": struct, "y": y, "in_node_dim": in_node_dim,
                })
                print(f"  {out_name}: {graph.num_nodes} nodes, {graph.edge_index.shape[1]} edges, in_node_dim={in_node_dim}")

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved {len(manifest)} graphs to {out_dir}")


if __name__ == "__main__":
    main()
