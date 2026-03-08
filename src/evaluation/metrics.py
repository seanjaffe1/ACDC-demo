"""Core evaluation metrics."""

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance_numpy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Symmetric Chamfer distance in mm (assumes coords are in mm).
    pred, target: (N, 3)
    """
    tree_t = cKDTree(target)
    tree_p = cKDTree(pred)
    d_pt, _ = tree_t.query(pred)
    d_tp, _ = tree_p.query(target)
    return float((d_pt.mean() + d_tp.mean()) / 2)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """95th-percentile Hausdorff distance."""
    tree_t = cKDTree(target)
    tree_p = cKDTree(pred)
    d_pt, _ = tree_t.query(pred)
    d_tp, _ = tree_p.query(target)
    return float(max(np.percentile(d_pt, 95), np.percentile(d_tp, 95)))


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """
    Signed volume via divergence theorem (assumes closed, consistently oriented mesh).
    Returns volume in mm^3 (if vertices in mm).
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    signed_vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
    return abs(float(signed_vol))


def ejection_fraction(lv_ed_verts: np.ndarray, lv_ed_faces: np.ndarray,
                       lv_es_verts: np.ndarray, lv_es_faces: np.ndarray) -> float:
    """EF = (EDV - ESV) / EDV * 100 [%]"""
    edv = mesh_volume(lv_ed_verts, lv_ed_faces)
    esv = mesh_volume(lv_es_verts, lv_es_faces)
    return (edv - esv) / edv * 100.0


def one_nearest_neighbour_accuracy(real: np.ndarray, generated: np.ndarray) -> float:
    """
    1-NNA: for each shape in real∪generated, find nearest neighbour.
    Perfect generation → 0.5. Memorisation/mode collapse → 1.0.
    real, generated: (N, D) flattened shape arrays.
    """
    n_real = len(real)
    n_gen = len(generated)
    all_shapes = np.concatenate([real, generated], axis=0)
    labels = np.array([0] * n_real + [1] * n_gen)

    tree = cKDTree(all_shapes)
    # Query 2 neighbours (first is self)
    _, idx = tree.query(all_shapes, k=2)
    nn_idx = idx[:, 1]
    nn_labels = labels[nn_idx]

    correct = (nn_labels == labels).sum()
    return float(correct) / len(labels)


def linear_probe_accuracy(latents: np.ndarray, labels: np.ndarray, n_classes: int = 5) -> float:
    """
    Fit logistic regression on latent codes and return 5-fold CV accuracy.
    Tests pathology separability of the latent space.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(latents)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    accs = []
    for train_idx, val_idx in cv.split(X, labels):
        clf.fit(X[train_idx], labels[train_idx])
        accs.append(clf.score(X[val_idx], labels[val_idx]))
    return float(np.mean(accs))
