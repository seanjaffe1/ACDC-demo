"""ACDC dataset loader — parses NIfTI volumes and segmentation masks."""

from pathlib import Path
import nibabel as nib
import numpy as np


# ACDC label conventions
LABEL_RV = 1
LABEL_MYO = 2
LABEL_LV = 3

PATHOLOGY_CLASSES = {
    "NOR": 0,
    "MINF": 1,
    "DCM": 2,
    "HCM": 3,
    "RV": 4,
}


class ACDCSubject:
    def __init__(self, subject_dir: Path):
        self.subject_dir = Path(subject_dir)
        self.subject_id = self.subject_dir.name
        self._info = self._parse_info()

    def _parse_info(self) -> dict:
        info_path = self.subject_dir / "Info.cfg"
        info = {}
        if info_path.exists():
            for line in info_path.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    info[k.strip()] = v.strip()
        return info

    @property
    def ed_frame(self) -> int:
        return int(self._info.get("ED", 1))

    @property
    def es_frame(self) -> int:
        return int(self._info.get("ES", 1))

    @property
    def pathology(self) -> str:
        return self._info.get("Group", "NOR")

    @property
    def pathology_label(self) -> int:
        return PATHOLOGY_CLASSES.get(self.pathology, 0)

    def load_volume(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (image, voxel_spacing) for a given frame index."""
        pattern = f"{self.subject_id}_frame{frame:02d}.nii.gz"
        path = self.subject_dir / pattern
        img = nib.load(str(path))
        spacing = img.header.get_zooms()[:3]
        return img.get_fdata(), np.array(spacing, dtype=np.float32)

    def load_segmentation(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (seg_mask, voxel_spacing) for a given frame index."""
        pattern = f"{self.subject_id}_frame{frame:02d}_gt.nii.gz"
        path = self.subject_dir / pattern
        seg = nib.load(str(path))
        spacing = seg.header.get_zooms()[:3]
        return seg.get_fdata().astype(np.uint8), np.array(spacing, dtype=np.float32)

    def load_ed(self):
        return self.load_volume(self.ed_frame), self.load_segmentation(self.ed_frame)

    def load_es(self):
        return self.load_volume(self.es_frame), self.load_segmentation(self.es_frame)


class ACDCDataset:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.subjects = sorted(
            [ACDCSubject(d) for d in self.root.iterdir() if d.is_dir() and d.name.startswith("patient")],
            key=lambda s: s.subject_id,
        )

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, idx) -> ACDCSubject:
        return self.subjects[idx]

    def by_pathology(self, pathology: str) -> list[ACDCSubject]:
        return [s for s in self.subjects if s.pathology == pathology]

    def split(self, train_frac=0.7, val_frac=0.15, seed=42):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(self.subjects))
        n_train = int(len(idx) * train_frac)
        n_val = int(len(idx) * val_frac)
        train = [self.subjects[i] for i in idx[:n_train]]
        val = [self.subjects[i] for i in idx[n_train : n_train + n_val]]
        test = [self.subjects[i] for i in idx[n_train + n_val :]]
        return train, val, test
