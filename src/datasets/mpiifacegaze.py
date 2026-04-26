"""MPIIFaceGaze dataset loader.

Produces square face crops (matching HRNet-W18 256x256 input convention) plus
the 3D gaze direction (normalized vector from the 3D face center to the 3D
gaze target, per the dataset readme: gaze_direction = gt - fc). Intended for
two uses:

1. Batch inference: feed the cropped image to HRNet-W18 to obtain 98 predicted
   landmark coordinates in the cropped-image frame; saved as the per-sample
   landmark array used by the gaze regressor.
2. Downstream gaze evaluation: the gaze regressor itself does not re-read
   images — it consumes the saved (N, 98, 2) landmark array. This Dataset is
   only instantiated for the landmark-prediction step.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class MPIIFaceGazeRecord:
    participant: str
    image: str
    bbox: tuple[float, float, float, float]
    gaze_target_3d: np.ndarray       # (3,) — gt, camera coords
    face_center_3d: np.ndarray       # (3,) — fc, camera coords (gaze origin)
    landmarks_6: np.ndarray          # (6, 2) in raw-image coords
    eye_side: str                    # "left" or "right" (from readme)


class MPIIFaceGazeDataset(Dataset):
    """Face-cropped MPIIFaceGaze samples.

    Returns per item:
        image: (3, H, W) float tensor, normalized to [-1, 1]
        crop_bbox: (4,) bbox used for the crop in raw-image coords
        gaze_direction: (3,) unit vector in camera coords (gaze_target - face_center)
        participant: str
    """

    def __init__(
        self,
        root: str | Path,
        image_size: int = 256,
        crop_scale: float = 1.0,
        participants: list[str] | None = None,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.crop_scale = crop_scale

        json_path = self.root / "processed" / "samples.json"
        with json_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        if participants is not None:
            wanted = set(participants)
            raw = [r for r in raw if r["participant"] in wanted]
        self.records: list[MPIIFaceGazeRecord] = [
            MPIIFaceGazeRecord(
                participant=r["participant"],
                image=r["image"],
                bbox=tuple(r["bbox_from_landmarks"]),
                gaze_target_3d=np.asarray(r["gaze_target_3d"], dtype=np.float32),
                face_center_3d=np.asarray(r["face_center_3d"], dtype=np.float32),
                landmarks_6=np.asarray(r["landmarks_6"], dtype=np.float32),
                eye_side=r["eye_side"],
            )
            for r in raw
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rec: MPIIFaceGazeRecord) -> np.ndarray:
        path = self.root / rec.image
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _crop(
        self,
        img: np.ndarray,
        bbox: tuple[float, float, float, float],
    ) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        size = max(x2 - x1, y2 - y1) * self.crop_scale
        x1c = int(round(cx - size / 2))
        y1c = int(round(cy - size / 2))
        side = int(round(size))
        x2c = x1c + side
        y2c = y1c + side

        H, W = img.shape[:2]
        pad_l = max(0, -x1c)
        pad_t = max(0, -y1c)
        pad_r = max(0, x2c - W)
        pad_b = max(0, y2c - H)
        if pad_l or pad_t or pad_r or pad_b:
            img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x1c += pad_l
            x2c += pad_l
            y1c += pad_t
            y2c += pad_t

        crop = img[y1c:y2c, x1c:x2c]
        crop = cv2.resize(crop, (self.image_size, self.image_size),
                          interpolation=cv2.INTER_LINEAR)
        return crop, (x1c - pad_l, y1c - pad_t, x2c - pad_l, y2c - pad_t)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = self._load_image(rec)
        crop, crop_bbox = self._crop(img, rec.bbox)
        img_t = torch.from_numpy(crop.transpose(2, 0, 1).copy()).float() / 255.0
        img_t = (img_t - 0.5) / 0.5

        direction = rec.gaze_target_3d - rec.face_center_3d
        norm = float(np.linalg.norm(direction))
        if norm > 1e-8:
            direction = direction / norm

        return {
            "image": img_t,
            "crop_bbox": torch.tensor(crop_bbox, dtype=torch.float32),
            "gaze_direction": torch.from_numpy(direction.astype(np.float32)),
            "participant": rec.participant,
            "index": idx,
        }
