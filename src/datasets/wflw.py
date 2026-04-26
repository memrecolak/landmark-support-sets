"""WFLW dataset loader with bbox-crop, augmentation, and heatmap targets."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


# WFLW-98 horizontal-flip correspondence.
# Midpoint landmarks (16 on contour, 51-54 nose bridge, 57 nose tip, 79, 85, 94 mouth center) are self-paired.
WFLW_FLIP_PAIRS: tuple[tuple[int, int], ...] = tuple(
    [(i, 32 - i) for i in range(16)]
    + [(33, 46), (34, 45), (35, 44), (36, 43), (37, 42),
       (38, 50), (39, 49), (40, 48), (41, 47),
       (55, 59), (56, 58),
       (60, 72), (61, 71), (62, 70), (63, 69), (64, 68),
       (65, 75), (66, 74), (67, 73),
       (76, 82), (77, 81), (78, 80),
       (87, 83), (86, 84),
       (88, 92), (89, 91),
       (95, 93),
       (96, 97)]
)

ATTR_ORDER = ("pose", "expression", "illumination", "makeup", "occlusion", "blur")


@dataclass
class WFLWRecord:
    image: str
    bbox: tuple[float, float, float, float]
    landmarks: np.ndarray  # (98, 2) float32
    attributes: dict[str, int]


class WFLWDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: int = 256,
        heatmap_size: int = 64,
        heatmap_sigma: float = 1.5,
        augment: bool = False,
        aug_cfg: dict | None = None,
        crop_scale: float = 1.25,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.heatmap_sigma = heatmap_sigma
        self.augment = augment
        self.aug_cfg = aug_cfg or {}
        self.crop_scale = crop_scale

        json_path = self.root / "processed" / f"{split}.json"
        with json_path.open("r", encoding="utf-8") as fp:
            raw = json.load(fp)
        self.records: list[WFLWRecord] = [
            WFLWRecord(
                image=r["image"],
                bbox=tuple(r["bbox"]),
                landmarks=np.asarray(r["landmarks"], dtype=np.float32),
                attributes=r["attributes"],
            )
            for r in raw
        ]

    def __len__(self) -> int:
        return len(self.records)

    def _load_image(self, rec: WFLWRecord) -> np.ndarray:
        path = self.root / "WFLW_images" / rec.image
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to load {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _crop(
        self,
        img: np.ndarray,
        bbox: tuple[float, float, float, float],
        landmarks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            x1c += pad_l
            x2c += pad_l
            y1c += pad_t
            y2c += pad_t
            landmarks = landmarks.copy()
            landmarks[:, 0] += pad_l
            landmarks[:, 1] += pad_t

        crop = img[y1c:y2c, x1c:x2c]
        lm = landmarks.copy()
        lm[:, 0] -= x1c
        lm[:, 1] -= y1c

        factor = self.image_size / crop.shape[0]
        crop = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        lm *= factor
        return crop, lm

    def _augment(self, img: np.ndarray, lm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cfg = self.aug_cfg
        # Horizontal flip with landmark permutation
        if cfg.get("flip", False) and np.random.rand() < 0.5:
            img = img[:, ::-1, :].copy()
            lm = lm.copy()
            lm[:, 0] = self.image_size - 1 - lm[:, 0]
            for a, b in WFLW_FLIP_PAIRS:
                lm[[a, b]] = lm[[b, a]]

        # Affine: rotation + scale + translation
        rot_deg = float(cfg.get("rotate_deg", 0.0))
        scale_range = cfg.get("scale", [1.0, 1.0])
        trans_frac = float(cfg.get("translate", 0.0))
        angle = np.random.uniform(-rot_deg, rot_deg) if rot_deg > 0 else 0.0
        s = float(np.random.uniform(scale_range[0], scale_range[1]))
        tx = float(np.random.uniform(-trans_frac, trans_frac) * self.image_size) if trans_frac > 0 else 0.0
        ty = float(np.random.uniform(-trans_frac, trans_frac) * self.image_size) if trans_frac > 0 else 0.0
        center = (self.image_size / 2.0, self.image_size / 2.0)
        M = cv2.getRotationMatrix2D(center, angle, s)
        M[0, 2] += tx
        M[1, 2] += ty
        img = cv2.warpAffine(
            img, M, (self.image_size, self.image_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        ones = np.ones((lm.shape[0], 1), dtype=np.float32)
        lm = (np.concatenate([lm, ones], axis=1) @ M.T).astype(np.float32)

        # Color jitter (brightness + contrast)
        cj = float(cfg.get("color_jitter", 0.0))
        if cj > 0:
            brightness = 1.0 + float(np.random.uniform(-cj, cj))
            contrast = 1.0 + float(np.random.uniform(-cj, cj))
            imgf = img.astype(np.float32)
            mean = imgf.mean()
            imgf = (imgf - mean) * contrast + mean
            imgf = imgf * brightness
            img = np.clip(imgf, 0, 255).astype(np.uint8)
        return img, lm

    def _make_heatmaps(self, lm: np.ndarray) -> np.ndarray:
        K = lm.shape[0]
        H = W = self.heatmap_size
        sigma = self.heatmap_sigma
        scale = self.heatmap_size / self.image_size
        hmap = np.zeros((K, H, W), dtype=np.float32)
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)
        radius = 3.0 * sigma
        for k in range(K):
            cx = lm[k, 0] * scale
            cy = lm[k, 1] * scale
            if cx < -radius or cy < -radius or cx > W + radius or cy > H + radius:
                continue
            hmap[k] = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma))
        return hmap

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        img = self._load_image(rec)
        img, lm = self._crop(img, rec.bbox, rec.landmarks)
        if self.augment:
            img, lm = self._augment(img, lm)
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float() / 255.0
        img_t = (img_t - 0.5) / 0.5
        hmap = torch.from_numpy(self._make_heatmaps(lm))
        attrs = torch.tensor([rec.attributes[k] for k in ATTR_ORDER], dtype=torch.uint8)
        return {
            "image": img_t,
            "heatmap": hmap,
            "landmarks": torch.from_numpy(lm),
            "attributes": attrs,
        }
