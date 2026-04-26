"""Run HRNet-W18 on MPIIFaceGaze face crops and save predicted 98 landmarks.

Usage:
    python -m scripts.predict_landmarks_mpiifacegaze \
        --checkpoint experiments/wflw_hrnet_w18_baseline/best.pth \
        --mpii-root data/mpiifacegaze \
        --out results/mpiifacegaze_landmarks.npz

The saved file has:
    landmarks:       (N, 98, 2) float32 — in cropped-image pixel coords
    gaze_direction:  (N, 3)     float32 — unit vector, camera coords
    participants:    (N,)       str     — participant id (p00..p14)
    index:           (N,)       int64   — row index into processed/samples.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.mpiifacegaze import MPIIFaceGazeDataset
from src.models.hrnet import HRNetLandmark
from src.training.metrics import decode_heatmaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mpii-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    num_landmarks = cfg["data"]["num_landmarks"]
    image_size = cfg["data"]["image_size"]
    heatmap_size = cfg["data"]["heatmap_size"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetLandmark(num_landmarks=num_landmarks, pretrained=False).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    ds = MPIIFaceGazeDataset(root=args.mpii_root, image_size=image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)
    scale = image_size / heatmap_size

    all_lm = np.zeros((len(ds), num_landmarks, 2), dtype=np.float32)
    all_gaze = np.zeros((len(ds), 3), dtype=np.float32)
    all_part = [None] * len(ds)
    idx_cursor = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="predict"):
            img = batch["image"].to(device, non_blocking=True)
            hmap = model(img)
            coords_hm = decode_heatmaps(hmap)               # (B, K, 2) in heatmap px
            coords = (coords_hm.float() * scale).cpu().numpy()
            B = coords.shape[0]
            all_lm[idx_cursor:idx_cursor + B] = coords
            all_gaze[idx_cursor:idx_cursor + B] = batch["gaze_direction"].numpy()
            for j, p in enumerate(batch["participant"]):
                all_part[idx_cursor + j] = p
            idx_cursor += B

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        landmarks=all_lm,
        gaze_direction=all_gaze,
        participants=np.array(all_part, dtype=object),
        index=np.arange(len(ds), dtype=np.int64),
    )
    print(f"Saved {len(ds)} MPIIFaceGaze landmark predictions to {args.out}")


if __name__ == "__main__":
    main()
