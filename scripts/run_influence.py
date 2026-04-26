"""Compute per-landmark influence matrix from a trained WFLW checkpoint.

Usage:
    python -m scripts.run_influence \
        --checkpoint experiments/wflw_hrnet_w18_baseline/best.pth \
        --out results/influence_matrix.npz \
        --radius 8 --batch-size 32
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ablation.influence import compute_per_sample_influence_stack
from src.datasets.wflw import WFLWDataset
from src.models.hrnet import HRNetLandmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--radius", type=float, default=8.0,
        help="Occlusion radius in image pixels (at image_size resolution)",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--save-stack", action="store_true",
                        help="Also save the full (K+1, N, K) per-sample NME stack")
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetLandmark(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ck["model"])

    val_ds = WFLWDataset(
        root=cfg["data"]["root"],
        split="test",
        image_size=cfg["data"]["image_size"],
        heatmap_size=cfg["data"]["heatmap_size"],
        heatmap_sigma=cfg["data"]["heatmap_sigma"],
        augment=False,
    )
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    scale_factor = cfg["data"]["image_size"] / cfg["data"]["heatmap_size"]
    res = compute_per_sample_influence_stack(
        model=model,
        loader=loader,
        num_landmarks=cfg["data"]["num_landmarks"],
        radius=args.radius,
        scale_factor=scale_factor,
        device=device,
        amp=args.amp,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "influence": res["influence"],
        "baseline_nme": res["baseline_nme"],
        "radius": np.float32(args.radius),
    }
    if args.save_stack:
        payload["baseline_per_sample"] = res["baseline_per_sample"]
        payload["occluded_stack"] = res["occluded_stack"]
    np.savez(args.out, **payload)
    print(f"Saved influence matrix ({res['influence'].shape}) to {args.out}")
    print(f"Baseline mean NME: {res['baseline_nme'].mean():.4f}")
    print(f"Max |influence|: {np.abs(res['influence']).max():.4f}")


if __name__ == "__main__":
    main()
