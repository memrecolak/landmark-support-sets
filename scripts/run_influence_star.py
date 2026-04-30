"""Per-landmark influence matrix from the STAR-Loss WFLW checkpoint.

Mirrors scripts/run_influence.py but swaps HRNet for STARLandmark. Used for
the T2 cross-architecture sanity check: do |S|<98, contour-largest, and
contour > nose hold on a Stacked-Hourglass + AAM detector trained with
STAR loss, the same way they hold on our HRNet-W18 baseline?

Same WFLW test split (N=2500), same crop convention, same r=8 occlusion,
same downstream NME (decode_heatmaps + scale_factor=4) — only the detector
weights/architecture differ.

Usage:
    python -m scripts.run_influence_star \
        --weights external/STAR/weights/star_wflw.pkl \
        --out results/influence_matrix_star.npz \
        --radius 8 --batch-size 32 --save-stack
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ablation.influence import compute_per_sample_influence_stack
from src.datasets.wflw import WFLWDataset
from src.models.star import STARLandmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path,
        default=Path("external/STAR/weights/star_wflw.pkl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/wflw"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--save-stack", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}, amp: {args.amp}")

    model = STARLandmark(weights_path=args.weights, use_ema=True).to(device)
    model.eval()
    print("loaded STAR (net_ema)")

    val_ds = WFLWDataset(
        root=args.data_root,
        split="test",
        image_size=STARLandmark.INPUT_SIZE,
        heatmap_size=STARLandmark.HEATMAP_SIZE,
        heatmap_sigma=1.5,
        augment=False,
    )
    print(f"WFLW test: {len(val_ds)} samples")

    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    scale_factor = STARLandmark.INPUT_SIZE / STARLandmark.HEATMAP_SIZE
    res = compute_per_sample_influence_stack(
        model=model,
        loader=loader,
        num_landmarks=STARLandmark.NUM_LANDMARKS,
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
    print(f"\nSaved influence matrix ({res['influence'].shape}) to {args.out}")
    print(f"Baseline mean NME: {res['baseline_nme'].mean():.4f}")
    print(f"Max |influence|: {np.abs(res['influence']).max():.4f}")


if __name__ == "__main__":
    main()
