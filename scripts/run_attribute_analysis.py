"""Post-hoc attribute-aware aggregation of a saved per-sample influence stack.

Requires an influence .npz that was produced with --save-stack. Loads the
WFLW test split to recover attribute flags (in record order), then computes
one K×K influence matrix per attribute subset (positive + negative) and
saves both raw matrices and comparison plots.

Usage:
    python -m scripts.run_attribute_analysis \
        --stack results/influence_matrix.npz \
        --checkpoint experiments/wflw_hrnet_w18_baseline/best.pth \
        --out-dir results/attribute_analysis
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.analysis.attributes import aggregate_influence_by_attribute, attribute_masks
from src.analysis.visualize import plot_influence_matrix, plot_region_influence
from src.datasets.wflw import ATTR_ORDER, WFLWDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stack", type=Path, required=True,
                        help="NPZ produced by run_influence.py --save-stack")
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Checkpoint whose config names the dataset root + regions")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    data = np.load(args.stack)
    if "occluded_stack" not in data.files or "baseline_per_sample" not in data.files:
        raise SystemExit("Stack NPZ missing per-sample arrays. "
                         "Re-run run_influence.py with --save-stack.")
    baseline_per_sample = data["baseline_per_sample"]  # (N, K)
    occluded = data["occluded_stack"]                   # (K, N, K)
    # Rebuild full (K+1, N, K)
    full = np.concatenate([baseline_per_sample[None, :, :], occluded], axis=0)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    ds = WFLWDataset(
        root=cfg["data"]["root"], split="test",
        image_size=cfg["data"]["image_size"],
        heatmap_size=cfg["data"]["heatmap_size"],
        heatmap_sigma=cfg["data"]["heatmap_sigma"],
        augment=False,
    )
    assert len(ds) == full.shape[1], (
        f"Dataset size {len(ds)} ≠ stack sample dim {full.shape[1]}")

    masks = attribute_masks(ds)
    regions = cfg["eval"]["regions"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Overall (all samples)
    overall_mask = np.ones(full.shape[1], dtype=bool)
    overall = aggregate_influence_by_attribute(full, overall_mask)
    np.save(args.out_dir / "influence_overall.npy", overall)
    plot_influence_matrix(overall, args.out_dir / "influence_overall.png", regions)
    plot_region_influence(overall, regions, args.out_dir / "region_influence_overall.png")

    # Per-attribute positive and negative subsets
    for attr in ATTR_ORDER:
        pos_mask = masks[attr]
        neg_mask = ~pos_mask
        n_pos, n_neg = int(pos_mask.sum()), int(neg_mask.sum())
        if n_pos == 0 or n_neg == 0:
            print(f"  {attr}: skipping (n_pos={n_pos}, n_neg={n_neg})")
            continue
        inf_pos = aggregate_influence_by_attribute(full, pos_mask)
        inf_neg = aggregate_influence_by_attribute(full, neg_mask)
        delta = inf_pos - inf_neg
        np.save(args.out_dir / f"influence_{attr}_pos.npy", inf_pos)
        np.save(args.out_dir / f"influence_{attr}_neg.npy", inf_neg)
        np.save(args.out_dir / f"influence_{attr}_delta.npy", delta)
        plot_influence_matrix(inf_pos, args.out_dir / f"influence_{attr}_pos.png", regions)
        plot_influence_matrix(inf_neg, args.out_dir / f"influence_{attr}_neg.png", regions)
        plot_influence_matrix(delta,   args.out_dir / f"influence_{attr}_delta.png", regions)
        plot_region_influence(inf_pos, regions, args.out_dir / f"region_influence_{attr}_pos.png")
        plot_region_influence(inf_neg, regions, args.out_dir / f"region_influence_{attr}_neg.png")
        plot_region_influence(delta,   regions, args.out_dir / f"region_influence_{attr}_delta.png")
        print(f"  {attr}: n_pos={n_pos} n_neg={n_neg} "
              f"max|delta|={float(np.abs(delta).max()):.4f}")

    print(f"\nSaved attribute analyses to {args.out_dir}")


if __name__ == "__main__":
    main()
