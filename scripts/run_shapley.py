"""Run Monte Carlo Shapley attribution on a trained WFLW checkpoint.

Iterates the target regions defined in the checkpoint's config and produces
one Shapley vector per region, saved to a compact NPZ. Target-region members
receive zero by convention (they are never occluded).

Usage:
    python -m scripts.run_shapley \
        --checkpoint experiments/wflw_hrnet_w18_baseline/best.pth \
        --out results/shapley.npz \
        --permutations 200 --subsample 256 --batch-size 32 \
        --regions left_eye right_eye
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from src.ablation.shapley import permutation_shapley
from src.datasets.wflw import WFLWDataset
from src.models.hrnet import HRNetLandmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument("--subsample", type=int, default=256,
                        help="Number of test images to evaluate each prefix on")
    parser.add_argument("--truncation-eps", type=float, default=5e-5)
    parser.add_argument("--min-prefix", type=int, default=8,
                        help="Minimum prefix length before truncation is allowed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--regions", nargs="+", default=None,
                        help="Subset of target-region names (default: all in config)")
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetLandmark(
        num_landmarks=cfg["data"]["num_landmarks"], pretrained=False,
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    val_ds = WFLWDataset(
        root=cfg["data"]["root"], split="test",
        image_size=cfg["data"]["image_size"], heatmap_size=cfg["data"]["heatmap_size"],
        heatmap_sigma=cfg["data"]["heatmap_sigma"], augment=False,
    )
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(val_ds), size=min(args.subsample, len(val_ds)), replace=False)
    subset = Subset(val_ds, idx.tolist())
    loader = DataLoader(
        subset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    scale_factor = cfg["data"]["image_size"] / cfg["data"]["heatmap_size"]
    regions_cfg: dict = cfg["eval"]["regions"]
    if args.regions:
        regions_cfg = {name: regions_cfg[name] for name in args.regions}

    region_names = list(regions_cfg.keys())
    K = cfg["data"]["num_landmarks"]
    shapley_all = np.zeros((len(region_names), K), dtype=np.float32)
    counts_all = np.zeros((len(region_names), K), dtype=np.int64)
    base_scores: dict[str, float] = {}
    truncated: dict[str, int] = {}

    for ri, (name, target_indices) in enumerate(regions_cfg.items()):
        print(f"\n=== Shapley for target region: {name} ({len(target_indices)} landmarks) ===")
        res = permutation_shapley(
            model=model, loader=loader,
            num_landmarks=K, target_indices=target_indices,
            radius=args.radius, scale_factor=scale_factor, device=device,
            num_permutations=args.permutations,
            truncation_eps=args.truncation_eps,
            min_prefix_before_truncate=args.min_prefix,
            seed=args.seed,
        )
        shapley_all[ri] = res["shapley"]
        counts_all[ri] = res["counts"]
        base_scores[name] = res["base_score"]
        truncated[name] = res["truncated_permutations"]
        topk = np.argsort(res["shapley"])[-10:][::-1]
        print(f"  base NME: {res['base_score']:.4f}, "
              f"truncated: {res['truncated_permutations']}/{args.permutations}")
        print(f"  top-10 by Shapley: {topk.tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        shapley=shapley_all,
        counts=counts_all,
        region_names=np.array(region_names),
        base_scores=np.array([base_scores[n] for n in region_names], dtype=np.float32),
        truncated=np.array([truncated[n] for n in region_names], dtype=np.int64),
        radius=np.float32(args.radius),
        permutations=np.int32(args.permutations),
        subsample=np.int32(args.subsample),
    )
    print(f"\nSaved Shapley values to {args.out}")
    print(f"  shape: {shapley_all.shape} (regions × landmarks)")


if __name__ == "__main__":
    main()
