"""Run influence-ordered backward elimination for every target region defined
in the checkpoint's config, and emit both raw trajectories and plots.

Usage:
    python -m scripts.run_elimination \
        --checkpoint experiments/wflw_hrnet_w18_baseline/best.pth \
        --influence results/influence_matrix.npz \
        --out-dir results/elimination
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.ablation.greedy import influence_ordered_elimination, support_set_at_tolerance
from src.analysis.visualize import plot_elimination_trajectories
from src.datasets.wflw import WFLWDataset
from src.models.hrnet import HRNetLandmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--influence", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Target-NME slack for support-set extraction")
    parser.add_argument("--regions", nargs="+", default=None,
                        help="Subset of target-region names to run (default: all in config)")
    args = parser.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    influence = np.load(args.influence)["influence"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetLandmark(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=False,
    ).to(device)
    model.load_state_dict(ck["model"])

    val_ds = WFLWDataset(
        root=cfg["data"]["root"], split="test",
        image_size=cfg["data"]["image_size"], heatmap_size=cfg["data"]["heatmap_size"],
        heatmap_sigma=cfg["data"]["heatmap_sigma"], augment=False,
    )
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    scale_factor = cfg["data"]["image_size"] / cfg["data"]["heatmap_size"]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = cfg["eval"]["regions"]
    if args.regions:
        regions = {name: regions[name] for name in args.regions}
    num_landmarks = cfg["data"]["num_landmarks"]
    all_results = {}
    for name, idx in regions.items():
        print(f"\n=== Elimination target: {name} ({len(idx)} landmarks) ===")
        res = influence_ordered_elimination(
            model=model, loader=loader, influence=influence,
            target_indices=idx, radius=args.radius,
            scale_factor=scale_factor, device=device,
        )
        all_results[name] = res
        support = support_set_at_tolerance(res["trajectory"], num_landmarks, args.tolerance)
        print(f"  baseline NME: {res['baseline_nme']:.4f}")
        print(f"  support set ({len(support)} landmarks @ tol={args.tolerance}): {support}")

    # Serialize
    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            "baseline_nme": res["baseline_nme"],
            "order": res["order"],
            "trajectory": res["trajectory"],
            "support_set": support_set_at_tolerance(res["trajectory"], num_landmarks, args.tolerance),
            "tolerance": args.tolerance,
        }
    with (args.out_dir / "trajectories.json").open("w") as fp:
        json.dump(serializable, fp, indent=2)

    plot_elimination_trajectories(all_results, args.out_dir / "trajectories.png")
    print(f"\nSaved trajectories.json and trajectories.png to {args.out_dir}")


if __name__ == "__main__":
    main()
