"""T2: backward-greedy elimination on STAR for the cross-arch sanity check.

Mirrors scripts/run_elimination.py but uses STARLandmark + STAR's influence
matrix. Reuses the same WFLW-test split, the same r=8 occlusion radius,
the same τ=0.005 NME tolerance, and the same 8 region definitions baked
into the HRNet baseline config.

The 3 robust HRNet claims (which survive every (seed, radius, τ) cell of
the robustness grid) are:
  (i)   |S_region| < 98 for every region
  (ii)  contour requires the largest support
  (iii) contour > nose

After this script writes results/star/trajectories_star.json, the companion
check_robust_claims_star.py (run separately) prints whether each claim
holds on STAR's support sets.

Usage:
    python -m scripts.run_elimination_star \
        --weights external/STAR/weights/star_wflw.pkl \
        --influence results/star/influence_matrix_star.npz \
        --out-dir results/star
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
from src.models.star import STARLandmark


# Same regions as the HRNet baseline (extracted from
# experiments/wflw_hrnet_w18_baseline/best.pth :: cfg.eval.regions).
WFLW_REGIONS: dict[str, list[int]] = {
    "contour":     list(range(0, 33)),
    "left_brow":   [33, 34, 35, 36, 37, 38, 39, 40, 41],
    "right_brow":  [42, 43, 44, 45, 46, 47, 48, 49, 50],
    "nose":        [51, 52, 53, 54, 55, 56, 57, 58, 59],
    "left_eye":    [60, 61, 62, 63, 64, 65, 66, 67, 96],
    "right_eye":   [68, 69, 70, 71, 72, 73, 74, 75, 97],
    "mouth_outer": [76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87],
    "mouth_inner": [88, 89, 90, 91, 92, 93, 94, 95],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path,
        default=Path("external/STAR/weights/star_wflw.pkl"),
    )
    parser.add_argument(
        "--influence", type=Path,
        default=Path("results/star/influence_matrix_star.npz"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/wflw"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/star"))
    parser.add_argument("--radius", type=float, default=8.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--regions", nargs="+", default=None)
    args = parser.parse_args()

    influence = np.load(args.influence)["influence"]
    assert influence.shape == (98, 98), influence.shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = STARLandmark(weights_path=args.weights, use_ema=True).to(device).eval()

    val_ds = WFLWDataset(
        root=args.data_root, split="test",
        image_size=STARLandmark.INPUT_SIZE,
        heatmap_size=STARLandmark.HEATMAP_SIZE,
        heatmap_sigma=1.5, augment=False,
    )
    loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    scale_factor = STARLandmark.INPUT_SIZE / STARLandmark.HEATMAP_SIZE

    args.out_dir.mkdir(parents=True, exist_ok=True)
    regions = WFLW_REGIONS
    if args.regions:
        regions = {name: regions[name] for name in args.regions}

    all_results = {}
    for name, idx in regions.items():
        print(f"\n=== Elimination target: {name} ({len(idx)} landmarks) ===")
        res = influence_ordered_elimination(
            model=model, loader=loader, influence=influence,
            target_indices=idx, radius=args.radius,
            scale_factor=scale_factor, device=device,
        )
        all_results[name] = res
        support = support_set_at_tolerance(res["trajectory"], 98, args.tolerance)
        print(f"  baseline NME: {res['baseline_nme']:.4f}")
        print(f"  support set ({len(support)} landmarks @ tol={args.tolerance})")

    serializable = {}
    for name, res in all_results.items():
        serializable[name] = {
            "baseline_nme": res["baseline_nme"],
            "order": res["order"],
            "trajectory": res["trajectory"],
            "support_set": support_set_at_tolerance(res["trajectory"], 98, args.tolerance),
            "tolerance": args.tolerance,
        }
    with (args.out_dir / "trajectories_star.json").open("w") as fp:
        json.dump(serializable, fp, indent=2)

    plot_elimination_trajectories(all_results, args.out_dir / "trajectories_star.png")
    print(f"\nSaved trajectories_star.{{json,png}} -> {args.out_dir}")


if __name__ == "__main__":
    main()
