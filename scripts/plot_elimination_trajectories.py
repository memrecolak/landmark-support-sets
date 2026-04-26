"""Render per-region backward-elimination trajectories (Fig. 2).

For each of the eight WFLW anatomical regions, plot the target-region NME
as a function of the number of landmarks masked, in ascending-influence
order, for the three random seeds {42, 7, 13}. Mark the tau=0.005 tolerance
line and the support-set boundary |S| where the trajectory first leaves the
[baseline, baseline+tau] band.

Usage:
    python -m scripts.plot_elimination_trajectories
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.ablation.greedy import support_set_at_tolerance

NUM_LANDMARKS = 98
TAU = 0.005

PATHS = {
    42: Path("results/elimination/trajectories.json"),
    7:  Path("results/seed7/elimination/trajectories.json"),
    13: Path("results/seed13/elimination/trajectories.json"),
}

OUT = Path("figures/elimination_trajectories.png")

REGION_ORDER = [
    "contour", "nose", "left_brow", "right_brow",
    "left_eye", "right_eye", "mouth_outer", "mouth_inner",
]
SEED_COLOR = {42: "#1f77b4", 7: "#ff7f0e", 13: "#2ca02c"}


def main() -> None:
    trajs = {seed: json.loads(p.read_text(encoding="utf-8"))
             for seed, p in PATHS.items()}

    fig, axes = plt.subplots(2, 4, figsize=(15, 6.5), sharey=False)
    axes = axes.flatten()

    for ax, region in zip(axes, REGION_ORDER):
        for seed in (42, 7, 13):
            rtraj = trajs[seed][region]["trajectory"]
            baseline = trajs[seed][region]["baseline_nme"]
            steps = np.array([t["step"] for t in rtraj])
            nme = np.array([t["target_nme"] for t in rtraj])
            ax.plot(steps, nme, color=SEED_COLOR[seed],
                    linewidth=1.4, alpha=0.85,
                    label=f"seed {seed}")
            # Tolerance band
            ax.axhline(baseline, color=SEED_COLOR[seed],
                       linestyle=":", linewidth=0.6, alpha=0.5)
            # Support-set boundary marker
            support = support_set_at_tolerance(rtraj, NUM_LANDMARKS, TAU)
            n_masked_at_S = NUM_LANDMARKS - len(support)
            ax.axvline(n_masked_at_S, color=SEED_COLOR[seed],
                       linestyle="--", linewidth=0.6, alpha=0.4)

        # Tau line (above the largest seed-baseline)
        max_baseline = max(trajs[seed][region]["baseline_nme"]
                           for seed in (42, 7, 13))
        ax.axhline(max_baseline + TAU, color="#444444",
                   linestyle="-.", linewidth=0.7,
                   label=f"max baseline+tau" if region == REGION_ORDER[0] else None)

        ax.set_title(region, fontsize=10)
        ax.set_xlabel("# landmarks masked")
        ax.set_ylabel("target NME")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(left=0)

    # Single legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=len(labels), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        "Influence-ordered backward elimination, per region (3 seeds; "
        f"tau={TAU}; r=8 px). Dashed verticals mark |S|.",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
