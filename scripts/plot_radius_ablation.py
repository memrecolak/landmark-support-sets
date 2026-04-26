"""Render the 3-seed x 3-radius support-set grid (Fig. 3).

Reads results/radius_ablation_table.json (built by
scripts/aggregate_radius_ablation.py) and emits a grouped bar chart:
one cluster per region, three bars per cluster (r=4 / r=8 / r=12),
each bar = cross-seed mean |S| with stdev error bars.

Usage:
    python -m scripts.plot_radius_ablation
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN = Path("results/radius_ablation_table.json")
OUT = Path("figures/radius_ablation.png")

REGION_ORDER = [
    "contour", "nose", "left_brow", "right_brow",
    "left_eye", "right_eye", "mouth_outer", "mouth_inner",
]
RADII = [4, 8, 12]
RADIUS_COLOR = {4: "#bcd9ed", 8: "#5a8fb5", 12: "#1f4d6a"}


def main() -> None:
    blob = json.loads(IN.read_text(encoding="utf-8"))
    cs = blob["cross_seed_per_radius"]

    means = {r: [cs[reg][str(r)]["mean"] for reg in REGION_ORDER]
             for r in RADII}
    stds  = {r: [cs[reg][str(r)]["stdev"] for reg in REGION_ORDER]
             for r in RADII}

    x = np.arange(len(REGION_ORDER))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    for i, r in enumerate(RADII):
        offset = (i - 1) * width
        ax.bar(x + offset, means[r], width=width,
               yerr=stds[r], capsize=3,
               color=RADIUS_COLOR[r], edgecolor="#222222", linewidth=0.4,
               label=f"r = {r} px")

    ax.set_xticks(x)
    ax.set_xticklabels(REGION_ORDER, rotation=20, ha="right")
    ax.set_ylabel("|S|  (support-set size at tau=0.005)")
    ax.set_ylim(0, 100)
    ax.axhline(98, color="#999999", linewidth=0.7, linestyle=":")
    ax.text(len(REGION_ORDER) - 0.5, 98.7, "K = 98",
            fontsize=8, color="#666666", ha="right")
    ax.set_title(
        "Per-region support-set size as a function of occlusion radius "
        "(3 seeds; mean +/- std)",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
