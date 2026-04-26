"""Render the MPIIFaceGaze 3-bar comparison (Fig. 4).

3 bars (all-98 / eye-support / non-support), 15-fold LOPO mean +/- std,
with the 4-5 deg appearance-based reference band drawn as a shaded
horizontal stripe and the 8.5-14.95 deg landmark-only within-domain band
drawn as a second stripe.

Usage:
    python -m scripts.plot_gaze
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN = Path("results/gaze/gaze_results.json")
OUT = Path("figures/gaze_comparison.png")

ORDER = [
    ("all98",       "All 98",       "#999999"),
    ("eye_support", "Eye support",  "#1f77b4"),
    ("non_support", "Non-support",  "#d62728"),
]


def main() -> None:
    blob = json.loads(IN.read_text(encoding="utf-8"))["results"]

    sizes = [blob[k]["subset_size"] for k, _, _ in ORDER]
    means = [blob[k]["mean_deg"] for k, _, _ in ORDER]
    stds  = [blob[k]["std_deg"]  for k, _, _ in ORDER]
    colors = [c for _, _, c in ORDER]
    labels = [f"{lbl}\n(|S|={n})" for n, (_, lbl, _) in zip(sizes, ORDER)]

    fig, ax = plt.subplots(figsize=(7.5, 5.4))

    # Reference bands behind bars
    ax.axhspan(4.0, 5.0, color="#cce5cc", alpha=0.45, zorder=0,
               label="appearance-based 4-5 deg [Cheng et al., TPAMI 2024]")
    ax.axhspan(8.5, 14.95, color="#fbe5b3", alpha=0.45, zorder=0,
               label="landmark-only 8.5-14.95 deg [arXiv:2603.24724]")

    x = np.arange(len(ORDER))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=colors, edgecolor="#222222", linewidth=0.6,
                  zorder=2)

    # Annotate bar tops
    for xi, m, s in zip(x, means, stds):
        ax.text(xi, m + s + 0.25, f"{m:.2f}+/-{s:.2f}",
                ha="center", fontsize=9, color="#222222")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("3D angular error  (deg, lower better)")
    ax.set_ylim(0, 16)
    ax.set_title(
        "MPIIFaceGaze 15-fold LOPO gaze regression on three landmark subsets",
        fontsize=11,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
