"""Plotting helpers for influence matrices and elimination trajectories."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_influence_matrix(
    influence: np.ndarray,
    out_path: Path,
    regions: dict[str, list[int]] | None = None,
) -> None:
    """Full K×K influence heatmap, with optional region-boundary overlays."""
    fig, ax = plt.subplots(figsize=(10, 9))
    vmax = float(np.abs(influence).max()) or 1.0
    im = ax.imshow(influence, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xlabel("j (target landmark)")
    ax.set_ylabel("k (occluded landmark)")
    ax.set_title(r"Influence matrix $I[k, j] = \Delta$NME$_j$ when masking $k$")
    plt.colorbar(im, ax=ax)
    if regions:
        bounds = sorted({min(v) for v in regions.values()} | {max(v) + 1 for v in regions.values()})
        for o in bounds:
            ax.axhline(o - 0.5, color="black", lw=0.4, alpha=0.3)
            ax.axvline(o - 0.5, color="black", lw=0.4, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_region_influence(
    influence: np.ndarray,
    regions: dict[str, list[int]],
    out_path: Path,
) -> None:
    """Per-region aggregate: mean influence from each landmark onto each region."""
    K = influence.shape[0]
    names = list(regions.keys())
    agg = np.zeros((K, len(names)))
    for r_idx, name in enumerate(names):
        agg[:, r_idx] = influence[:, regions[name]].mean(axis=1)
    fig, ax = plt.subplots(figsize=(6, 14))
    vmax = float(np.abs(agg).max()) or 1.0
    im = ax.imshow(agg, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("k (occluded landmark)")
    ax.set_title("Mean influence of masking k on each target region")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()


def plot_elimination_trajectories(trajectories: dict, out_path: Path) -> None:
    """Line plot of target-region NME as landmarks are progressively masked."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in trajectories.items():
        steps = [t["step"] for t in res["trajectory"]]
        nme = [t["target_nme"] for t in res["trajectory"]]
        ax.plot(steps, nme, label=name, linewidth=1.5)
    ax.set_xlabel("# landmarks masked (ascending by target influence)")
    ax.set_ylabel("Target-region NME")
    ax.set_title("Elimination trajectory per target region")
    ax.legend(ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
