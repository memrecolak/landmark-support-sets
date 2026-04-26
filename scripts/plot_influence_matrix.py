"""Render the 98x98 per-landmark influence matrix as a paper figure.

Loads results/influence_matrix.npz and writes two PNGs + one combined figure
to results/: the full matrix, and the off-diagonal (cross-landmark) matrix
with self-dependencies zeroed out — the latter is the publication headline
that shows mask(k) hurting landmark j != k.

Usage:
    python -m scripts.plot_influence_matrix \
        --influence results/influence_matrix.npz \
        --config configs/wflw_hrnet_w18.yaml \
        --out-dir results
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.colors import TwoSlopeNorm


def region_blocks(regions: dict[str, list[int]]) -> list[tuple[str, int, int]]:
    """Return (name, start, end_inclusive) for each region's maximal contiguous
    prefix in natural-index order. Any trailing non-contiguous indices (e.g.
    pupils 96/97 that the config assigns to the eye regions) are reported as
    'pupils' in a separate tail block."""
    named_blocks: list[tuple[str, int, int]] = []
    covered: set[int] = set()
    for name, idx in regions.items():
        idx_sorted = sorted(idx)
        start = idx_sorted[0]
        end = start
        for v in idx_sorted[1:]:
            if v == end + 1:
                end = v
            else:
                break
        named_blocks.append((name, start, end))
        covered.update(range(start, end + 1))

    named_blocks.sort(key=lambda t: t[1])
    max_idx = max(max(v) for v in regions.values())
    tail = sorted(i for i in range(max_idx + 1) if i not in covered)
    if tail:
        named_blocks.append(("pupils", tail[0], tail[-1]))
    return named_blocks


def draw_matrix(
    ax,
    mat: np.ndarray,
    blocks: list[tuple[str, int, int]],
    title: str,
    vmax: float,
    center_at_zero: bool,
):
    if center_at_zero:
        norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
        im = ax.imshow(mat, cmap="RdBu_r", norm=norm,
                       aspect="equal", origin="upper", interpolation="nearest")
    else:
        im = ax.imshow(mat, cmap="magma", vmin=0.0, vmax=vmax,
                       aspect="equal", origin="upper", interpolation="nearest")
    # Region separators
    for _, _, end in blocks[:-1]:
        ax.axhline(end + 0.5, color="#303030", linewidth=0.5)
        ax.axvline(end + 0.5, color="#303030", linewidth=0.5)
    # Tick labels at block centers
    centers = [(s + e) / 2.0 for _, s, e in blocks]
    labels = [n for n, _, _ in blocks]
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("target landmark j")
    ax.set_ylabel("masked landmark k")
    ax.set_title(title, fontsize=10)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--influence", type=Path, default=Path("results/influence_matrix.npz"))
    ap.add_argument("--config", type=Path, default=Path("configs/wflw_hrnet_w18.yaml"))
    ap.add_argument("--out-dir", type=Path, default=Path("figures"))
    args = ap.parse_args()

    blob = np.load(args.influence)
    influence = blob["influence"].astype(np.float32)
    baseline_nme = float(blob["baseline_nme"].mean())
    radius = float(blob["radius"])

    with args.config.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)
    regions = cfg["eval"]["regions"]
    blocks = region_blocks(regions)

    off_diag = influence.copy()
    np.fill_diagonal(off_diag, 0.0)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Panel A: full matrix
    vmax_full = float(max(abs(influence.min()), influence.max()))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = draw_matrix(ax, influence, blocks,
                     f"Full 98x98 importance I[k, j]  "
                     f"(baseline NME={baseline_nme:.4f}, radius={radius:.1f} px)",
                     vmax=vmax_full, center_at_zero=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta NME")
    fig.tight_layout()
    fig.savefig(args.out_dir / "influence_matrix.png", dpi=220)
    plt.close(fig)

    # Panel B: off-diagonal only
    vmax_off = float(max(abs(off_diag.min()), off_diag.max()))
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = draw_matrix(ax, off_diag, blocks,
                     "Off-diagonal influence (self-dependencies zeroed)",
                     vmax=vmax_off, center_at_zero=True)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Delta NME")
    fig.tight_layout()
    fig.savefig(args.out_dir / "influence_matrix_offdiag.png", dpi=220)
    plt.close(fig)

    # Combined figure (the paper's headline)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    im0 = draw_matrix(axes[0], influence, blocks, "(a) Full influence I[k, j]",
                      vmax=vmax_full, center_at_zero=True)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Delta NME")
    im1 = draw_matrix(axes[1], off_diag, blocks, "(b) Off-diagonal only",
                      vmax=vmax_off, center_at_zero=True)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Delta NME")
    fig.suptitle("WFLW-98 intervention-based per-landmark importance matrix "
                 "(HRNet-W18, seed 42)", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out_dir / "influence_matrix_figure.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote:")
    print(f"  {args.out_dir / 'influence_matrix.png'}")
    print(f"  {args.out_dir / 'influence_matrix_offdiag.png'}")
    print(f"  {args.out_dir / 'influence_matrix_figure.png'}")
    print(f"Region blocks: {blocks}")


if __name__ == "__main__":
    main()
