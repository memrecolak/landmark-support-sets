"""T1: scatter STAR σ̄_k vs HRNet per-landmark support cardinality.

Per-landmark support cardinality = number of the 8 regional minimum support
sets (contour, brows, nose, eyes, mouth_outer, mouth_inner) that include
landmark k. Range: {0, 1, ..., 8}. This is the natural per-landmark
projection of our region-based support-set framework.

STAR σ̄_k = mean over the WFLW-test set of the spatial standard deviation
of STAR's predicted heatmap for landmark k (output of extract_star_sigma.py).

We report Spearman ρ + 95% bootstrap CI (B=10000), test the null ρ=0 with
a permutation test (B=10000), and save:
  * results/star/sigma_vs_support_scatter.png
  * results/star/sigma_vs_support_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def per_landmark_support_count(trajectories: dict, num_landmarks: int = 98) -> np.ndarray:
    """For each landmark k in [0, K), count how many regional support sets
    include k. Returns (K,) int array in {0, ..., 8}."""
    counts = np.zeros(num_landmarks, dtype=np.int32)
    for region, payload in trajectories.items():
        for k in payload["support_set"]:
            counts[k] += 1
    return counts


def per_landmark_support_count_multi(traj_paths: list[Path], num_landmarks: int = 98) -> np.ndarray:
    """Mean membership count across multiple seed trajectory files. Returns
    a continuous (K,) array in [0, 8] = mean over seeds of the per-seed
    integer membership count."""
    stacks = []
    for p in traj_paths:
        with p.open() as f:
            stacks.append(per_landmark_support_count(json.load(f), num_landmarks))
    return np.stack(stacks).mean(axis=0)


def bootstrap_spearman_ci(
    x: np.ndarray, y: np.ndarray, B: int = 10000, alpha: float = 0.05, seed: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(x)
    rhos = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, N, size=N)
        rhos[b] = stats.spearmanr(x[idx], y[idx]).statistic
    return float(np.percentile(rhos, 100 * alpha / 2)), float(np.percentile(rhos, 100 * (1 - alpha / 2)))


def perm_test_spearman(
    x: np.ndarray, y: np.ndarray, observed: float, B: int = 10000, seed: int = 1,
) -> float:
    rng = np.random.default_rng(seed)
    abs_obs = abs(observed)
    hits = 0
    y_perm = y.copy()
    for _ in range(B):
        rng.shuffle(y_perm)
        rho_perm = stats.spearmanr(x, y_perm).statistic
        if abs(rho_perm) >= abs_obs:
            hits += 1
    return (hits + 1) / (B + 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sigma", type=Path,
        default=Path("results/star/sigma_per_landmark.npz"),
    )
    parser.add_argument(
        "--trajectories", type=Path, nargs="+",
        default=[
            Path("results/elimination/trajectories.json"),
            Path("results/seed7/elimination/trajectories.json"),
            Path("results/seed13/elimination/trajectories.json"),
        ],
        help="One or more trajectory JSON files. If multiple, membership count is the mean over seeds (continuous).",
    )
    parser.add_argument(
        "--out-fig", type=Path,
        default=Path("results/star/sigma_vs_support_scatter.png"),
    )
    parser.add_argument(
        "--out-stats", type=Path,
        default=Path("results/star/sigma_vs_support_stats.json"),
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()

    sigma = np.load(args.sigma)["sigma_mean"]  # (98,)
    if len(args.trajectories) == 1:
        with args.trajectories[0].open() as f:
            support = per_landmark_support_count(json.load(f)).astype(np.float32)
        support_label = f"HRNet support cardinality (regions $\\ni k$)"
    else:
        support = per_landmark_support_count_multi(args.trajectories)
        support_label = f"HRNet support cardinality, {len(args.trajectories)}-seed mean (regions $\\ni k$)"

    rho_s, p_s = stats.spearmanr(sigma, support)
    rho_p, p_p = stats.pearsonr(sigma, support)
    ci_lo, ci_hi = bootstrap_spearman_ci(sigma, support, B=args.bootstrap)
    p_perm = perm_test_spearman(sigma, support, rho_s, B=args.bootstrap)

    stats_out = {
        "n_landmarks": int(len(sigma)),
        "spearman_rho": float(rho_s),
        "spearman_p_scipy": float(p_s),
        "spearman_ci95_bootstrap": [ci_lo, ci_hi],
        "spearman_p_perm": float(p_perm),
        "pearson_r": float(rho_p),
        "pearson_p": float(p_p),
        "sigma_range_heatmap_px": [float(sigma.min()), float(sigma.max())],
        "support_count_range": [int(support.min()), int(support.max())],
        "bootstrap_B": int(args.bootstrap),
    }
    args.out_stats.parent.mkdir(parents=True, exist_ok=True)
    with args.out_stats.open("w") as f:
        json.dump(stats_out, f, indent=2)
    print(json.dumps(stats_out, indent=2))

    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    jitter_scale = 0.05 if len(args.trajectories) > 1 else 0.15
    jitter = np.random.default_rng(0).uniform(-jitter_scale, jitter_scale, size=len(sigma))
    ax.scatter(support + jitter, sigma, s=18, alpha=0.65, edgecolor="none")
    ax.set_xlabel(support_label)
    ax.set_ylabel("STAR mean heatmap $\\sigma$ (heatmap px)")
    ax.text(
        0.05, 0.95,
        f"Spearman $\\rho={rho_s:+.2f}$\n"
        f"95% CI [{ci_lo:+.2f}, {ci_hi:+.2f}]\n"
        f"perm $p={p_perm:.3f}$  ($N={len(sigma)}$)",
        transform=ax.transAxes, va="top", ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9),
    )
    fig.tight_layout()
    args.out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=200)
    print(f"Saved scatter -> {args.out_fig}")


if __name__ == "__main__":
    main()
