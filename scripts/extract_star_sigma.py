"""Extract per-landmark heatmap σ from STAR predictions on WFLW-test.

For each test sample we compute STAR's last-stack fusion heatmap H (B, 98, 64, 64),
treat each H_k as an unnormalised spatial distribution, and take

    p_k(x, y) = H_k(x, y) / sum H_k
    μ_k       = E_{p_k}[(x, y)]
    Σ_k       = E_{p_k}[(x, y)(x, y)^T] - μ_k μ_k^T
    σ_k       = sqrt(trace(Σ_k) / 2)   in heatmap pixels

This is the same scalar dispersion used by MCUDN (Wan et al. 2026) as their
predicted-uncertainty signal. Output:

    sigma_per_sample : (N, 98)   per-sample, per-landmark σ
    sigma_mean       : (98,)     mean over samples
    sigma_median     : (98,)     median over samples

Used downstream (T1) to scatter STAR-σ̄_k against HRNet support cardinality
|S_k|, with Spearman ρ + 95% CI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.datasets.wflw import WFLWDataset
from src.models.star import STARLandmark


@torch.no_grad()
def heatmap_sigma(heatmap: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-landmark σ = sqrt(trace(Σ_xy) / 2) in heatmap-pixel units.

    heatmap: (B, K, H, W), nonneg.
    returns: (B, K)
    """
    B, K, H, W = heatmap.shape
    # Force nonneg in case of slight negative values from in+relu boundary.
    h = heatmap.clamp(min=0.0)
    h_sum = h.sum(dim=(-2, -1)).clamp(min=eps)  # (B, K)

    ys = torch.arange(H, device=h.device, dtype=h.dtype).view(1, 1, H, 1)
    xs = torch.arange(W, device=h.device, dtype=h.dtype).view(1, 1, 1, W)

    mean_x = (h * xs).sum(dim=(-2, -1)) / h_sum
    mean_y = (h * ys).sum(dim=(-2, -1)) / h_sum

    var_x = (h * (xs - mean_x.view(B, K, 1, 1)) ** 2).sum(dim=(-2, -1)) / h_sum
    var_y = (h * (ys - mean_y.view(B, K, 1, 1)) ** 2).sum(dim=(-2, -1)) / h_sum

    return torch.sqrt(0.5 * (var_x + var_y) + eps)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--weights", type=Path,
        default=Path("external/STAR/weights/star_wflw.pkl"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/wflw"))
    parser.add_argument("--out", type=Path,
                        default=Path("results/star/sigma_per_landmark.npz"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = STARLandmark(weights_path=args.weights, use_ema=True).to(device).eval()

    ds = WFLWDataset(
        root=args.data_root,
        split="test",
        image_size=STARLandmark.INPUT_SIZE,
        heatmap_size=STARLandmark.HEATMAP_SIZE,
        heatmap_sigma=1.5,
        augment=False,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )
    print(f"WFLW test: {len(ds)} samples")

    chunks: list[np.ndarray] = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        hm = model(img)  # (B, 98, 64, 64) — last-stack fusion
        sig = heatmap_sigma(hm)  # (B, 98)
        chunks.append(sig.cpu().numpy())

    sigma_per_sample = np.concatenate(chunks, axis=0).astype(np.float32)
    sigma_mean = sigma_per_sample.mean(axis=0)
    sigma_median = np.median(sigma_per_sample, axis=0).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.out,
        sigma_per_sample=sigma_per_sample,
        sigma_mean=sigma_mean.astype(np.float32),
        sigma_median=sigma_median,
    )
    print(f"\nSaved -> {args.out}")
    print(f"  sigma_per_sample shape: {sigma_per_sample.shape}")
    print(f"  σ̄ range: [{sigma_mean.min():.3f}, {sigma_mean.max():.3f}]  (heatmap px)")
    print(f"  σ̄ mean:  {sigma_mean.mean():.3f}")


if __name__ == "__main__":
    main()
