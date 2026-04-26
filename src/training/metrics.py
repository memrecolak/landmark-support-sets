"""Per-region NME with inter-ocular normalization for WFLW."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def decode_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    """argmax + 1-px sign-offset sub-pixel refinement. (B,K,H,W) -> (B,K,2) in heatmap pixels."""
    B, K, H, W = heatmaps.shape
    flat = heatmaps.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    px = (idx % W).long()
    py = (idx // W).long()

    b_idx = torch.arange(B, device=heatmaps.device).view(B, 1).expand(B, K)
    k_idx = torch.arange(K, device=heatmaps.device).view(1, K).expand(B, K)

    px_l = (px - 1).clamp(min=0)
    px_r = (px + 1).clamp(max=W - 1)
    py_u = (py - 1).clamp(min=0)
    py_d = (py + 1).clamp(max=H - 1)

    dx = heatmaps[b_idx, k_idx, py, px_r] - heatmaps[b_idx, k_idx, py, px_l]
    dy = heatmaps[b_idx, k_idx, py_d, px] - heatmaps[b_idx, k_idx, py_u, px]
    border = (px == 0) | (px == W - 1) | (py == 0) | (py == H - 1)
    ox = torch.where(border, torch.zeros_like(dx), 0.25 * torch.sign(dx))
    oy = torch.where(border, torch.zeros_like(dy), 0.25 * torch.sign(dy))

    return torch.stack([px.float() + ox, py.float() + oy], dim=-1)


def _gaussian_blur_2d(x: torch.Tensor, kernel: int, sigma: float) -> torch.Tensor:
    """Depthwise 2D Gaussian blur over (B, K, H, W). Reflect-padded; max-preserving."""
    K = x.shape[1]
    half = kernel // 2
    coords = torch.arange(kernel, device=x.device, dtype=x.dtype) - half
    g1 = torch.exp(-(coords ** 2) / (2.0 * sigma * sigma))
    g1 = g1 / g1.sum()
    weight = (g1.view(1, kernel) * g1.view(kernel, 1)).view(1, 1, kernel, kernel).expand(K, 1, kernel, kernel)

    orig_max = x.amax(dim=(-2, -1), keepdim=True)
    x_padded = F.pad(x, (half, half, half, half), mode="reflect")
    y = F.conv2d(x_padded, weight, groups=K)
    new_max = y.amax(dim=(-2, -1), keepdim=True).clamp(min=1e-10)
    return y * (orig_max / new_max)


@torch.no_grad()
def decode_heatmaps_dark(
    heatmaps: torch.Tensor,
    blur_kernel: int = 11,
    blur_sigma: float = 2.0,
    log_eps: float = 1e-10,
) -> torch.Tensor:
    """DARK decode (Zhang et al., CVPR 2020).

    Gaussian-modulates the heatmap then applies a Taylor-expansion Newton
    step in log-space at the argmax, yielding sub-pixel sub-sign-offset
    accuracy. Falls back to zero offset at borders (< 2 px from edge) or
    when the Hessian is numerically singular. Signature identical to
    ``decode_heatmaps``: (B,K,H,W) -> (B,K,2) in heatmap pixels.
    """
    B, K, H, W = heatmaps.shape
    device = heatmaps.device

    smoothed = _gaussian_blur_2d(heatmaps, blur_kernel, blur_sigma)

    flat = smoothed.view(B, K, -1)
    idx = flat.argmax(dim=-1)
    px = (idx % W).long()
    py = (idx // W).long()

    log_h = torch.log(smoothed.clamp(min=log_eps))

    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, K)
    k_idx = torch.arange(K, device=device).view(1, K).expand(B, K)

    def sample(dx_off: int, dy_off: int) -> torch.Tensor:
        xx = (px + dx_off).clamp(0, W - 1)
        yy = (py + dy_off).clamp(0, H - 1)
        return log_h[b_idx, k_idx, yy, xx]

    dx = 0.5 * (sample(1, 0) - sample(-1, 0))
    dy = 0.5 * (sample(0, 1) - sample(0, -1))
    dxx = 0.25 * (sample(2, 0) - 2.0 * sample(0, 0) + sample(-2, 0))
    dyy = 0.25 * (sample(0, 2) - 2.0 * sample(0, 0) + sample(0, -2))
    dxy = 0.25 * (sample(1, 1) - sample(1, -1) - sample(-1, 1) + sample(-1, -1))

    det = dxx * dyy - dxy * dxy
    safe = det.abs() > 1e-10
    interior = (px >= 2) & (px <= W - 3) & (py >= 2) & (py <= H - 3)
    usable = interior & safe

    inv_det = torch.where(usable, 1.0 / det.clamp(min=1e-10), torch.zeros_like(det))
    ox = (dxy * dy - dyy * dx) * inv_det
    oy = (dxy * dx - dxx * dy) * inv_det
    ox = torch.where(usable, ox, torch.zeros_like(ox))
    oy = torch.where(usable, oy, torch.zeros_like(oy))
    # Clamp pathological offsets to one pixel (DARK's Newton step is only
    # meaningful near the mode; larger offsets mean the quadratic fit is bad).
    ox = ox.clamp(-1.0, 1.0)
    oy = oy.clamp(-1.0, 1.0)

    return torch.stack([px.float() + ox, py.float() + oy], dim=-1)


@torch.no_grad()
def interocular_distance(landmarks: torch.Tensor, idx_left: int = 60, idx_right: int = 72) -> torch.Tensor:
    """Distance between outer eye corners (WFLW indices 60 and 72). Returns (B,)."""
    return (landmarks[:, idx_left] - landmarks[:, idx_right]).norm(dim=-1)


@torch.no_grad()
def nme(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalize: torch.Tensor,
    indices: list[int] | None = None,
) -> torch.Tensor:
    if indices is not None:
        pred = pred[:, indices]
        target = target[:, indices]
    dists = (pred - target).norm(dim=-1)
    per_sample = dists.mean(dim=-1) / normalize
    return per_sample.mean()


@torch.no_grad()
def nme_per_region(
    pred: torch.Tensor,
    target: torch.Tensor,
    normalize: torch.Tensor,
    regions: dict[str, list[int]],
) -> dict[str, float]:
    out = {"overall": float(nme(pred, target, normalize))}
    for name, idx in regions.items():
        out[name] = float(nme(pred, target, normalize, indices=idx))
    return out
