"""Image-region masking utilities for landmark ablation."""

from __future__ import annotations

import torch


def circular_mask(h: int, w: int, centers: torch.Tensor, radius: float) -> torch.Tensor:
    """Return (N, H, W) boolean mask, True inside the circle for each center.

    centers: (N, 2) as (x, y) in pixel coordinates.
    """
    device = centers.device
    N = centers.size(0)
    ys = torch.arange(h, device=device).float().view(1, h, 1)
    xs = torch.arange(w, device=device).float().view(1, 1, w)
    cx = centers[:, 0].view(N, 1, 1)
    cy = centers[:, 1].view(N, 1, 1)
    return ((xs - cx) ** 2 + (ys - cy) ** 2) <= (radius * radius)


def occlude_landmark(
    images: torch.Tensor,        # (B, 3, H, W), normalized to [-1, 1]
    landmarks: torch.Tensor,     # (B, K, 2) in image-pixel coords
    landmark_idx: int,
    radius: float,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Zero out pixels within `radius` of landmark `landmark_idx` in each image."""
    B, C, H, W = images.shape
    centers = landmarks[:, landmark_idx]
    mask = circular_mask(H, W, centers, radius)  # (B, H, W)
    out = images.clone()
    out.masked_fill_(mask.unsqueeze(1), fill_value)
    return out


def occlude_landmarks(
    images: torch.Tensor,
    landmarks: torch.Tensor,
    landmark_indices: list[int],
    radius: float,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Zero out pixels within `radius` of every landmark in `landmark_indices`."""
    B, C, H, W = images.shape
    out = images.clone()
    for k in landmark_indices:
        centers = landmarks[:, k]
        mask = circular_mask(H, W, centers, radius)
        out.masked_fill_(mask.unsqueeze(1), fill_value)
    return out
