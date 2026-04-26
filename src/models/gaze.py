"""Landmark-conditioned gaze regressor.

Input: a subset of 98 predicted facial landmarks, flattened to (2 * K,) where
K is the subset size. Output: 3-D unit gaze direction (rescaled to unit norm
at forward time).

Intentionally small — the paper's claim is that support-set *landmark
coordinates alone* carry enough signal for gaze, so the regressor is a plain
MLP with no image features.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LandmarkGazeRegressor(nn.Module):
    def __init__(self, num_landmarks: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        in_dim = 2 * num_landmarks
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 3),
        )

    def forward(self, landmarks_xy: torch.Tensor) -> torch.Tensor:
        """(B, K, 2) or (B, 2*K) -> (B, 3) unit vector."""
        if landmarks_xy.dim() == 3:
            landmarks_xy = landmarks_xy.flatten(1)
        raw = self.net(landmarks_xy)
        return raw / raw.norm(dim=-1, keepdim=True).clamp(min=1e-8)


@torch.no_grad()
def angular_error_deg(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Angle between unit vectors, in degrees. (B, 3), (B, 3) -> (B,)."""
    cos = (pred * target).sum(dim=-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos))
