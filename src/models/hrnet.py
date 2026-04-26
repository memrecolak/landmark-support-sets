"""HRNet-W18 landmark heatmap regressor.

Uses timm's HRNet backbone in features_only mode; all multi-scale features are
upsampled to the highest resolution, concatenated, then projected to a
K-channel heatmap via a small conv head.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class HRNetLandmark(nn.Module):
    def __init__(self, num_landmarks: int = 98, pretrained: bool = True, variant: str = "hrnet_w18"):
        super().__init__()
        self.backbone = timm.create_model(variant, pretrained=pretrained, features_only=True)
        # Skip the stride-2 stem feature; anchor head at stride-4 (matches 64x64 heatmap for 256 input).
        chs = [f["num_chs"] for f in self.backbone.feature_info][1:]
        total = sum(chs)
        self.head = nn.Sequential(
            nn.Conv2d(total, total, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(total),
            nn.ReLU(inplace=True),
            nn.Conv2d(total, num_landmarks, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)[1:]
        target_size = feats[0].shape[-2:]
        up = [feats[0]]
        for f in feats[1:]:
            up.append(F.interpolate(f, size=target_size, mode="bilinear", align_corners=False))
        return self.head(torch.cat(up, dim=1))
