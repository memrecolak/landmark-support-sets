"""Adaptive Wing Loss (Wang et al., 2019) for heatmap regression."""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveWingLoss(nn.Module):
    def __init__(
        self,
        omega: float = 14.0,
        theta: float = 0.5,
        epsilon: float = 1.0,
        alpha: float = 2.1,
    ):
        super().__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        delta = (target - pred).abs()
        alpha_t = self.alpha - target
        ratio = self.theta / self.epsilon
        A = (
            self.omega
            * (1.0 / (1.0 + torch.pow(ratio, alpha_t)))
            * alpha_t
            * torch.pow(ratio, alpha_t - 1.0)
            / self.epsilon
        )
        C = self.theta * A - self.omega * torch.log(1.0 + torch.pow(ratio, alpha_t))
        loss_small = self.omega * torch.log(1.0 + torch.pow(delta / self.epsilon, alpha_t))
        loss_large = A * delta - C
        return torch.where(delta < self.theta, loss_small, loss_large).mean()
