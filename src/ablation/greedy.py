"""Backward elimination of landmarks via input occlusion, ordered by
influence-matrix scores.

Given an influence matrix I (K×K, where I[k, j] is the NME-change of j when k
is occluded) and a target landmark subset T, each candidate k is scored by

    s(k) = sum_{j in T} I[k, j]

Low s(k) means masking k barely hurts T → safe to mask early.
High s(k) means masking k hurts T → keep visible until late.

Candidates are masked in ascending s(k) order; target-region NME is re-evaluated
after each addition, yielding a monotone-by-construction trajectory that reveals
the minimal support set for each target region.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ablation.masking import occlude_landmarks
from src.training.metrics import decode_heatmaps, interocular_distance


@torch.no_grad()
def _evaluate_with_mask(
    model: torch.nn.Module,
    loader: DataLoader,
    masked: list[int],
    radius: float,
    scale_factor: float,
    device: torch.device,
    target_indices: list[int] | None,
    amp: bool,
) -> float:
    total = 0.0
    count = 0
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        lm_gt = batch["landmarks"].to(device, non_blocking=True)
        if masked:
            img = occlude_landmarks(img, lm_gt, masked, radius)
        with torch.amp.autocast("cuda", enabled=amp):
            hm = model(img).float()
        pred = decode_heatmaps(hm) * scale_factor
        iod = interocular_distance(lm_gt)
        dists = (pred - lm_gt).norm(dim=-1)
        if target_indices is not None:
            dists = dists[:, target_indices]
        per_sample = dists.mean(dim=-1) / iod
        total += per_sample.sum().item()
        count += img.size(0)
    return total / count


@torch.no_grad()
def influence_ordered_elimination(
    model: torch.nn.Module,
    loader: DataLoader,
    influence: np.ndarray,
    target_indices: list[int],
    radius: float,
    scale_factor: float,
    device: torch.device,
    amp: bool = True,
) -> dict:
    """Mask landmarks in order of least aggregate target-region influence,
    recording the target-NME trajectory."""
    K = influence.shape[0]
    target_set = set(target_indices)
    candidates = [k for k in range(K) if k not in target_set]
    scores = influence[candidates][:, list(target_indices)].sum(axis=1)
    order = [candidates[i] for i in np.argsort(scores)]

    masked: list[int] = []
    base_nme = _evaluate_with_mask(
        model, loader, masked, radius, scale_factor, device, target_indices, amp,
    )
    trajectory = [{"step": 0, "added": None, "masked": [], "target_nme": base_nme}]
    for i, k in enumerate(tqdm(order, desc="elimination"), start=1):
        masked.append(k)
        score = _evaluate_with_mask(
            model, loader, masked, radius, scale_factor, device, target_indices, amp,
        )
        trajectory.append({"step": i, "added": k, "masked": list(masked), "target_nme": score})
    return {"trajectory": trajectory, "order": order, "baseline_nme": base_nme}


def support_set_at_tolerance(trajectory: list[dict], num_landmarks: int, tolerance: float) -> list[int]:
    """Minimal visual-support set: landmarks whose occlusion would push target NME
    beyond (baseline + tolerance)."""
    base = trajectory[0]["target_nme"]
    last_ok = 0
    for i, entry in enumerate(trajectory):
        if entry["target_nme"] - base <= tolerance:
            last_ok = i
    safely_masked = set(trajectory[last_ok]["masked"])
    return [k for k in range(num_landmarks) if k not in safely_masked]
