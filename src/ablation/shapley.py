"""Permutation-sampling Shapley attribution for landmark importance.

For a target region T (a subset of landmark indices), compute the Shapley
value of each non-target landmark k — the expected marginal effect of adding
k to a random set of already-occluded landmarks, on the target region's NME.

Algorithm (Castro et al. 2009): sample M random permutations of the candidate
set; for each permutation, walk through prefixes and record the marginal NME
change contributed by the landmark at that position. Average over
permutations yields an unbiased Shapley estimate.

Truncation: if adding further landmarks barely changes the NME, skip the
remaining tail of the permutation. Standard practice (Jia et al. 2019); the
truncated landmarks have near-zero marginal contribution by construction, so
the bias is small and bounded by `truncation_eps`.

Cost: M * K * T_eval, where T_eval is one dataset pass. For M=200, K=97, and
evaluation on a 256-image subsample at batch 32 (~0.3 s), budget ≈ 1.6 h per
target region. Reduce M or subsample size for faster runs.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ablation.masking import occlude_landmarks
from src.training.metrics import decode_heatmaps, interocular_distance


@torch.no_grad()
def _score_mask(
    model: torch.nn.Module,
    loader: DataLoader,
    masked: list[int],
    radius: float,
    scale_factor: float,
    device: torch.device,
    target_indices: list[int],
    amp: bool,
) -> float:
    """Mean target-region NME over the loader when `masked` landmarks are occluded."""
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
def permutation_shapley(
    model: torch.nn.Module,
    loader: DataLoader,
    num_landmarks: int,
    target_indices: list[int],
    radius: float,
    scale_factor: float,
    device: torch.device,
    num_permutations: int = 200,
    truncation_eps: float = 5e-5,
    min_prefix_before_truncate: int = 8,
    amp: bool = True,
    seed: int = 42,
) -> dict:
    """Permutation-sampling Shapley values.

    Shapley value at index k = mean marginal increase in target NME when k is
    added to the already-occluded prefix. High values mean k's visual context
    is important for the target region.

    Target landmarks are excluded from candidates (they are never occluded).

    Returns:
        shapley (np.ndarray): shape (K,), Shapley value per landmark; zero for target members.
        counts (np.ndarray): shape (K,), number of permutations in which each landmark was evaluated.
        base_score (float): mean target NME with no landmarks occluded.
        truncated_permutations (int): how many of the M permutations hit the truncation condition.
    """
    rng = np.random.default_rng(seed)
    K = num_landmarks
    target_set = set(target_indices)
    candidates = [k for k in range(K) if k not in target_set]

    shapley = np.zeros(K, dtype=np.float64)
    counts = np.zeros(K, dtype=np.int64)

    base_score = _score_mask(
        model, loader, [], radius, scale_factor, device, target_indices, amp,
    )

    truncated = 0
    for _ in tqdm(range(num_permutations), desc=f"shapley (|T|={len(target_indices)})"):
        order = candidates.copy()
        rng.shuffle(order)

        masked: list[int] = []
        prev_score = base_score
        hit_truncation = False
        for i, k in enumerate(order):
            masked.append(k)
            score = _score_mask(
                model, loader, masked, radius, scale_factor, device, target_indices, amp,
            )
            delta = score - prev_score
            shapley[k] += delta
            counts[k] += 1
            prev_score = score
            if i >= min_prefix_before_truncate and abs(delta) < truncation_eps:
                hit_truncation = True
                break
        if hit_truncation:
            truncated += 1

    with np.errstate(divide="ignore", invalid="ignore"):
        shapley = np.where(counts > 0, shapley / counts, 0.0)

    return {
        "shapley": shapley.astype(np.float32),
        "counts": counts,
        "base_score": float(base_score),
        "truncated_permutations": int(truncated),
    }
