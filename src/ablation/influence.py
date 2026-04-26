"""Per-landmark influence via input-region occlusion.

Two outputs:
  * compute_per_sample_influence_stack: returns the full (K+1, N, K) tensor
    of per-sample NME values — row 0 is the no-mask baseline, rows 1..K are
    the NME after occluding landmark k-1. Enables arbitrary downstream
    aggregation (overall, per-attribute, per-subgroup).
  * compute_influence_matrix: thin wrapper returning only the mean K×K
    influence matrix for convenience.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ablation.masking import occlude_landmark
from src.training.metrics import decode_heatmaps, interocular_distance


@torch.no_grad()
def _per_sample_nme(
    model: torch.nn.Module,
    loader: DataLoader,
    scale_factor: float,
    device: torch.device,
    amp: bool,
    occlude_k: int | None,
    radius: float,
) -> np.ndarray:
    """Single pass: return per-sample (N, K) NME array, normalised by
    inter-ocular distance."""
    chunks: list[np.ndarray] = []
    for batch in loader:
        img = batch["image"].to(device, non_blocking=True)
        lm_gt = batch["landmarks"].to(device, non_blocking=True)
        if occlude_k is not None:
            img = occlude_landmark(img, lm_gt, occlude_k, radius)
        with torch.amp.autocast("cuda", enabled=amp):
            hm = model(img).float()
        pred = decode_heatmaps(hm) * scale_factor
        iod = interocular_distance(lm_gt)
        per_lm = (pred - lm_gt).norm(dim=-1) / iod.unsqueeze(-1)
        chunks.append(per_lm.cpu().numpy())
    return np.concatenate(chunks, axis=0)


@torch.no_grad()
def compute_per_sample_influence_stack(
    model: torch.nn.Module,
    loader: DataLoader,
    num_landmarks: int,
    radius: float,
    scale_factor: float,
    device: torch.device,
    amp: bool = True,
) -> dict:
    """Return dict with:
      baseline_per_sample: (N, K) NME with no occlusion
      occluded_stack:      (K, N, K) NME after occluding each landmark
      influence:           (K, K) mean influence matrix
      baseline_nme:        (K,)   mean per-landmark NME, no occlusion
    """
    model.eval()
    K = num_landmarks

    baseline = _per_sample_nme(model, loader, scale_factor, device, amp, None, 0.0)
    N = baseline.shape[0]

    stack = np.zeros((K, N, K), dtype=np.float32)
    for k in tqdm(range(K), desc="per-k occlusion"):
        stack[k] = _per_sample_nme(model, loader, scale_factor, device, amp, k, radius)

    mean_occ = stack.mean(axis=1)
    mean_base = baseline.mean(axis=0)
    influence = (mean_occ - mean_base[None, :]).astype(np.float32)
    return {
        "baseline_per_sample": baseline,
        "occluded_stack": stack,
        "influence": influence,
        "baseline_nme": mean_base.astype(np.float32),
    }


@torch.no_grad()
def compute_influence_matrix(
    model: torch.nn.Module,
    loader: DataLoader,
    num_landmarks: int,
    radius: float,
    scale_factor: float,
    device: torch.device,
    amp: bool = True,
) -> dict:
    """Convenience wrapper around compute_per_sample_influence_stack that
    returns only the mean K×K influence matrix and baseline NME."""
    res = compute_per_sample_influence_stack(
        model, loader, num_landmarks, radius, scale_factor, device, amp,
    )
    return {"influence": res["influence"], "baseline_nme": res["baseline_nme"]}
