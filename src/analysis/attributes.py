"""Attribute-aware aggregation over the WFLW test split.

WFLW tags every image with six binary attributes (pose, expression,
illumination, makeup, occlusion, blur). The per-sample per-landmark NME
matrices emitted by influence.py can be grouped by attribute to ask:
  "does landmark k's influence on target region R change under large pose?"
"""

from __future__ import annotations

import numpy as np

from src.datasets.wflw import ATTR_ORDER, WFLWDataset


def split_indices_by_attribute(
    dataset: WFLWDataset, attribute: str
) -> tuple[list[int], list[int]]:
    """Return (positive_sample_indices, negative_sample_indices) for `attribute`."""
    if attribute not in ATTR_ORDER:
        raise ValueError(f"Unknown attribute {attribute!r}; expected one of {ATTR_ORDER}")
    pos: list[int] = []
    neg: list[int] = []
    for i, rec in enumerate(dataset.records):
        (pos if rec.attributes[attribute] == 1 else neg).append(i)
    return pos, neg


def aggregate_influence_by_attribute(
    per_sample_nme: np.ndarray,   # (K+1, N, K) — rows: no-mask, mask-0, ..., mask-K-1
    attribute_mask: np.ndarray,   # (N,) bool
) -> np.ndarray:
    """Aggregate a per-sample NME stack over a boolean sample-mask,
    returning the K×K influence matrix restricted to those samples."""
    sel = per_sample_nme[:, attribute_mask, :]  # (K+1, n, K)
    mean = sel.mean(axis=1)                      # (K+1, K)
    base = mean[0]                               # (K,)
    influence = mean[1:] - base[None, :]          # (K, K)
    return influence


def attribute_masks(dataset: WFLWDataset) -> dict[str, np.ndarray]:
    """Return a dict of attribute_name -> (N,) boolean vector for the full dataset."""
    N = len(dataset)
    out: dict[str, np.ndarray] = {}
    for a in ATTR_ORDER:
        mask = np.zeros(N, dtype=bool)
        for i, rec in enumerate(dataset.records):
            mask[i] = bool(rec.attributes[a])
        out[a] = mask
    return out
