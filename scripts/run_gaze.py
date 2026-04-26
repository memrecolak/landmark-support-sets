"""Leave-one-participant-out gaze regression across three landmark subsets.

Reads:
    - results/mpiifacegaze_landmarks.npz (from predict_landmarks_mpiifacegaze.py)
    - results/elimination/trajectories.json (for the eye support set)

Trains the landmark-conditioned gaze regressor three times per held-out
participant — once on all 98 landmarks, once on the eye support set, once on
the complementary non-support set — and reports mean angular error.

The paper's pillar-4 claim is that the support set preserves task utility
while the non-support set does not.

Usage:
    python -m scripts.run_gaze \
        --landmarks results/mpiifacegaze_landmarks.npz \
        --trajectories results/elimination/trajectories.json \
        --out results/gaze \
        --target-region left_eye right_eye
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.gaze import LandmarkGazeRegressor, angular_error_deg


def eye_support_set(trajectories: dict, region_names: list[str]) -> list[int]:
    union: set[int] = set()
    for name in region_names:
        if name not in trajectories:
            raise SystemExit(f"Region {name!r} not in trajectories.json")
        union.update(trajectories[name]["support_set"])
    return sorted(union)


def complement(support: list[int], total: int) -> list[int]:
    keep = set(range(total)) - set(support)
    return sorted(keep)


def standardize(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True).clip(min=1e-6)
    return (x - mean) / std, mean, std


def train_one_fold(
    train_xy: np.ndarray, train_gaze: np.ndarray,
    test_xy: np.ndarray, test_gaze: np.ndarray,
    num_landmarks: int, *, epochs: int, lr: float, batch_size: int,
    device: torch.device,
) -> float:
    train_xy_z, mu, sd = standardize(train_xy)
    test_xy_z = (test_xy - mu) / sd

    tr_ds = TensorDataset(torch.from_numpy(train_xy_z.astype(np.float32)),
                          torch.from_numpy(train_gaze.astype(np.float32)))
    te_ds = TensorDataset(torch.from_numpy(test_xy_z.astype(np.float32)),
                          torch.from_numpy(test_gaze.astype(np.float32)))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    te_loader = DataLoader(te_ds, batch_size=512, shuffle=False, num_workers=0)

    model = LandmarkGazeRegressor(num_landmarks=num_landmarks).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for _ in range(epochs):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = 1.0 - (pred * yb).sum(dim=-1).mean()  # cosine-margin loss
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    errs: list[float] = []
    with torch.no_grad():
        for xb, yb in te_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            errs.append(angular_error_deg(pred, yb).cpu())
    return float(torch.cat(errs).mean())


def run_subset(
    name: str, subset: list[int],
    lm: np.ndarray, gaze: np.ndarray, parts: np.ndarray,
    *, epochs: int, lr: float, batch_size: int, device: torch.device,
) -> dict:
    per_participant: dict[str, float] = {}
    unique = sorted(np.unique(parts).tolist())
    for held in unique:
        mask = parts == held
        tr_xy = lm[~mask][:, subset].reshape(int((~mask).sum()), -1)
        tr_g = gaze[~mask]
        te_xy = lm[mask][:, subset].reshape(int(mask.sum()), -1)
        te_g = gaze[mask]
        err = train_one_fold(tr_xy, tr_g, te_xy, te_g,
                             num_landmarks=len(subset),
                             epochs=epochs, lr=lr, batch_size=batch_size,
                             device=device)
        per_participant[held] = err
        print(f"  [{name}] holdout={held} err={err:.2f} deg")
    mean = float(np.mean(list(per_participant.values())))
    std = float(np.std(list(per_participant.values())))
    print(f"  [{name}] mean={mean:.2f} +/- {std:.2f} deg  (K={len(subset)})")
    return {"subset_size": len(subset), "subset": subset,
            "per_participant": per_participant,
            "mean_deg": mean, "std_deg": std}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--landmarks", type=Path, required=True)
    ap.add_argument("--trajectories", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--target-region", nargs="+", default=["left_eye", "right_eye"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args()

    blob = np.load(args.landmarks, allow_pickle=True)
    lm = blob["landmarks"].astype(np.float32)           # (N, 98, 2)
    gaze = blob["gaze_direction"].astype(np.float32)    # (N, 3)
    parts = blob["participants"]
    N, K, _ = lm.shape

    with args.trajectories.open("r", encoding="utf-8") as fp:
        trajectories = json.load(fp)
    support = eye_support_set(trajectories, args.target_region)
    non_support = complement(support, K)
    all_idx = list(range(K))
    print(f"Eye support set (|S|={len(support)}) from {args.target_region}")
    print(f"Complement (|S'|={len(non_support)})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = {}
    for name, subset in [("all98", all_idx),
                         ("eye_support", support),
                         ("non_support", non_support)]:
        print(f"\n=== Subset: {name} ({len(subset)} landmarks) ===")
        results[name] = run_subset(
            name, subset, lm, gaze, parts,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            device=device,
        )

    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "gaze_results.json").open("w", encoding="utf-8") as fp:
        json.dump({"target_region": args.target_region, "results": results}, fp, indent=2)
    print(f"\nSaved results to {args.out / 'gaze_results.json'}")


if __name__ == "__main__":
    main()
