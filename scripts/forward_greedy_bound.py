"""Forward-greedy / linear-additivity optimality bound on support-set sizes.

Audit-mandated claim-1 defense (see paper.md milestone 13): reviewers will
ask how tight the backward-greedy support-set sizes are. This script runs
three comparisons per region per seed, all from the influence matrix + the
already-saved HRNet trajectories (no new HRNet forward passes):

1. **Backward-greedy HRNet** (paper default) — reads support sizes straight
   from `trajectories.json` via `support_set_at_tolerance`.
2. **Backward-greedy LINEAR** — predicts target-region NME under the
   additive-influence approximation
        NME_J(M) ≈ baseline_J + (1/|J|) Σ_{k in M} Σ_{j in J} I[k, j]
   and returns the largest prefix of the ascending-out_J ordering whose
   linear NME stays ≤ baseline + τ. Under linearity this IS the optimum
   of the 1-D knapsack-style problem, so backward = forward = optimal.
3. **Random-order LINEAR baseline** — closed-form expected support size
   when candidates are masked in random order under the same linear model.
   Shows how much of the compression is due to the influence-ordered
   heuristic versus trivial. Lower-is-better for the greedy; the gap
   (greedy << random) is the claim-1 evidence that ordering matters.

Outputs:
    results/forward_greedy_bound.{json,md}

Usage:
    python -m scripts.forward_greedy_bound
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.ablation.greedy import support_set_at_tolerance

REGIONS = yaml.safe_load(Path("configs/wflw_hrnet_w18.yaml").read_text(
    encoding="utf-8"))["eval"]["regions"]

TAU = 0.005
NUM_LANDMARKS = 98

SEEDS = {
    42: {
        "influence": Path("results/influence_matrix.npz"),
        "trajectories": Path("results/elimination/trajectories.json"),
    },
    7: {
        "influence": Path("results/seed7/influence_matrix.npz"),
        "trajectories": Path("results/seed7/elimination/trajectories.json"),
    },
    13: {
        "influence": Path("results/seed13/influence_matrix.npz"),
        "trajectories": Path("results/seed13/elimination/trajectories.json"),
    },
}

OUT_JSON = Path("results/forward_greedy_bound.json")
OUT_MD = Path("results/forward_greedy_bound.md")


def backward_linear_support(influence: np.ndarray, target_idx: list[int],
                            tau: float) -> int:
    """|S| under linear-additivity: mask in ascending out_J order until
    predicted mean-per-target NME delta exceeds tau."""
    target_set = set(target_idx)
    candidates = [k for k in range(influence.shape[0]) if k not in target_set]
    out_to_target = influence[candidates][:, target_idx].sum(axis=1)
    order = np.argsort(out_to_target)
    per_target = out_to_target[order] / len(target_idx)
    cumulative = np.cumsum(per_target)
    # largest m with cumulative[m-1] <= tau
    m = int(np.searchsorted(cumulative, tau, side="right"))
    return NUM_LANDMARKS - m


def random_linear_expected_support(influence: np.ndarray, target_idx: list[int],
                                   tau: float) -> float:
    """E[|S|] under random masking order, linear-additive NME model.
    Expected delta after masking the first m of K_c random candidates is
    (m / K_c) * sum(out_to_target) / |J|.  Solve for threshold."""
    target_set = set(target_idx)
    candidates = [k for k in range(influence.shape[0]) if k not in target_set]
    total_out = influence[candidates][:, target_idx].sum()
    mean_per_target_total = total_out / len(target_idx)
    k_c = len(candidates)
    if mean_per_target_total <= 0:
        return NUM_LANDMARKS  # degenerate: influence matrix all zero
    m_expected = tau * k_c / mean_per_target_total
    m_expected = min(max(m_expected, 0.0), float(k_c))
    return NUM_LANDMARKS - m_expected


def main() -> None:
    # Regions come from the trajectories.json of any seed
    ref_traj = json.loads(Path(SEEDS[42]["trajectories"]).read_text(encoding="utf-8"))
    regions = list(ref_traj.keys())

    rows: list[dict] = []
    per_seed_agg: dict[str, dict[str, dict]] = {}

    for seed, paths in SEEDS.items():
        influence = np.load(paths["influence"])["influence"]
        trajs = json.loads(Path(paths["trajectories"]).read_text(encoding="utf-8"))
        for region in regions:
            rtraj = trajs[region]["trajectory"]
            target_idx = REGIONS[region]
            hrnet_support_ids = support_set_at_tolerance(rtraj, NUM_LANDMARKS, TAU)
            s_hrnet = len(hrnet_support_ids)
            s_linear = backward_linear_support(influence, target_idx, TAU)
            s_random_exp = random_linear_expected_support(influence, target_idx, TAU)
            rows.append({
                "seed": seed, "region": region,
                "s_hrnet": int(s_hrnet), "s_linear": int(s_linear),
                "s_random_expected": float(s_random_exp),
                "gap_hrnet_linear": int(s_hrnet - s_linear),
                "greedy_vs_random_savings": float(s_random_exp - s_hrnet),
            })

    # Aggregate: mean across seeds per region
    by_region: dict[str, dict] = {}
    for region in regions:
        region_rows = [r for r in rows if r["region"] == region]
        by_region[region] = {
            "s_hrnet_mean": float(np.mean([r["s_hrnet"] for r in region_rows])),
            "s_hrnet_std": float(np.std([r["s_hrnet"] for r in region_rows], ddof=1)),
            "s_linear_mean": float(np.mean([r["s_linear"] for r in region_rows])),
            "s_linear_std": float(np.std([r["s_linear"] for r in region_rows], ddof=1)),
            "s_random_mean": float(np.mean([r["s_random_expected"] for r in region_rows])),
            "gap_mean": float(np.mean([r["gap_hrnet_linear"] for r in region_rows])),
            "savings_vs_random": float(np.mean([r["greedy_vs_random_savings"] for r in region_rows])),
            "per_seed": {r["seed"]: {
                "hrnet": r["s_hrnet"], "linear": r["s_linear"],
                "random": round(r["s_random_expected"], 1),
            } for r in region_rows},
        }

    out = {"tau": TAU, "per_row": rows, "per_region_agg": by_region}
    OUT_JSON.write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = []
    md.append(f"## Forward-greedy / linear-additivity bound on support sizes "
              f"(tau={TAU}, 3 seeds)\n")
    md.append("**Columns:** HRNet-backward (paper), LINEAR-backward "
              "(optimal under additivity → forward-greedy = backward-greedy on "
              "the 1-D thresholding problem), RANDOM-order expected "
              "(closed-form under the same linear model).\n")
    md.append("| Region | HRNet mean±std | LINEAR mean±std | RANDOM E[|S|] | "
              "HRNet − LINEAR | RANDOM − HRNet (savings) |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for region, agg in by_region.items():
        md.append(
            f"| {region} | {agg['s_hrnet_mean']:.1f} ± {agg['s_hrnet_std']:.1f} "
            f"| {agg['s_linear_mean']:.1f} ± {agg['s_linear_std']:.1f} "
            f"| {agg['s_random_mean']:.1f} "
            f"| {agg['gap_mean']:+.1f} "
            f"| {agg['savings_vs_random']:+.1f} |"
        )
    md.append("")
    md.append("### Per-seed detail (HRNet / LINEAR / RANDOM)\n")
    md.append("| Region | " + " | ".join(f"seed {s}" for s in SEEDS) + " |")
    md.append("| --- | " + " | ".join("---" for _ in SEEDS) + " |")
    for region, agg in by_region.items():
        cells = []
        for s in SEEDS:
            ps = agg["per_seed"][s]
            cells.append(f"{ps['hrnet']} / {ps['linear']} / {ps['random']}")
        md.append(f"| {region} | " + " | ".join(cells) + " |")
    md.append("")
    md.append("**Interpretation (see paper.md §Forward-greedy optimality bound):**")
    md.append("- `HRNet − LINEAR`: negative means the linear model predicts a "
              "larger support set than HRNet actually needs (HRNet is more "
              "forgiving than additive theory predicts); positive means the "
              "linear model is optimistic and HRNet enforces a bigger set.")
    md.append("- `RANDOM − HRNet`: savings of influence-ordered greedy over "
              "uninformed random masking. Large positive numbers here are "
              "the claim-1 evidence that ordering encodes structure.")
    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
