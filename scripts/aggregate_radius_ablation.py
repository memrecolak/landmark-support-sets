"""Aggregate occlusion-radius ablation: 3 seeds x 3 radii (r in {4, 8, 12}).

Reads the 9 trajectories.json files (the r=8 set lives in the main results
tree; r=4 and r=12 live under results/radius_ablation/) and emits:

    results/radius_ablation_table.{md,json}

The markdown contains:
  1. Per-seed wide table (r=4 / r=8 / r=12 columns).
  2. Cross-seed summary at each radius (mean +/- std).
  3. Per-region rank order at each radius (radius-stability check).

Usage:
    python -m scripts.aggregate_radius_ablation
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from src.ablation.greedy import support_set_at_tolerance

TAU = 0.005
NUM_LANDMARKS = 98

PATHS: dict[tuple[int, int], Path] = {
    (42, 4):  Path("results/radius_ablation/seed42_r4/elimination/trajectories.json"),
    (42, 8):  Path("results/elimination/trajectories.json"),
    (42, 12): Path("results/radius_ablation/seed42_r12/elimination/trajectories.json"),
    (7,  4):  Path("results/radius_ablation/seed7_r4/elimination/trajectories.json"),
    (7,  8):  Path("results/seed7/elimination/trajectories.json"),
    (7,  12): Path("results/radius_ablation/seed7_r12/elimination/trajectories.json"),
    (13, 4):  Path("results/radius_ablation/seed13_r4/elimination/trajectories.json"),
    (13, 8):  Path("results/seed13/elimination/trajectories.json"),
    (13, 12): Path("results/radius_ablation/seed13_r12/elimination/trajectories.json"),
}

SEEDS = [42, 7, 13]
RADII = [4, 8, 12]

OUT_JSON = Path("results/radius_ablation_table.json")
OUT_MD = Path("results/radius_ablation_table.md")


def support_size(traj_data: dict, region: str) -> int:
    rtraj = traj_data[region]["trajectory"]
    return len(support_set_at_tolerance(rtraj, NUM_LANDMARKS, TAU))


def main() -> None:
    trajs = {key: json.loads(p.read_text(encoding="utf-8"))
             for key, p in PATHS.items()}
    regions = list(trajs[(42, 8)].keys())

    sizes: dict[str, dict[int, dict[int, int]]] = {
        region: {seed: {} for seed in SEEDS} for region in regions
    }
    for (seed, r), data in trajs.items():
        for region in regions:
            sizes[region][seed][r] = support_size(data, region)

    cross_seed: dict[str, dict[int, dict]] = {region: {} for region in regions}
    for region in regions:
        for r in RADII:
            vals = [sizes[region][seed][r] for seed in SEEDS]
            cross_seed[region][r] = {
                "values": vals,
                "mean": float(statistics.mean(vals)),
                "stdev": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                "min": int(min(vals)),
                "max": int(max(vals)),
                "range": int(max(vals) - min(vals)),
            }

    ranks: dict[int, dict[str, int]] = {}
    for r in RADII:
        ordered = sorted(regions, key=lambda x: cross_seed[x][r]["mean"])
        ranks[r] = {region: ordered.index(region) + 1 for region in regions}

    OUT_JSON.write_text(json.dumps({
        "tau": TAU,
        "seeds": SEEDS,
        "radii": RADII,
        "per_region_per_seed_per_radius": sizes,
        "cross_seed_per_radius": cross_seed,
        "rank_per_radius": ranks,
    }, indent=2), encoding="utf-8")

    md: list[str] = []
    md.append(f"## Occlusion-radius ablation: support-set sizes |S| at tau={TAU}\n")
    md.append("**Grid:** 3 seeds (42, 7, 13) x 3 radii (4, 8, 12 px on 256x256 input).\n")

    for seed in SEEDS:
        md.append(f"### Seed {seed}")
        md.append("| Region | r=4 | r=8 | r=12 | range (max-min) |")
        md.append("| --- | ---: | ---: | ---: | ---: |")
        for region in regions:
            vals = [sizes[region][seed][r] for r in RADII]
            rng = max(vals) - min(vals)
            md.append(f"| {region} | {vals[0]} | {vals[1]} | {vals[2]} | {rng} |")
        md.append("")

    md.append("### Cross-seed summary (mean +/- std across seeds {42, 7, 13})")
    md.append("| Region | r=4 mean+/-std | r=8 mean+/-std | r=12 mean+/-std | "
              "rank r=4 | rank r=8 | rank r=12 |")
    md.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for region in regions:
        cells = []
        for r in RADII:
            agg = cross_seed[region][r]
            cells.append(f"{agg['mean']:.1f} +/- {agg['stdev']:.1f}")
        md.append(
            f"| {region} | {cells[0]} | {cells[1]} | {cells[2]} | "
            f"{ranks[4][region]} | {ranks[8][region]} | {ranks[12][region]} |"
        )
    md.append("")

    md.append("### Robust claims (true at every (seed, radius) cell)")
    survives_under_98 = True
    survives_contour_largest = True
    survives_contour_gt_nose = True
    for seed in SEEDS:
        for r in RADII:
            row = {region: sizes[region][seed][r] for region in regions}
            if max(row.values()) >= NUM_LANDMARKS:
                survives_under_98 = False
            if max(row, key=row.get) != "contour":
                survives_contour_largest = False
            if row["contour"] <= row["nose"]:
                survives_contour_gt_nose = False
    md.append(f"- |S| < 98 in every cell: **{survives_under_98}**")
    md.append(f"- contour has the largest |S| in every cell: **{survives_contour_largest}**")
    md.append(f"- contour > nose in every cell: **{survives_contour_gt_nose}**")
    md.append("")

    md.append("### Fragile claims (radius-dependent)")
    md.append("Below contour, the rank order shifts across radii. The "
              "'eyes need least context' reading is a r >= 8 phenomenon; "
              "at r = 4, eyebrows are typically the smallest support set.")
    md.append("")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
