"""Aggregate per-region support-set sizes across seeds at each tau.

Reads results/<name>/elimination/tau_sweep.json for each configured seed,
and emits a single 3-seed mean/std/min/max table suitable for the paper.

Usage:
    python -m scripts.aggregate_cross_seed_tau
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

SEED_PATHS = {
    42: Path("results/elimination/tau_sweep.json"),
    7: Path("results/seed7/elimination/tau_sweep.json"),
    13: Path("results/seed13/elimination/tau_sweep.json"),
}
OUT_MD = Path("results/cross_seed_tau_table.md")
OUT_JSON = Path("results/cross_seed_tau_table.json")
FOCUS_TAU = "0.005"
RULE = "support_sizes_last_ok"


def main() -> None:
    per_seed: dict[int, dict] = {}
    for seed, path in SEED_PATHS.items():
        with path.open("r", encoding="utf-8") as fp:
            per_seed[seed] = json.load(fp)

    any_data = next(iter(per_seed.values()))
    taus = [f"{t:.3f}" for t in any_data["taus"]]
    regions = list(any_data[RULE].keys())

    agg: dict[str, dict[str, dict[str, float]]] = {}
    for region in regions:
        agg[region] = {}
        for tau in taus:
            vals = [per_seed[s][RULE][region][tau] for s in SEED_PATHS]
            agg[region][tau] = {
                "seed42": per_seed[42][RULE][region][tau],
                "seed7": per_seed[7][RULE][region][tau],
                "seed13": per_seed[13][RULE][region][tau],
                "min": min(vals),
                "max": max(vals),
                "mean": statistics.fmean(vals),
                "stdev": statistics.stdev(vals),
                "range": max(vals) - min(vals),
            }

    OUT_JSON.write_text(json.dumps(agg, indent=2), encoding="utf-8")

    lines = []
    lines.append(f"## Cross-seed support-set sizes at τ={FOCUS_TAU} (last-OK rule)\n")
    lines.append("| Region | seed 42 | seed 7 | seed 13 | min | max | mean ± std | range |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for region, row in agg.items():
        r = row[FOCUS_TAU]
        lines.append(
            f"| {region} | {r['seed42']} | {r['seed7']} | {r['seed13']} | "
            f"{r['min']} | {r['max']} | {r['mean']:.1f} ± {r['stdev']:.1f} | {r['range']} |"
        )
    lines.append("")
    lines.append("## Full sweep — mean ± std (last-OK rule)\n")
    lines.append("| Region | " + " | ".join(f"τ={t}" for t in taus) + " |")
    lines.append("| --- | " + " | ".join("---:" for _ in taus) + " |")
    for region, row in agg.items():
        cells = [f"{row[t]['mean']:.1f}±{row[t]['stdev']:.1f}" for t in taus]
        lines.append(f"| {region} | " + " | ".join(cells) + " |")

    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
