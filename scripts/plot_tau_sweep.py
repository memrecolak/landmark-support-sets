"""Support-set-size-vs-tolerance sweep from an elimination trajectory.

Reads `results/.../trajectories.json` (produced by run_elimination.py),
computes per-region support-set cardinality as a function of NME tolerance
tau, and emits:
    - results/.../tau_sweep.json     — numerical table
    - results/.../tau_sweep.png      — curve per region
    - results/.../tau_sweep_table.md — markdown table for paper insertion

Audit-mandated claim-1 defense (see paper.md, 2026-04-23): reviewers will
ask how sensitive per-region support sizes are to the tau choice. This
script answers that without re-running HRNet — the trajectory already
contains target-NME at every intermediate mask size.

Usage:
    python -m scripts.plot_tau_sweep \
        --trajectories results/elimination/trajectories.json \
        --out-dir results/elimination
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_TAUS = (0.001, 0.002, 0.003, 0.005, 0.007, 0.010, 0.015, 0.020)
NUM_LANDMARKS = 98


def support_size_at_tau(traj_entries: list[dict], baseline: float, tau: float) -> int:
    """|S| at tolerance tau using the same rule as
    src.ablation.greedy.support_set_at_tolerance: the LAST step across the
    trajectory whose NME stays within baseline + tau. If NME crosses the
    threshold and later dips back under, the later (tighter) config wins.
    Support-set size after removing s landmarks is NUM_LANDMARKS - s.
    """
    threshold = baseline + tau
    last_ok_step = 0
    for e in traj_entries:
        if e["target_nme"] <= threshold:
            last_ok_step = e["step"]
    return NUM_LANDMARKS - last_ok_step


def support_size_at_tau_first_break(traj_entries: list[dict], baseline: float, tau: float) -> int:
    """Conservative variant: stops at the FIRST step where NME exceeds
    baseline + tau. Upper-bounds the true support set and is monotone in tau."""
    threshold = baseline + tau
    last_ok_step = 0
    for e in traj_entries:
        if e["target_nme"] <= threshold:
            last_ok_step = e["step"]
        else:
            break
    return NUM_LANDMARKS - last_ok_step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trajectories", type=Path,
                    default=Path("results/elimination/trajectories.json"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/elimination"))
    ap.add_argument("--taus", type=float, nargs="+", default=list(DEFAULT_TAUS))
    args = ap.parse_args()

    with args.trajectories.open("r", encoding="utf-8") as fp:
        trajs = json.load(fp)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Numerical tables: per-region support size at each tau (both rules)
    table_last: dict[str, dict[str, int]] = {}
    table_first: dict[str, dict[str, int]] = {}
    for region, data in trajs.items():
        baseline = float(data["baseline_nme"])
        entries = data["trajectory"]
        table_last[region] = {f"{t:.3f}": support_size_at_tau(entries, baseline, t)
                              for t in args.taus}
        table_first[region] = {f"{t:.3f}": support_size_at_tau_first_break(entries, baseline, t)
                               for t in args.taus}

    out_json = args.out_dir / "tau_sweep.json"
    with out_json.open("w", encoding="utf-8") as fp:
        json.dump({"taus": list(args.taus),
                   "support_sizes_last_ok": table_last,
                   "support_sizes_first_break": table_first}, fp, indent=2)

    md_lines = []
    md_lines.append("## Support-set size vs NME tolerance — *last-OK rule* (paper default)\n")
    md_lines.append("| Region | " + " | ".join(f"τ={t:.3f}" for t in args.taus) + " |")
    md_lines.append("| --- | " + " | ".join("---:" for _ in args.taus) + " |")
    for region, row in table_last.items():
        md_lines.append(
            f"| {region} | " + " | ".join(str(row[f'{t:.3f}']) for t in args.taus) + " |"
        )
    md_lines.append("")
    md_lines.append("## Support-set size vs NME tolerance — *first-break rule* (conservative upper bound)\n")
    md_lines.append("| Region | " + " | ".join(f"τ={t:.3f}" for t in args.taus) + " |")
    md_lines.append("| --- | " + " | ".join("---:" for _ in args.taus) + " |")
    for region, row in table_first.items():
        md_lines.append(
            f"| {region} | " + " | ".join(str(row[f'{t:.3f}']) for t in args.taus) + " |"
        )
    out_md = args.out_dir / "tau_sweep_table.md"
    out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Plot: |S| vs tau per region
    colors = plt.get_cmap("tab10").colors
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for i, (region, data) in enumerate(trajs.items()):
        baseline = float(data["baseline_nme"])
        entries = data["trajectory"]
        sizes = [support_size_at_tau(entries, baseline, t) for t in args.taus]
        ax.plot(args.taus, sizes, marker="o", color=colors[i % len(colors)],
                label=region, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("NME tolerance τ (added to region baseline)")
    ax.set_ylabel("Support-set size |S| (out of 98 landmarks)")
    ax.set_title("Per-region support-set cardinality vs tolerance")
    ax.axvline(0.005, color="#606060", linestyle="--", linewidth=0.8,
               label="τ = 0.005 (paper)")
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_png = args.out_dir / "tau_sweep.png"
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
