"""Aggregate cross-seed gaze results and run paired Wilcoxon on the
15 leave-one-person-out folds.

Reads:
    results/gaze/gaze_results.json          (seed 42)
    results/gaze/seed7/gaze_results.json
    results/gaze/seed13/gaze_results.json

Writes:
    results/gaze/cross_seed_summary.json
    results/gaze/cross_seed_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
GAZE = ROOT / "results" / "gaze"

SEED_FILES = {
    "seed42": GAZE / "gaze_results.json",
    "seed7":  GAZE / "seed7" / "gaze_results.json",
    "seed13": GAZE / "seed13" / "gaze_results.json",
}


def per_fold(blob: dict, key: str) -> np.ndarray:
    pp = blob[key]["per_participant"]
    return np.array([pp[p] for p in sorted(pp)])


def main() -> None:
    summary = {"seeds": {}, "cross_seed": {}}

    for seed, path in SEED_FILES.items():
        results = json.loads(path.read_text())["results"]
        a = per_fold(results, "all98")
        s = per_fold(results, "eye_support")
        n = per_fold(results, "non_support")
        es_gap = s - a
        ns_gap = n - a
        summary["seeds"][seed] = {
            "support_size": results["eye_support"]["subset_size"],
            "non_support_size": results["non_support"]["subset_size"],
            "all98_mean": float(a.mean()),
            "all98_std": float(a.std()),
            "eye_support_mean": float(s.mean()),
            "eye_support_std": float(s.std()),
            "non_support_mean": float(n.mean()),
            "non_support_std": float(n.std()),
            "eye_support_gap_mean": float(es_gap.mean()),
            "non_support_gap_mean": float(ns_gap.mean()),
            "wilcoxon_eye_vs_all98_p": float(wilcoxon(es_gap).pvalue),
            "wilcoxon_non_vs_all98_p": float(wilcoxon(ns_gap).pvalue),
        }

    means = np.array([summary["seeds"][s]["eye_support_gap_mean"] for s in SEED_FILES])
    summary["cross_seed"] = {
        "eye_support_gap_mean_across_seeds": float(means.mean()),
        "eye_support_gap_max_abs": float(np.max(np.abs(means))),
        "non_support_gap_mean_across_seeds": float(np.mean(
            [summary["seeds"][s]["non_support_gap_mean"] for s in SEED_FILES])),
        "non_support_gap_min_across_seeds": float(np.min(
            [summary["seeds"][s]["non_support_gap_mean"] for s in SEED_FILES])),
    }

    out_json = GAZE / "cross_seed_summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out_json}")

    lines = []
    lines.append("## Cross-seed gaze regression on MPIIFaceGaze (15-fold LOPO)\n")
    lines.append("| Seed | |S_eye| | all98 mean ± std | eye_support mean ± std | non_support mean ± std | gap eye-all98 | gap non-all98 | Wilcoxon p (eye) | Wilcoxon p (non) |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for seed in SEED_FILES:
        d = summary["seeds"][seed]
        lines.append(
            f"| {seed} | {d['support_size']} | "
            f"{d['all98_mean']:.2f} ± {d['all98_std']:.2f}° | "
            f"{d['eye_support_mean']:.2f} ± {d['eye_support_std']:.2f}° | "
            f"{d['non_support_mean']:.2f} ± {d['non_support_std']:.2f}° | "
            f"{d['eye_support_gap_mean']:+.2f}° | "
            f"{d['non_support_gap_mean']:+.2f}° | "
            f"{d['wilcoxon_eye_vs_all98_p']:.3f} | "
            f"{d['wilcoxon_non_vs_all98_p']:.4f} |"
        )
    cs = summary["cross_seed"]
    lines.append("")
    lines.append(f"- **Eye-support gap, mean across seeds:** {cs['eye_support_gap_mean_across_seeds']:+.3f}° "
                 f"(max |gap| over seeds: {cs['eye_support_gap_max_abs']:.3f}°)")
    lines.append(f"- **Non-support gap, mean across seeds:** {cs['non_support_gap_mean_across_seeds']:+.3f}° "
                 f"(min over seeds: {cs['non_support_gap_min_across_seeds']:+.3f}°)")
    out_md = GAZE / "cross_seed_summary.md"
    out_md.write_text("\n".join(lines) + "\n")
    print(f"Saved {out_md}")


if __name__ == "__main__":
    main()
