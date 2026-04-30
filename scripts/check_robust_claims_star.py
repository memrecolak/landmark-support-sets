"""T2: check the 3 robust HRNet claims on STAR's elimination output.

Reads results/star/trajectories_star.json and prints whether each of the
following holds on STAR (HRNet's claims survive every (seed, radius, τ) cell
of the robustness grid):

    (i)   |S_region| < 98 for every region
    (ii)  contour requires the largest support
    (iii) contour > nose

Also prints the side-by-side STAR vs HRNet support sizes for direct
comparison, and writes a markdown summary table to
results/star/robust_claims_summary.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_supports(path: Path) -> dict[str, int]:
    with path.open() as f:
        d = json.load(f)
    return {region: len(payload["support_set"]) for region, payload in d.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--star", type=Path,
        default=Path("results/star/trajectories_star.json"),
    )
    parser.add_argument(
        "--hrnet", type=Path,
        default=Path("results/elimination/trajectories.json"),
    )
    parser.add_argument(
        "--out-md", type=Path,
        default=Path("results/star/robust_claims_summary.md"),
    )
    args = parser.parse_args()

    star = load_supports(args.star)
    hrnet = load_supports(args.hrnet)

    regions = list(star.keys())
    rows = []
    for r in regions:
        rows.append((r, hrnet.get(r, "—"), star[r]))

    # Claims
    claim_i_hrnet = all(v < 98 for v in hrnet.values())
    claim_i_star = all(v < 98 for v in star.values())

    largest_hrnet = max(hrnet, key=hrnet.get)
    largest_star = max(star, key=star.get)
    claim_ii_hrnet = (largest_hrnet == "contour")
    claim_ii_star = (largest_star == "contour")

    claim_iii_hrnet = hrnet["contour"] > hrnet["nose"]
    claim_iii_star = star["contour"] > star["nose"]

    md = ["# T2: STAR cross-architecture robust-claim check", ""]
    md.append("## Per-region support cardinality (HRNet vs STAR, WFLW-test, r=8, tau=0.005)")
    md.append("")
    md.append("| region       | HRNet |S| | STAR |S| |")
    md.append("|--------------|----------:|---------:|")
    for r, h, s in rows:
        md.append(f"| {r:12s} | {h!s:>9} | {s!s:>8} |")
    md.append("")
    md.append("## Claims")
    md.append("")
    md.append("| claim | HRNet | STAR |")
    md.append("|-------|:-----:|:----:|")
    md.append(f"| (i) every \\|S\\| < 98 | {'OK' if claim_i_hrnet else 'FAIL'} | {'OK' if claim_i_star else 'FAIL'} |")
    md.append(f"| (ii) contour largest | {'OK' if claim_ii_hrnet else f'FAIL ({largest_hrnet})'} | {'OK' if claim_ii_star else f'FAIL ({largest_star})'} |")
    md.append(f"| (iii) contour > nose | {'OK' if claim_iii_hrnet else 'FAIL'} | {'OK' if claim_iii_star else 'FAIL'} |")
    md.append("")
    md.append(f"Largest-support region — HRNet: **{largest_hrnet}** ({hrnet[largest_hrnet]});  "
              f"STAR: **{largest_star}** ({star[largest_star]}).")

    out = "\n".join(md)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(out, encoding="utf-8")
    print(out)
    print(f"\nWrote {args.out_md}")


if __name__ == "__main__":
    main()
