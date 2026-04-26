"""Re-evaluate trained HRNet-W18 checkpoints with DARK decoding.

Per-seed: load checkpoint, run a single forward pass over the WFLW test
set, decode the heatmap stack with both the standard sign-offset decoder
and the DARK decoder, compute overall + per-region NME for each, and
write a side-by-side comparison.

This is the milestone-11 minimal rerun: the intervention pipeline is
Delta-on-Delta over the same decoder so support-set sizes / influence
matrices are unaffected. Only the absolute baseline-NME table changes.

Outputs:
    results/dark_decoding_eval.{json,md}

Usage:
    python -m scripts.eval_dark
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.datasets.wflw import WFLWDataset
from src.models.hrnet import HRNetLandmark
from src.training.metrics import (
    decode_heatmaps,
    decode_heatmaps_dark,
    interocular_distance,
    nme_per_region,
)

CKPTS = {
    42: Path("experiments/wflw_hrnet_w18_baseline/best.pth"),
    7:  Path("experiments/wflw_hrnet_w18_seed7/best.pth"),
    13: Path("experiments/wflw_hrnet_w18_seed13/best.pth"),
}

OUT_JSON = Path("results/dark_decoding_eval.json")
OUT_MD = Path("results/dark_decoding_eval.md")


def evaluate(checkpoint: Path) -> dict:
    """Run one forward pass; return per-region NME under both decoders."""
    ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetLandmark(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=False,
    ).to(device).eval()
    model.load_state_dict(ck["model"])

    val_ds = WFLWDataset(
        root=cfg["data"]["root"], split="test",
        image_size=cfg["data"]["image_size"],
        heatmap_size=cfg["data"]["heatmap_size"],
        heatmap_sigma=cfg["data"]["heatmap_sigma"],
        augment=False,
    )
    loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=0, pin_memory=True,
    )
    scale_factor = cfg["data"]["image_size"] / cfg["data"]["heatmap_size"]
    regions = cfg["eval"]["regions"]

    preds_signoff = []
    preds_dark = []
    targets = []
    iod = []
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device, non_blocking=True)
            lm = batch["landmarks"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                hm = model(img).float()
            preds_signoff.append((decode_heatmaps(hm) * scale_factor).cpu())
            preds_dark.append((decode_heatmaps_dark(hm) * scale_factor).cpu())
            targets.append(lm.cpu())
            iod.append(interocular_distance(lm).cpu())

    pred_signoff = torch.cat(preds_signoff, dim=0)
    pred_dark = torch.cat(preds_dark, dim=0)
    target = torch.cat(targets, dim=0)
    iod = torch.cat(iod, dim=0)

    return {
        "signoff": nme_per_region(pred_signoff, target, iod, regions),
        "dark":    nme_per_region(pred_dark,    target, iod, regions),
    }


def main() -> None:
    results: dict[int, dict] = {}
    for seed, ck_path in CKPTS.items():
        print(f"=== seed {seed}: {ck_path} ===")
        results[seed] = evaluate(ck_path)
        s_overall = results[seed]["signoff"]["overall"]
        d_overall = results[seed]["dark"]["overall"]
        print(f"  signoff overall NME: {s_overall:.4f}")
        print(f"  DARK    overall NME: {d_overall:.4f}  "
              f"(delta = {d_overall - s_overall:+.4f})")

    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Markdown table
    region_keys = ["overall", "contour", "left_brow", "right_brow", "nose",
                   "left_eye", "right_eye", "mouth_outer", "mouth_inner"]
    short = {"overall": "Overall", "contour": "Contour",
             "left_brow": "L-brow", "right_brow": "R-brow", "nose": "Nose",
             "left_eye": "L-eye", "right_eye": "R-eye",
             "mouth_outer": "Mouth-out", "mouth_inner": "Mouth-in"}

    md: list[str] = []
    md.append("## DARK vs sign-offset decoding (WFLW test, 3 seeds)\n")
    md.append("Same trained HRNet-W18 checkpoints, two decoder choices.")
    md.append("DARK fits a Gaussian to the heatmap mode (Zhang et al., CVPR 2020); "
              "sign-offset is argmax + 1-px sub-pixel refinement. Lower is better.\n")
    for decoder in ("signoff", "dark"):
        md.append(f"### {decoder.upper()}")
        md.append("| Seed | " + " | ".join(short[k] for k in region_keys) + " |")
        md.append("| ---: | " + " | ".join("---:" for _ in region_keys) + " |")
        for seed in CKPTS:
            row = results[seed][decoder]
            cells = [f"{row[k]:.4f}" for k in region_keys]
            md.append(f"| {seed} | " + " | ".join(cells) + " |")
        # 3-seed mean
        means = []
        for k in region_keys:
            vals = [results[seed][decoder][k] for seed in CKPTS]
            means.append(sum(vals) / len(vals))
        md.append("| **mean** | " +
                  " | ".join(f"**{m:.4f}**" for m in means) + " |")
        md.append("")

    md.append("### Decoder delta (DARK - signoff)")
    md.append("| Seed | " + " | ".join(short[k] for k in region_keys) + " |")
    md.append("| ---: | " + " | ".join("---:" for _ in region_keys) + " |")
    for seed in CKPTS:
        cells = []
        for k in region_keys:
            d = results[seed]["dark"][k] - results[seed]["signoff"][k]
            cells.append(f"{d:+.4f}")
        md.append(f"| {seed} | " + " | ".join(cells) + " |")
    md.append("")

    OUT_MD.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT_JSON}")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
