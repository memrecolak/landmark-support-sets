"""Validate and preprocess MPIIFaceGaze into a compact JSON.

MPIIFaceGaze (Zhang et al., CVPRW 2017) — downloadable from
https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/appearance-based-gaze-estimation-in-the-wild/

Expected layout under data/mpiifacegaze/ after manual download + extraction:

    data/mpiifacegaze/
      p00/
        p00.txt
        day01/0001.jpg
        ...
      p01/
      ...
      p14/

Each pXX.txt has one line per sample with 28 whitespace-separated columns
(per the dataset readme):

    col 1       image file path (relative to pXX/)
    cols 2-3    gaze location on screen (pixels, screen coords)
    cols 4-15   6 face landmarks (x, y) in the image frame — four eye corners
                and two mouth corners
    cols 16-18  head-pose rotation vector (camera coords)
    cols 19-21  head-pose translation vector
    cols 22-24  face center (fc) in camera coords — averaged 3D position of
                the 6-landmark face model
    cols 25-27  gaze target (gt) in camera coords
    col 28      "left" or "right" — eye used for the evaluation subset

Per the readme: gaze_direction = gt - fc (normalized).

This script parses every pXX.txt into a single records list at
data/mpiifacegaze/processed/samples.json with, per record:

    {"participant": "p00", "image": "p00/day01/0005.jpg",
     "landmarks_6":    [[x, y], ...],
     "face_center_3d": [x, y, z],   "gaze_target_3d": [x, y, z],
     "head_rotation":  [rx, ry, rz], "head_translation": [x, y, z],
     "eye_side": "left" | "right",
     "bbox_from_landmarks": [x1, y1, x2, y2]}

We compute bbox_from_landmarks by padding the 6-landmark hull — used by the
dataset loader to crop a face region for HRNet-W18 inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

EXPECTED_FIELDS = 28  # per readme: image + 2 + 12 + 3 + 3 + 3 + 3 + 1


def parse_annotation_line(line: str, participant: str) -> dict | None:
    parts = line.strip().split()
    if not parts:
        return None
    if len(parts) < EXPECTED_FIELDS:
        raise ValueError(
            f"[{participant}] expected >= {EXPECTED_FIELDS} fields, got {len(parts)}: {line[:120]}..."
        )
    image = parts[0]
    lm6 = [[float(parts[3 + 2 * i]), float(parts[4 + 2 * i])] for i in range(6)]
    head_rotation = [float(parts[15]), float(parts[16]), float(parts[17])]
    head_translation = [float(parts[18]), float(parts[19]), float(parts[20])]
    face_center_3d = [float(parts[21]), float(parts[22]), float(parts[23])]
    gaze_target_3d = [float(parts[24]), float(parts[25]), float(parts[26])]
    eye_side = parts[27]

    xs = [p[0] for p in lm6]
    ys = [p[1] for p in lm6]
    pad = 0.6 * (max(xs) - min(xs))  # 60 % hull width padding on each side
    bbox = [min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad]

    return {
        "participant": participant,
        "image": f"{participant}/{image}",
        "landmarks_6": lm6,
        "face_center_3d": face_center_3d,
        "gaze_target_3d": gaze_target_3d,
        "head_rotation": head_rotation,
        "head_translation": head_translation,
        "eye_side": eye_side,
        "bbox_from_landmarks": bbox,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/mpiifacegaze"))
    args = parser.parse_args()

    root: Path = args.root
    if not root.exists():
        raise SystemExit(
            f"Expected {root} to exist. Download MPIIFaceGaze manually from "
            "https://www.mpi-inf.mpg.de/.../appearance-based-gaze-estimation-in-the-wild/ "
            "and extract under data/mpiifacegaze/."
        )

    participants = sorted(p.name for p in root.iterdir()
                          if p.is_dir() and p.name.startswith("p") and p.name != "processed")
    if not participants:
        raise SystemExit(f"No participant folders (p00..p14) found under {root}.")

    all_records: list[dict] = []
    skipped = 0
    for pid in participants:
        ann = root / pid / f"{pid}.txt"
        if not ann.exists():
            print(f"[warn] missing annotation {ann}; skipping")
            skipped += 1
            continue
        with ann.open("r", encoding="utf-8") as fp:
            for ln in fp:
                rec = parse_annotation_line(ln, pid)
                if rec is not None:
                    all_records.append(rec)

    out = root / "processed" / "samples.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fp:
        json.dump(all_records, fp)

    print(f"Wrote {len(all_records)} records across {len(participants) - skipped} "
          f"participants to {out}")


if __name__ == "__main__":
    main()
