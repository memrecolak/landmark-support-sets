"""Fetch and preprocess WFLW (Wider Facial Landmarks in-the-wild).

WFLW is hosted at https://wywu.github.io/projects/LAB/WFLW.html
Images and annotations must be downloaded manually (Google Drive / Baidu).

Expected layout under data/wflw/ after manual download + extraction:

    data/wflw/
      WFLW_images/...
      WFLW_annotations/
        list_98pt_rect_attr_train_test/
          list_98pt_rect_attr_train.txt
          list_98pt_rect_attr_test.txt
        list_98pt_test/
          list_98pt_test_largepose.txt
          list_98pt_test_expression.txt
          list_98pt_test_illumination.txt
          list_98pt_test_makeup.txt
          list_98pt_test_occlusion.txt
          list_98pt_test_blur.txt

This script validates the layout, parses annotations into a compact format
(one JSON per split with normalized 98-point coordinates + face bbox +
attribute flags), and writes them to data/wflw/processed/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ANNOT_DIR = Path("WFLW_annotations/list_98pt_rect_attr_train_test")
TRAIN_LIST = ANNOT_DIR / "list_98pt_rect_attr_train.txt"
TEST_LIST = ANNOT_DIR / "list_98pt_rect_attr_test.txt"

ATTR_NAMES = ["pose", "expression", "illumination", "makeup", "occlusion", "blur"]


def parse_annotation_line(line: str) -> dict:
    """Each WFLW annotation line: 196 coords + 4 bbox + 6 attrs + image_path."""
    parts = line.strip().split()
    assert len(parts) == 196 + 4 + 6 + 1, f"Unexpected line length: {len(parts)}"
    coords = [float(x) for x in parts[:196]]
    landmarks = [(coords[2 * i], coords[2 * i + 1]) for i in range(98)]
    bbox = [float(x) for x in parts[196:200]]
    attrs = {name: int(parts[200 + i]) for i, name in enumerate(ATTR_NAMES)}
    image_path = parts[-1]
    return {
        "image": image_path,
        "bbox": bbox,
        "landmarks": landmarks,
        "attributes": attrs,
    }


def process_split(list_path: Path, out_path: Path) -> int:
    with list_path.open("r", encoding="utf-8") as fp:
        lines = [ln for ln in fp if ln.strip()]
    records = [parse_annotation_line(ln) for ln in lines]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(records, fp)
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/wflw"))
    args = parser.parse_args()

    root: Path = args.root
    if not (root / "WFLW_images").exists():
        raise SystemExit(
            f"Expected {root / 'WFLW_images'} to exist. "
            "Download WFLW images+annotations manually from "
            "https://wywu.github.io/projects/LAB/WFLW.html and extract under data/wflw/."
        )
    if not (root / TRAIN_LIST).exists() or not (root / TEST_LIST).exists():
        raise SystemExit(f"Missing annotation files under {root / ANNOT_DIR}.")

    processed = root / "processed"
    n_train = process_split(root / TRAIN_LIST, processed / "train.json")
    n_test = process_split(root / TEST_LIST, processed / "test.json")
    print(f"Wrote {n_train} train / {n_test} test records to {processed}/")


if __name__ == "__main__":
    main()
