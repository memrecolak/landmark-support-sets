"""STAR-Loss WFLW landmark detector wrapper.

Loads the released ZhenglinZhou/STAR checkpoint (Stacked Hourglass V1 backbone +
AAM + STAR loss) and exposes a forward contract compatible with our
intervention pipeline:

  forward(x):  (B, 3, 256, 256) in [-1, 1] -> (B, 98, 64, 64) heatmaps
               (last-stack fusion heatmap)
  predict_coords(x):  (B, 3, 256, 256) in [-1, 1] -> (B, 98, 2) pixel coords
                      using STAR's own soft-argmax decoder

The checkpoint is downloaded from the public Google Drive link in the STAR
README and stored at external/STAR/weights/star_wflw.pkl.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

from src.models._star_backbone.stacked_hg import StackedHGNetV1

_WFLW_EDGE_INFO = (
    (False, tuple(range(0, 33))),                                # FaceContour
    (True,  (33, 34, 35, 36, 37, 38, 39, 40, 41)),               # RightEyebrow
    (True,  (42, 43, 44, 45, 46, 47, 48, 49, 50)),               # LeftEyebrow
    (False, (51, 52, 53, 54)),                                   # NoseLine
    (False, (55, 56, 57, 58, 59)),                               # Nose
    (True,  (60, 61, 62, 63, 64, 65, 66, 67)),                   # RightEye
    (True,  (68, 69, 70, 71, 72, 73, 74, 75)),                   # LeftEye
    (True,  (76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87)),   # OuterLip
    (True,  (88, 89, 90, 91, 92, 93, 94, 95)),                   # InnerLip
)


class STARLandmark(nn.Module):
    """Wraps StackedHGNetV1 (4 stacks, AAM, add_coord) at WFLW config."""

    HEATMAP_SIZE = 64
    INPUT_SIZE = 256
    NUM_LANDMARKS = 98

    def __init__(
        self,
        weights_path: str | Path,
        use_ema: bool = True,
    ):
        super().__init__()
        cfg = SimpleNamespace(
            width=self.INPUT_SIZE,
            height=self.INPUT_SIZE,
            use_AAM=True,
        )
        self.net = StackedHGNetV1(
            config=cfg,
            classes_num=[98, 9, 98],
            edge_info=_WFLW_EDGE_INFO,
            nstack=4,
            nlevels=4,
            in_channel=256,
            increase=0,
            add_coord=True,
            decoder_type="default",
        )
        ck = torch.load(str(weights_path), map_location="cpu", weights_only=False)
        state = ck["net_ema"] if (use_ema and "net_ema" in ck) else ck["net"]
        missing, unexpected = self.net.load_state_dict(state, strict=False)
        if missing:
            raise RuntimeError(f"missing keys when loading STAR weights: {missing[:5]} ...")
        if unexpected:
            raise RuntimeError(f"unexpected keys when loading STAR weights: {unexpected[:5]} ...")
        self.net.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return last-stack fusion heatmap (B, 98, 64, 64).

        STAR's native output is the soft-argmax of this map (returned by
        `predict_coords`). Provided here so that our existing
        decode_heatmaps + scale_factor pipeline can probe STAR uniformly.
        """
        _y, fusionmaps, _landmarks = self.net(x)
        return fusionmaps[-1]

    @torch.no_grad()
    def predict_coords(self, x: torch.Tensor) -> torch.Tensor:
        """Return STAR's own predicted coords in pixel space (B, 98, 2).

        Uses STAR's soft-argmax decoder; coords in [-1, 1] are mapped to
        [0, INPUT_SIZE - 1] via STAR's align_corners=True convention so the
        result matches what STAR's evaluate.py reports.
        """
        _y, _fusionmaps, landmarks = self.net(x)
        return ((landmarks + 1.0) / 2.0) * (self.INPUT_SIZE - 1)
