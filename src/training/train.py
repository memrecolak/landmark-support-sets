"""HRNet-W18 landmark-detection training entry point.

Usage (from project root):
    python -m src.training.train --config configs/wflw_hrnet_w18.yaml
    python -m src.training.train --config configs/wflw_hrnet_w18_smoke.yaml --subset 64
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.datasets.wflw import WFLWDataset
from src.models.hrnet import HRNetLandmark
from src.training.losses import AdaptiveWingLoss
from src.training.metrics import decode_heatmaps, interocular_distance, nme_per_region


def resolve_output_dir(cfg: dict, config_path: Path) -> Path:
    raw = cfg["experiment"]["output_dir"]
    out = Path(raw.replace("${experiment.name}", cfg["experiment"]["name"]))
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(config_path, out / "config.yaml")
    return out


def build_dataloaders(cfg: dict, subset: int | None) -> tuple[DataLoader, DataLoader]:
    dcfg = cfg["data"]
    train_ds = WFLWDataset(
        root=dcfg["root"], split="train",
        image_size=dcfg["image_size"], heatmap_size=dcfg["heatmap_size"],
        heatmap_sigma=dcfg["heatmap_sigma"],
        augment=True, aug_cfg=dcfg.get("augment", {}),
    )
    val_ds = WFLWDataset(
        root=dcfg["root"], split="test",
        image_size=dcfg["image_size"], heatmap_size=dcfg["heatmap_size"],
        heatmap_sigma=dcfg["heatmap_sigma"],
        augment=False,
    )
    if subset is not None:
        train_ds.records = train_ds.records[:subset]
        val_ds.records = val_ds.records[: max(subset // 4, 4)]
    nw = cfg["train"].get("num_workers", 0)
    train_loader = DataLoader(
        train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=nw > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=nw > 0,
    )
    return train_loader, val_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--subset", type=int, default=None, help="Limit samples for smoke testing")
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as fp:
        cfg = yaml.safe_load(fp)

    seed = cfg["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    out_dir = resolve_output_dir(cfg, args.config)
    writer = SummaryWriter(str(out_dir / "tb"))

    train_loader, val_loader = build_dataloaders(cfg, args.subset)
    print(f"train samples: {len(train_loader.dataset)}  val samples: {len(val_loader.dataset)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HRNetLandmark(
        num_landmarks=cfg["data"]["num_landmarks"],
        pretrained=(cfg["model"].get("pretrained") == "imagenet"),
    ).to(device)

    criterion = AdaptiveWingLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    amp_enabled = bool(cfg["train"].get("amp", True))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    scale_factor = cfg["data"]["image_size"] / cfg["data"]["heatmap_size"]
    regions = cfg["eval"]["regions"]

    best_nme = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg['train']['epochs']}")
        for batch in pbar:
            img = batch["image"].to(device, non_blocking=True)
            hm = batch["heatmap"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                pred = model(img)
                loss = criterion(pred, hm)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")
        scheduler.step()
        writer.add_scalar("train/loss", total_loss / max(n_batches, 1), epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        # Validation
        model.eval()
        agg: dict[str, float] = {}
        count = 0
        for batch in val_loader:
            img = batch["image"].to(device, non_blocking=True)
            lm_img = batch["landmarks"].to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled), torch.no_grad():
                pred_hm = model(img).float()
            pred_lm = decode_heatmaps(pred_hm) * scale_factor
            iod = interocular_distance(lm_img)
            per = nme_per_region(pred_lm, lm_img, iod, regions)
            B = img.size(0)
            for k, v in per.items():
                agg[k] = agg.get(k, 0.0) + v * B
            count += B
        val = {k: v / max(count, 1) for k, v in agg.items()}
        for k, v in val.items():
            writer.add_scalar(f"val/nme_{k}", v, epoch)
        print(f"[epoch {epoch+1}] " + "  ".join(f"{k}={v:.4f}" for k, v in val.items()))

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "val_metrics": val,
            "cfg": cfg,
        }
        torch.save(ckpt, out_dir / "last.pth")
        if val["overall"] < best_nme:
            best_nme = val["overall"]
            torch.save(ckpt, out_dir / "best.pth")
            (out_dir / "best_metrics.json").write_text(json.dumps(val, indent=2))

    writer.close()
    print(f"Done. Best overall NME: {best_nme:.4f}")


if __name__ == "__main__":
    main()
