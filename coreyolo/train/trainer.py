"""Training loop with cosine LR, warmup, and Apple silicon MPS by default."""

from __future__ import annotations

import json
import math
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from coreyolo.data.dataset import YOLODetectionDataset, collate_fn
from coreyolo.data.yaml import YOLODatasetYAML
from coreyolo.infer.nms import non_max_suppression
from coreyolo.nn.decode import xywh_to_xyxy
from coreyolo.nn.model import build_model, normalize_family
from coreyolo.nn.modules import normalize_act
from coreyolo.train.loss import DetectionLoss, SegmentationLoss
from coreyolo.train.metrics import ap_per_class
from coreyolo.utils import (
    increment_path,
    load_checkpoint,
    MIT_WEIGHT_LICENSE,
    origin_metadata,
    place_module,
    save_checkpoint,
    select_device,
)


@dataclass
class TrainConfig:
    data: str
    model: str = "n"
    epochs: int = 100
    batch: int = 16
    imgsz: int = 640
    device: str = "gpu"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 5e-4
    warmup_epochs: float = 3.0
    workers: int = 4
    project: str = "runs/detect"
    name: str = "train"
    act: str = "relu"
    family: str = "gelan"
    task: str = "detect"
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    mosaic: float = 1.0
    close_mosaic: int = 10
    hsv_h: float = 0.015
    hsv_s: float = 0.7
    hsv_v: float = 0.4
    degrees: float = 0.0
    translate: float = 0.1
    scale: float = 0.5
    fliplr: float = 0.5
    patience: int = 50
    val_period: int = 1
    resume: str | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        self.family = normalize_family(self.family)
        self.task = str(self.task).lower()
        self.act = normalize_act(self.act)


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.ema = deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        d = self.decay
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(d).add_(msd[k].detach(), alpha=1 - d)


def _cosine_lr(epoch: int, epochs: int, lr0: float, lrf: float) -> float:
    return lr0 * ((1 - math.cos(math.pi * epoch / max(epochs, 1))) / 2 * (lrf - 1) + 1)


def _sgd_param_groups(model: torch.nn.Module, lr: float, momentum: float, weight_decay: float) -> torch.optim.SGD:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.SGD(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=momentum,
        nesterov=True,
    )


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    cfg.act = normalize_act(cfg.act)
    cfg.family = normalize_family(cfg.family)
    device = select_device(cfg.device)
    task = str(cfg.task).lower()
    print(f"device: {device}  family={cfg.family}  task={task}  act={cfg.act}")
    spec = YOLODatasetYAML(cfg.data)
    project = cfg.project
    if task == "segment" and project == "runs/detect":
        project = "runs/segment"
    save_dir = increment_path(Path(project) / cfg.name)
    wdir = save_dir / "weights"
    wdir.mkdir(parents=True, exist_ok=True)
    (save_dir / "args.json").write_text(json.dumps(asdict(cfg), indent=2))
    (save_dir / "names.json").write_text(json.dumps(spec.names, indent=2))

    train_ds = YOLODetectionDataset(
        spec,
        "train",
        imgsz=cfg.imgsz,
        augment=True,
        mosaic=cfg.mosaic,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        fliplr=cfg.fliplr,
        task=task,
    )
    try:
        val_ds = YOLODetectionDataset(spec, "val", imgsz=cfg.imgsz, augment=False, mosaic=0.0, task=task)
    except FileNotFoundError:
        val_ds = None

    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=min(cfg.batch, len(train_ds)),
        shuffle=True,
        num_workers=cfg.workers,
        collate_fn=collate_fn,
        pin_memory=pin,
        drop_last=len(train_ds) >= cfg.batch,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=min(cfg.batch, len(val_ds)),
            shuffle=False,
            num_workers=cfg.workers,
            collate_fn=collate_fn,
            pin_memory=pin,
        )

    model, device = place_module(
        build_model(nc=spec.nc, scale=cfg.model, act=cfg.act, family=cfg.family, task=task),
        device,
    )
    origin: dict = {}
    if cfg.resume:
        ckpt = load_checkpoint(cfg.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        origin = origin_metadata(ckpt)
        if origin:
            license_id = str(origin.get("weights_license") or "")
            if license_id.upper().startswith("AGPL"):
                print(
                    "warning: --resume is converted YOLOv9 tensors (AGPL-3.0). "
                    "Fine-tunes stay AGPL-3.0; they are not MIT. See docs/licenses.md.",
                    file=sys.stderr,
                )
            else:
                print(
                    "warning: --resume is converted MultimediaTechLab tensors (MIT). "
                    "Keep the Kin-Yiu Wong / Hao-Tang Tsui copyright notice. See docs/licenses.md.",
                    file=sys.stderr,
                )

    if task == "segment":
        criterion = SegmentationLoss(model, box=cfg.box, cls=cfg.cls, dfl=cfg.dfl)
    else:
        criterion = DetectionLoss(model, box=cfg.box, cls=cfg.cls, dfl=cfg.dfl)
    optimizer = _sgd_param_groups(model, cfg.lr0, cfg.momentum, cfg.weight_decay)
    try:
        ema = ModelEMA(model)
    except Exception:
        ema = None

    best_fitness = -1.0
    stale = 0
    history: list[dict] = []

    for epoch in range(cfg.epochs):
        if cfg.close_mosaic and epoch == cfg.epochs - cfg.close_mosaic:
            train_ds.mosaic = 0.0
        model.train()
        lr = _cosine_lr(epoch, cfg.epochs, cfg.lr0, cfg.lrf)
        for g in optimizer.param_groups:
            g["lr"] = lr
        meters = {"loss": 0.0, "box": 0.0, "cls": 0.0, "dfl": 0.0}
        if task == "segment":
            meters["mask"] = 0.0
        pbar = tqdm(train_loader, desc=f"epoch {epoch + 1}/{cfg.epochs}", leave=False)
        n_batches = 0
        nw = max(round(cfg.warmup_epochs * len(train_loader)), 100)
        for i, batch in enumerate(pbar):
            ni = epoch * len(train_loader) + i
            if ni <= nw:
                xi = [0, nw]
                for j, g in enumerate(optimizer.param_groups):
                    g["lr"] = np.interp(ni, xi, [0.0, _cosine_lr(epoch, cfg.epochs, cfg.lr0, cfg.lrf)])
                    if "momentum" in g:
                        g["momentum"] = np.interp(ni, xi, [0.8, cfg.momentum])

            imgs = batch["img"].to(device, non_blocking=True)
            batch["labels"] = batch["labels"].to(device, non_blocking=True)
            if "masks" in batch:
                batch["masks"] = batch["masks"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(imgs)
            loss, items = criterion(preds, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            n_batches += 1
            for k in meters:
                meters[k] += items.get(k, 0.0)
            pbar.set_postfix({k: f"{meters[k] / n_batches:.3f}" for k in meters})

        row = {k: v / max(n_batches, 1) for k, v in meters.items()}
        row["epoch"] = epoch + 1
        row["lr"] = lr

        did_val = False
        if val_loader is not None and (
            (epoch + 1) % max(cfg.val_period, 1) == 0 or epoch + 1 == cfg.epochs
        ):
            eval_model = ema.ema if ema is not None else model
            metrics = validate(eval_model, val_loader, device, spec.nc)
            row.update(metrics)
            fitness = metrics["mAP50"]
            did_val = True
        else:
            fitness = best_fitness
            row["mAP50"] = history[-1].get("mAP50") if history else None
            row["mAP50-95"] = history[-1].get("mAP50-95") if history else None

        history.append(row)
        ckpt = {
            "epoch": epoch,
            "model": (ema.ema if ema is not None else model).state_dict(),
            "raw_model": model.state_dict(),
            "nc": spec.nc,
            "names": spec.names,
            "scale": cfg.model,
            "act": cfg.act,
            "family": cfg.family,
            "task": task,
            "nm": int(getattr(model.head, "nm", 0) or 0),
            "end2end": bool(getattr(model.head, "end2end", False)),
            "reg_max": int(getattr(model.head, "reg_max", 16)),
            "imgsz": cfg.imgsz,
            "args": asdict(cfg),
            "weights_license": MIT_WEIGHT_LICENSE,
        }
        if origin:
            ckpt.update(origin)
        save_checkpoint(wdir / "last.pt", ckpt)
        if did_val:
            if fitness >= best_fitness:
                best_fitness = fitness
                stale = 0
                save_checkpoint(wdir / "best.pt", ckpt)
            else:
                stale += 1
        elif not (wdir / "best.pt").exists():
            save_checkpoint(wdir / "best.pt", ckpt)
        tqdm.write(
            f"epoch {epoch + 1}/{cfg.epochs}  loss={row['loss']:.4f}  "
            f"mAP50={row.get('mAP50')}  mAP50-95={row.get('mAP50-95')}"
        )
        if did_val and cfg.patience and stale >= cfg.patience:
            tqdm.write(f"early stop after {cfg.patience} epochs without improvement")
            break

    (save_dir / "results.json").write_text(json.dumps(history, indent=2))
    return save_dir


@torch.no_grad()
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, nc: int) -> dict[str, float]:
    model.eval()
    dets, labs = [], []
    for batch in loader:
        imgs = batch["img"].to(device)
        pred = model(imgs)
        if isinstance(pred, (tuple, list)) and not torch.is_tensor(pred):
            pred = pred[0]
        if torch.is_tensor(pred) and pred.ndim == 3 and pred.shape[-1] == 6:
            nms = [row[row[:, 4] > 0.001] for row in pred]
        else:
            nms = non_max_suppression(pred, conf_thres=0.001, iou_thres=0.6, max_det=300)
        labels = batch["labels"]
        for i, det in enumerate(nms):
            d = det.cpu().numpy()
            dets.append(d)
            m = labels[:, 0] == i
            lab = labels[m]
            if lab.numel() == 0:
                labs.append(np.zeros((0, 5), dtype=np.float32))
                continue
            xyxy = xywh_to_xyxy(lab[:, 2:6]).cpu().numpy()
            packed = np.concatenate((lab[:, 1:2].cpu().numpy(), xyxy), 1)
            labs.append(packed)
    return ap_per_class(dets, labs, nc)
