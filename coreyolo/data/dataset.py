"""Roboflow YOLO detection dataset: one ``.txt`` label file per image."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset

from coreyolo.data.augment import hsv_jitter, letterbox, mosaic4, random_perspective, hflip
from coreyolo.data.yaml import YOLODatasetYAML
from coreyolo.utils import IMAGE_EXTS


def load_yolo_labels(path: Path) -> np.ndarray:
    """Read Roboflow/YOLO labels: ``class x_center y_center width height`` (normalized).

    Extra trailing values (segmentation polygons, OBB, etc.) are ignored so a
    detect trainer can still consume Roboflow YOLO-Seg exports.
    """
    if not path.is_file():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = parts[:5]
        rows.append([float(cls), float(xc), float(yc), float(w), float(h)])
    if not rows:
        return np.zeros((0, 5), dtype=np.float32)
    arr = np.asarray(rows, dtype=np.float32)
    arr[:, 1:] = arr[:, 1:].clip(0, 1)
    return arr


def _xywh_to_poly(xc: float, yc: float, w: float, h: float) -> np.ndarray:
    x1, y1 = xc - w / 2, yc - h / 2
    x2, y2 = xc + w / 2, yc + h / 2
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


def load_yolo_instances(path: Path) -> tuple[np.ndarray, list[np.ndarray]]:
    """Load YOLO detect or seg labels.

    Detect lines are ``cls xc yc w h``. Seg lines are ``cls x1 y1 x2 y2 ...``
    (normalized polygon). ``cls xc yc w h`` plus extra polygon points is also
    accepted (Roboflow mixed export). Boxes without polygons become rectangles.
    """
    if not path.is_file():
        return np.zeros((0, 5), dtype=np.float32), []
    labels: list[list[float]] = []
    polygons: list[np.ndarray] = []
    for line in path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = float(parts[0])
        vals = np.asarray(parts[1:], dtype=np.float32)
        if vals.size == 4:
            xc, yc, w, h = vals.tolist()
            labels.append([cls, xc, yc, w, h])
            polygons.append(_xywh_to_poly(xc, yc, w, h))
            continue
        rest = None
        if vals.size >= 8 and vals.size % 2 == 0:
            xc, yc, bw, bh = vals[:4]
            looks_xywh = 0.0 <= xc <= 1.0 and 0.0 <= yc <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0
            extra = vals[4:]
            if looks_xywh and extra.size >= 6 and extra.size % 2 == 0:
                rest = extra.reshape(-1, 2)
                labels.append([cls, float(xc), float(yc), float(bw), float(bh)])
                polygons.append(np.clip(rest, 0, 1))
                continue
        if vals.size >= 6 and vals.size % 2 == 0:
            poly = np.clip(vals.reshape(-1, 2), 0, 1)
            x1, y1 = poly.min(0)
            x2, y2 = poly.max(0)
            labels.append([cls, float((x1 + x2) / 2), float((y1 + y2) / 2), float(x2 - x1), float(y2 - y1)])
            polygons.append(poly)
            continue
        xc, yc, w, h = vals[:4].tolist()
        labels.append([cls, xc, yc, w, h])
        polygons.append(_xywh_to_poly(xc, yc, w, h))
    if not labels:
        return np.zeros((0, 5), dtype=np.float32), []
    arr = np.asarray(labels, dtype=np.float32)
    arr[:, 1:] = arr[:, 1:].clip(0, 1)
    return arr, polygons


def polygons_to_masks(polys: list[np.ndarray], imgsz: int, down: int = 4) -> np.ndarray:
    """Rasterize letterboxed pixel polygons onto ``imgsz/down`` binary masks."""
    mh = mw = max(imgsz // down, 1)
    if not polys:
        return np.zeros((0, mh, mw), dtype=np.float32)
    masks = np.zeros((len(polys), mh, mw), dtype=np.uint8)
    sx = mw / imgsz
    sy = mh / imgsz
    for i, poly in enumerate(polys):
        if poly is None or len(poly) < 3:
            continue
        pts = [(float(x * sx), float(y * sy)) for x, y in poly]
        canvas = Image.new("L", (mw, mh), 0)
        ImageDraw.Draw(canvas).polygon(pts, fill=1)
        masks[i] = np.asarray(canvas, dtype=np.uint8)
    return masks.astype(np.float32)


def normalized_to_pixels(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    if labels.size == 0:
        return labels
    out = labels.copy()
    out[:, 1] *= width
    out[:, 2] *= height
    out[:, 3] *= width
    out[:, 4] *= height
    return out


def pixels_to_normalized(labels: np.ndarray, width: int, height: int) -> np.ndarray:
    if labels.size == 0:
        return labels
    out = labels.copy()
    out[:, 1] /= width
    out[:, 2] /= height
    out[:, 3] /= width
    out[:, 4] /= height
    return out


class YOLODetectionDataset(Dataset):
    """Images + YOLO txt labels as used by Roboflow's YOLO export."""

    def __init__(
        self,
        spec: YOLODatasetYAML,
        split: str = "train",
        imgsz: int = 640,
        augment: bool = False,
        mosaic: float = 1.0,
        hsv_h: float = 0.015,
        hsv_s: float = 0.7,
        hsv_v: float = 0.4,
        degrees: float = 0.0,
        translate: float = 0.1,
        scale: float = 0.5,
        fliplr: float = 0.5,
        task: str = "detect",
    ) -> None:
        self.spec = spec
        self.split = split
        self.imgsz = imgsz
        self.augment = augment
        self.mosaic = mosaic if task != "segment" else 0.0
        self.hsv_h, self.hsv_s, self.hsv_v = hsv_h, hsv_s, hsv_v
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.fliplr = fliplr
        self.task = task

        images_dir, labels_dir = spec.split(split)
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.files = sorted(
            p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in {images_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def _label_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def _load_pair(self, index: int) -> tuple[Image.Image, np.ndarray]:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        labels = load_yolo_labels(self._label_path(path))
        labels = normalized_to_pixels(labels, image.size[0], image.size[1])
        return image, labels

    def _load_instances(self, index: int) -> tuple[Image.Image, np.ndarray, list[np.ndarray]]:
        path = self.files[index]
        image = Image.open(path).convert("RGB")
        w, h = image.size
        labels, polys = load_yolo_instances(self._label_path(path))
        labels = normalized_to_pixels(labels, w, h)
        pixel_polys = [p * np.array([w, h], dtype=np.float32) for p in polys]
        return image, labels, pixel_polys

    def __getitem__(self, index: int) -> dict:
        if self.task == "segment":
            image, labels, polys = self._load_instances(index)
        elif self.augment and self.mosaic > 0 and random.random() < self.mosaic:
            indices = [index] + [random.randrange(len(self)) for _ in range(3)]
            imgs, labs = zip(*(self._load_pair(i) for i in indices))
            image, labels = mosaic4(list(imgs), list(labs), self.imgsz)
            polys = []
        else:
            image, labels = self._load_pair(index)
            polys = []

        if self.augment:
            image = hsv_jitter(image, self.hsv_h, self.hsv_s, self.hsv_v)
            if self.task != "segment":
                image, labels = random_perspective(
                    image, labels, degrees=self.degrees, translate=self.translate, scale=self.scale
                )
            if random.random() < self.fliplr:
                image, labels = hflip(image, labels)
                if polys:
                    width = image.size[0]
                    flipped = []
                    for poly in polys:
                        q = poly.copy()
                        q[:, 0] = width - q[:, 0]
                        flipped.append(q)
                    polys = flipped

        canvas, ratio, (pad_x, pad_y) = letterbox(image, self.imgsz, scaleup=self.augment)
        if labels.size:
            labels = labels.copy()
            labels[:, 1] = labels[:, 1] * ratio + pad_x
            labels[:, 2] = labels[:, 2] * ratio + pad_y
            labels[:, 3] *= ratio
            labels[:, 4] *= ratio
        if polys:
            shifted = []
            for poly in polys:
                q = poly.copy()
                q[:, 0] = q[:, 0] * ratio + pad_x
                q[:, 1] = q[:, 1] * ratio + pad_y
                shifted.append(q)
            polys = shifted

        tensor = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255.0
        boxes = torch.from_numpy(labels) if labels.size else torch.zeros((0, 5), dtype=torch.float32)
        item = {
            "img": tensor,
            "labels": boxes,
            "path": str(self.files[index]),
            "ori_size": torch.tensor(Image.open(self.files[index]).size),  # (w, h)
        }
        if self.task == "segment":
            masks = polygons_to_masks(polys, self.imgsz, down=4)
            item["masks"] = torch.from_numpy(masks)
        return item


def collate_fn(batch: list[dict]) -> dict:
    imgs = torch.stack([b["img"] for b in batch], 0)
    labels = []
    for i, b in enumerate(batch):
        lab = b["labels"]
        if lab.numel() == 0:
            continue
        idx = torch.full((lab.shape[0], 1), i, dtype=lab.dtype)
        labels.append(torch.cat((idx, lab), 1))
    labels_t = torch.cat(labels, 0) if labels else torch.zeros((0, 6), dtype=torch.float32)
    out = {
        "img": imgs,
        "labels": labels_t,  # (n, 6) batch, cls, cx, cy, w, h  in pixels
        "path": [b["path"] for b in batch],
        "ori_size": torch.stack([b["ori_size"] for b in batch], 0),
    }
    if "masks" in batch[0]:
        packed = [b["masks"] for b in batch if b["masks"].numel()]
        mh = mw = batch[0]["img"].shape[-1] // 4
        out["masks"] = torch.cat(packed, 0) if packed else torch.zeros((0, mh, mw), dtype=torch.float32)
    return out
