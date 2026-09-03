"""Letterbox, mosaic, HSV jitter, and geometric augs for YOLO boxes."""

from __future__ import annotations

import math
import random

import numpy as np
from PIL import Image, ImageOps


def letterbox(
    image: Image.Image,
    imgsz: int,
    color: tuple[int, int, int] = (114, 114, 114),
    scaleup: bool = True,
) -> tuple[Image.Image, float, tuple[float, float]]:
    """Resize with unchanged aspect ratio and pad to a square ``imgsz``.

    Returns the canvas, the scale applied to width/height, and ``(pad_x, pad_y)``
    left/top padding in pixels.
    """
    w, h = image.size
    scale = min(imgsz / w, imgsz / h)
    if not scaleup:
        scale = min(scale, 1.0)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = image.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new("RGB", (imgsz, imgsz), color)
    pad_x = (imgsz - nw) / 2
    pad_y = (imgsz - nh) / 2
    canvas.paste(resized, (int(round(pad_x)), int(round(pad_y))))
    return canvas, scale, (pad_x, pad_y)


def hsv_jitter(image: Image.Image, h: float, s: float, v: float) -> Image.Image:
    """Random hue / saturation / value jitter. ``h`` is a fraction of 180."""
    if h == 0 and s == 0 and v == 0:
        return image
    img = image.convert("HSV")
    arr = np.asarray(img).astype(np.int16)
    dh = int((random.random() * 2 - 1) * h * 180)
    ds = 1.0 + (random.random() * 2 - 1) * s
    dv = 1.0 + (random.random() * 2 - 1) * v
    arr[..., 0] = (arr[..., 0] + dh) % 180
    arr[..., 1] = np.clip(arr[..., 1] * ds, 0, 255)
    arr[..., 2] = np.clip(arr[..., 2] * dv, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), mode="HSV").convert("RGB")


def random_perspective(
    image: Image.Image,
    labels: np.ndarray,
    degrees: float = 0.0,
    translate: float = 0.1,
    scale: float = 0.5,
    shear: float = 0.0,
    perspective: float = 0.0,
) -> tuple[Image.Image, np.ndarray]:
    """Affine warp that keeps YOLO ``xywh`` labels in pixel xyxy then back."""
    w, h = image.size
    # identity
    C = np.eye(3)
    C[0, 2] = -w / 2
    C[1, 2] = -h / 2

    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)
    P[2, 1] = random.uniform(-perspective, perspective)

    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = np.array(
        [[math.cos(math.radians(a)) * s, -math.sin(math.radians(a)) * s, 0],
         [math.sin(math.radians(a)) * s, math.cos(math.radians(a)) * s, 0]]
    )

    S = np.eye(3)
    S[0, 1] = math.tan(math.radians(random.uniform(-shear, shear)))
    S[1, 0] = math.tan(math.radians(random.uniform(-shear, shear)))

    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * w
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * h

    M = T @ S @ R @ P @ C
    # PIL affine is the inverse 2x3 of the 3x3
    try:
        Minv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        return image, labels
    affine = Minv[:2].flatten().tolist()
    warped = image.transform((w, h), Image.AFFINE, affine, resample=Image.BILINEAR, fillcolor=(114, 114, 114))

    if labels.size == 0:
        return warped, labels

    xyxy = labels.copy()
    # labels are pixel xywh
    cx, cy, bw, bh = xyxy[:, 1], xyxy[:, 2], xyxy[:, 3], xyxy[:, 4]
    x1, y1 = cx - bw / 2, cy - bh / 2
    x2, y2 = cx + bw / 2, cy + bh / 2
    corners = np.stack((x1, y1, x2, y1, x2, y2, x1, y2), 1).reshape(-1, 2)
    ones = np.ones((corners.shape[0], 1))
    corners_h = np.concatenate((corners, ones), 1)
    new = corners_h @ M.T
    new = new[:, :2] / np.clip(new[:, 2:3], 1e-6, None)
    new = new.reshape(-1, 8)
    x_coords = new[:, 0::2]
    y_coords = new[:, 1::2]
    nx1 = x_coords.min(1).clip(0, w)
    ny1 = y_coords.min(1).clip(0, h)
    nx2 = x_coords.max(1).clip(0, w)
    ny2 = y_coords.max(1).clip(0, h)
    n_w = nx2 - nx1
    n_h = ny2 - ny1
    keep = (n_w > 2) & (n_h > 2)
    out = labels[keep].copy()
    out[:, 1] = ((nx1 + nx2) / 2)[keep]
    out[:, 2] = ((ny1 + ny2) / 2)[keep]
    out[:, 3] = n_w[keep]
    out[:, 4] = n_h[keep]
    return warped, out


def mosaic4(
    images: list[Image.Image],
    labels_list: list[np.ndarray],
    imgsz: int,
) -> tuple[Image.Image, np.ndarray]:
    """2x2 mosaic onto a ``2*imgsz`` canvas, then cropped conceptually by letterbox later."""
    s = imgsz
    mosaic = Image.new("RGB", (s * 2, s * 2), (114, 114, 114))
    yc = int(random.uniform(s * 0.5, s * 1.5))
    xc = int(random.uniform(s * 0.5, s * 1.5))
    out_labels = []
    placements = (
        (xc, yc, True, True),    # top-left: right/bottom edges at xc, yc
        (xc, yc, False, True),   # top-right
        (xc, yc, True, False),   # bottom-left
        (xc, yc, False, False),  # bottom-right
    )
    for img, labels, (cx, cy, left, top) in zip(images, labels_list, placements):
        w, h = img.size
        if left and top:
            x1a, y1a, x2a, y2a = max(cx - w, 0), max(cy - h, 0), cx, cy
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif (not left) and top:
            x1a, y1a, x2a, y2a = cx, max(cy - h, 0), min(cx + w, s * 2), cy
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif left and (not top):
            x1a, y1a, x2a, y2a = max(cx - w, 0), cy, cx, min(s * 2, cy + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(h, y2a - y1a)
        else:
            x1a, y1a, x2a, y2a = cx, cy, min(cx + w, s * 2), min(s * 2, cy + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(h, y2a - y1a)
        crop = img.crop((x1b, y1b, x2b, y2b))
        mosaic.paste(crop, (x1a, y1a))
        padw, padh = x1a - x1b, y1a - y1b
        if labels.size:
            lab = labels.copy()
            lab[:, 1] = labels[:, 1] + padw
            lab[:, 2] = labels[:, 2] + padh
            out_labels.append(lab)
    labels = np.concatenate(out_labels, 0) if out_labels else np.zeros((0, 5), dtype=np.float32)
    # clip to mosaic canvas
    w_c, h_c = mosaic.size
    if labels.size:
        x1 = (labels[:, 1] - labels[:, 3] / 2).clip(0, w_c)
        y1 = (labels[:, 2] - labels[:, 4] / 2).clip(0, h_c)
        x2 = (labels[:, 1] + labels[:, 3] / 2).clip(0, w_c)
        y2 = (labels[:, 2] + labels[:, 4] / 2).clip(0, h_c)
        labels[:, 1] = (x1 + x2) / 2
        labels[:, 2] = (y1 + y2) / 2
        labels[:, 3] = x2 - x1
        labels[:, 4] = y2 - y1
        labels = labels[(labels[:, 3] > 2) & (labels[:, 4] > 2)]
    return mosaic, labels


def hflip(image: Image.Image, labels: np.ndarray) -> tuple[Image.Image, np.ndarray]:
    image = ImageOps.mirror(image)
    if labels.size:
        labels = labels.copy()
        labels[:, 1] = image.size[0] - labels[:, 1]
    return image, labels
