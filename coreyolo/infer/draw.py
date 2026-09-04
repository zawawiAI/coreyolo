"""Draw detections and save annotated images."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


PALETTE = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def annotate(
    image: Image.Image,
    detections: np.ndarray,
    names: list[str],
    conf_digits: int = 2,
    masks: np.ndarray | None = None,
) -> Image.Image:
    """detections: (k, 6) xyxy, conf, cls or (k, 7) with a track id."""
    out = image.convert("RGB").copy()
    arr = np.asarray(out).copy()
    if masks is not None and detections is not None and len(detections) and len(masks):
        for i, row in enumerate(detections):
            if i >= len(masks):
                break
            mask = np.asarray(masks[i])
            if mask.shape[0] != arr.shape[0] or mask.shape[1] != arr.shape[1]:
                continue
            color = np.array(PALETTE[int(row[5]) % len(PALETTE)], dtype=np.float32)
            hit = mask > 0
            if hit.any():
                arr[hit] = (arr[hit].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
        out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    if detections is None or len(detections) == 0:
        return out
    for row in detections:
        x1, y1, x2, y2, conf, cls = [float(v) for v in row[:6]]
        c = int(cls)
        color = PALETTE[c % len(PALETTE)]
        name = names[c] if 0 <= c < len(names) else str(c)
        if row.shape[0] >= 7:
            label = f"{name}#{int(row[6])} {conf:.{conf_digits}f}"
        else:
            label = f"{name} {conf:.{conf_digits}f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if font is not None:
            bbox = draw.textbbox((x1, y1), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1), label, fill=(255, 255, 255), font=font)
        else:
            draw.text((x1, y1), label, fill=color)
    return out


def save_annotated(image: Image.Image, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path


def annotate_bgr(
    frame: np.ndarray,
    detections: np.ndarray,
    names: list[str],
    conf_digits: int = 2,
    masks: np.ndarray | None = None,
) -> np.ndarray:
    """Draw boxes/masks on a BGR OpenCV frame. Mutates a copy."""
    out = np.ascontiguousarray(frame)
    if masks is not None and detections is not None and len(detections) and len(masks):
        for i, row in enumerate(detections):
            if i >= len(masks):
                break
            mask = np.asarray(masks[i])
            if mask.shape[0] != out.shape[0] or mask.shape[1] != out.shape[1]:
                continue
            rgb = PALETTE[int(row[5]) % len(PALETTE)]
            color = np.array((rgb[2], rgb[1], rgb[0]), dtype=np.float32)
            hit = mask > 0
            if hit.any():
                out[hit] = (out[hit].astype(np.float32) * 0.55 + color * 0.45).astype(np.uint8)
    if detections is None or len(detections) == 0:
        return out
    try:
        import cv2
    except ImportError:
        return out
    for row in detections:
        x1, y1, x2, y2, conf, cls = [float(v) for v in row[:6]]
        c = int(cls)
        rgb = PALETTE[c % len(PALETTE)]
        color = (int(rgb[2]), int(rgb[1]), int(rgb[0]))
        name = names[c] if 0 <= c < len(names) else str(c)
        if row.shape[0] >= 7:
            label = f"{name}#{int(row[6])} {conf:.{conf_digits}f}"
        else:
            label = f"{name} {conf:.{conf_digits}f}"
        p1 = (int(x1), int(y1))
        p2 = (int(x2), int(y2))
        cv2.rectangle(out, p1, p2, color, 2, cv2.LINE_AA)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (p1[0], max(0, p1[1] - th - 6)), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(
            out,
            label,
            (p1[0] + 2, max(th + 2, p1[1] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return out
