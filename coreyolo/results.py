"""Prediction results returned by the CoreYOLO Python SDK."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from coreyolo.infer.draw import annotate, save_annotated


class Result:
    """One image's detections (and optional masks).

    ``boxes`` is ``(k, 6)`` ``xyxy, conf, cls`` in original-image pixels.
    ``masks`` is ``(k, H, W)`` bool, aligned with ``boxes``, or ``None``.
    """

    def __init__(
        self,
        boxes: np.ndarray,
        names: list[str],
        masks: np.ndarray | None = None,
        path: str | None = None,
        orig: Image.Image | None = None,
        saved: str | None = None,
    ) -> None:
        if boxes is None:
            boxes = np.zeros((0, 6), dtype=np.float32)
        self.boxes = np.asarray(boxes, dtype=np.float32)
        self.names = names
        self.masks = masks
        self.path = path
        self.orig = orig.convert("RGB") if orig is not None else None
        self.saved = saved

    def __len__(self) -> int:
        return int(self.boxes.shape[0])

    def __repr__(self) -> str:
        src = self.path or "image"
        return f"Result({src}, n={len(self)})"

    @property
    def xyxy(self) -> np.ndarray:
        return self.boxes[:, :4] if len(self) else self.boxes.reshape(0, 4)

    @property
    def conf(self) -> np.ndarray:
        return self.boxes[:, 4] if len(self) else self.boxes.reshape(0)

    @property
    def cls(self) -> np.ndarray:
        return self.boxes[:, 5] if len(self) else self.boxes.reshape(0)

    def label(self, index: int) -> str:
        c = int(self.cls[index])
        return self.names[c] if 0 <= c < len(self.names) else str(c)

    def plot(self) -> Image.Image:
        if self.orig is None:
            raise ValueError("Result has no source image; pass a path or PIL image to predict()")
        return annotate(self.orig, self.boxes, self.names, masks=self.masks)

    def save(self, path: str | Path | None = None) -> Path:
        dest = Path(path) if path else Path(self.saved or "runs/predict/result.jpg")
        saved = save_annotated(self.plot(), dest)
        self.saved = str(saved)
        return saved
