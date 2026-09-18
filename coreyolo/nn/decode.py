"""Anchor generation and box encode/decode used by the head and loss."""

from __future__ import annotations

import torch


def make_static_anchors(
    imgsz: int,
    strides: list[int] | tuple[int, ...],
    offset: float = 0.5,
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid centers from Python ``imgsz`` / stride — no meshgrid in the export graph.

    ``h = w = imgsz // stride`` are ints, so every ``reshape`` is a static size.
    Call this once before ``torch.jit.trace`` and register the result as a buffer.
    """
    points, stride_col = [], []
    size = int(imgsz)
    for stride in strides:
        s = int(stride)
        h = w = size // s
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((gx, gy), -1).reshape(h * w, 2))
        stride_col.append(torch.full((h * w, 1), float(s), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_col)


def make_anchors(
    feats: list[torch.Tensor],
    strides: torch.Tensor,
    offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid centers in feature-map units, plus a matching stride column.

    Train / val path. Export uses :func:`make_static_anchors` so Core ML never
    sees ``arange`` / ``meshgrid``.
    """
    stride_list = [int(s) for s in strides.tolist()]
    h0, w0 = int(feats[0].shape[-2]), int(feats[0].shape[-1])
    imgsz = h0 * stride_list[0]
    if w0 * stride_list[0] != imgsz:
        # Non-square maps: fall back to per-feature sizes (still Python ints).
        points, stride_col = [], []
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(stride_list):
            h, w = int(feats[i].shape[-2]), int(feats[i].shape[-1])
            sy = torch.arange(h, device=device, dtype=dtype) + offset
            sx = torch.arange(w, device=device, dtype=dtype) + offset
            gy, gx = torch.meshgrid(sy, sx, indexing="ij")
            points.append(torch.stack((gx, gy), -1).reshape(h * w, 2))
            stride_col.append(torch.full((h * w, 1), float(stride), device=device, dtype=dtype))
        return torch.cat(points), torch.cat(stride_col)
    return make_static_anchors(imgsz, stride_list, offset, dtype=feats[0].dtype, device=feats[0].device)


def dist2bbox(
    distance: torch.Tensor,
    anchors: torch.Tensor,
    xywh: bool = True,
    dim: int = -1,
) -> torch.Tensor:
    """Convert ltrb distances (in grid units) to xywh or xyxy boxes."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchors - lt
    x2y2 = anchors + rb
    if xywh:
        return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)
    return torch.cat((x1y1, x2y2), dim)


def bbox2dist(anchors: torch.Tensor, boxes: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Convert xyxy boxes to ltrb distances, clamped to the DFL bin range."""
    x1y1, x2y2 = boxes.chunk(2, -1)
    return torch.cat((anchors - x1y1, x2y2 - anchors), -1).clamp_(0, reg_max - 0.01)


def xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    c, s = boxes.chunk(2, -1)
    return torch.cat((c - s / 2, c + s / 2), -1)


def xyxy_to_xywh(boxes: torch.Tensor) -> torch.Tensor:
    x1y1, x2y2 = boxes.chunk(2, -1)
    return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), -1)
