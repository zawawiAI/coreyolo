"""Anchor generation and box encode/decode used by the head and loss."""

from __future__ import annotations

import torch


def make_anchors(
    feats: list[torch.Tensor],
    strides: torch.Tensor,
    offset: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Grid centers in feature-map units, plus a matching stride column."""
    points, stride_col = [], []
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[-2:]
        sy = torch.arange(h, device=device, dtype=dtype) + offset
        sx = torch.arange(w, device=device, dtype=dtype) + offset
        gy, gx = torch.meshgrid(sy, sx, indexing="ij")
        points.append(torch.stack((gx, gy), -1).reshape(-1, 2))
        stride_col.append(torch.full((h * w, 1), float(stride), device=device, dtype=dtype))
    return torch.cat(points), torch.cat(stride_col)


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
