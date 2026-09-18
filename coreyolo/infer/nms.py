"""Class-aware NMS for decoded ``(B, 4+nc, N)`` xywh + score tensors."""

from __future__ import annotations

import torch
from torchvision.ops import nms

from coreyolo.nn.decode import xywh_to_xyxy


def non_max_suppression(
    pred: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    max_det: int = 300,
    classes: list[int] | None = None,
    extra: torch.Tensor | None = None,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Filter a batched decoded prediction.

    Parameters
    ----------
    pred:
        ``(B, 4 + nc, N)`` with ``xywh`` in pixels and per-class scores.
    extra:
        Optional ``(B, C, N)`` features (mask coefficients) gathered with NMS.
    Returns
    -------
    list of ``(k, 6)`` tensors: ``x1, y1, x2, y2, conf, cls``.
    If ``extra`` is set, also a list of ``(k, C)`` tensors.
    """
    if pred.ndim == 2:
        pred = pred.unsqueeze(0)
    batch, ch, _ = pred.shape
    nc = ch - 4
    out: list[torch.Tensor] = []
    extra_out: list[torch.Tensor] = []
    for i in range(batch):
        boxes = pred[i, :4].transpose(0, 1)
        scores = pred[i, 4:].transpose(0, 1)
        feat = extra[i].transpose(0, 1) if extra is not None else None
        if nc == 1:
            conf, cls_id = scores, torch.zeros(scores.shape[0], device=pred.device, dtype=torch.long)
            conf = conf.squeeze(-1)
        else:
            conf, cls_id = scores.max(1)
        keep = conf > conf_thres
        boxes, conf, cls_id = boxes[keep], conf[keep], cls_id[keep]
        if feat is not None:
            feat = feat[keep]
        if classes is not None:
            allow = torch.zeros_like(cls_id, dtype=torch.bool)
            for c in classes:
                allow |= cls_id == c
            boxes, conf, cls_id = boxes[allow], conf[allow], cls_id[allow]
            if feat is not None:
                feat = feat[allow]
        if boxes.numel() == 0:
            out.append(pred.new_zeros((0, 6)))
            if extra is not None:
                extra_out.append(extra.new_zeros((0, extra.shape[1])))
            continue
        # NMS math must be float32. Class offsets (cls * 7680) overflow float16
        # for COCO ids ≥ 9, which is why laptop/tv duplicates survived on MPS.
        xyxy = xywh_to_xyxy(boxes).float()
        conf_f = conf.float()
        cls_f = cls_id.float()
        max_wh = 7680
        nms_boxes = xyxy + cls_f[:, None] * max_wh
        keep_idx = nms(nms_boxes.cpu(), conf_f.cpu(), iou_thres).to(pred.device)
        keep_idx = keep_idx[:max_det]
        det = torch.cat((xyxy[keep_idx], conf_f[keep_idx, None], cls_f[keep_idx, None]), 1)
        out.append(det.to(dtype=pred.dtype))
        if extra is not None:
            extra_out.append(feat[keep_idx])
    if extra is not None:
        return out, extra_out
    return out


def scale_boxes(
    boxes: torch.Tensor,
    src_hw: tuple[int, int],
    dst_wh: tuple[int, int],
    pad: tuple[float, float],
    ratio: float,
) -> torch.Tensor:
    """Map letterboxed xyxy boxes back onto the original image."""
    boxes = boxes.clone()
    boxes[:, [0, 2]] -= pad[0]
    boxes[:, [1, 3]] -= pad[1]
    boxes[:, :4] /= ratio
    w, h = dst_wh
    boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, w)
    boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, h)
    return boxes
