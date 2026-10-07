"""Hungarian set-prediction loss for the RF-DETR family.

One query matches one ground-truth box. Unmatched queries are background.
The loss is not used at export; eval already returns decoded boxes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from coreyolo.nn.rfdetr import _cxcywh_to_xyxy


def linear_sum_assignment(cost: torch.Tensor) -> tuple[list[int], list[int]]:
    """Minimize a rectangular cost matrix. Jonker-Volgenant, CPU, no SciPy."""
    a = cost.detach().float().cpu()
    n, m = a.shape
    if n == 0 or m == 0:
        return [], []
    transposed = n > m
    if transposed:
        a = a.T
        n, m = m, n
    u = torch.zeros(n + 1)
    v = torch.zeros(m + 1)
    p = torch.zeros(m + 1, dtype=torch.long)
    way = torch.zeros(m + 1, dtype=torch.long)
    inf = torch.tensor(float("inf"))
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = torch.full((m + 1,), inf)
        used = torch.zeros(m + 1, dtype=torch.bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = inf
            j1 = 0
            for j in range(1, m + 1):
                if bool(used[j]):
                    continue
                cur = a[i0 - 1, j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            for j in range(m + 1):
                if bool(used[j]):
                    u[p[j]] = u[p[j]] + delta
                    v[j] = v[j] - delta
                else:
                    minv[j] = minv[j] - delta
            j0 = j1
            if int(p[j0]) == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    rows: list[int] = []
    cols: list[int] = []
    for j in range(1, m + 1):
        if int(p[j]) != 0:
            rows.append(int(p[j]) - 1)
            cols.append(j - 1)
    if transposed:
        return cols, rows
    return rows, cols


def _giou(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise GIoU for xyxy boxes shaped ``(N, 4)``."""
    a_x1, a_y1, a_x2, a_y2 = a.unbind(-1)
    b_x1, b_y1, b_x2, b_y2 = b.unbind(-1)
    inter_w = (torch.minimum(a_x2, b_x2) - torch.maximum(a_x1, b_x1)).clamp(min=0)
    inter_h = (torch.minimum(a_y2, b_y2) - torch.maximum(a_y1, b_y1)).clamp(min=0)
    inter = inter_w * inter_h
    area_a = (a_x2 - a_x1).clamp(min=0) * (a_y2 - a_y1).clamp(min=0)
    area_b = (b_x2 - b_x1).clamp(min=0) * (b_y2 - b_y1).clamp(min=0)
    union = area_a + area_b - inter + eps
    iou = inter / union
    cw = torch.maximum(a_x2, b_x2) - torch.minimum(a_x1, b_x1)
    ch = torch.maximum(a_y2, b_y2) - torch.minimum(a_y1, b_y1)
    enclose = cw.clamp(min=0) * ch.clamp(min=0) + eps
    return iou - (enclose - union) / enclose


class SetCriterion(torch.nn.Module):
    """Classification focal loss, L1, and GIoU. Averaged over decoder layers."""

    def __init__(self, cls_w: float = 1.0, box_w: float = 5.0, giou_w: float = 2.0) -> None:
        super().__init__()
        self.cls_w = cls_w
        self.box_w = box_w
        self.giou_w = giou_w

    def forward(
        self,
        outputs: list[tuple[torch.Tensor, torch.Tensor]],
        batch: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        labels = batch["labels"]
        imgsz = float(batch["img"].shape[-1])
        total = outputs[0][0].new_zeros(())
        parts = {"cls": 0.0, "box": 0.0, "giou": 0.0}
        for logits, boxes in outputs:
            loss, items = self._layer(logits, boxes, labels, imgsz)
            total = total + loss
            for key in parts:
                parts[key] += items[key]
        scale = 1.0 / max(len(outputs), 1)
        total = total * scale
        items = {key: value * scale for key, value in parts.items()}
        items["loss"] = float(total.detach())
        items["dfl"] = 0.0
        return total, items

    def _layer(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        imgsz: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        b, _, nc = logits.shape
        cls_loss = logits.new_zeros(())
        box_loss = logits.new_zeros(())
        giou_loss = logits.new_zeros(())
        n_boxes = 0
        for i in range(b):
            if labels.numel():
                take = labels[:, 0] == i
                lab = labels[take]
            else:
                lab = labels
            tgt_cls = lab[:, 1].long() if lab.numel() else logits.new_zeros((0,), dtype=torch.long)
            tgt_box = lab[:, 2:6] / imgsz if lab.numel() else boxes.new_zeros((0, 4))
            tgt_box = tgt_box.clamp(0, 1)
            target = torch.zeros_like(logits[i])
            if tgt_cls.numel():
                q_idx, g_idx = self._match(logits[i], boxes[i], tgt_cls, tgt_box)
                if q_idx:
                    qi = torch.tensor(q_idx, device=logits.device, dtype=torch.long)
                    gi = torch.tensor(g_idx, device=logits.device, dtype=torch.long)
                    matched_cls = tgt_cls[gi]
                    target[qi, matched_cls] = 1
                    pred_b = boxes[i, qi]
                    gt_b = tgt_box[gi]
                    box_loss = box_loss + (pred_b - gt_b).abs().sum()
                    giou = _giou(_cxcywh_to_xyxy(pred_b), _cxcywh_to_xyxy(gt_b))
                    giou_loss = giou_loss + (1 - giou).sum()
                    n_boxes += int(qi.numel())
            cls_loss = cls_loss + _sigmoid_focal(logits[i], target)
        denom = max(n_boxes, 1)
        cls_loss = cls_loss / denom
        box_loss = box_loss / denom
        giou_loss = giou_loss / denom
        total = self.cls_w * cls_loss + self.box_w * box_loss + self.giou_w * giou_loss
        return total, {
            "cls": float((self.cls_w * cls_loss).detach()),
            "box": float((self.box_w * box_loss).detach()),
            "giou": float((self.giou_w * giou_loss).detach()),
        }

    @torch.no_grad()
    def _match(
        self,
        logits: torch.Tensor,
        boxes: torch.Tensor,
        tgt_cls: torch.Tensor,
        tgt_box: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        prob = logits.sigmoid()
        cost_cls = -prob[:, tgt_cls]
        cost_box = torch.cdist(boxes, tgt_box, p=1)
        cost_giou = -_giou(_cxcywh_to_xyxy(boxes)[:, None, :], _cxcywh_to_xyxy(tgt_box)[None, :, :])
        cost = self.cls_w * cost_cls + self.box_w * cost_box + self.giou_w * cost_giou
        return linear_sum_assignment(cost)


def _sigmoid_focal(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    return (alpha_t * ce * (1 - p_t).pow(gamma)).sum()
