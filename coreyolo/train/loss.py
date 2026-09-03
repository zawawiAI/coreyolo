"""Detection loss: Task-Aligned assignment, CIoU, DFL, and BCE classification."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreyolo.nn.decode import bbox2dist, dist2bbox, make_anchors, xywh_to_xyxy


def bbox_ciou(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete IoU between xyxy boxes. Broadcasts on the last dim."""
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.unbind(-1)
    w1 = (b1_x2 - b1_x1).clamp(min=0)
    h1 = (b1_y2 - b1_y1).clamp(min=0)
    w2 = (b2_x2 - b2_x1).clamp(min=0)
    h2 = (b2_y2 - b2_y1).clamp(min=0)

    inter_w = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(min=0)
    inter_h = (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(min=0)
    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
    ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4
    v = (4 / math.pi**2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


def _select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """``True`` where each anchor center lies inside a GT box.

    xy_centers: (n, 2), gt_bboxes: (b, max_gt, 4) xyxy → (b, max_gt, n)
    """
    n_anchors = xy_centers.shape[0]
    b, n_gt, _ = gt_bboxes.shape
    lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)  # (b*n_gt, 1, 2)
    bbox_deltas = torch.cat((xy_centers[None] - lt, rb - xy_centers[None]), dim=2).view(b, n_gt, n_anchors, 4)
    return bbox_deltas.amin(3).gt_(eps)


class TaskAlignedAssigner(nn.Module):
    """TOOD-style alignment: ``score^alpha * iou^beta``, top-k per GT."""

    def __init__(self, topk: int = 13, alpha: float = 1.0, beta: float = 6.0) -> None:
        super().__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta

    @torch.no_grad()
    def forward(
        self,
        pred_scores: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anc_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_bboxes: torch.Tensor,
        mask_gt: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        n_max = gt_bboxes.shape[1]
        b, n_anchors = pred_scores.shape[:2]

        if n_max == 0:
            fg = pred_scores.new_zeros((b, n_anchors), dtype=torch.bool)
            target_labels = pred_scores.new_full((b, n_anchors), 0, dtype=torch.long)
            target_bboxes = pred_scores.new_zeros((b, n_anchors, 4))
            target_scores = pred_scores.new_zeros((b, n_anchors, pred_scores.shape[-1]))
            fg_mask = fg.float()
            target_gt_idx = pred_scores.new_zeros((b, n_anchors), dtype=torch.long)
            return target_labels, target_bboxes, target_scores, fg, fg_mask, target_gt_idx

        mask_pos, align_metric, overlaps = self._get_pos_mask(
            pred_scores, pred_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self._select_highest_overlaps(mask_pos, overlaps, n_max)
        target_labels, target_bboxes, target_scores = self._get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask, pred_scores.shape[-1]
        )
        align_metric *= mask_pos
        pos_align = align_metric.amax(-1, keepdim=True).clamp(min=1e-9)
        pos_overlaps = (overlaps * mask_pos).amax(-1, keepdim=True)
        norm_align = (align_metric * pos_overlaps / pos_align).amax(-2).unsqueeze(-1)
        target_scores = target_scores * norm_align
        return target_labels, target_bboxes, target_scores, fg_mask.bool(), fg_mask.float(), target_gt_idx

    def _get_pos_mask(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = _select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self._alignment(pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        mask_topk = self._select_topk(align_metric, self.topk)
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def _alignment(self, pred_scores, pred_bboxes, gt_labels, gt_bboxes, mask_gt):
        b, n_max = gt_labels.shape[:2]
        na = pred_scores.shape[1]
        # gather predicted score for each GT class: (b, n_max, na)
        idx = gt_labels.squeeze(-1).clamp(min=0).long()
        scores = pred_scores.permute(0, 2, 1)
        bbox_scores = scores.gather(1, idx.unsqueeze(-1).expand(-1, -1, na))

        overlaps = bbox_ciou(
            pred_bboxes.unsqueeze(1).expand(-1, n_max, -1, -1),
            gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1),
        ).clamp(min=0)
        align = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta) * mask_gt
        return align, overlaps

    def _select_topk(self, metrics: torch.Tensor, topk: int) -> torch.Tensor:
        topk = min(topk, metrics.shape[-1])
        values, indices = torch.topk(metrics, topk, dim=-1, largest=True)
        mask = torch.zeros_like(metrics, dtype=torch.bool)
        mask.scatter_(-1, indices, values > 0)
        return mask.float()

    def _select_highest_overlaps(self, mask_pos, overlaps, n_max):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            # anchors assigned to multiple GTs: keep the highest IoU
            max_overlaps_idx = overlaps.argmax(1)
            is_multi = (fg_mask > 1).float()
            mask_multi = torch.zeros_like(mask_pos)
            mask_multi.scatter_(1, max_overlaps_idx.unsqueeze(1), 1)
            mask_pos = mask_pos * (1 - is_multi.unsqueeze(1)) + mask_multi * is_multi.unsqueeze(1)
            fg_mask = mask_pos.sum(-2)
        target_gt_idx = mask_pos.argmax(-2)
        return target_gt_idx, fg_mask, mask_pos

    def _get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask, num_classes):
        batch_ind = torch.arange(end=gt_labels.shape[0], device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * gt_labels.shape[1]
        target_labels = gt_labels.long().flatten()[target_gt_idx]
        target_bboxes = gt_bboxes.view(-1, 4)[target_gt_idx]
        target_labels = target_labels.clamp(min=0)
        target_scores = F.one_hot(target_labels.long(), num_classes).float()
        fg = fg_mask.bool().unsqueeze(-1)
        target_scores = torch.where(fg, target_scores, torch.zeros_like(target_scores))
        target_labels = torch.where(fg.squeeze(-1), target_labels, torch.zeros_like(target_labels))
        return target_labels, target_bboxes, target_scores


class DetectionLoss(nn.Module):
    def __init__(
        self,
        model,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        tal_topk: int = 10,
    ) -> None:
        super().__init__()
        head = model.head
        self.nc = head.nc
        self.reg_max = head.reg_max
        self.stride = head.stride
        self.no = head.no
        self.box_w = box
        self.cls_w = cls
        self.dfl_w = dfl
        self.assigner = TaskAlignedAssigner(topk=tal_topk)
        self.assigner_o2o = TaskAlignedAssigner(topk=1)
        self.proj = torch.arange(max(self.reg_max, 1), dtype=torch.float32)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.reg_bins = max(int(self.reg_max), 1)

    def forward(self, preds, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(preds, dict):
            loss_m, m = self._loss_one(preds["one2many"], batch, self.assigner)
            loss_o, o = self._loss_one(preds["one2one"], batch, self.assigner_o2o)
            loss = loss_m + loss_o
            return loss, {
                "loss": float(loss.detach()),
                "box": float((m["box"] + o["box"]).detach()),
                "cls": float((m["cls"] + o["cls"]).detach()),
                "dfl": float((m["dfl"] + o["dfl"]).detach()),
            }
        loss, items = self._loss_one(preds, batch, self.assigner)
        return loss, {k: float(v.detach()) if torch.is_tensor(v) else v for k, v in {"loss": loss, **items}.items()}

    def _loss_one(
        self,
        feats: list[torch.Tensor],
        batch: dict,
        assigner: TaskAlignedAssigner,
        pred_mc: torch.Tensor | None = None,
        proto: torch.Tensor | None = None,
    ):
        device = feats[0].device
        pred_cat = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2)
        pred_distri, pred_scores = pred_cat.split((self.reg_bins * 4, self.nc), 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchors, strides = make_anchors(feats, self.stride.to(device), 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(feats[0].shape[2:], device=device, dtype=dtype) * self.stride[0]

        targets = self._prepare_targets(batch["labels"], batch_size, imgsz)
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)

        pred_bboxes = dist2bbox(self._dfl_decode(pred_distri), anchors.unsqueeze(0), xywh=False, dim=-1)
        pred_bboxes_pix = pred_bboxes * strides

        assigner = assigner.to(device)
        _, target_bboxes, target_scores, fg_mask, _, target_gt_idx = assigner(
            pred_scores.detach().sigmoid(),
            pred_bboxes_pix.detach(),
            anchors * strides,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )

        target_bboxes_pix = target_bboxes.clone()
        target_bboxes = target_bboxes / strides
        target_scores_sum = max(target_scores.sum(), 1)

        loss_cls = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum
        if fg_mask.sum():
            loss_iou, loss_dfl = self._bbox_loss(
                pred_distri, pred_bboxes, anchors, target_bboxes, target_scores, target_scores_sum, fg_mask
            )
        else:
            loss_iou = pred_scores.sum() * 0
            loss_dfl = pred_scores.sum() * 0

        loss = self.box_w * loss_iou + self.cls_w * loss_cls + self.dfl_w * loss_dfl
        items = {"box": loss_iou, "cls": loss_cls, "dfl": loss_dfl}
        if pred_mc is not None and proto is not None:
            loss_mask = self._mask_loss(
                pred_mc, proto, fg_mask, target_gt_idx, target_bboxes_pix, batch, imgsz
            )
            loss = loss + getattr(self, "mask_w", 1.0) * loss_mask
            items["mask"] = loss_mask
        return loss, items

    def _dfl_decode(self, pred_distri: torch.Tensor) -> torch.Tensor:
        if self.reg_max <= 1:
            return pred_distri
        b, a, c = pred_distri.shape
        pred = pred_distri.view(b, a, 4, c // 4).softmax(3)
        proj = self.proj.to(pred.device, pred.dtype)
        return pred.matmul(proj)

    def _bbox_loss(self, pred_distri, pred_bboxes, anchors, target_bboxes, target_scores, target_scores_sum, fg_mask):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou = bbox_ciou(pred_bboxes[fg_mask], target_bboxes[fg_mask])
        loss_iou = ((1.0 - iou) * weight.squeeze(-1)).sum() / target_scores_sum
        if self.reg_max <= 1:
            return loss_iou, loss_iou * 0
        target_ltrb = bbox2dist(anchors, target_bboxes, self.reg_max)
        loss_dfl = self._df_loss(pred_distri[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask].view(-1)) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tl = target.long()
        tr = tl + 1
        wl = tr - target
        wr = 1 - wl
        tl = tl.clamp(0, self.reg_max - 1)
        tr = tr.clamp(0, self.reg_max - 1)
        loss = F.cross_entropy(pred_dist, tl, reduction="none") * wl + F.cross_entropy(pred_dist, tr, reduction="none") * wr
        return loss.view(-1, 4).mean(-1, keepdim=True)

    def _prepare_targets(self, labels: torch.Tensor, batch_size: int, imgsz: torch.Tensor) -> torch.Tensor:
        """labels: (n, 6) batch, cls, cx, cy, w, h in pixels → padded (b, max_gt, 5) cls+xyxy."""
        device = labels.device
        if labels.numel() == 0:
            return torch.zeros((batch_size, 0, 5), device=device)
        out = []
        max_n = 0
        per = []
        for i in range(batch_size):
            m = labels[:, 0] == i
            lab = labels[m]
            if lab.numel() == 0:
                per.append(torch.zeros((0, 5), device=device))
                continue
            xyxy = xywh_to_xyxy(lab[:, 2:6])
            packed = torch.cat((lab[:, 1:2], xyxy), 1)
            per.append(packed)
            max_n = max(max_n, packed.shape[0])
        if max_n == 0:
            return torch.zeros((batch_size, 0, 5), device=device)
        padded = torch.zeros((batch_size, max_n, 5), device=device)
        for i, p in enumerate(per):
            if p.shape[0]:
                padded[i, : p.shape[0]] = p
        return padded


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero mask pixels outside xyxy boxes (boxes in mask-pixel coordinates)."""
    n, h, w = masks.shape
    x1, y1, x2, y2 = boxes.unbind(1)
    r = torch.arange(w, device=masks.device, dtype=masks.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=masks.dtype)[None, :, None]
    x1, y1, x2, y2 = x1[:, None, None], y1[:, None, None], x2[:, None, None], y2[:, None, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class SegmentationLoss(DetectionLoss):
    """Detection loss plus cropped mask BCE against proto coefficients."""

    def __init__(
        self,
        model,
        box: float = 7.5,
        cls: float = 0.5,
        dfl: float = 1.5,
        mask: float = 1.0,
        tal_topk: int = 10,
    ) -> None:
        super().__init__(model, box=box, cls=cls, dfl=dfl, tal_topk=tal_topk)
        self.mask_w = mask

    def forward(self, preds, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        det, pred_mc, proto = preds
        if isinstance(det, dict):
            loss_m, m = self._loss_one(det["one2many"], batch, self.assigner, pred_mc, proto)
            loss_o, o = self._loss_one(det["one2one"], batch, self.assigner_o2o)
            loss = loss_m + loss_o
            mask_term = m.get("mask", m["box"] * 0)
            return loss, {
                "loss": float(loss.detach()),
                "box": float((m["box"] + o["box"]).detach()),
                "cls": float((m["cls"] + o["cls"]).detach()),
                "dfl": float((m["dfl"] + o["dfl"]).detach()),
                "mask": float(mask_term.detach()) if torch.is_tensor(mask_term) else float(mask_term),
            }
        loss, items = self._loss_one(det, batch, self.assigner, pred_mc, proto)
        return loss, {k: float(v.detach()) if torch.is_tensor(v) else v for k, v in {"loss": loss, **items}.items()}

    def _mask_loss(
        self,
        pred_mc: torch.Tensor,
        proto: torch.Tensor,
        fg_mask: torch.Tensor,
        target_gt_idx: torch.Tensor,
        target_bboxes_pix: torch.Tensor,
        batch: dict,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        pred_mc = pred_mc.permute(0, 2, 1).contiguous()
        masks_gt = batch.get("masks")
        labels = batch["labels"]
        if masks_gt is None or masks_gt.numel() == 0:
            return proto.sum() * 0 + pred_mc.sum() * 0
        proto_h, proto_w = proto.shape[-2:]
        img_h, img_w = float(imgsz[0]), float(imgsz[1])
        loss = proto.sum() * 0
        for i in range(fg_mask.shape[0]):
            fg = fg_mask[i]
            if not fg.any():
                loss = loss + pred_mc[i].sum() * 0
                continue
            gt_m = masks_gt[labels[:, 0] == i]
            if gt_m.numel() == 0:
                loss = loss + pred_mc[i].sum() * 0
                continue
            idx = target_gt_idx[i][fg].long().clamp(0, gt_m.shape[0] - 1)
            gt = gt_m[idx].to(device=proto.device, dtype=proto.dtype)
            if gt.shape[-2:] != (proto_h, proto_w):
                gt = F.interpolate(gt.unsqueeze(1), (proto_h, proto_w), mode="nearest").squeeze(1)
            xyxy = target_bboxes_pix[i][fg]
            gain = xyxy.new_tensor([proto_w / img_w, proto_h / img_h, proto_w / img_w, proto_h / img_h])
            xyxy_m = xyxy * gain
            area = (xyxy_m[:, 2] - xyxy_m[:, 0]).clamp(min=1.0) * (xyxy_m[:, 3] - xyxy_m[:, 1]).clamp(min=1.0)
            pred_mask = torch.einsum("in,nhw->ihw", pred_mc[i][fg], proto[i])
            per = F.binary_cross_entropy_with_logits(pred_mask, gt.clamp(0, 1), reduction="none")
            per = crop_mask(per, xyxy_m).mean((1, 2)) / area
            loss = loss + per.sum()
        return loss / fg_mask.sum().clamp(min=1)
