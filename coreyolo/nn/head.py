"""Anchor-free detection head: DFL (v8) or NMS-free one-to-one (v26)."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn

from coreyolo.nn.decode import dist2bbox, make_anchors
from coreyolo.nn.modules import Conv, DFL, Proto


def _cls_branch(c: int, c3: int, nc: int, act: str, legacy: bool) -> nn.Sequential:
    if legacy:
        return nn.Sequential(Conv(c, c3, 3, act=act), Conv(c3, c3, 3, act=act), nn.Conv2d(c3, nc, 1))
    return nn.Sequential(
        nn.Sequential(Conv(c, c, 3, act=act, g=c), Conv(c, c3, 1, act=act)),
        nn.Sequential(Conv(c3, c3, 3, act=act, g=c3), Conv(c3, c3, 1, act=act)),
        nn.Conv2d(c3, nc, 1),
    )


def nms_free_topk(
    decoded: torch.Tensor, max_det: int = 300, return_idx: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Turn ``(B, 4+nc, A)`` xyxy+scores into ``(B, K, 6)`` xyxy, conf, cls."""
    batch, ch, anchors = decoded.shape
    nc = ch - 4
    k = min(max_det, anchors)
    preds = decoded.permute(0, 2, 1).contiguous()
    boxes, scores = preds.split((4, nc), dim=-1)
    keep = scores.amax(-1).topk(k, dim=1).indices
    boxes = boxes.gather(1, keep.unsqueeze(-1).expand(-1, -1, 4))
    scores = scores.gather(1, keep.unsqueeze(-1).expand(-1, -1, nc))
    scores, flat = scores.flatten(1).topk(k, dim=1)
    row = torch.arange(batch, device=decoded.device)[:, None]
    out = torch.cat(
        (boxes[row, flat // nc], scores.unsqueeze(-1), (flat % nc).unsqueeze(-1).float()),
        dim=-1,
    )
    if return_idx:
        return out, keep.gather(1, flat // nc)
    return out


class Detect(nn.Module):
    """Three-scale decoupled head (P3 / P4 / P5).

    Training returns raw per-scale maps (or a one2many/one2one dict when
    ``end2end``). Inference concatenates decoded boxes with sigmoid class
    scores. End-to-end mode selects top-300 boxes in-graph (NMS-free).
    """

    def __init__(
        self,
        nc: int,
        ch: tuple[int, ...],
        reg_max: int = 16,
        act: str = "relu",
        end2end: bool = False,
        legacy: bool = True,
        max_det: int = 300,
    ) -> None:
        super().__init__()
        self.nc = nc
        self.nl = len(ch)
        self.reg_max = reg_max
        self.no = nc + 4 * max(reg_max, 1)
        self.export = False
        self.end2end = end2end
        self.max_det = max_det
        c2 = max(16, ch[0] // 4, max(reg_max, 1) * 4)
        c3 = max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(c, c2, 3, act=act), Conv(c2, c2, 3, act=act), nn.Conv2d(c2, 4 * max(reg_max, 1), 1))
            for c in ch
        )
        self.cv3 = nn.ModuleList(_cls_branch(c, c3, nc, act, legacy) for c in ch)
        self.dfl = DFL(reg_max) if reg_max > 1 else nn.Identity()
        self.stride = torch.zeros(self.nl)
        self._export_anchors: torch.Tensor | None = None
        self._export_strides: torch.Tensor | None = None
        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    def prepare_export(self, feats: list[torch.Tensor]) -> None:
        """Cache grid anchors so the Core ML graph has no meshgrid / arange."""
        anchors, strides = make_anchors(feats, self.stride.to(feats[0].device), 0.5)
        self._export_anchors = anchors.transpose(0, 1).unsqueeze(0)
        self._export_strides = strides.transpose(0, 1)

    def _heads(self, feats: list[torch.Tensor], box_head: nn.ModuleList, cls_head: nn.ModuleList) -> list[torch.Tensor]:
        return [torch.cat((box_head[i](feats[i]), cls_head[i](feats[i])), 1) for i in range(self.nl)]

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor | list[torch.Tensor] | dict[str, list[torch.Tensor]]:
        if self.end2end:
            feats_o2o = [f.detach() for f in feats] if self.training else feats
            one2one = self._heads(feats_o2o, self.one2one_cv2, self.one2one_cv3)
            one2many = self._heads(feats, self.cv2, self.cv3)
            if self.training:
                return {"one2many": one2many, "one2one": one2one}
            return self._decode(one2one)
        outputs = self._heads(feats, self.cv2, self.cv3)
        if self.training:
            return outputs
        return self._decode(outputs)

    def _decode(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        b = outputs[0].shape[0]
        x = torch.cat([o.view(b, self.no, -1) for o in outputs], 2)
        box, cls = x.split((max(self.reg_max, 1) * 4, self.nc), 1)
        if self.export and self._export_anchors is not None:
            anchors = self._export_anchors.to(x.device, x.dtype)
            strides = self._export_strides.to(x.device, x.dtype)
        else:
            anchors, strides = make_anchors(outputs, self.stride.to(x.device), 0.5)
            anchors = anchors.transpose(0, 1).unsqueeze(0)
            strides = strides.transpose(0, 1)
        xywh = not self.end2end
        dbox = dist2bbox(self.dfl(box), anchors, xywh=xywh, dim=1) * strides
        decoded = torch.cat((dbox, cls.sigmoid()), 1)
        if self.end2end:
            return nms_free_topk(decoded, self.max_det)
        return decoded

    def bias_init(self) -> None:
        """Stable starting point for box and class logits."""
        branches = [(self.cv2, self.cv3)]
        if self.end2end:
            branches.append((self.one2one_cv2, self.one2one_cv3))
        for box_list, cls_list in branches:
            for box_branch, cls_branch, stride in zip(box_list, cls_list, self.stride):
                box_branch[-1].bias.data[:] = 1.0
                cls_branch[-1].bias.data[: self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)


class Segment(Detect):
    """Detect plus a proto mask branch (YOLOv8-seg style).

    Training returns ``(det, mask_coeff, proto)``. Inference returns decoded
    boxes plus per-anchor coefficients ``(B, nm, N)`` (or ``(B, K, nm)`` when
    NMS-free) and proto maps ``(B, nm, H, W)``.
    """

    def __init__(
        self,
        nc: int,
        ch: tuple[int, ...],
        nm: int = 32,
        npr: int = 256,
        reg_max: int = 16,
        act: str = "relu",
        end2end: bool = False,
        legacy: bool = True,
        max_det: int = 300,
    ) -> None:
        super().__init__(nc, ch, reg_max=reg_max, act=act, end2end=end2end, legacy=legacy, max_det=max_det)
        self.nm = nm
        self.npr = npr
        self.proto = Proto(ch[0], npr, nm, act=act)
        c4 = max(ch[0] // 4, nm)
        self.cv4 = nn.ModuleList(
            nn.Sequential(Conv(c, c4, 3, act=act), Conv(c4, c4, 3, act=act), nn.Conv2d(c4, nm, 1)) for c in ch
        )

    def _mask_coeff(self, feats: list[torch.Tensor]) -> torch.Tensor:
        b = feats[0].shape[0]
        return torch.cat([self.cv4[i](feats[i]).view(b, self.nm, -1) for i in range(self.nl)], 2)

    def forward(
        self, feats: list[torch.Tensor]
    ) -> tuple[torch.Tensor | list[torch.Tensor] | dict[str, list[torch.Tensor]], torch.Tensor, torch.Tensor]:
        proto = self.proto(feats[0])
        mc = self._mask_coeff(feats)
        if self.end2end:
            feats_o2o = [f.detach() for f in feats] if self.training else feats
            one2one = self._heads(feats_o2o, self.one2one_cv2, self.one2one_cv3)
            one2many = self._heads(feats, self.cv2, self.cv3)
            if self.training:
                return {"one2many": one2many, "one2one": one2one}, mc, proto
            decoded = self._decode_maps(one2one)
            topk, idx = nms_free_topk(decoded, self.max_det, return_idx=True)
            mc_k = mc.permute(0, 2, 1).gather(1, idx.unsqueeze(-1).expand(-1, -1, self.nm))
            return topk, mc_k, proto
        outputs = self._heads(feats, self.cv2, self.cv3)
        if self.training:
            return outputs, mc, proto
        return self._decode(outputs), mc, proto

    def _decode_maps(self, outputs: list[torch.Tensor]) -> torch.Tensor:
        """Decoded ``(B, 4+nc, N)`` before NMS-free top-k (xyxy + scores)."""
        b = outputs[0].shape[0]
        x = torch.cat([o.view(b, self.no, -1) for o in outputs], 2)
        box, cls = x.split((max(self.reg_max, 1) * 4, self.nc), 1)
        if self.export and self._export_anchors is not None:
            anchors = self._export_anchors.to(x.device, x.dtype)
            strides = self._export_strides.to(x.device, x.dtype)
        else:
            anchors, strides = make_anchors(outputs, self.stride.to(x.device), 0.5)
            anchors = anchors.transpose(0, 1).unsqueeze(0)
            strides = strides.transpose(0, 1)
        dbox = dist2bbox(self.dfl(box), anchors, xywh=False, dim=1) * strides
        return torch.cat((dbox, cls.sigmoid()), 1)
