"""Turn proto coefficients into instance masks and map them off the letterbox."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def crop_mask(masks: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Zero mask pixels outside xyxy boxes (boxes in mask-pixel coordinates)."""
    _, h, w = masks.shape
    x1, y1, x2, y2 = boxes.unbind(1)
    r = torch.arange(w, device=masks.device, dtype=masks.dtype)[None, None, :]
    c = torch.arange(h, device=masks.device, dtype=masks.dtype)[None, :, None]
    x1, y1, x2, y2 = x1[:, None, None], y1[:, None, None], x2[:, None, None], y2[:, None, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


def process_mask(
    proto: torch.Tensor,
    coeff: torch.Tensor,
    boxes_xyxy: torch.Tensor,
    imgsz: int,
) -> torch.Tensor:
    """``proto`` (nm, H, W), ``coeff`` (k, nm), boxes on the letterboxed canvas → (k, imgsz, imgsz) bool."""
    if coeff.numel() == 0:
        return proto.new_zeros((0, imgsz, imgsz), dtype=torch.bool)
    nm, mh, mw = proto.shape
    masks = (coeff @ proto.view(nm, -1)).sigmoid().view(-1, mh, mw)
    gain = boxes_xyxy.new_tensor([mw / imgsz, mh / imgsz, mw / imgsz, mh / imgsz])
    masks = crop_mask(masks, boxes_xyxy * gain)
    masks = F.interpolate(masks.unsqueeze(0), (imgsz, imgsz), mode="bilinear", align_corners=False)[0]
    return masks > 0.5


def scale_masks(
    masks: torch.Tensor,
    orig_wh: tuple[int, int],
    pad: tuple[float, float],
    ratio: float,
    imgsz: int,
) -> torch.Tensor:
    """Map letterboxed ``(k, imgsz, imgsz)`` masks onto the original image."""
    if masks.numel() == 0:
        w, h = orig_wh
        return masks.new_zeros((0, h, w), dtype=torch.bool)
    pad_x, pad_y = pad
    orig_w, orig_h = orig_wh
    nw = max(int(round(orig_w * ratio)), 1)
    nh = max(int(round(orig_h * ratio)), 1)
    x1 = int(round(pad_x))
    y1 = int(round(pad_y))
    x2 = min(x1 + nw, imgsz)
    y2 = min(y1 + nh, imgsz)
    cropped = masks[:, y1:y2, x1:x2].float()
    if cropped.shape[-2] == 0 or cropped.shape[-1] == 0:
        return masks.new_zeros((masks.shape[0], orig_h, orig_w), dtype=torch.bool)
    scaled = F.interpolate(cropped.unsqueeze(1), (orig_h, orig_w), mode="bilinear", align_corners=False)[:, 0]
    return scaled > 0.5
