from PIL import Image

from coreyolo.data.augment import letterbox
from coreyolo.infer.nms import scale_boxes
import torch


def test_letterbox_matches_ios_contract() -> None:
    """Swift Letterbox.swift must use the same scale / pad as this function."""
    image = Image.new("RGB", (80, 60), (10, 20, 30))
    canvas, scale, pad = letterbox(image, 64)
    assert canvas.size == (64, 64)
    # min(64/80, 64/60) = 0.8 → 64×48, pad_x=0, pad_y=8
    assert abs(scale - 0.8) < 1e-9
    assert abs(pad[0] - 0.0) < 1e-9
    assert abs(pad[1] - 8.0) < 1e-9
    px = canvas.getpixel((0, 0))
    assert px == (114, 114, 114)
    # pasted region starts at round(pad)
    inner = canvas.getpixel((0, 8))
    assert inner == (10, 20, 30)


def test_scale_boxes_inverts_letterbox() -> None:
    boxes = torch.tensor([[0.0, 8.0, 64.0, 56.0]])
    out = scale_boxes(boxes, (64, 64), (80, 60), pad=(0.0, 8.0), ratio=0.8)
    assert abs(float(out[0, 0]) - 0.0) < 1e-4
    assert abs(float(out[0, 1]) - 0.0) < 1e-4
    assert abs(float(out[0, 2]) - 80.0) < 1e-4
    assert abs(float(out[0, 3]) - 60.0) < 1e-4
