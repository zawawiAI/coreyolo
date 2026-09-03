import torch

from coreyolo.export.convert import detect_ultralytics_family, infer_v8_scale, remap_v8_state
from coreyolo.nn.model import build_model


def test_remap_v8_stem_and_dfl() -> None:
    src = {
        "model.0.conv.weight": torch.randn(16, 3, 3, 3),
        "model.22.dfl.conv.weight": torch.arange(16, dtype=torch.float32).view(1, 16, 1, 1),
        "model.2.cv1.conv.weight": torch.randn(32, 32, 1, 1),
        "model.22.cv3.0.2.weight": torch.randn(80, 16, 1, 1),
    }
    out = remap_v8_state(src)
    assert "stem.conv.weight" in out
    assert out["stem.conv.weight"].shape == (16, 3, 3, 3)
    assert out["head.dfl.proj"].shape == (16,)
    assert torch.equal(out["head.dfl.proj"], torch.arange(16, dtype=torch.float32))
    assert "stage2.1.cv1.conv.weight" in out
    assert "head.cv3.0.2.weight" in out
    # model.22 must not be classified as model.2
    assert "stage2.1.dfl.conv.weight" not in out


def test_detect_families() -> None:
    assert detect_ultralytics_family(["model.22.dfl.conv.weight"]) == "yolov8"
    assert detect_ultralytics_family(["model.23.dfl.conv.weight"]) == "yolo11"
    assert detect_ultralytics_family(["model.23.one2one_cv2.0.0.weight"]) == "yolo26"


def test_infer_v8_scale() -> None:
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(16, 3, 3, 3)}) == "n"
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(32, 3, 3, 3)}) == "s"


def test_v8_map_covers_coreyolo_n() -> None:
    dst = build_model(80, "n", act="silu").state_dict()
    fake = {("model.22." + k[len("head.") :] if k.startswith("head.") else k): v for k, v in dst.items()}
    # invert is hard; just ensure remap of a full v8-style copy of dst via reverse names
    src = {}
    inv = {
        "stem.": "model.0.",
        "stage2.0.": "model.1.",
        "stage2.1.": "model.2.",
        "stage3.0.": "model.3.",
        "stage3.1.": "model.4.",
        "stage4.0.": "model.5.",
        "stage4.1.": "model.6.",
        "stage5.0.": "model.7.",
        "stage5.1.": "model.8.",
        "stage5.2.": "model.9.",
        "n4.": "model.12.",
        "n3.": "model.15.",
        "d4.": "model.16.",
        "n4b.": "model.18.",
        "d5.": "model.19.",
        "n5b.": "model.21.",
        "head.": "model.22.",
    }
    for k, v in dst.items():
        uk = k
        for c, u in inv.items():
            if k.startswith(c):
                uk = u + k[len(c) :]
                break
        if uk.endswith("dfl.proj"):
            uk = uk.replace("dfl.proj", "dfl.conv.weight")
            v = v.view(1, -1, 1, 1)
        src[uk] = v
    mapped = remap_v8_state(src)
    model = build_model(80, "n", act="silu")
    model.load_state_dict(mapped, strict=True)
