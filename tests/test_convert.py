import torch

from coreyolo.export.convert import (
    CONVERT_LICENSE_NOTICE,
    detect_ultralytics_family,
    infer_gelan_scale,
    infer_v8_scale,
    remap_v8_state,
)
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
    assert detect_ultralytics_family(["model.9.cv5.conv.weight", "model.22.dfl.conv.weight"]) == "yolov9"
    assert detect_ultralytics_family(["model.23.dfl.conv.weight"]) == "yolo11"
    assert detect_ultralytics_family(["model.23.one2one_cv2.0.0.weight"]) == "yolo26"


def test_infer_v8_scale() -> None:
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(16, 3, 3, 3)}) == "n"
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(32, 3, 3, 3)}) == "s"
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(48, 3, 3, 3)}) == "m"
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(64, 3, 3, 3)}) == "l"
    assert infer_v8_scale({"model.0.conv.weight": torch.zeros(80, 3, 3, 3)}) == "x"


def test_infer_gelan_scale() -> None:
    assert infer_gelan_scale({"model.0.conv.weight": torch.zeros(16, 3, 3, 3)}) == "n"
    assert infer_gelan_scale(
        {"model.0.conv.weight": torch.zeros(32, 3, 3, 3), "model.3.cv1.conv.weight": torch.zeros(128, 32, 3, 3)}
    ) == "s"
    assert infer_gelan_scale(
        {"model.0.conv.weight": torch.zeros(32, 3, 3, 3), "model.3.cv1.conv.weight": torch.zeros(240, 128, 3, 3)}
    ) == "m"
    assert infer_gelan_scale({"model.0.conv.weight": torch.zeros(64, 3, 3, 3)}) == "l"


def _fake_ultralytics_from_coreyolo(scale: str, family: str = "dfl") -> dict[str, torch.Tensor]:
    dst = build_model(80, scale, act="silu", family=family).state_dict()
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
    src: dict[str, torch.Tensor] = {}
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
    return src


def test_v8_map_covers_coreyolo_n() -> None:
    mapped = remap_v8_state(_fake_ultralytics_from_coreyolo("n", family="dfl"))
    model = build_model(80, "n", act="silu", family="dfl")
    model.load_state_dict(mapped, strict=True)


def test_v8_map_covers_coreyolo_l() -> None:
    mapped = remap_v8_state(_fake_ultralytics_from_coreyolo("l", family="dfl"))
    model = build_model(80, "l", act="silu", family="dfl")
    model.load_state_dict(mapped, strict=True)


def test_gelan_map_covers_coreyolo_n() -> None:
    mapped = remap_v8_state(_fake_ultralytics_from_coreyolo("n", family="gelan"))
    model = build_model(80, "n", act="silu", family="gelan")
    model.load_state_dict(mapped, strict=True)
    fake = _fake_ultralytics_from_coreyolo("n", family="gelan")
    assert detect_ultralytics_family(list(fake)) == "yolov9"
    assert infer_gelan_scale(fake) == "n"


def test_convert_license_notice_covers_agpl_saas_and_commercial() -> None:
    text = CONVERT_LICENSE_NOTICE.lower()
    assert "agpl-3.0" in text
    assert "saas" in text or "network" in text
    assert "commercial license" in text
    assert "does not relicense" in text
    assert "docs/licenses.md" in CONVERT_LICENSE_NOTICE
