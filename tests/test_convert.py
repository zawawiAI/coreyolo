import torch

from coreyolo.export.convert import (
    CONVERT_LICENSE_NOTICE,
    MTL_CONVERT_LICENSE_NOTICE,
    detect_pt_family,
    expand_grouped_to_dense,
    infer_gelan_scale,
    infer_mtl_gelan_scale,
    infer_v8_scale,
    remap_mtl_state,
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
    assert detect_pt_family(["model.22.dfl.conv.weight"]) == "yolov8"
    assert detect_pt_family(["model.9.cv5.conv.weight", "model.22.dfl.conv.weight"]) == "yolov9"
    assert detect_pt_family(["22.heads.0.anchor_conv.0.conv.weight", "0.conv.weight"]) == "yolov9-mtl"
    assert detect_pt_family(["model.23.dfl.conv.weight"]) == "yolo11"
    assert detect_pt_family(["model.23.one2one_cv2.0.0.weight"]) == "yolo26"


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


def _fake_sequential_from_coreyolo(scale: str, family: str = "dfl") -> dict[str, torch.Tensor]:
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
    mapped = remap_v8_state(_fake_sequential_from_coreyolo("n", family="dfl"))
    model = build_model(80, "n", act="silu", family="dfl")
    model.load_state_dict(mapped, strict=True)


def test_v8_map_covers_coreyolo_l() -> None:
    mapped = remap_v8_state(_fake_sequential_from_coreyolo("l", family="dfl"))
    model = build_model(80, "l", act="silu", family="dfl")
    model.load_state_dict(mapped, strict=True)


def test_gelan_map_covers_coreyolo_n() -> None:
    mapped = remap_v8_state(_fake_sequential_from_coreyolo("n", family="gelan"))
    model = build_model(80, "n", act="silu", family="gelan")
    model.load_state_dict(mapped, strict=True)
    fake = _fake_sequential_from_coreyolo("n", family="gelan")
    assert detect_pt_family(list(fake)) == "yolov9"
    assert infer_gelan_scale(fake) == "n"


def test_gelan_map_covers_coreyolo_s_and_m() -> None:
    for scale in ("s", "m"):
        fake = _fake_sequential_from_coreyolo(scale, family="gelan")
        mapped = remap_v8_state(fake)
        model = build_model(80, scale, act="silu", family="gelan")
        model.load_state_dict(mapped, strict=True)
        assert infer_gelan_scale(fake) == scale


def test_convert_license_notice_covers_agpl_saas_and_commercial() -> None:
    text = CONVERT_LICENSE_NOTICE.lower()
    assert "agpl-3.0" in text
    assert "saas" in text or "network" in text
    assert "commercial license" in text
    assert "does not relicense" in text
    assert "docs/licenses.md" in CONVERT_LICENSE_NOTICE


def test_mtl_convert_notice_keeps_mit_attribution() -> None:
    text = MTL_CONVERT_LICENSE_NOTICE.lower()
    assert "mit" in text
    assert "multimediatechlab" in text
    assert "kin-yiu" in text
    assert "agpl-3.0" in text
    assert "docs/licenses.md" in MTL_CONVERT_LICENSE_NOTICE


def test_expand_grouped_to_dense_block_diagonal() -> None:
    grouped = torch.arange(4, dtype=torch.float32).view(4, 1, 1, 1)
    dense = expand_grouped_to_dense(grouped, groups=4)
    assert dense.shape == (4, 4, 1, 1)
    for i in range(4):
        assert float(dense[i, i]) == float(grouped[i, 0])
        for j in range(4):
            if i != j:
                assert float(dense[i, j]) == 0.0


def test_remap_mtl_detect_and_aconv() -> None:
    src = {
        "0.conv.weight": torch.zeros(16, 3, 3, 3),
        "3.conv.conv.weight": torch.zeros(64, 32, 3, 3),
        "22.heads.0.anchor_conv.0.conv.weight": torch.zeros(64, 64, 3, 3),
        "22.heads.0.class_conv.2.weight": torch.zeros(80, 80, 1, 1),
        "22.heads.0.anc2vec.anc2vec.weight": torch.zeros(1, 16, 1, 1, 1),
        "23.conv1.conv.weight": torch.zeros(64, 128, 1, 1),
    }
    out = remap_mtl_state(src, "t")
    assert infer_mtl_gelan_scale(src) == "n"
    assert "stem.conv.weight" in out
    assert "stage3.0.cv1.conv.weight" in out
    assert "head.cv2.0.0.conv.weight" in out
    assert "head.cv3.0.2.weight" in out
    assert not any("anc2vec" in k for k in out)
    assert not any(k.startswith("head.") and "23" in k for k in out)


def test_convert_mtl_v9t_stamps_mit(tmp_path) -> None:
    from pathlib import Path

    from coreyolo.export.convert import convert_yolo_pt
    from coreyolo.utils import load_checkpoint

    src = Path("weights/mit-src/v9-t.pt")
    if not src.is_file():
        return
    dest = tmp_path / "coreyolo-n-coco.coreyolo"
    convert_yolo_pt(src, dest)
    ckpt = load_checkpoint(dest)
    assert ckpt["weights_license"] == "MIT"
    assert ckpt["source_vendor"] == "MultimediaTechLab/YOLO"
    assert "Kin-Yiu" in str(ckpt.get("source_copyright"))
    assert ckpt["family"] == "gelan"
    assert ckpt["scale"] == "n"


def test_convert_ultralytics_stamps_agpl(tmp_path) -> None:
    from pathlib import Path

    from coreyolo.export.convert import convert_yolo_pt
    from coreyolo.utils import load_checkpoint

    src = Path("weights/yolov9s.pt")
    if not src.is_file():
        return
    dest = tmp_path / "from-ultralytics.coreyolo"
    convert_yolo_pt(src, dest)
    ckpt = load_checkpoint(dest)
    assert ckpt["weights_license"] == "AGPL-3.0"
    assert ckpt["source_vendor"] == "ultralytics"
