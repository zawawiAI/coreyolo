"""Core ML export + load roundtrip. Runs on macOS only (Core ML compile)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from coreyolo.nn.model import build_model, is_e2e_family
from coreyolo.utils import save_checkpoint

pytestmark = [
    pytest.mark.coreml,
    pytest.mark.skipif(sys.platform != "darwin", reason="Core ML compile requires macOS"),
]


def _tiny_ckpt(tmp_path: Path, family: str = "dfl", task: str = "detect", act: str = "relu") -> Path:
    nc = 2
    model = build_model(nc=nc, scale="n", act=act, family=family, task=task)
    path = tmp_path / f"tiny-{family}-{task}-{act}.pt"
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "nc": nc,
            "names": ["a", "b"],
            "scale": "n",
            "act": act,
            "family": family,
            "task": task,
            "imgsz": 64,
            "nm": 32,
            "end2end": is_e2e_family(family),
        },
    )
    return path


def _export(tmp_path: Path, family: str, task: str = "detect") -> tuple[Path, dict]:
    from coreyolo.export.coreml import export_coreml

    weights = _tiny_ckpt(tmp_path, family=family, task=task)
    out = tmp_path / f"tiny-{family}-{task}.mlpackage"
    export_coreml(weights, out=out, imgsz=64, fp16=False, image_input=True)
    assert (out / "Manifest.json").is_file()
    return out, {"family": family, "task": task}


def test_export_dfl_roundtrip(tmp_path: Path) -> None:
    from coreyolo.export.engine import CoreMLEngine

    pkg, _ = _export(tmp_path, "dfl")
    engine = CoreMLEngine(pkg, device="cpu")
    assert engine.imgsz == 64
    assert engine.nc == 2
    assert engine.task == "detect"
    assert engine.end2end is False
    canvas = Image.new("RGB", (64, 64), (0, 0, 0))
    pred = engine.predict_letterboxed(canvas)
    arr = np.asarray(pred)
    assert arr.ndim == 3
    assert arr.shape[0] == 1
    assert arr.shape[1] == 6  # xywh + 2 classes
    # 64/8² + 64/16² + 64/32² = 64+16+4
    assert arr.shape[2] == 84
    assert np.isfinite(arr).all()

    ckpt = tmp_path / "tiny-dfl-detect-relu.pt"
    model = build_model(nc=2, scale="n", act="relu", family="dfl")
    from coreyolo.utils import load_checkpoint

    loaded = load_checkpoint(ckpt)
    model.load_state_dict(loaded["model"], strict=False)
    model.eval()
    model.fuse()
    model.prepare_export(64)
    dummy = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        pt = model(dummy).numpy()
    # FP32 Core ML vs PyTorch; MIL lowering can shift logits. Shapes must match.
    assert pt.shape == arr.shape
    corr = np.corrcoef(pt.ravel(), arr.ravel())[0, 1]
    assert np.isfinite(corr)
    assert corr > 0.95 or float(np.max(np.abs(pt - arr))) < 0.25


def test_export_e2e_topk_layout(tmp_path: Path) -> None:
    from coreyolo.export.engine import CoreMLEngine

    pkg, _ = _export(tmp_path, "e2e")
    engine = CoreMLEngine(pkg, device="cpu")
    assert engine.end2end is True
    canvas = Image.new("RGB", (64, 64), (114, 114, 114))
    pred = engine.predict_letterboxed(canvas)
    arr = np.asarray(pred)
    assert arr.ndim == 3
    assert arr.shape[0] == 1
    assert arr.shape[2] == 6  # xyxy, conf, cls
    assert arr.shape[1] <= 300
    assert np.isfinite(arr).all()


def test_export_segment_three_outputs(tmp_path: Path) -> None:
    from coreyolo.export.engine import CoreMLEngine

    pkg, _ = _export(tmp_path, "dfl", task="segment")
    engine = CoreMLEngine(pkg, device="cpu")
    assert engine.task == "segment"
    canvas = Image.new("RGB", (64, 64), (20, 20, 20))
    packed = engine.predict_letterboxed(canvas)
    assert isinstance(packed, dict)
    assert "detections" in packed and "mask_coeff" in packed and "proto" in packed
    det = np.asarray(packed["detections"])
    proto = np.asarray(packed["proto"])
    assert det.shape[1] == 6
    assert proto.ndim == 4


def test_export_fp16_relu_records_ane_optimize(tmp_path: Path) -> None:
    from coreyolo.export.coreml import ane_preferred_act, export_coreml
    from coreyolo.export.engine import CoreMLEngine

    assert ane_preferred_act("relu")
    assert ane_preferred_act("star")
    assert not ane_preferred_act("silu")
    weights = _tiny_ckpt(tmp_path, "gelan")
    out = tmp_path / "tiny-gelan-fp16.mlpackage"
    export_coreml(weights, out=out, imgsz=64, fp16=True, image_input=True)
    engine = CoreMLEngine(out, device="cpu")
    meta = dict(engine.model.user_defined_metadata)
    assert meta.get("preferred_compute") == "ane"
    assert meta.get("optimize") in {"palettize8", "fp16"}
    pred = engine.predict_letterboxed(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert np.isfinite(np.asarray(pred)).all()


def test_export_fp16_silu_stays_dense_gpu(tmp_path: Path) -> None:
    from coreyolo.export.coreml import export_coreml
    from coreyolo.export.engine import CoreMLEngine

    weights = _tiny_ckpt(tmp_path, "gelan", act="silu")
    out = tmp_path / "tiny-gelan-silu-fp16.mlpackage"
    export_coreml(weights, out=out, imgsz=64, fp16=True, image_input=True)
    engine = CoreMLEngine(out, device="cpu")
    meta = dict(engine.model.user_defined_metadata)
    assert meta.get("preferred_compute") == "gpu"
    assert meta.get("optimize") == "fp16"
    pred = engine.predict_letterboxed(Image.new("RGB", (64, 64), (0, 0, 0)))
    assert np.isfinite(np.asarray(pred)).all()
