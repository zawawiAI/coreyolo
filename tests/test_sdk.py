from pathlib import Path

import numpy as np
from PIL import Image

from coreyolo import YOLO, Result
from coreyolo.data.dummy import write_dummy_dataset


def test_sdk_family_aliases() -> None:
    dfl = YOLO("n", nc=2, family="8", device="cpu", imgsz=64)
    assert dfl.family == "dfl"
    assert dfl.end2end is False
    e2e = YOLO("n", nc=2, family="26", device="cpu", imgsz=64)
    assert e2e.family == "e2e"
    assert e2e.end2end is True


def test_sdk_predict_scale_and_result(tmp_path: Path) -> None:
    model = YOLO("n", nc=2, names=["red_box", "blue_box"], device="cpu", imgsz=64)
    image = Image.new("RGB", (80, 60), (20, 20, 20))
    results = model.predict(image, conf=0.001)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, Result)
    assert r.boxes.ndim == 2 and r.boxes.shape[1] >= 6
    vis = r.plot()
    assert vis.size == (80, 60)
    saved = r.save(tmp_path / "out.jpg")
    assert saved.is_file()


def test_sdk_predict_numpy_and_dir(tmp_path: Path) -> None:
    yaml_path = write_dummy_dataset(tmp_path, n_train=1, n_val=1, size=64)
    model = YOLO("n", nc=2, device="cpu", imgsz=64)
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    results = model(arr, conf=0.9)
    assert len(results) == 1
    folder = tmp_path / "valid" / "images"
    more = model.predict(folder, save=True, save_dir=tmp_path / "pred", conf=0.9)
    assert more
    assert yaml_path.is_file()


def test_sdk_train_then_predict(tmp_path: Path) -> None:
    data = write_dummy_dataset(tmp_path / "data", n_train=8, n_val=4, size=64)
    model = YOLO("n", nc=2, device="cpu", imgsz=64)
    save_dir = model.train(
        data,
        epochs=1,
        batch=2,
        workers=0,
        mosaic=0.0,
        patience=0,
        project=str(tmp_path / "runs"),
        name="sdk",
        device="cpu",
        imgsz=64,
    )
    assert (save_dir / "weights" / "best.pt").is_file()
    assert model.weights is not None
    metrics = model.val(data, batch=2, workers=0, device="cpu", imgsz=64)
    assert "mAP50" in metrics
    img = tmp_path / "data" / "valid" / "images"
    results = model.predict(img, conf=0.001, device="cpu")
    assert results
    assert model.info()["task"] == "detect"


def test_sdk_native_coreyolo_roundtrip(tmp_path: Path) -> None:
    src = YOLO("n", nc=3, names=["a", "b", "c"], device="cpu", imgsz=64)
    path = src.save(tmp_path / "app.coreyolo")
    assert path.suffix == ".coreyolo"
    loaded = YOLO(path, device="cpu", imgsz=64)
    assert loaded.names == ["a", "b", "c"]
    assert loaded.info().get("format") == "coreyolo" or loaded.weights.suffix == ".coreyolo"
    ckpt = __import__("torch").load(path, map_location="cpu", weights_only=False)
    assert ckpt["format"] == "coreyolo"
    assert any(k.startswith("stem.") for k in ckpt["model"])


def test_rejects_ultralytics_pt(tmp_path: Path) -> None:
    import torch

    from coreyolo.utils import load_checkpoint

    fake = tmp_path / "yolov8n.pt"
    torch.save({"model": {"model.22.dfl.conv.weight": torch.zeros(1, 16, 1, 1)}}, fake)
    try:
        load_checkpoint(fake)
        raised = False
    except TypeError as exc:
        raised = True
        assert "Ultralytics" in str(exc)
    assert raised
    try:
        YOLO(fake, device="cpu")
        yolo_raised = False
    except TypeError:
        yolo_raised = True
    assert yolo_raised
