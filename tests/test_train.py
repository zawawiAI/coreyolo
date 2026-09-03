from pathlib import Path

import torch

from coreyolo.data.dataset import YOLODetectionDataset, collate_fn
from coreyolo.data.dummy import write_dummy_dataset
from coreyolo.data.yaml import YOLODatasetYAML
from coreyolo.nn.model import build_model
from coreyolo.train.loss import DetectionLoss
from coreyolo.train.trainer import TrainConfig, train


def test_one_train_step(tmp_path: Path) -> None:
    yaml_path = write_dummy_dataset(tmp_path, n_train=4, n_val=2, size=64)
    spec = YOLODatasetYAML(yaml_path)
    ds = YOLODetectionDataset(spec, "train", imgsz=64, augment=False, mosaic=0.0)
    batch = collate_fn([ds[0], ds[1]])
    model = build_model(nc=spec.nc, scale="n")
    model.train()
    criterion = DetectionLoss(model)
    preds = model(batch["img"])
    loss, items = criterion(preds, batch)
    assert torch.isfinite(loss)
    assert "box" in items
    loss.backward()


def test_one_train_step_e2e(tmp_path: Path) -> None:
    yaml_path = write_dummy_dataset(tmp_path, n_train=4, n_val=2, size=64)
    spec = YOLODatasetYAML(yaml_path)
    ds = YOLODetectionDataset(spec, "train", imgsz=64, augment=False, mosaic=0.0)
    batch = collate_fn([ds[0], ds[1]])
    model = build_model(nc=spec.nc, scale="n", family="e2e")
    model.train()
    criterion = DetectionLoss(model)
    preds = model(batch["img"])
    loss, items = criterion(preds, batch)
    assert torch.isfinite(loss)
    assert "box" in items
    loss.backward()


def test_train_one_epoch(tmp_path: Path) -> None:
    yaml_path = write_dummy_dataset(tmp_path / "data", n_train=8, n_val=4, size=64)
    save_dir = train(
        TrainConfig(
            data=str(yaml_path),
            model="n",
            epochs=1,
            batch=2,
            imgsz=64,
            device="cpu",
            workers=0,
            project=str(tmp_path / "runs"),
            name="t",
            mosaic=0.0,
            patience=0,
        )
    )
    assert (save_dir / "weights" / "best.pt").is_file()
    assert (save_dir / "weights" / "last.pt").is_file()
