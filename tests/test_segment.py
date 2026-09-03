import torch

from coreyolo.data.dataset import YOLODetectionDataset, collate_fn
from coreyolo.data.dummy import write_dummy_dataset
from coreyolo.data.yaml import YOLODatasetYAML
from coreyolo.nn.model import build_model
from coreyolo.train.loss import SegmentationLoss
from coreyolo.train.trainer import TrainConfig, train


def test_segment_forward_shapes() -> None:
    model = build_model(nc=3, scale="n", task="segment")
    assert model.task == "segment"
    assert any(type(m).__name__ == "Proto" for m in model.modules())
    x = torch.zeros(2, 3, 64, 64)
    model.train()
    det, mc, proto = model(x)
    assert len(det) == 3
    assert mc.shape[0] == 2
    assert mc.shape[1] == 32
    assert proto.shape[:2] == (2, 32)
    model.eval()
    decoded, mc_e, proto_e = model(x)
    assert decoded.shape[1] == 4 + 3
    assert mc_e.shape[1] == 32
    assert proto_e.shape[:2] == (2, 32)


def test_segment_e2e_eval_topk() -> None:
    model = build_model(nc=3, scale="n", family="e2e", task="segment")
    model.eval()
    decoded, mc, proto = model(torch.zeros(1, 3, 64, 64))
    assert decoded.shape[-1] == 6
    assert mc.shape[0] == 1 and mc.shape[-1] == 32
    assert proto.shape[1] == 32


def test_one_segment_train_step(tmp_path) -> None:
    yaml_path = write_dummy_dataset(tmp_path, n_train=4, n_val=2, size=64, segment=True)
    spec = YOLODatasetYAML(yaml_path)
    ds = YOLODetectionDataset(spec, "train", imgsz=64, augment=False, mosaic=0.0, task="segment")
    batch = collate_fn([ds[0], ds[1]])
    assert "masks" in batch
    assert batch["masks"].ndim == 3
    model = build_model(nc=spec.nc, scale="n", task="segment")
    model.train()
    criterion = SegmentationLoss(model)
    preds = model(batch["img"])
    loss, items = criterion(preds, batch)
    assert torch.isfinite(loss)
    assert "mask" in items
    loss.backward()


def test_segment_train_one_epoch(tmp_path) -> None:
    yaml_path = write_dummy_dataset(tmp_path / "data", n_train=8, n_val=4, size=64, segment=True)
    save_dir = train(
        TrainConfig(
            data=str(yaml_path),
            model="n",
            task="segment",
            epochs=1,
            batch=2,
            imgsz=64,
            device="cpu",
            workers=0,
            project=str(tmp_path / "runs"),
            name="seg",
            mosaic=0.0,
            patience=0,
        )
    )
    assert (save_dir / "weights" / "best.pt").is_file()
