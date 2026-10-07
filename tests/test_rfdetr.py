import torch

from coreyolo.nn.model import build_model, is_rfdetr_family, normalize_family
from coreyolo.nn.rfdetr import check_rfdetr_imgsz, rfdetr_native_imgsz
from coreyolo.train.set_loss import SetCriterion, linear_sum_assignment
from coreyolo.utils import (
    APACHE_WEIGHT_LICENSE,
    PML_WEIGHT_LICENSE,
    checkpoint_weight_license,
    origin_metadata,
    refuse_upstream_rfdetr,
    save_checkpoint,
)
from coreyolo.zoo import assert_zoo_license_labels


def test_family_alias_and_native_sizes() -> None:
    assert normalize_family("rf-detr") == "rfdetr"
    assert normalize_family("RF_DETR") == "rfdetr"
    assert is_rfdetr_family("rfdetr")
    assert rfdetr_native_imgsz("n") == 384
    assert rfdetr_native_imgsz("s") == 512
    assert rfdetr_native_imgsz("m") == 576
    assert rfdetr_native_imgsz("l") == 704
    for scale, imgsz in (("n", 384), ("s", 512), ("m", 576), ("l", 704)):
        check_rfdetr_imgsz(scale, imgsz)


def test_pml_scales_refused() -> None:
    try:
        build_model(nc=2, scale="x", family="rfdetr", act="gelu")
        raised = False
    except ValueError as exc:
        raised = True
        assert "PML" in str(exc) or "Platform Model License" in str(exc)
    assert raised


def test_forward_train_and_eval() -> None:
    model = build_model(nc=3, scale="n", family="rfdetr", act="gelu")
    assert model.family == "rfdetr"
    assert model.end2end is True
    assert model.info()["queries"] == 100
    x = torch.zeros(2, 3, 64, 64)
    model.train()
    outs = model(x)
    assert isinstance(outs, list)
    assert len(outs) == 2
    logits, boxes = outs[-1]
    assert logits.shape == (2, 100, 3)
    assert boxes.shape == (2, 100, 4)
    model.eval()
    decoded = model(x)
    assert decoded.shape == (2, 100, 6)
    assert torch.isfinite(decoded).all()


def test_bad_imgsz_names_nearest() -> None:
    try:
        check_rfdetr_imgsz("s", 96)
        raised = False
    except ValueError as exc:
        raised = True
        text = str(exc)
        assert "64" in text and "128" in text
    assert raised


def test_hungarian_known_optimum() -> None:
    cost = torch.tensor([[4.0, 1.0, 3.0], [2.0, 0.0, 5.0], [3.0, 2.0, 2.0]])
    rows, cols = linear_sum_assignment(cost)
    paired = sorted(zip(rows, cols))
    assert paired == [(0, 1), (1, 0), (2, 2)]
    total = sum(cost[r, c].item() for r, c in paired)
    assert total == 5


def test_set_loss_backward() -> None:
    model = build_model(nc=2, scale="n", family="rfdetr", act="gelu")
    model.train()
    images = torch.rand(2, 3, 64, 64)
    labels = torch.tensor(
        [
            [0, 0, 20, 20, 10, 12],
            [1, 1, 40, 30, 8, 8],
        ],
        dtype=torch.float32,
    )
    loss, items = SetCriterion()(model(images), {"img": images, "labels": labels})
    assert torch.isfinite(loss)
    assert items["box"] >= 0
    loss.backward()


def test_upstream_apache_pickle_refused(tmp_path) -> None:
    path = tmp_path / "rf-detr-nano.pth"
    torch.save({"model": {"backbone.0.encoder.weight": torch.zeros(1)}}, path)
    try:
        refuse_upstream_rfdetr(path, torch.load(path, weights_only=False))
        raised = False
    except TypeError as exc:
        raised = True
        assert "Apache-2.0" in str(exc)
    assert raised


def test_upstream_pml_refused(tmp_path) -> None:
    path = tmp_path / "rf-detr-xlarge.pth"
    torch.save({"model": {"weight": torch.zeros(1)}}, path)
    try:
        refuse_upstream_rfdetr(path, {"model": {"weight": torch.zeros(1)}})
        raised = False
    except TypeError as exc:
        raised = True
        assert "PML" in str(exc) or "Platform Model License" in str(exc)
    assert raised


def test_scratch_checkpoint_stays_mit(tmp_path) -> None:
    model = build_model(nc=2, scale="n", family="rfdetr", act="gelu")
    path = tmp_path / "coreyolo-rfdetr-n.coreyolo"
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "nc": 2,
            "scale": "n",
            "act": "gelu",
            "family": "rfdetr",
            "weights_license": "MIT",
        },
    )
    from coreyolo.utils import load_checkpoint

    ckpt = load_checkpoint(path)
    assert ckpt["family"] == "rfdetr"
    assert checkpoint_weight_license(ckpt) == "MIT"
    loaded = build_model(nc=2, scale="n", family="rfdetr", weights=str(path))
    assert loaded.num_queries == 100


def test_apache_stamp_survives_origin() -> None:
    ckpt = {
        "weights_license": APACHE_WEIGHT_LICENSE,
        "source_vendor": "Roboflow",
        "family": "rfdetr",
        "scale": "s",
    }
    assert checkpoint_weight_license(ckpt) == APACHE_WEIGHT_LICENSE
    origin = origin_metadata(ckpt)
    assert origin["weights_license"] == APACHE_WEIGHT_LICENSE
    assert origin["source_family"] == "rfdetr"
    xlarge = {"source_vendor": "Roboflow", "scale": "xlarge"}
    assert checkpoint_weight_license(xlarge) == PML_WEIGHT_LICENSE


def test_zoo_rejects_pml_and_mit_roboflow(tmp_path) -> None:
    dest = tmp_path / "manifest.json"
    dest.write_text(
        '{"models":[{"id":"pml","how":"convert","weights_origin":"rf-detr-xlarge",'
        '"weights_license":"Apache-2.0"},'
        '{"id":"rehost","how":"convert","weights_origin":"Roboflow rf-detr-nano",'
        '"weights_license":"MIT"}]}'
    )
    try:
        assert_zoo_license_labels(dest)
        raised = False
    except ValueError as exc:
        raised = True
        text = str(exc)
        assert "PML" in text or "XLarge" in text
        assert "Apache-2.0" in text
    assert raised
