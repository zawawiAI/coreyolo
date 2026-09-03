import torch

from coreyolo.nn.model import build_model
from coreyolo.infer.nms import non_max_suppression


def test_forward_shapes_train_and_eval() -> None:
    model = build_model(nc=3, scale="n")
    x = torch.zeros(2, 3, 64, 64)
    model.train()
    outs = model(x)
    assert len(outs) == 3
    assert outs[0].shape[0] == 2
    # P3 is 64/8 = 8
    assert outs[0].shape[-1] == 8
    model.eval()
    decoded = model(x)
    assert decoded.shape[0] == 2
    assert decoded.shape[1] == 4 + 3  # xywh + classes


def test_family_aliases() -> None:
    from coreyolo.nn.model import is_e2e_family, normalize_family

    assert normalize_family(None) == "dfl"
    assert normalize_family("8") == "dfl"
    assert normalize_family("v8") == "dfl"
    assert normalize_family("dfl") == "dfl"
    assert normalize_family("26") == "e2e"
    assert normalize_family("v26") == "e2e"
    assert normalize_family("e2e") == "e2e"
    assert is_e2e_family("26") is True
    assert is_e2e_family("8") is False
    dfl = build_model(nc=3, scale="n", family="8")
    assert dfl.family == "dfl"
    assert dfl.end2end is False


def test_e2e_c3k2_c2psa() -> None:
    model = build_model(nc=3, scale="n", family="e2e")
    info = model.info()
    assert info["family"] == "e2e"
    assert info["reg_max"] == 1
    assert info["end2end"] is True
    assert any(type(m).__name__ == "C3k2" for m in model.modules())
    assert any(type(m).__name__ == "C2PSA" for m in model.modules())
    x = torch.zeros(2, 3, 64, 64)
    model.train()
    outs = model(x)
    assert set(outs) == {"one2many", "one2one"}
    assert len(outs["one2one"]) == 3
    assert outs["one2one"][0].shape[1] == 4 + 3
    model.eval()
    decoded = model(x)
    assert decoded.shape[0] == 2
    assert decoded.shape[-1] == 6
    assert decoded.shape[1] <= 300


def test_nms_empty_and_peaked() -> None:
    pred = torch.zeros(1, 6, 4)
    out = non_max_suppression(pred, conf_thres=0.5)
    assert out[0].shape[0] == 0
    pred[0, 0, 0] = 32  # cx
    pred[0, 1, 0] = 32
    pred[0, 2, 0] = 10
    pred[0, 3, 0] = 10
    pred[0, 4, 0] = 0.9
    out = non_max_suppression(pred, conf_thres=0.25)
    assert out[0].shape[0] == 1


def test_gpu_device_aliases() -> None:
    from coreyolo.export.engine import resolve_compute_units
    from coreyolo.utils import select_device

    kind, unit = resolve_compute_units("gpu")
    assert kind == "gpu"
    assert "GPU" in unit.name
    kind, unit = resolve_compute_units("all")
    assert kind == "all"
    assert unit.name == "ALL"
    cpu = select_device("cpu")
    assert cpu.type == "cpu"
    gpu = select_device("gpu")
    assert gpu.type in {"mps", "cuda", "cpu"}
    ane = select_device("ane")
    assert ane.type in {"mps", "cuda", "cpu"}
    if torch.backends.mps.is_available():
        from coreyolo.utils import mps_usable

        if mps_usable():
            assert select_device("gpu").type == "mps"
