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
    from coreyolo.nn.model import is_e2e_family, is_gelan_family, normalize_family

    assert normalize_family(None) == "gelan"
    assert normalize_family("8") == "dfl"
    assert normalize_family("v8") == "dfl"
    assert normalize_family("dfl") == "dfl"
    assert normalize_family("9") == "gelan"
    assert normalize_family("v9") == "gelan"
    assert normalize_family("yolov9") == "gelan"
    assert normalize_family("26") == "e2e"
    assert normalize_family("v26") == "e2e"
    assert normalize_family("e2e") == "e2e"
    assert is_e2e_family("26") is True
    assert is_e2e_family("8") is False
    assert is_gelan_family("9") is True
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


def test_gelan_yolov9n_graph() -> None:
    model = build_model(nc=3, scale="n", family="gelan")
    info = model.info()
    assert info["family"] == "gelan"
    assert info["reg_max"] == 16
    assert info["end2end"] is False
    names = {type(m).__name__ for m in model.modules()}
    assert "ELAN1" in names
    assert "RepNCSPELAN4" in names
    assert "SPPELAN" in names
    assert "AConv" in names
    x = torch.zeros(2, 3, 64, 64)
    model.train()
    outs = model(x)
    assert len(outs) == 3
    assert outs[0].shape[-1] == 8
    model.eval()
    decoded = model(x)
    assert decoded.shape[0] == 2
    assert decoded.shape[1] == 4 + 3
    fused = model.fuse()
    fused.eval()
    out = fused(torch.zeros(1, 3, 64, 64))
    assert out.shape[0] == 1


def test_gelan_s_and_m_match_public_yolov9() -> None:
    """s is 2× n channels. m AConv[180] is make_divisible to 184, as in yolov9m.pt."""
    s = build_model(nc=80, scale="s", family="gelan")
    m = build_model(nc=80, scale="m", family="gelan")
    assert s.stem.conv.out_channels == 32
    assert m.stem.conv.out_channels == 32
    assert m.d4.cv1.conv.out_channels == 184
    s.eval()
    m.eval()
    x = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        assert s(x).shape == (1, 84, 84)
        assert m(x).shape == (1, 84, 84)


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


def test_nms_fp16_matches_float() -> None:
    pred = torch.zeros(1, 6, 4)
    pred[0, 0, 0] = 32
    pred[0, 1, 0] = 32
    pred[0, 2, 0] = 10
    pred[0, 3, 0] = 10
    pred[0, 4, 0] = 0.9
    extra = torch.randn(1, 32, 4)
    extra[0, :, 0] = 1.0
    dets, coeffs = non_max_suppression(pred.half(), conf_thres=0.25, extra=extra.half())
    assert dets[0].shape[0] == 1
    assert coeffs[0].shape[0] == 1


def test_nms_fp16_collapses_high_class_duplicates() -> None:
    """cls * 7680 overflows float16; laptop/tv (62/63) must still NMS."""
    pred = torch.zeros(1, 84, 3)
    for i, cx in enumerate((40.0, 41.0, 200.0)):
        pred[0, 0, i] = cx
        pred[0, 1, i] = 40.0
        pred[0, 2, i] = 20.0
        pred[0, 3, i] = 20.0
        pred[0, 4 + 63, i] = 0.9 if i < 2 else 0.8
    out = non_max_suppression(pred.half(), conf_thres=0.25, iou_thres=0.5)
    assert out[0].shape[0] == 2
    assert int(out[0][0, 5].item()) == 63


def test_gpu_device_aliases() -> None:
    from coreyolo.export.engine import resolve_compute_units
    from coreyolo.utils import select_device

    kind, unit = resolve_compute_units("gpu")
    assert kind == "gpu"
    assert "GPU" in unit.name
    kind, unit = resolve_compute_units("auto")
    assert kind == "auto"
    assert "NE" in unit.name or unit.name == "ALL"
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


def test_modern_activations_forward() -> None:
    from coreyolo.nn.modules import StarReLU, make_act, normalize_act

    assert normalize_act("GELU") == "gelu"
    assert normalize_act("starrelu") == "star"
    assert normalize_act("swish") == "silu"
    assert isinstance(make_act("star"), StarReLU)
    x = torch.linspace(-3, 3, 16)
    gelu = make_act("gelu")(x)
    silu = make_act("silu")(x)
    star = make_act("star")(x)
    assert gelu.shape == x.shape
    assert not torch.allclose(gelu, silu)
    assert float(star.min()) >= -1.0
    for act in ("gelu", "star", "hardswish"):
        model = build_model(nc=3, scale="n", act=act)
        assert model.act == act
        model.eval()
        out = model(torch.zeros(1, 3, 64, 64))
        assert out.shape[0] == 1


def test_unknown_activation_rejected() -> None:
    from coreyolo.nn.modules import normalize_act

    try:
        normalize_act("swiglu")
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_fuse_keeps_parameter_device() -> None:
    from coreyolo.utils import mps_usable

    if not mps_usable():
        return
    model = build_model(nc=3, scale="n", family="dfl").to("mps")
    model.eval()
    model.fuse()
    weight = next(model.parameters())
    assert weight.device.type == "mps"
    out = model(torch.zeros(1, 3, 64, 64, device="mps"))
    assert out.device.type == "mps"


def test_e2e_eval_skips_one2many_aux() -> None:
    model = build_model(nc=3, scale="n", family="e2e")
    model.eval()
    calls = {"aux": 0}
    orig = model.head.cv2[0].forward

    def counted(x):
        calls["aux"] += 1
        return orig(x)

    model.head.cv2[0].forward = counted  # type: ignore[method-assign]
    with torch.no_grad():
        out = model(torch.zeros(1, 3, 64, 64))
    assert calls["aux"] == 0
    assert out.shape[-1] == 6


def test_static_anchors_match_feature_grids() -> None:
    from coreyolo.nn.decode import make_anchors, make_static_anchors

    model = build_model(nc=3, scale="n", family="gelan")
    model.eval()
    dummy = torch.zeros(1, 3, 64, 64)
    with torch.no_grad():
        feats = list(model.forward_neck(dummy))
    dynamic, dstride = make_anchors(feats, model.head.stride, 0.5)
    static, sstride = make_static_anchors(64, [int(s) for s in model.head.stride.tolist()], 0.5)
    assert torch.allclose(dynamic, static)
    assert torch.allclose(dstride, sstride)
    model.prepare_export(64)
    assert model.head._export_hw == [(8, 8), (4, 4), (2, 2)]
    with torch.no_grad():
        decoded = model(dummy)
    assert decoded.shape == (1, 7, 84)


def test_export_trace_has_no_dynamic_grids() -> None:
    model = build_model(nc=3, scale="n", family="gelan")
    model.prepare_export(64)
    dummy = torch.zeros(1, 3, 64, 64)
    traced = torch.jit.trace(model, dummy, strict=False)
    graph = str(traced.graph)
    assert "meshgrid" not in graph
    assert "aten::arange" not in graph
