"""CoreYOLO detector: DFL (C2f), GELAN (YOLOv9), or E2E (C3k2 + C2PSA)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from coreyolo.nn.head import Detect, Segment
from coreyolo.nn.modules import (
    AConv,
    ADown,
    C2f,
    C2PSA,
    C3k2,
    Conv,
    ELAN1,
    RepNCSPELAN4,
    SPPF,
    SPPELAN,
    fuse_model,
    normalize_act,
)

# depth, width, max_channels
SCALES_DFL: dict[str, tuple[float, float, int]] = {
    "n": (0.33, 0.25, 1024),
    "s": (0.33, 0.50, 1024),
    "m": (0.67, 0.75, 768),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.25, 512),
}

# E2E compound scaling (depth is 0.50 on n/s/m, not 0.33)
SCALES_E2E: dict[str, tuple[float, float, int]] = {
    "n": (0.50, 0.25, 1024),
    "s": (0.50, 0.50, 1024),
    "m": (0.50, 1.00, 512),
    "l": (1.00, 1.00, 512),
    "x": (1.00, 1.50, 512),
}

SCALES = SCALES_DFL
FAMILIES = ("dfl", "gelan", "e2e")
FAMILY_ALIASES = {
    "dfl": "dfl",
    "c2f": "dfl",
    "8": "dfl",
    "gelan": "gelan",
    "9": "gelan",
    "yolov9": "gelan",
    "e2e": "e2e",
    "c3k2": "e2e",
    "26": "e2e",
}

# Compact GELAN (Ultralytics YOLOv9 t/s/m/c). Scale n matches yolov9t — there is
# no official yolov9n.yaml; that is the nano-class graph people mean by "v9n".
# Tuple layout: (kind, c2, c3, c4, n) for ELAN blocks; (kind, c2) for downs.
GELAN_SCALES: dict[str, dict[str, Any]] = {
    "n": {
        "stem": 16,
        "s2": 32,
        "s2_block": ("elan1", 32, 32, 16, 1),
        "s3": ("aconv", 64),
        "s3_block": ("rep", 64, 64, 32, 3),
        "s4": ("aconv", 96),
        "s4_block": ("rep", 96, 96, 48, 3),
        "s5": ("aconv", 128),
        "s5_block": ("rep", 128, 128, 64, 3),
        "spp": (128, 64),
        "n4": (96, 96, 48, 3),
        "n3": (64, 64, 32, 3),
        "d4": ("aconv", 48),
        "n4b": (96, 96, 48, 3),
        "d5": ("aconv", 64),
        "n5b": (128, 128, 64, 3),
        "head": (64, 96, 128),
    },
    "s": {
        "stem": 32,
        "s2": 64,
        "s2_block": ("elan1", 64, 64, 32, 1),
        "s3": ("aconv", 128),
        "s3_block": ("rep", 128, 128, 64, 3),
        "s4": ("aconv", 192),
        "s4_block": ("rep", 192, 192, 96, 3),
        "s5": ("aconv", 256),
        "s5_block": ("rep", 256, 256, 128, 3),
        "spp": (256, 128),
        "n4": (192, 192, 96, 3),
        "n3": (128, 128, 64, 3),
        "d4": ("aconv", 96),
        "n4b": (192, 192, 96, 3),
        "d5": ("aconv", 128),
        "n5b": (256, 256, 128, 3),
        "head": (128, 192, 256),
    },
    "m": {
        "stem": 32,
        "s2": 64,
        "s2_block": ("rep", 128, 128, 64, 1),
        "s3": ("aconv", 240),
        "s3_block": ("rep", 240, 240, 120, 1),
        "s4": ("aconv", 360),
        "s4_block": ("rep", 360, 360, 180, 1),
        "s5": ("aconv", 480),
        "s5_block": ("rep", 480, 480, 240, 1),
        "spp": (480, 240),
        "n4": (360, 360, 180, 1),
        "n3": (240, 240, 120, 1),
        "d4": ("aconv", 180),
        "n4b": (360, 360, 180, 1),
        "d5": ("aconv", 240),
        "n5b": (480, 480, 240, 1),
        "head": (240, 360, 480),
    },
    "l": {
        "stem": 64,
        "s2": 128,
        "s2_block": ("rep", 256, 128, 64, 1),
        "s3": ("adown", 256),
        "s3_block": ("rep", 512, 256, 128, 1),
        "s4": ("adown", 512),
        "s4_block": ("rep", 512, 512, 256, 1),
        "s5": ("adown", 512),
        "s5_block": ("rep", 512, 512, 256, 1),
        "spp": (512, 256),
        "n4": (512, 512, 256, 1),
        "n3": (256, 256, 128, 1),
        "d4": ("adown", 256),
        "n4b": (512, 512, 256, 1),
        "d5": ("adown", 512),
        "n5b": (512, 512, 256, 1),
        "head": (256, 512, 512),
    },
}
GELAN_SCALES["x"] = GELAN_SCALES["l"]


def normalize_family(family: str | None = None) -> str:
    """Map a family id to ``dfl``, ``gelan``, or ``e2e``. ``8`` / ``9`` / ``26`` remain aliases."""
    raw = str(family or "gelan").strip().lower()
    if raw.startswith("v") and raw[1:].isdigit():
        raw = raw[1:]
    key = FAMILY_ALIASES.get(raw, raw)
    if key not in FAMILIES:
        raise ValueError(
            f"Unknown family {family!r}. Choose from {list(FAMILIES)} (aliases: 8→dfl, 9→gelan, 26→e2e)"
        )
    return key


def is_e2e_family(family: str | None) -> bool:
    return normalize_family(family) == "e2e"


def is_gelan_family(family: str | None) -> bool:
    return normalize_family(family) == "gelan"


def _elan_block(kind: str, c1: int, c2: int, c3: int, c4: int, n: int, act: str) -> nn.Module:
    if kind == "elan1":
        return ELAN1(c1, c2, c3, c4, act=act)
    return RepNCSPELAN4(c1, c2, c3, c4, n, act=act)


def _down(kind: str, c1: int, c2: int, act: str) -> nn.Module:
    if kind == "adown":
        return ADown(c1, c2, act=act)
    return AConv(c1, c2, act=act)


def _make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(value + divisor / 2) // divisor * divisor)


def _depth(n: int, gd: float) -> int:
    return max(1, round(n * gd)) if n > 1 else n


def _width(c: int, gw: float, max_channels: int) -> int:
    return _make_divisible(min(c, max_channels) * gw)


class CoreYOLO(nn.Module):
    """Anchor-free one-stage detector or instance segmenter.

    ``family='gelan'`` is the default YOLOv9 GELAN graph (scale n matches
    Ultralytics ``yolov9t``). ``family='dfl'`` is the older C2f + DFL graph.
    ``family='e2e'`` is C3k2 + C2PSA with an NMS-free top-300 head.
    ``task='segment'`` adds a proto mask branch. Checkpoint ids ``8`` / ``9`` /
    ``26`` still load.
    """

    def __init__(
        self,
        nc: int = 80,
        scale: str = "n",
        act: str = "relu",
        reg_max: int | None = None,
        family: str = "gelan",
        task: str = "detect",
        nm: int = 32,
    ) -> None:
        super().__init__()
        family = normalize_family(family)
        if family not in FAMILIES:
            raise ValueError(f"Unknown family '{family}'. Choose from {list(FAMILIES)}")
        task = str(task).lower()
        if task not in {"detect", "segment"}:
            raise ValueError(f"Unknown task '{task}'. Choose detect or segment")
        if family == "gelan":
            if scale not in GELAN_SCALES:
                raise ValueError(f"Unknown scale '{scale}'. Choose from {list(GELAN_SCALES)}")
            spec = GELAN_SCALES[scale]
            gd, gw, max_ch = 1.0, 1.0, 1024
        else:
            table = SCALES_E2E if family == "e2e" else SCALES_DFL
            if scale not in table:
                raise ValueError(f"Unknown scale '{scale}'. Choose from {list(table)}")
            spec = None
            gd, gw, max_ch = table[scale]
        self.nc = nc
        self.scale = scale
        self.act = normalize_act(act)
        self.family = family
        self.task = task
        self.nm = nm
        if family == "gelan":
            self.npr = int(spec["head"][0]) if task == "segment" else 0
        else:
            self.npr = _width(256, gw, max_ch) if task == "segment" else 0
        self.reg_max = 1 if family == "e2e" else (16 if reg_max is None else reg_max)
        self.end2end = family == "e2e"

        if family == "e2e":
            self._build_e2e(nc, gd, gw, max_ch, act)
        elif family == "gelan":
            self._build_gelan(nc, spec, act)
        else:
            self._build_dfl(nc, gd, gw, max_ch, act)

        self._init_strides()
        self.head.bias_init()

    def _build_dfl(self, nc: int, gd: float, gw: float, max_ch: int, act: str) -> None:
        c2 = _width(64, gw, max_ch)
        c3 = _width(128, gw, max_ch)
        c4 = _width(256, gw, max_ch)
        c5 = _width(512, gw, max_ch)
        c6 = _width(1024, gw, max_ch)
        n3, n4, n5, n6 = _depth(3, gd), _depth(6, gd), _depth(6, gd), _depth(3, gd)
        nd = _depth(3, gd)

        self.stem = Conv(3, c2, 3, 2, act=act)
        self.stage2 = nn.Sequential(Conv(c2, c3, 3, 2, act=act), C2f(c3, c3, n3, True, act=act))
        self.stage3 = nn.Sequential(Conv(c3, c4, 3, 2, act=act), C2f(c4, c4, n4, True, act=act))
        self.stage4 = nn.Sequential(Conv(c4, c5, 3, 2, act=act), C2f(c5, c5, n5, True, act=act))
        self.stage5 = nn.Sequential(
            Conv(c5, c6, 3, 2, act=act),
            C2f(c6, c6, n6, True, act=act),
            SPPF(c6, c6, 5, act=act),
        )
        self.psa = nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.n4 = C2f(c6 + c5, c5, nd, False, act=act)
        self.n3 = C2f(c5 + c4, c4, nd, False, act=act)
        self.d4 = Conv(c4, c4, 3, 2, act=act)
        self.n4b = C2f(c4 + c5, c5, nd, False, act=act)
        self.d5 = Conv(c5, c5, 3, 2, act=act)
        self.n5b = C2f(c5 + c6, c6, nd, False, act=act)
        self.head = self._make_head(nc, (c4, c5, c6), act, end2end=False, legacy=True)

    def _build_gelan(self, nc: int, spec: dict[str, Any], act: str) -> None:
        s2k, s2o, s2c3, s2c4, s2n = spec["s2_block"]
        s3k, s3c = spec["s3"]
        s3bk, s3o, s3c3, s3c4, s3n = spec["s3_block"]
        s4k, s4c = spec["s4"]
        s4bk, s4o, s4c3, s4c4, s4n = spec["s4_block"]
        s5k, s5c = spec["s5"]
        s5bk, s5o, s5c3, s5c4, s5n = spec["s5_block"]
        spp_c2, spp_c3 = spec["spp"]
        n4o, n4c3, n4c4, n4n = spec["n4"]
        n3o, n3c3, n3c4, n3n = spec["n3"]
        d4k, d4c = spec["d4"]
        n4bo, n4bc3, n4bc4, n4bn = spec["n4b"]
        d5k, d5c = spec["d5"]
        n5bo, n5bc3, n5bc4, n5bn = spec["n5b"]

        self.stem = Conv(3, spec["stem"], 3, 2, act=act)
        self.stage2 = nn.Sequential(
            Conv(spec["stem"], spec["s2"], 3, 2, act=act),
            _elan_block(s2k, spec["s2"], s2o, s2c3, s2c4, s2n, act),
        )
        self.stage3 = nn.Sequential(
            _down(s3k, s2o, s3c, act),
            _elan_block(s3bk, s3c, s3o, s3c3, s3c4, s3n, act),
        )
        self.stage4 = nn.Sequential(
            _down(s4k, s3o, s4c, act),
            _elan_block(s4bk, s4c, s4o, s4c3, s4c4, s4n, act),
        )
        self.stage5 = nn.Sequential(
            _down(s5k, s4o, s5c, act),
            _elan_block(s5bk, s5c, s5o, s5c3, s5c4, s5n, act),
            SPPELAN(s5o, spp_c2, spp_c3, act=act),
        )
        self.psa = nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.n4 = _elan_block("rep", spp_c2 + s4o, n4o, n4c3, n4c4, n4n, act)
        self.n3 = _elan_block("rep", n4o + s3o, n3o, n3c3, n3c4, n3n, act)
        self.d4 = _down(d4k, n3o, d4c, act)
        self.n4b = _elan_block("rep", d4c + n4o, n4bo, n4bc3, n4bc4, n4bn, act)
        self.d5 = _down(d5k, n4bo, d5c, act)
        self.n5b = _elan_block("rep", d5c + spp_c2, n5bo, n5bc3, n5bc4, n5bn, act)
        self.head = self._make_head(nc, tuple(spec["head"]), act, end2end=False, legacy=True)

    def _build_e2e(self, nc: int, gd: float, gw: float, max_ch: int, act: str) -> None:
        c64 = _width(64, gw, max_ch)
        c128 = _width(128, gw, max_ch)
        c256 = _width(256, gw, max_ch)
        c512 = _width(512, gw, max_ch)
        c1024 = _width(1024, gw, max_ch)
        n2 = _depth(2, gd)

        self.stem = Conv(3, c64, 3, 2, act=act)
        self.stage2 = nn.Sequential(
            Conv(c64, c128, 3, 2, act=act),
            C3k2(c128, c256, n2, c3k=False, e=0.25, act=act),
        )
        self.stage3 = nn.Sequential(
            Conv(c256, c256, 3, 2, act=act),
            C3k2(c256, c512, n2, c3k=False, e=0.25, act=act),
        )
        self.stage4 = nn.Sequential(
            Conv(c512, c512, 3, 2, act=act),
            C3k2(c512, c512, n2, c3k=True, act=act),
        )
        self.stage5 = nn.Sequential(
            Conv(c512, c1024, 3, 2, act=act),
            C3k2(c1024, c1024, n2, c3k=True, act=act),
            SPPF(c1024, c1024, 5, act=act, shortcut=True),
        )
        self.psa = C2PSA(c1024, c1024, n2, act=act)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.n4 = C3k2(c1024 + c512, c512, n2, c3k=True, act=act)
        self.n3 = C3k2(c512 + c512, c256, n2, c3k=True, act=act)
        self.d4 = Conv(c256, c256, 3, 2, act=act)
        self.n4b = C3k2(c256 + c512, c512, n2, c3k=True, act=act)
        self.d5 = Conv(c512, c512, 3, 2, act=act)
        self.n5b = C3k2(c512 + c1024, c1024, _depth(1, gd), c3k=True, e=0.5, attn=True, act=act)
        self.head = self._make_head(nc, (c256, c512, c1024), act, end2end=True, legacy=False)

    def _make_head(
        self, nc: int, ch: tuple[int, ...], act: str, end2end: bool, legacy: bool
    ) -> Detect:
        kwargs = dict(nc=nc, ch=ch, reg_max=self.reg_max, act=act, end2end=end2end, legacy=legacy)
        if self.task == "segment":
            return Segment(nm=self.nm, npr=self.npr, **kwargs)
        return Detect(**kwargs)

    def _init_strides(self, size: int = 256) -> None:
        was_training = self.training
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, size, size)
            p3, p4, p5 = self.forward_neck(dummy)
            for i, feat in enumerate((p3, p4, p5)):
                self.head.stride[i] = size / feat.shape[-2]
        self.train(was_training)

    def forward_neck(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        p3 = self.stage3(x)
        p4 = self.stage4(p3)
        p5 = self.psa(self.stage5(p4))

        u4 = self.n4(torch.cat((self.up(p5), p4), 1))
        u3 = self.n3(torch.cat((self.up(u4), p3), 1))
        d4 = self.n4b(torch.cat((self.d4(u3), u4), 1))
        d5 = self.n5b(torch.cat((self.d5(d4), p5), 1))
        return u3, d4, d5

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor] | dict[str, list[torch.Tensor]]:
        return self.head(list(self.forward_neck(x)))

    def fuse(self) -> "CoreYOLO":
        fuse_model(self)
        return self

    def info(self) -> dict[str, Any]:
        n_params = sum(p.numel() for p in self.parameters())
        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "family": self.family,
            "scale": self.scale,
            "nc": self.nc,
            "act": self.act,
            "reg_max": self.reg_max,
            "end2end": self.end2end,
            "task": self.task,
            "nm": self.nm if self.task == "segment" else 0,
            "npr": self.npr if self.task == "segment" else 0,
            "stride": self.head.stride.tolist(),
            "params": n_params,
            "trainable": n_train,
        }


def build_model(
    nc: int,
    scale: str = "n",
    act: str = "relu",
    weights: str | None = None,
    family: str = "gelan",
    task: str = "detect",
    nm: int = 32,
) -> CoreYOLO:
    if weights:
        from coreyolo.utils import load_checkpoint

        ckpt = load_checkpoint(weights, map_location="cpu")
        if isinstance(ckpt, dict):
            family = str(ckpt.get("family", family))
            scale = ckpt.get("scale", scale)
            act = ckpt.get("act", act)
            nc = int(ckpt.get("nc", nc))
            task = str(ckpt.get("task", task))
            nm = int(ckpt.get("nm", nm))
    model = CoreYOLO(nc=nc, scale=scale, act=act, family=family, task=task, nm=nm)
    if weights:
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state, strict=False)
    return model
