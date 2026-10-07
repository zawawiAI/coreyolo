"""Core ML RF-DETR: windowed ViT encoder, dense query decoder, no NMS.

The export contract follows LibreYOLO's Core ML path for this family: one
fixed RGB image (``scale=1/255``), ImageNet mean/std inside the graph, set
prediction with no NMS, detection only, sizes n / s / m / l. LibreYOLO leaves
precision at FP32 and ``compute_units=all``. CoreYOLO pins FP16 and
``CPU_AND_NE`` so the Neural Engine is the ship path. Palettes stay off.

This file does not vendor LibreYOLO or ``roboflow/rf-detr`` and it does not
load those checkpoints. Nano–Large weights on that graph stay Apache-2.0.
XLarge and 2XLarge stay Platform Model License 1.0 and are refused.

Core ML choices, fixed at trace time:

* ImageNet mean/std is the first op, so export keeps the usual ``scale=1/255``
  image input.
* Positions are a buffer resized with bilinear interpolate. Forward does not
  call ``meshgrid`` or ``arange``.
* Window attention is reshape and permute. Global layers add a fixed register
  token count.
* Cross-attention is dense multi-head attention over projector tokens.
  Multi-scale deformable sampling (``grid_sample``) is omitted so the MIL
  program stays matmul and softmax, which ``CPU_AND_NE`` can schedule.
* Encoder FFN uses tanh-GELU. Decoder FFN uses ReLU.
* Eval returns ``(B, Q, 6)`` xyxy + conf + cls. There is no NMS op.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from coreyolo.nn.modules import make_act, normalize_act

# patch × windows. Input size must be a multiple of this.
RFDETR_SCALES: dict[str, dict[str, Any]] = {
    "n": {
        "imgsz": 384,
        "patch": 16,
        "windows": 2,
        "depth": 6,
        "dim": 192,
        "heads": 3,
        "dec_layers": 2,
        "dec_dim": 128,
        "dec_heads": 4,
        "queries": 100,
        "registers": 4,
        "out_indexes": (1, 3, 5),
    },
    "s": {
        "imgsz": 512,
        "patch": 16,
        "windows": 4,
        "depth": 8,
        "dim": 256,
        "heads": 4,
        "dec_layers": 3,
        "dec_dim": 192,
        "dec_heads": 6,
        "queries": 200,
        "registers": 4,
        "out_indexes": (2, 5, 7),
    },
    "m": {
        "imgsz": 576,
        "patch": 16,
        "windows": 4,
        "depth": 10,
        "dim": 320,
        "heads": 5,
        "dec_layers": 3,
        "dec_dim": 256,
        "dec_heads": 8,
        "queries": 300,
        "registers": 4,
        "out_indexes": (3, 6, 9),
    },
    "l": {
        "imgsz": 704,
        "patch": 16,
        "windows": 4,
        "depth": 12,
        "dim": 384,
        "heads": 6,
        "dec_layers": 4,
        "dec_dim": 256,
        "dec_heads": 8,
        "queries": 300,
        "registers": 4,
        "out_indexes": (3, 7, 11),
    },
}

_PML_SCALES = {"x", "xl", "xlarge", "2xl", "2xlarge", "xx"}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def rfdetr_native_imgsz(scale: str) -> int:
    spec = _spec(scale)
    return int(spec["imgsz"])


def rfdetr_size_unit(scale: str) -> int:
    spec = _spec(scale)
    return int(spec["patch"]) * int(spec["windows"])


def check_rfdetr_imgsz(scale: str, imgsz: int) -> None:
    """Raise when ``imgsz`` is not a whole patch/window grid."""
    spec = _spec(scale)
    imgsz = int(imgsz)
    unit = int(spec["patch"]) * int(spec["windows"])
    if imgsz > 0 and imgsz % unit == 0:
        return
    lower = imgsz - (imgsz % unit) if imgsz > 0 else 0
    upper = lower + unit
    native = int(spec["imgsz"])
    raise ValueError(
        f"rfdetr scale {scale} needs imgsz divisible by {unit} "
        f"(patch {spec['patch']} × {spec['windows']} windows). "
        f"Nearest: {lower} or {upper}. Native size is {native}."
    )


def _spec(scale: str) -> dict[str, Any]:
    key = str(scale).strip().lower()
    if key in _PML_SCALES:
        raise ValueError(
            f"Scale {scale!r} is outside RF-DETR n/s/m/l. "
            "Upstream XLarge and 2XLarge checkpoints are Platform Model License 1.0. "
            "CoreYOLO does not load them. Choose n, s, m, or l."
        )
    if key not in RFDETR_SCALES:
        raise ValueError(f"Unknown rfdetr scale {scale!r}. Choose from {list(RFDETR_SCALES)}")
    return RFDETR_SCALES[key]


def _sincos_grid(height: int, width: int, dim: int) -> torch.Tensor:
    """``(1, dim, H, W)`` sine-cosine grid. Built once; forward only interpolates."""
    if dim % 4 != 0:
        raise ValueError(f"position dim {dim} must be divisible by 4")
    y = torch.arange(height, dtype=torch.float32)
    x = torch.arange(width, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    omega = torch.arange(dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4, 1)))
    x_embed = xx.reshape(-1, 1) * omega.reshape(1, -1)
    y_embed = yy.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat((x_embed.sin(), x_embed.cos(), y_embed.sin(), y_embed.cos()), dim=1)
    return pos.T.reshape(1, dim, height, width)


class _HeadMeta:
    """Fields the trainer reads off ``model.head``."""

    def __init__(self) -> None:
        self.nm = 0
        self.end2end = True
        self.reg_max = 1


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, n, c = query.shape
        m = memory.shape[1]
        q = self.q(query).reshape(b, n, self.heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(memory).reshape(b, m, 2, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, act: str, *, windowed: bool, registers: int) -> None:
        super().__init__()
        self.windowed = windowed
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.act = make_act(act)
        self.registers = None
        if not windowed and registers > 0:
            self.registers = nn.Parameter(torch.zeros(1, registers, dim))
            nn.init.normal_(self.registers, std=0.02)

    def forward(self, x: torch.Tensor, height: int, width: int, window: int) -> torch.Tensor:
        h = self.norm1(x)
        if self.windowed:
            h = self._window(h, height, width, window)
        else:
            h = self._global(h)
        x = x + h
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x

    def _global(self, x: torch.Tensor) -> torch.Tensor:
        if self.registers is None:
            return self.attn(x)
        regs = self.registers.expand(x.shape[0], -1, -1)
        tokens = torch.cat((regs, x), dim=1)
        tokens = self.attn(tokens)
        return tokens[:, regs.shape[1] :]

    def _window(self, x: torch.Tensor, height: int, width: int, window: int) -> torch.Tensor:
        b, _, c = x.shape
        x = x.view(b, height, width, c)
        x = x.view(b, height // window, window, width // window, window, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        nwin = x.shape[1] * x.shape[2]
        x = x.reshape(b * nwin, window * window, c)
        x = self.attn(x)
        x = x.view(b, height // window, width // window, window, window, c)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, height * width, c)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.self_attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.cross_attn = CrossAttention(dim, heads)
        self.norm3 = nn.LayerNorm(dim, eps=1e-6)
        hidden = dim * 4
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        query = query + self.self_attn(self.norm1(query))
        query = query + self.cross_attn(self.norm2(query), memory)
        query = query + self.fc2(F.relu(self.fc1(self.norm3(query))))
        return query


class RFDETR(nn.Module):
    """Set-prediction detector. Scales ``n`` / ``s`` / ``m`` / ``l`` only."""

    def __init__(self, nc: int = 80, scale: str = "n", act: str = "gelu") -> None:
        super().__init__()
        spec = _spec(scale)
        act = normalize_act(act)
        self.nc = int(nc)
        self.scale = str(scale).strip().lower()
        self.act = act
        self.family = "rfdetr"
        self.task = "detect"
        self.end2end = True
        self.reg_max = 1
        self.nm = 0
        self.npr = 0
        self.head = _HeadMeta()
        self.imgsz = int(spec["imgsz"])
        self.patch = int(spec["patch"])
        self.windows = int(spec["windows"])
        self.num_queries = int(spec["queries"])
        dim = int(spec["dim"])
        dec_dim = int(spec["dec_dim"])
        depth = int(spec["depth"])
        self.out_indexes = tuple(int(i) for i in spec["out_indexes"])

        self.patch_embed = nn.Conv2d(3, dim, kernel_size=self.patch, stride=self.patch)
        grid = self.imgsz // self.patch
        self.register_buffer("pos", _sincos_grid(grid, grid, dim), persistent=False)
        self.register_buffer("mem_pos", _sincos_grid(grid, grid, dec_dim), persistent=False)
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("pixel_mean", mean, persistent=False)
        self.register_buffer("pixel_std", std, persistent=False)

        registers = int(spec["registers"])
        blocks = []
        for i in range(depth):
            windowed = (i % 2 == 0) and i != depth - 1
            blocks.append(
                EncoderBlock(dim, int(spec["heads"]), act, windowed=windowed, registers=registers)
            )
        self.blocks = nn.ModuleList(blocks)
        self.projs = nn.ModuleList(nn.Conv2d(dim, dec_dim, kernel_size=1) for _ in self.out_indexes)
        self.down = nn.Conv2d(dec_dim, dec_dim, kernel_size=3, stride=2, padding=1)
        self.query = nn.Parameter(torch.zeros(1, self.num_queries, dec_dim))
        nn.init.normal_(self.query, std=0.02)
        self.decoder = nn.ModuleList(
            DecoderLayer(dec_dim, int(spec["dec_heads"])) for _ in range(int(spec["dec_layers"]))
        )
        self._export_imgsz: int | None = None
        self.cls = nn.Linear(dec_dim, self.nc)
        self.bbox = nn.Sequential(
            nn.Linear(dec_dim, dec_dim),
            nn.ReLU(),
            nn.Linear(dec_dim, dec_dim),
            nn.ReLU(),
            nn.Linear(dec_dim, 4),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        prior = 0.01
        nn.init.constant_(self.cls.bias, -math.log((1 - prior) / prior))

    def prepare_export(self, imgsz: int) -> "RFDETR":
        check_rfdetr_imgsz(self.scale, imgsz)
        self._export_imgsz = int(imgsz)
        self.eval()
        return self

    def fuse(self) -> "RFDETR":
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
            "end2end": True,
            "task": self.task,
            "nm": 0,
            "npr": 0,
            "imgsz": self.imgsz,
            "queries": self.num_queries,
            "patch": self.patch,
            "windows": self.windows,
            "stride": [float(self.patch)],
            "params": n_params,
            "trainable": n_train,
        }

    def _window(self, height: int) -> int:
        return height // self.windows

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        x = self.patch_embed(x)
        if self._export_imgsz and not self.training:
            height = width = self._export_imgsz // self.patch
        else:
            height = int(x.shape[-2])
            width = int(x.shape[-1])
            if height % self.windows != 0 or width % self.windows != 0:
                check_rfdetr_imgsz(self.scale, height * self.patch)
        pos = F.interpolate(self.pos, size=(height, width), mode="bilinear", align_corners=False)
        tokens = (x + pos).flatten(2).transpose(1, 2)
        window = self._window(height)
        selected: list[torch.Tensor] = []
        pick = 0
        deepest: torch.Tensor | None = None
        for i, block in enumerate(self.blocks):
            tokens = block(tokens, height, width, window)
            if i in self.out_indexes:
                feat = tokens.transpose(1, 2).reshape(x.shape[0], -1, height, width)
                mapped = self.projs[pick](feat)
                mem_pos = F.interpolate(self.mem_pos, size=(height, width), mode="bilinear", align_corners=False)
                selected.append((mapped + mem_pos).flatten(2).transpose(1, 2))
                deepest = mapped
                pick += 1
        if deepest is None:
            raise RuntimeError("rfdetr encoder produced no projector maps")
        low = self.down(deepest)
        low_pos = F.interpolate(self.mem_pos, size=low.shape[-2:], mode="bilinear", align_corners=False)
        selected.append((low + low_pos).flatten(2).transpose(1, 2))
        return torch.cat(selected, dim=1)

    def _decode(self, memory: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
        query = self.query.expand(memory.shape[0], -1, -1)
        outputs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer in self.decoder:
            query = layer(query, memory)
            if self.training or layer is self.decoder[-1]:
                outputs.append((self.cls(query), self.bbox(query).sigmoid()))
        return outputs

    def forward(self, x: torch.Tensor) -> torch.Tensor | list[tuple[torch.Tensor, torch.Tensor]]:
        if self._export_imgsz and not self.training:
            side = self._export_imgsz
        else:
            side = int(x.shape[-1])
            check_rfdetr_imgsz(self.scale, side)
            if side != int(x.shape[-2]):
                raise ValueError(f"rfdetr expects a square input, got {tuple(x.shape)}")
        outputs = self._decode(self._encode(x))
        if self.training:
            return outputs
        logits, boxes = outputs[-1]
        scores = logits.sigmoid()
        conf, cls = scores.max(dim=-1)
        xyxy = _cxcywh_to_xyxy(boxes) * float(side)
        packed = torch.cat((xyxy, conf.unsqueeze(-1), cls.unsqueeze(-1).to(dtype=xyxy.dtype)), dim=-1)
        return packed


def _cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), dim=-1)
