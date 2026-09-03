"""CoreML-friendly building blocks for a YOLO-style detector.

Activations default to ReLU so fused Conv-BN-ReLU graphs map cleanly onto the
Apple Neural Engine. SiLU remains available for training accuracy experiments.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def make_act(name: str = "relu") -> nn.Module:
    name = name.lower()
    if name in {"relu", "relu6"}:
        return nn.ReLU(inplace=True) if name == "relu" else nn.ReLU6(inplace=True)
    if name in {"silu", "swish"}:
        return nn.SiLU(inplace=True)
    if name in {"hswish", "hardswish"}:
        return nn.Hardswish(inplace=True)
    if name in {"identity", "none"}:
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class Conv(nn.Module):
    """Conv2d + BatchNorm + activation. BN can be fused for Core ML export."""

    def __init__(
        self,
        c1: int,
        c2: int,
        k: int = 3,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
        act: str = "relu",
    ) -> None:
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = make_act(act)
        self.fused = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def fuse(self) -> "Conv":
        if self.fused:
            return self
        conv, bn = self.conv, self.bn
        w = conv.weight
        gamma = bn.weight / torch.sqrt(bn.running_var + bn.eps)
        fused_w = w * gamma.reshape(-1, 1, 1, 1)
        fused_b = bn.bias - bn.running_mean * gamma
        fused = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.stride,
            conv.padding,
            groups=conv.groups,
            bias=True,
        )
        fused.weight.data.copy_(fused_w)
        fused.bias.data.copy_(fused_b)
        self.conv = fused
        self.bn = nn.Identity()
        self.fused = True
        return self

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class Bottleneck(nn.Module):
    """Residual conv pair used inside CSP stages."""

    def __init__(
        self,
        c: int,
        shortcut: bool = True,
        act: str = "relu",
        c2: int | None = None,
        k: tuple[int, int] = (3, 3),
        g: int = 1,
        e: float = 1.0,
    ) -> None:
        super().__init__()
        c2 = c if c2 is None else c2
        hidden = int(c2 * e)
        self.cv1 = Conv(c, hidden, k[0], 1, act=act)
        self.cv2 = Conv(hidden, c2, k[1], 1, g=g, act=act)
        self.add = shortcut and c == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C2f(nn.Module):
    """Cross-stage partial block with bottleneck reuse (YOLO-family C2f)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        act: str = "relu",
    ) -> None:
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1, act=act)
        self.m = nn.ModuleList(Bottleneck(self.c, shortcut, act=act) for _ in range(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    """Spatial pyramid pooling, fast variant (three 5x5 max-pools)."""

    def __init__(self, c1: int, c2: int, k: int = 5, act: str = "relu", shortcut: bool = False) -> None:
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1, act=act)
        self.cv2 = Conv(c_ * 4, c2, 1, 1, act=act)
        self.pool = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        y1 = self.pool(y)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        out = self.cv2(torch.cat((y, y1, y2, y3), 1))
        return out + x if self.add else out


class C3(nn.Module):
    """CSP bottleneck with a bypass conv (YOLOv5-style C3)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        act: str = "relu",
        k: int = 3,
    ) -> None:
        super().__init__()
        hidden = int(c2 * e)
        self.cv1 = Conv(c1, hidden, 1, 1, act=act)
        self.cv2 = Conv(c1, hidden, 1, 1, act=act)
        self.cv3 = Conv(2 * hidden, c2, 1, 1, act=act)
        self.m = nn.Sequential(
            *(Bottleneck(hidden, True, act, c2=hidden, k=(k, k), e=1.0) for _ in range(n))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3 whose inner bottlenecks use a configurable kernel (default 3×3)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = True,
        e: float = 0.5,
        act: str = "relu",
        k: int = 3,
    ) -> None:
        super().__init__(c1, c2, n, shortcut, e, act, k)


class Attention(nn.Module):
    """Multi-head attention with a depthwise positional conv (PSA)."""

    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5) -> None:
        super().__init__()
        self.num_heads = max(num_heads, 1)
        self.head_dim = dim // self.num_heads
        self.key_dim = max(int(self.head_dim * attn_ratio), 1)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * self.num_heads
        self.qkv = Conv(dim, dim + nh_kd * 2, 1, 1, act="none")
        self.proj = Conv(dim, dim, 1, 1, act="none")
        self.pe = Conv(dim, dim, 3, 1, g=dim, act="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Do not do `n = h * w` then `.view(..., n)`. Core ML's PyTorch frontend
        # inserts `int()` on that 1-element shape tensor and raises
        # "only 0-dimensional arrays can be converted to Python scalars".
        qkv = self.qkv(x).flatten(2)
        qkv = qkv.unflatten(1, (self.num_heads, self.key_dim * 2 + self.head_dim))
        q, k, v = qkv.split((self.key_dim, self.key_dim, self.head_dim), dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        mixed = (v @ attn.transpose(-2, -1)).flatten(1, 2)
        out = mixed.reshape_as(x) + self.pe(v.flatten(1, 2).reshape_as(x))
        return self.proj(out)


class PSABlock(nn.Module):
    """Position-sensitive attention plus a residual FFN."""

    def __init__(self, c: int, attn_ratio: float = 0.5, act: str = "relu") -> None:
        super().__init__()
        heads = max(c // 64, 1)
        self.attn = Attention(c, num_heads=heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1, 1, act=act), Conv(c * 2, c, 1, 1, act="none"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x)
        return x + self.ffn(x)


class C2PSA(nn.Module):
    """CSP wrapper around stacked PSABlock modules (YOLO11 / YOLO26)."""

    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5, act: str = "relu") -> None:
        super().__init__()
        if c1 != c2:
            raise ValueError("C2PSA requires c1 == c2")
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1, act=act)
        self.cv2 = Conv(2 * self.c, c1, 1, 1, act=act)
        self.m = nn.Sequential(*(PSABlock(self.c, act=act) for _ in range(n)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class C3k2(C2f):
    """C2f whose inner units are Bottleneck, C3k, or Bottleneck+PSA (YOLO11 / 26)."""

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        c3k: bool = False,
        e: float = 0.5,
        attn: bool = False,
        shortcut: bool = True,
        act: str = "relu",
    ) -> None:
        super().__init__(c1, c2, n, shortcut, e, act)
        if attn:
            self.m = nn.ModuleList(
                nn.Sequential(Bottleneck(self.c, shortcut, act), PSABlock(self.c, act=act)) for _ in range(n)
            )
        elif c3k:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, e=0.5, act=act) for _ in range(n))
        else:
            self.m = nn.ModuleList(Bottleneck(self.c, shortcut, act) for _ in range(n))


class Proto(nn.Module):
    """Mask prototype branch: upsample P3 2× and emit ``nm`` prototype maps."""

    def __init__(self, c1: int, c_: int = 256, c2: int = 32, act: str = "relu") -> None:
        super().__init__()
        self.cv1 = Conv(c1, c_, 3, act=act)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, 3, act=act)
        self.cv3 = Conv(c_, c2, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class DFL(nn.Module):
    """Integral projection for Distribution Focal Loss (Generalized Focal Loss)."""

    def __init__(self, reg_max: int = 16) -> None:
        super().__init__()
        self.reg_max = reg_max
        self.register_buffer("proj", torch.arange(reg_max, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4 * reg_max, A) → expected bin index per side (B, 4, A)
        b, _, a = x.shape
        x = x.view(b, 4, self.reg_max, a).softmax(2)
        return (x * self.proj.view(1, 1, -1, 1)).sum(2)


def fuse_model(model: nn.Module) -> nn.Module:
    """Fuse every Conv+BN pair so Core ML sees a single convolution."""
    for module in model.modules():
        if isinstance(module, Conv):
            module.fuse()
            module.forward = module.forward_fuse  # type: ignore[method-assign]
    return model
