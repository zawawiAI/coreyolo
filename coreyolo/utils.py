"""Shared helpers: version, device selection, and checkpoint I/O."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch

# Let ops without MPS kernels fall back instead of aborting a GPU run.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

__version__ = "0.2.1"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CHECKPOINT_FORMAT = "coreyolo"
NATIVE_WEIGHT_EXTS = {".pt", ".coreyolo"}
AGPL_WEIGHT_LICENSE = "AGPL-3.0"
MIT_WEIGHT_LICENSE = "MIT"

# Printed when loading/exporting converted third-party YOLOv9 tensors.
AGPL_WEIGHTS_NOTICE = """\
license: these tensors inherit AGPL-3.0 from their source. They are NOT covered by
the MIT License that applies to CoreYOLO code. Converting the file does not change
its applicable terms. See weights/LICENSE_NOTICE.txt and docs/licenses.md.
"""

MTL_WEIGHTS_NOTICE = """\
license: these tensors inherit MIT from MultimediaTechLab/YOLO, copyright Kin-Yiu Wong
and Hao-Tang Tsui. Keep the license text and that copyright notice with any copy.
Converting the file does not change its applicable terms.
"""

_GPU_ALIASES = {
    "auto",
    "gpu",
    "mps",
    "ane",
    "ne",
    "all",
    "cpu_and_gpu",
    "cpu_and_ne",
}


def mps_usable() -> bool:
    """True when a real MPS allocation succeeds, not only when the backend exists."""
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_built() or not mps.is_available():
        return False
    try:
        x = torch.zeros(1, device="mps")
        x = x + 1
        if hasattr(torch, "mps") and hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()
        del x
        return True
    except Exception:
        return False


def select_device(device: str | None = None) -> torch.device:
    """Pick a GPU whenever one works: MPS, then CUDA, then CPU.

    ``gpu`` / ``auto`` / ``mps`` / Core ML labels (``all``, ``ane``) all mean
    "use the GPU" on the PyTorch path. ``cpu`` stays on CPU.
    """
    raw = (device or "gpu").strip().lower().replace("-", "_")
    if raw in {"cpu", "cpu_only"}:
        return torch.device("cpu")

    if raw in _GPU_ALIASES:
        if mps_usable():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        if raw == "mps":
            raise RuntimeError("MPS GPU is not available")
        print("GPU (MPS/CUDA) not available; falling back to CPU", file=sys.stderr)
        return torch.device("cpu")

    if raw == "cuda" or raw.startswith("cuda:"):
        if torch.cuda.is_available():
            return torch.device(raw if raw.startswith("cuda:") else "cuda")
        if mps_usable():
            print("CUDA not available; using MPS GPU", file=sys.stderr)
            return torch.device("mps")
        print("CUDA not available; falling back to CPU", file=sys.stderr)
        return torch.device("cpu")

    return torch.device(raw)


def place_module(module: torch.nn.Module, device: torch.device) -> tuple[torch.nn.Module, torch.device]:
    """Move ``module`` onto ``device``, falling back to CPU if the GPU rejects it."""
    try:
        return module.to(device), device
    except Exception as exc:
        if device.type == "cpu":
            raise
        print(f"failed to place model on {device} ({exc}); using CPU", file=sys.stderr)
        return module.to("cpu"), torch.device("cpu")


def increment_path(path: str | Path) -> Path:
    """Return ``path`` or ``path2``, ``path3``, ... if the directory exists."""
    path = Path(path)
    if not path.exists():
        return path
    stem, parent = path.name, path.parent
    for i in range(2, 10_000):
        candidate = parent / f"{stem}{i}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not allocate a unique path under {parent}")


def _checkpoint_state_keys(ckpt: Any) -> list[str]:
    if not isinstance(ckpt, dict):
        return []
    model = ckpt.get("model", ckpt)
    if hasattr(model, "state_dict") and not isinstance(model, dict):
        try:
            return list(model.state_dict())
        except Exception:
            return []
    if isinstance(model, dict):
        return [str(k) for k in model]
    return []


def is_coreyolo_checkpoint(ckpt: Any) -> bool:
    """True for a native CoreYOLO dict (not a sequential ``model.N.*`` pickle)."""
    if not isinstance(ckpt, dict):
        return False
    if ckpt.get("format") == CHECKPOINT_FORMAT:
        return True
    keys = _checkpoint_state_keys(ckpt)
    return any(k.startswith("stem.") or k.startswith("head.") or k.startswith("stage2.") for k in keys)


def is_sequential_yolo_checkpoint(ckpt: Any) -> bool:
    """True when tensors use upstream YOLOv9 names (Ultralytics or MultimediaTechLab)."""
    if is_coreyolo_checkpoint(ckpt):
        return False
    keys = _checkpoint_state_keys(ckpt)
    if any("one2one" in k for k in keys):
        return True
    if any(k.startswith("model.22.") or k.startswith("model.23.") or k.startswith("model.0.") for k in keys):
        return True
    if any(k.startswith("0.conv.") for k in keys) or any(".heads." in k and "anchor_conv" in k for k in keys):
        return True
    if isinstance(ckpt, dict) and "0.conv.weight" in ckpt:
        return True
    return False


def checkpoint_weight_license(ckpt: Any) -> str:
    """License on the numeric tensors, not on CoreYOLO source."""
    if not isinstance(ckpt, dict):
        return MIT_WEIGHT_LICENSE
    for key in ("weights_license", "source_license"):
        raw = ckpt.get(key)
        if raw:
            return str(raw)
    vendor = str(ckpt.get("source_vendor") or "")
    if "multimediatechlab" in vendor.lower():
        return MIT_WEIGHT_LICENSE
    family = str(ckpt.get("source_family") or "")
    if family.startswith("yolov8") or family.startswith("yolov9") or vendor.lower() in {"yolov9", "yolov8", "ultralytics"}:
        return AGPL_WEIGHT_LICENSE
    return MIT_WEIGHT_LICENSE


def is_converted_agpl_weights(ckpt: Any) -> bool:
    """True when this CoreYOLO file still carries third-party AGPL YOLOv9 tensors."""
    return str(checkpoint_weight_license(ckpt)).upper().startswith("AGPL")


def origin_metadata(ckpt: Any) -> dict[str, Any]:
    """Fields that must follow converted tensors through save, train, and export."""
    if not isinstance(ckpt, dict):
        return {}
    vendor = str(ckpt.get("source_vendor") or "")
    copyright = ckpt.get("source_copyright")
    if not is_converted_agpl_weights(ckpt) and not vendor and not copyright:
        return {}
    license_id = checkpoint_weight_license(ckpt)
    out: dict[str, Any] = {
        "weights_license": license_id,
        "source_license": str(ckpt.get("source_license") or license_id),
        "source_family": ckpt.get("source_family") or "yolov9",
    }
    for key in ("source", "source_version", "source_vendor", "source_copyright"):
        if ckpt.get(key) is not None:
            out[key] = ckpt[key]
    return out


def warn_converted_weights(ckpt: Any, *, stream: Any = None) -> None:
    """Remind that a CoreYOLO file can still be third-party YOLOv9 numbers."""
    if is_converted_agpl_weights(ckpt):
        print(AGPL_WEIGHTS_NOTICE, end="", file=stream or sys.stderr)
        return
    if isinstance(ckpt, dict) and ckpt.get("source_vendor"):
        print(MTL_WEIGHTS_NOTICE, end="", file=stream or sys.stderr)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Write a native CoreYOLO checkpoint (``.pt`` or ``.coreyolo``)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamped = dict(payload)
    stamped.setdefault("format", CHECKPOINT_FORMAT)
    torch.save(stamped, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a CoreYOLO checkpoint. Sequential ``yolov9t.pt`` pickles are rejected."""
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if is_sequential_yolo_checkpoint(ckpt):
        raise TypeError(
            f"{path.name} is an upstream YOLOv9 pickle, not CoreYOLO.\n"
            "Do not load it in app code. Convert once (remap names only), then load the CoreYOLO file:\n"
            "  coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo\n"
            "  Detector('weights/coreyolo-n-coco.coreyolo')\n"
            "MultimediaTechLab v9-*.pt stays MIT. Ultralytics yolov9*.pt stays AGPL-3.0. "
            "Converting the file does not change its applicable terms. See docs/licenses.md."
        )
    if isinstance(ckpt, dict) and not is_coreyolo_checkpoint(ckpt) and "model" in ckpt:
        keys = _checkpoint_state_keys(ckpt)
        if keys and not any(k.startswith("stem.") for k in keys):
            raise TypeError(
                f"{path.name} is not a CoreYOLO checkpoint. Expected stem.*/head.* tensors "
                f"or format={CHECKPOINT_FORMAT!r}."
            )
    warn_converted_weights(ckpt)
    return ckpt
