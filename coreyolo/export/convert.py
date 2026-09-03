"""Convert Ultralytics COCO checkpoints into a CoreYOLO ``.pt``.

CoreYOLO-n is a YOLOv8-n graph (C2f + DFL ``reg_max=16``). That is a 1:1
tensor rename from ``yolov8n.pt``. YOLO26n (C3k2, no DFL, end-to-end head)
and YOLO11n (C3k2 + C2PSA) do not share that layout and cannot be copied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from coreyolo.data.coco import COCO_NAMES
from coreyolo.nn.model import build_model
from coreyolo.utils import save_checkpoint

# Ultralytics sequential index → CoreYOLO named module.
# Longer indices must be applied first so ``model.22`` is not eaten by ``model.2``.
V8_LAYER_MAP: dict[int, str] = {
    0: "stem.",
    1: "stage2.0.",
    2: "stage2.1.",
    3: "stage3.0.",
    4: "stage3.1.",
    5: "stage4.0.",
    6: "stage4.1.",
    7: "stage5.0.",
    8: "stage5.1.",
    9: "stage5.2.",
    12: "n4.",
    15: "n3.",
    16: "d4.",
    18: "n4b.",
    19: "d5.",
    21: "n5b.",
    22: "head.",
}


def detect_ultralytics_family(keys: list[str]) -> str:
    if any("one2one" in k for k in keys):
        return "yolo26"
    if any(k.startswith("model.23.dfl") for k in keys):
        return "yolo11"
    if any(k.startswith("model.22.dfl") for k in keys):
        return "yolov8"
    return "unknown"


def remap_v8_state(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename YOLOv8 ``model.N.*`` tensors to CoreYOLO names. DFL conv → ``proj``."""
    out: dict[str, torch.Tensor] = {}
    order = sorted(V8_LAYER_MAP, reverse=True)
    for key, tensor in src.items():
        dest = None
        for idx in order:
            prefix = f"model.{idx}."
            if key.startswith(prefix):
                dest = V8_LAYER_MAP[idx] + key[len(prefix) :]
                break
        if dest is None:
            continue
        if dest == "head.dfl.conv.weight":
            dest = "head.dfl.proj"
            tensor = tensor.detach().reshape(-1).float().contiguous()
        else:
            tensor = tensor.detach().float().contiguous()
        out[dest] = tensor
    return out


def _extract_state(ckpt: Any) -> tuple[dict[str, torch.Tensor], list[str], str]:
    if isinstance(ckpt, dict) and "model" in ckpt:
        model = ckpt["model"]
        version = str(ckpt.get("version", ""))
    else:
        model = ckpt
        version = ""
    if hasattr(model, "float"):
        try:
            model = model.float()
        except Exception:
            pass
    if hasattr(model, "state_dict"):
        src = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    elif isinstance(model, dict):
        src = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in model.items()}
    else:
        raise TypeError(f"Cannot read tensors from {type(model)}")
    names = list(COCO_NAMES)
    if hasattr(model, "names") and model.names:
        raw = model.names
        if isinstance(raw, dict):
            names = [str(raw[i]) for i in sorted(raw, key=lambda x: int(x))]
        else:
            names = [str(n) for n in raw]
    return src, names, version


def infer_v8_scale(src: dict[str, torch.Tensor]) -> str:
    stem = src.get("model.0.conv.weight")
    if stem is None:
        return "n"
    width = {16: "n", 32: "s", 48: "m", 64: "l", 80: "x"}
    return width.get(int(stem.shape[0]), "n")


def convert_ultralytics(
    weights: str | Path,
    out: str | Path | None = None,
    scale: str | None = None,
) -> Path:
    """Load an Ultralytics ``.pt`` and write a CoreYOLO checkpoint.

    Only YOLOv8 detect weights match CoreYOLO. YOLO26/YOLO11 raise ``ValueError``.
    """
    weights = Path(weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    src, names, version = _extract_state(ckpt)
    family = detect_ultralytics_family(list(src))
    if family != "yolov8":
        raise ValueError(
            f"{weights.name} looks like {family or 'an unknown'} checkpoint, not YOLOv8. "
            "CoreYOLO-n is a C2f + DFL (reg_max=16) graph, the same as YOLOv8n. "
            "YOLO26n uses C3k2, drops DFL, and adds an NMS-free one-to-one head "
            "(~2.5M params, 708 tensors) — those weights cannot be copied. "
            "YOLO11n is also C3k2 + C2PSA. Convert COCO init from YOLOv8n instead:\n"
            "  coreyolo convert --weights yolov8n.pt --out weights/coreyolo-n-coco.pt"
        )
    mapped = remap_v8_state(src)
    inferred = infer_v8_scale(src)
    if scale is None:
        scale = inferred
    elif scale != inferred:
        raise ValueError(
            f"--model {scale} does not match this checkpoint (stem width implies {inferred})"
        )
    task = "segment" if any(k.startswith("model.22.proto") or k.startswith("model.22.cv4") for k in src) else "detect"
    nm = 32
    if task == "segment":
        for key, tensor in src.items():
            if key.endswith("proto.cv3.conv.weight"):
                nm = int(tensor.shape[0])
                break
    model = build_model(nc=len(names), scale=scale, act="silu", task=task, nm=nm)
    dest_keys = set(model.state_dict())
    extra = sorted(set(mapped) - dest_keys)
    missing = sorted(dest_keys - set(mapped))
    if missing:
        raise ValueError(f"YOLOv8 remap is missing {len(missing)} CoreYOLO tensors, e.g. {missing[:8]}")
    for key in extra:
        mapped.pop(key)
    model.load_state_dict(mapped, strict=True)
    dest = Path(out) if out else Path("weights") / (
        "coreyolo-n-coco-seg.coreyolo" if task == "segment" else "coreyolo-n-coco.coreyolo"
    )
    save_checkpoint(
        dest,
        {
            "model": model.state_dict(),
            "nc": len(names),
            "names": names,
            "scale": scale,
            "act": "silu",
            "family": "dfl",
            "task": task,
            "nm": nm if task == "segment" else 0,
            "end2end": False,
            "reg_max": 16,
            "imgsz": 640,
            "source": str(weights),
            "source_family": "yolov8-seg" if task == "segment" else "yolov8",
            "ultralytics_version": version,
        },
    )
    print(
        f"converted {weights} → {dest}  ({len(mapped)} tensors, act=silu, family=dfl, "
        f"task={task}, scale={scale})"
    )
    return dest
