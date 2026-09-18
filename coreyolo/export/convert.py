"""Convert a compact YOLOv9 COCO checkpoint into a CoreYOLO file.

Two public GELAN sources exist:

* MultimediaTechLab/YOLO ``v9-t.pt`` / ``v9-s.pt`` / ``v9-m.pt`` / ``v9-c.pt``
  (MIT, copyright Kin-Yiu Wong and Hao-Tang Tsui). This is the LibreYOLO
  path: remap names, keep MIT, keep the copyright notice.
* Ultralytics ``yolov9t.pt`` / ``s`` / ``m`` / ``c`` (AGPL-3.0). Remap names
  only — that is not a relicensing.

YOLO26n (C3k2, no DFL, end-to-end head) and YOLO11n (C3k2 + C2PSA) cannot
be copied. See ``docs/licenses.md``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import torch

from coreyolo.data.coco import COCO_NAMES
from coreyolo.nn.model import build_model
from coreyolo.utils import (
    AGPL_WEIGHT_LICENSE,
    MIT_WEIGHT_LICENSE,
    save_checkpoint,
)

# Printed when converting an Ultralytics / AGPL pickle. Not a relicensing.
CONVERT_LICENSE_NOTICE = """\
license: CoreYOLO code is MIT. Converted tensors still come from a third-party YOLOv9
  checkpoint (AGPL-3.0). Renaming, fine-tuning, or Core ML export does not relicense
  the weights. SaaS use of those tensors typically requires releasing application
  source under AGPL-3.0; a closed product usually needs a commercial license from
  the copyright holders. Do not rehost converted files as MIT. Prefer MultimediaTechLab
  v9-*.pt (MIT) or train CoreYOLO from scratch. See docs/licenses.md.
"""

# Printed when converting MultimediaTechLab/YOLO MIT weights (LibreYOLO source).
MTL_CONVERT_LICENSE_NOTICE = """\
license: converted tensors come from MultimediaTechLab/YOLO (MIT), copyright
  Kin-Yiu Wong and Hao-Tang Tsui. CoreYOLO remaps names only and keeps that MIT
  license plus the copyright notice. Ultralytics yolov9*.pt is a different file
  (AGPL-3.0) and is not this path. See docs/licenses.md.
"""

MTL_VENDOR = "MultimediaTechLab/YOLO"
MTL_COPYRIGHT = "Copyright (c) 2024 Kin-Yiu Wong and Hao-Tang Tsui"

# Compact YOLOv9 GELAN sequential index → CoreYOLO named module.
# Longer indices must be applied first so ``model.22`` is not eaten by ``model.2``.
LAYER_MAP: dict[int, str] = {
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
V8_LAYER_MAP = LAYER_MAP

_MTL_HEAD_RE = re.compile(r"(?:^|\.)heads\.\d+\.(?:anchor_conv|class_conv)\.")


def detect_pt_family(keys: list[str]) -> str:
    if any("one2one" in k for k in keys):
        return "yolo26"
    if any(_MTL_HEAD_RE.search(k) for k in keys):
        return "yolov9-mtl"
    if any(k.startswith("model.23.dfl") for k in keys):
        return "yolo11"
    if any(k.startswith("model.9.cv5") for k in keys):
        return "yolov9"
    if any(k.startswith("model.22.dfl") for k in keys):
        return "yolov8"
    return "unknown"


def is_ultralytics_checkpoint(ckpt: Any) -> bool:
    """True for an Ultralytics pickle (AGPL-3.0 tensors)."""
    if not isinstance(ckpt, dict):
        return False
    license_id = str(ckpt.get("license") or "")
    docs = str(ckpt.get("docs") or "")
    if "agpl" in license_id.lower() or "ultralytics" in docs.lower():
        return True
    model = ckpt.get("model")
    module = str(getattr(type(model), "__module__", "") or "")
    return module.startswith("ultralytics")


def remap_v8_state(src: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Rename sequential ``model.N.*`` tensors to CoreYOLO names. DFL conv → ``proj``."""
    out: dict[str, torch.Tensor] = {}
    order = sorted(LAYER_MAP, reverse=True)
    for key, tensor in src.items():
        dest = None
        for idx in order:
            prefix = f"model.{idx}."
            if key.startswith(prefix):
                dest = LAYER_MAP[idx] + key[len(prefix) :]
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


remap_gelan_state = remap_v8_state


def expand_grouped_to_dense(weight: torch.Tensor, groups: int) -> torch.Tensor:
    """Reparameterize grouped conv weights into an equivalent dense kernel."""
    out_c, in_per_group, *spatial = weight.shape
    if groups < 2 or out_c % groups or in_per_group < 1:
        return weight
    dense = weight.new_zeros((out_c, in_per_group * groups, *spatial))
    out_per_group = out_c // groups
    for group in range(groups):
        out_slice = slice(group * out_per_group, (group + 1) * out_per_group)
        in_slice = slice(group * in_per_group, (group + 1) * in_per_group)
        dense[out_slice, in_slice] = weight[out_slice]
    return dense


def _map_mtl_elan(suffix: str) -> str:
    return re.sub(r"^conv([1234])\.", r"cv\1.", suffix)


def _map_mtl_aconv(suffix: str) -> str:
    return re.sub(r"^conv\.", "cv1.", suffix)


def _map_mtl_adown(suffix: str) -> str:
    return suffix.replace("conv1", "cv1").replace("conv2", "cv2")


def _map_mtl_sppelan(suffix: str) -> str:
    return suffix.replace("conv1.", "cv1.").replace("conv5.", "cv5.")


def _map_mtl_rep(suffix: str) -> str:
    suffix = re.sub(r"^conv([1234])\.", r"cv\1.", suffix)
    suffix = re.sub(r"(cv[23]\.0)\.conv([123])\.", r"\1.cv\2.", suffix)
    suffix = suffix.replace(".bottleneck.", ".m.")
    return re.sub(r"\.m\.(\d+)\.conv([12])\.", r".m.\1.cv\2.", suffix)


def _map_mtl_detect(suffix: str) -> str | None:
    if "anc2vec" in suffix:
        return None
    suffix = re.sub(r"^heads\.(\d+)\.anchor_conv\.", r"cv2.\1.", suffix)
    suffix = re.sub(r"^heads\.(\d+)\.class_conv\.", r"cv3.\1.", suffix)
    return suffix


def _mtl_layer_type(idx: int, config: str) -> str:
    if idx in (0, 1):
        return "conv"
    if idx == 2:
        return "elan" if config in ("t", "s") else "rep"
    if idx in (3, 5, 7, 16, 19):
        return "adown" if config == "c" else "aconv"
    if idx in (4, 6, 8, 12, 15, 18, 21):
        return "rep"
    if idx == 9:
        return "sppelan"
    if idx == 22:
        return "detect"
    return "skip"


_MTL_MAPPERS = {
    "conv": lambda suffix: suffix,
    "elan": _map_mtl_elan,
    "aconv": _map_mtl_aconv,
    "adown": _map_mtl_adown,
    "rep": _map_mtl_rep,
    "sppelan": _map_mtl_sppelan,
    "detect": _map_mtl_detect,
}


def infer_mtl_config(src: dict[str, torch.Tensor]) -> str:
    """Infer MultimediaTechLab v9-t/s/m/c from stem / first-block width."""
    stem = src.get("0.conv.weight")
    if stem is None:
        return "t"
    ch = int(stem.shape[0])
    if ch == 16:
        return "t"
    if ch == 64:
        return "c"
    if ch == 32:
        block = src.get("2.conv1.conv.weight")
        if block is not None and int(block.shape[0]) == 128:
            return "m"
        return "s"
    return "t"


def infer_mtl_gelan_scale(src: dict[str, torch.Tensor]) -> str:
    """Map MultimediaTechLab v9-t/s/m/c onto CoreYOLO GELAN n/s/m/l."""
    return {"t": "n", "s": "s", "m": "m", "c": "l"}[infer_mtl_config(src)]


def remap_mtl_state(src: dict[str, torch.Tensor], config: str | None = None) -> dict[str, torch.Tensor]:
    """Rename MultimediaTechLab numbered keys onto CoreYOLO GELAN names."""
    config = config or infer_mtl_config(src)
    out: dict[str, torch.Tensor] = {}
    for key, tensor in src.items():
        idx_s, _, suffix = key.partition(".")
        if not idx_s.isdigit():
            continue
        idx = int(idx_s)
        if idx >= 23:
            continue
        prefix = LAYER_MAP.get(idx)
        kind = _mtl_layer_type(idx, config)
        if prefix is None or kind == "skip":
            continue
        dest_suffix = _MTL_MAPPERS[kind](suffix)
        if dest_suffix is None:
            continue
        out[prefix + dest_suffix] = tensor.detach().float().contiguous()
    return out


def _fit_grouped_head(mapped: dict[str, torch.Tensor], dest: dict[str, torch.Tensor]) -> None:
    """Expand MTL grouped box-head convs into CoreYOLO's dense Detect kernels."""
    for key, tensor in list(mapped.items()):
        want = dest.get(key)
        if want is None or tuple(tensor.shape) == tuple(want.shape):
            continue
        if tensor.ndim == want.ndim == 4 and tensor.shape[0] == want.shape[0]:
            if want.shape[1] > tensor.shape[1] and want.shape[1] % tensor.shape[1] == 0:
                groups = want.shape[1] // tensor.shape[1]
                mapped[key] = expand_grouped_to_dense(tensor, groups)


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
    if hasattr(model, "state_dict") and not isinstance(model, dict):
        src = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    elif isinstance(model, dict):
        src = {k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in model.items() if torch.is_tensor(v)}
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


def infer_gelan_scale(src: dict[str, torch.Tensor]) -> str:
    """Map compact YOLOv9 stem / P3 width to CoreYOLO GELAN n/s/m/l."""
    stem = src.get("model.0.conv.weight")
    if stem is None:
        return infer_mtl_gelan_scale(src) if "0.conv.weight" in src else "n"
    ch = int(stem.shape[0])
    if ch == 16:
        return "n"
    if ch == 64:
        return "l"
    if ch == 32:
        p3 = src.get("model.3.cv1.conv.weight")
        if p3 is not None and int(p3.shape[0]) == 240:
            return "m"
        return "s"
    return "n"


def convert_yolo_pt(
    weights: str | Path,
    out: str | Path | None = None,
    scale: str | None = None,
) -> Path:
    """Load a compact YOLOv9 ``.pt`` and write a CoreYOLO checkpoint.

    MultimediaTechLab ``v9-*.pt`` maps onto ``family=gelan`` and stays MIT.
    Ultralytics ``yolov9*.pt`` maps onto the same graph and stays AGPL-3.0.
    YOLO26/YOLO11 raise ``ValueError``.
    """
    weights = Path(weights)
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    src, names, version = _extract_state(ckpt)
    family = detect_pt_family(list(src))
    ultralytics = is_ultralytics_checkpoint(ckpt)
    if family == "yolov9-mtl" and not ultralytics:
        core_family = "gelan"
        inferred = infer_mtl_gelan_scale(src)
        source_tag = "yolov9"
        license_id = MIT_WEIGHT_LICENSE
        vendor = MTL_VENDOR
        copyright = MTL_COPYRIGHT
        notice = MTL_CONVERT_LICENSE_NOTICE
        mapped = remap_mtl_state(src)
    elif family == "yolov9":
        core_family = "gelan"
        inferred = infer_gelan_scale(src)
        source_tag = "yolov9"
        license_id = AGPL_WEIGHT_LICENSE
        vendor = "ultralytics"
        copyright = None
        notice = CONVERT_LICENSE_NOTICE
        mapped = remap_v8_state(src)
    elif family == "yolov8":
        core_family = "dfl"
        inferred = infer_v8_scale(src)
        source_tag = "yolov8"
        license_id = AGPL_WEIGHT_LICENSE
        vendor = "ultralytics"
        copyright = None
        notice = CONVERT_LICENSE_NOTICE
        mapped = remap_v8_state(src)
    else:
        raise ValueError(
            f"{weights.name} looks like {family or 'an unknown'} checkpoint, not compact YOLOv9. "
            "MIT path: MultimediaTechLab v9-t.pt (nano; CoreYOLO scale n), v9-s.pt, v9-m.pt, v9-c.pt (scale l). "
            "AGPL path: Ultralytics yolov9t.pt / s / m / yolov9c.pt. YOLO26n uses C3k2, drops DFL, and adds an "
            "NMS-free one-to-one head — those weights cannot be copied. YOLO11n is also C3k2 + C2PSA.\n"
            "  coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo"
        )
    if ultralytics and license_id == MIT_WEIGHT_LICENSE:
        raise ValueError(
            f"{weights.name} is an Ultralytics pickle (AGPL-3.0). Convert will stamp AGPL-3.0; "
            "it cannot relicense those tensors as MIT. Use MultimediaTechLab v9-*.pt for the MIT path."
        )
    if scale is None:
        scale = inferred
    elif scale != inferred:
        raise ValueError(
            f"--model {scale} does not match this checkpoint (stem width implies {inferred})"
        )
    task = "detect"
    if any(k.startswith("model.22.proto") or k.startswith("model.22.cv4") for k in src) or any(
        k.startswith("22.heads.") and "mask" in k for k in src
    ):
        task = "segment"
    nm = 32
    if task == "segment":
        for key, tensor in src.items():
            if key.endswith("proto.cv3.conv.weight"):
                nm = int(tensor.shape[0])
                break
        source_tag = f"{source_tag}-seg"
    model = build_model(nc=len(names), scale=scale, act="silu", family=core_family, task=task, nm=nm)
    dest_state = model.state_dict()
    _fit_grouped_head(mapped, dest_state)
    if "head.dfl.proj" not in mapped and "head.dfl.proj" in dest_state:
        mapped["head.dfl.proj"] = dest_state["head.dfl.proj"].detach().float().contiguous()
    dest_keys = set(dest_state)
    extra = sorted(set(mapped) - dest_keys)
    missing = sorted(dest_keys - set(mapped))
    if missing:
        raise ValueError(
            f"{family} remap is missing {len(missing)} CoreYOLO tensors, e.g. {missing[:8]}"
        )
    for key in extra:
        mapped.pop(key)
    model.load_state_dict(mapped, strict=True)
    dest = Path(out) if out else Path("weights") / (
        f"coreyolo-{scale}-coco-seg.coreyolo" if task == "segment" else f"coreyolo-{scale}-coco.coreyolo"
    )
    payload: dict[str, Any] = {
        "model": model.state_dict(),
        "nc": len(names),
        "names": names,
        "scale": scale,
        "act": "silu",
        "family": core_family,
        "task": task,
        "nm": nm if task == "segment" else 0,
        "end2end": False,
        "reg_max": 16,
        "imgsz": 640,
        "source": str(weights),
        "source_family": source_tag,
        "source_vendor": vendor,
        "source_license": license_id,
        "weights_license": license_id,
        "source_version": version,
    }
    if copyright:
        payload["source_copyright"] = copyright
    save_checkpoint(dest, payload)
    print(
        f"converted {weights} → {dest}  ({len(mapped)} tensors, act=silu, family={core_family}, "
        f"task={task}, scale={scale}, weights_license={license_id})"
    )
    print(notice, end="", file=sys.stderr)
    return dest
