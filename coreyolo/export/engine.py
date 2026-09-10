"""Load a Core ML package on GPU, Neural Engine, or CPU."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image


def resolve_compute_units(device: str | None = "gpu"):
    """Map a device string to a Core ML ``ComputeUnit``.

    ``gpu`` / ``mps`` → ``CPU_AND_GPU``
    ``auto`` → ANE-first (``CPU_AND_NE``), then GPU
    ``all`` → ``ALL`` (CPU + GPU + Neural Engine)
    ``ane`` → ``CPU_AND_NE``
    ``cpu`` → ``CPU_ONLY``
    """
    import coremltools as ct

    raw = (device or "gpu").lower().replace("-", "_")
    alias = {
        "auto": "auto",
        "gpu": "gpu",
        "mps": "gpu",
        "cuda": "gpu",
        "cpu_and_gpu": "gpu",
        "all": "all",
        "ane": "ane",
        "ne": "ane",
        "cpu_and_ne": "ane",
        "cpu": "cpu",
        "cpu_only": "cpu",
    }
    kind = alias.get(raw, "gpu")
    mapping = {
        "gpu": getattr(ct.ComputeUnit, "CPU_AND_GPU", ct.ComputeUnit.ALL),
        "auto": getattr(ct.ComputeUnit, "CPU_AND_NE", ct.ComputeUnit.ALL),
        "all": ct.ComputeUnit.ALL,
        "ane": getattr(ct.ComputeUnit, "CPU_AND_NE", ct.ComputeUnit.CPU_ONLY),
        "cpu": ct.ComputeUnit.CPU_ONLY,
    }
    return kind, mapping[kind]


def _fallback_chain(kind: str):
    import coremltools as ct

    gpu = getattr(ct.ComputeUnit, "CPU_AND_GPU", None)
    ane = getattr(ct.ComputeUnit, "CPU_AND_NE", None)
    order = {
        "gpu": [gpu, ct.ComputeUnit.ALL, ane, ct.ComputeUnit.CPU_ONLY],
        "auto": [ane, gpu, ct.ComputeUnit.ALL, ct.ComputeUnit.CPU_ONLY],
        "all": [ct.ComputeUnit.ALL, gpu, ane, ct.ComputeUnit.CPU_ONLY],
        "ane": [ane, gpu, ct.ComputeUnit.CPU_ONLY],
        "cpu": [ct.ComputeUnit.CPU_ONLY],
    }[kind]
    seen: set[object] = set()
    out = []
    for unit in order:
        if unit is None or unit in seen:
            continue
        seen.add(unit)
        out.append(unit)
    return out


def _unit_name(unit) -> str:
    return getattr(unit, "name", str(unit))


class CoreMLEngine:
    """Run an exported CoreYOLO ``.mlpackage``. ``auto`` tries Neural Engine first."""

    def __init__(self, path: str | Path, device: str = "auto") -> None:
        import coremltools as ct

        self.path = Path(path)
        kind, _preferred = resolve_compute_units(device)
        last_error: Exception | None = None
        self.model = None
        self.compute_units = None
        for unit in _fallback_chain(kind):
            try:
                self.model = ct.models.MLModel(str(self.path), compute_units=unit)
                self.compute_units = unit
                break
            except Exception as exc:  # compile can abort on some GPU paths
                last_error = exc
        if self.model is None:
            raise RuntimeError(f"Failed to load {self.path} on GPU/ANE/CPU") from last_error

        spec = self.model.get_spec()
        self.input_name = spec.description.input[0].name
        self.output_name = spec.description.output[0].name
        self.output_names = [o.name for o in spec.description.output]
        meta = dict(self.model.user_defined_metadata)
        self.names = json.loads(meta["names"]) if "names" in meta else []
        self.nc = int(meta.get("nc", len(self.names) or 80))
        self.imgsz = int(meta.get("imgsz", 640))
        self.image_input = spec.description.input[0].type.HasField("imageType")
        self.device_name = _unit_name(self.compute_units)
        self.end2end = meta.get("nms", "host") in {"none", "end2end", "topk"}
        self.task = meta.get("task", "segment" if len(self.output_names) > 1 else "detect")

    def predict_letterboxed(self, canvas: Image.Image) -> np.ndarray:
        """``canvas`` must already be RGB ``imgsz x imgsz`` (letterboxed)."""
        canvas = canvas.convert("RGB").resize((self.imgsz, self.imgsz), Image.BILINEAR)
        if self.image_input:
            out = self.model.predict({self.input_name: canvas})
        else:
            arr = np.asarray(canvas).astype(np.float32) / 255.0
            arr = np.transpose(arr, (2, 0, 1))[None]
            out = self.model.predict({self.input_name: arr})
        pred = out[self.output_name]
        if len(self.output_names) == 1:
            return np.asarray(pred)
        packed = {name: np.asarray(out[name]) for name in self.output_names}
        if "detections" not in packed:
            packed["detections"] = packed[self.output_names[0]]
        return packed
