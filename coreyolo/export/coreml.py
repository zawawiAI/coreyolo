"""Trace a fused CoreYOLO graph and convert it to a Core ML ML Program."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from coreyolo.nn.model import CoreYOLO, build_model, is_e2e_family, normalize_family
from coreyolo.utils import __version__, MIT_WEIGHT_LICENSE, checkpoint_weight_license, load_checkpoint, origin_metadata


def _load_for_export(weights: str | Path) -> tuple[CoreYOLO, dict]:
    ckpt = load_checkpoint(weights, map_location="cpu")
    nc = int(ckpt.get("nc", 80))
    scale = ckpt.get("scale", "n")
    act = ckpt.get("act", "relu")
    family = normalize_family(ckpt.get("family", "gelan"))
    names = ckpt.get("names") or [f"class_{i}" for i in range(nc)]
    task = str(ckpt.get("task", "detect"))
    nm = int(ckpt.get("nm", 32))
    model = build_model(nc=nc, scale=scale, act=act, family=family, task=task, nm=nm)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    model.fuse()
    model.head.export = True
    end2end = bool(ckpt.get("end2end", is_e2e_family(family)))
    origin = origin_metadata(ckpt)
    return model, {
        "nc": nc,
        "scale": scale,
        "act": act,
        "names": names,
        "imgsz": ckpt.get("imgsz", 640),
        "family": family,
        "end2end": end2end,
        "task": task,
        "nm": nm,
        "weights_license": checkpoint_weight_license(ckpt),
        **origin,
    }


def export_coreml(
    weights: str | Path,
    out: str | Path | None = None,
    imgsz: int = 640,
    fp16: bool = True,
    quantize_8bit: bool = False,
    image_input: bool = True,
) -> Path:
    """Export ``best.pt`` to ``.mlpackage`` for Apple GPU / Neural Engine.

    The graph is static ``(1, 3, imgsz, imgsz)``. DFL exports decoded
    ``(1, 4+nc, N)`` xywh + scores (NMS on host). E2E exports NMS-free
    ``(1, 300, 6)`` xyxy + conf + cls.
    """
    import coremltools as ct

    model, meta = _load_for_export(weights)
    imgsz = int(imgsz or meta.get("imgsz") or 640)
    dummy = torch.zeros(1, 3, imgsz, imgsz)
    with torch.no_grad():
        feats = model.forward_neck(dummy)
        model.head.prepare_export(list(feats))
        traced = torch.jit.trace(model, dummy, strict=False)
        traced.eval()

    out = Path(out) if out else Path(weights).with_suffix(".mlpackage")
    out.parent.mkdir(parents=True, exist_ok=True)

    if image_input:
        inputs = [
            ct.ImageType(
                name="image",
                shape=(1, 3, imgsz, imgsz),
                scale=1 / 255.0,
                bias=[0.0, 0.0, 0.0],
                color_layout=ct.colorlayout.RGB,
            )
        ]
    else:
        inputs = [ct.TensorType(name="image", shape=(1, 3, imgsz, imgsz), dtype=np.float32)]

    precision = ct.precision.FLOAT16 if fp16 else ct.precision.FLOAT32
    segment = meta.get("task") == "segment"
    if segment:
        output_types = [
            ct.TensorType(name="detections"),
            ct.TensorType(name="mask_coeff"),
            ct.TensorType(name="proto"),
        ]
    else:
        output_types = [ct.TensorType(name="detections")]
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=output_types,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=precision,
    )

    if quantize_8bit:
        from coremltools.optimize.coreml import (
            OptimizationConfig,
            OpLinearQuantizerConfig,
            linear_quantize_weights,
        )

        qconfig = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric"))
        mlmodel = linear_quantize_weights(mlmodel, qconfig)

    mlmodel.short_description = (
        "CoreYOLO instance segmenter (boxes + mask coefficients + proto)"
        if segment
        else (
            "CoreYOLO E2E NMS-free detector (xyxy, conf, cls; top-300)"
            if meta.get("end2end")
            else "CoreYOLO object detector (decoded xywh + class scores, NMS on host)"
        )
    )
    mlmodel.author = "CoreYOLO"
    mlmodel.version = __version__
    mlmodel.user_defined_metadata["names"] = json.dumps(meta["names"])
    mlmodel.user_defined_metadata["nc"] = str(meta["nc"])
    mlmodel.user_defined_metadata["imgsz"] = str(imgsz)
    mlmodel.user_defined_metadata["scale"] = str(meta["scale"])
    mlmodel.user_defined_metadata["act"] = str(meta["act"])
    mlmodel.user_defined_metadata["family"] = str(meta.get("family", "gelan"))
    mlmodel.user_defined_metadata["task"] = str(meta.get("task", "detect"))
    if segment:
        mlmodel.user_defined_metadata["nm"] = str(meta.get("nm", 32))
        mlmodel.user_defined_metadata["layout"] = "detections + mask_coeff + proto"
        mlmodel.user_defined_metadata["nms"] = "end2end" if meta.get("end2end") else "host"
    elif meta.get("end2end"):
        mlmodel.user_defined_metadata["layout"] = "B,300,6  xyxy + conf + cls"
        mlmodel.user_defined_metadata["nms"] = "end2end"
    else:
        mlmodel.user_defined_metadata["layout"] = "B,4+nc,N  xywh_pixels + class_scores"
        mlmodel.user_defined_metadata["nms"] = "host"
    license_id = str(meta.get("weights_license") or MIT_WEIGHT_LICENSE)
    if hasattr(mlmodel, "license"):
        mlmodel.license = license_id
    mlmodel.user_defined_metadata["weights_license"] = license_id
    if license_id.upper().startswith("AGPL"):
        mlmodel.short_description = (
            "Contains Ultralytics tensors (AGPL-3.0). Name remap / Core ML export is not a relicensing. "
            + str(mlmodel.short_description)
        )
        if meta.get("source_vendor"):
            mlmodel.user_defined_metadata["source_vendor"] = str(meta["source_vendor"])
        if meta.get("source_family"):
            mlmodel.user_defined_metadata["source_family"] = str(meta["source_family"])
    mlmodel.save(str(out))
    return out
