"""Unified predictor for PyTorch checkpoints and Core ML packages."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from coreyolo.data.augment import letterbox
from coreyolo.infer.draw import annotate, save_annotated
from coreyolo.infer.masks import instance_masks
from coreyolo.infer.nms import non_max_suppression, scale_boxes
from coreyolo.nn.model import build_model, is_e2e_family, normalize_family
from coreyolo.utils import IMAGE_EXTS, load_checkpoint, place_module, select_device


def resolve_class_filter(names: list[str], classes: list[str] | None) -> list[int] | None:
    """Map ``person`` / ``0`` style filters onto class indices."""
    if not classes:
        return None
    lookup = {str(n).lower(): i for i, n in enumerate(names)}
    out: list[int] = []
    for raw in classes:
        token = str(raw).strip()
        if not token:
            continue
        if token.isdigit():
            out.append(int(token))
            continue
        key = token.lower()
        if key not in lookup:
            raise ValueError(f"Unknown class {token!r}. Available: {names[:12]}{'…' if len(names) > 12 else ''}")
        out.append(lookup[key])
    return out or None


def _letterbox_bgr(
    frame: np.ndarray,
    imgsz: int,
    color: int = 114,
    scaleup: bool = False,
) -> tuple[np.ndarray, float, tuple[float, float]]:
    """Same scale/pad contract as ``letterbox``, on an OpenCV BGR frame."""
    import cv2

    h, w = frame.shape[:2]
    scale = min(imgsz / w, imgsz / h)
    if not scaleup:
        scale = min(scale, 1.0)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), color, dtype=np.uint8)
    pad_x = (imgsz - nw) / 2
    pad_y = (imgsz - nh) / 2
    x0, y0 = int(round(pad_x)), int(round(pad_y))
    canvas[y0 : y0 + nh, x0 : x0 + nw] = resized
    return canvas, scale, (pad_x, pad_y)


class Predictor:
    def __init__(
        self,
        weights: str | Path | None = None,
        device: str = "gpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        names: list[str] | None = None,
        classes: list[str] | int | None = None,
        module: torch.nn.Module | None = None,
        task: str = "detect",
        end2end: bool = False,
    ) -> None:
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.names = names or []
        self.torch_model = None
        self.coreml_engine = None
        self.end2end = end2end
        self.task = task
        self.weights = Path(weights) if weights else None
        if module is not None:
            self.backend = "torch"
            self.device = select_device(device)
            self.task = str(getattr(module, "task", task))
            self.end2end = bool(getattr(module, "end2end", end2end))
            nc = int(getattr(module, "nc", 80))
            if not self.names:
                self.names = [f"class_{i}" for i in range(nc)]
            self.torch_model, self.device = place_module(module, self.device)
            self.torch_model.eval()
            self._dtype = next(self.torch_model.parameters()).dtype
        elif self.weights is None:
            raise ValueError("Predictor needs a weights path or an in-memory module")
        else:
            self.backend = (
                "coreml"
                if self.weights.suffix in {".mlpackage", ".mlmodel"}
                or (self.weights.is_dir() and self.weights.suffix != ".coreyolo" and self.weights.suffix != ".pt")
                else "torch"
            )
            if self.backend == "coreml":
                from coreyolo.export.engine import CoreMLEngine

                self.coreml_engine = CoreMLEngine(self.weights, device=device)
                self.device = self.coreml_engine.device_name
                self.imgsz = self.coreml_engine.imgsz or imgsz
                self.end2end = self.coreml_engine.end2end
                self.task = self.coreml_engine.task
                if not self.names:
                    self.names = self.coreml_engine.names
            else:
                self.device = select_device(device)
                ckpt = load_checkpoint(self.weights, map_location="cpu")
                nc = ckpt.get("nc", 80)
                scale = ckpt.get("scale", "n")
                act = ckpt.get("act", "relu")
                family = normalize_family(ckpt.get("family", "gelan"))
                self.imgsz = int(ckpt.get("imgsz", imgsz))
                self.end2end = bool(ckpt.get("end2end", is_e2e_family(family)))
                self.task = str(ckpt.get("task", "detect"))
                if not self.names:
                    self.names = ckpt.get("names") or [f"class_{i}" for i in range(nc)]
                self.torch_model = build_model(
                    nc=nc,
                    scale=scale,
                    act=act,
                    family=family,
                    task=self.task,
                    nm=int(ckpt.get("nm", 32)),
                )
                self.torch_model.load_state_dict(ckpt["model"], strict=False)
                self.torch_model, self.device = place_module(self.torch_model, self.device)
                self.torch_model.eval()
                self._prepare_torch_runtime()
        if isinstance(classes, (int, str)):
            classes = [classes]
        self.classes = resolve_class_filter(self.names, [str(c) for c in classes] if classes else None)

    def _prepare_torch_runtime(self) -> None:
        """Fuse Conv-BN, cache DFL anchors, and use FP16 on MPS/CUDA."""
        assert self.torch_model is not None
        if hasattr(self.torch_model, "fuse"):
            self.torch_model.fuse()
        if self.device.type in {"mps", "cuda"}:
            self.torch_model.half()
        self._dtype = next(self.torch_model.parameters()).dtype
        with torch.inference_mode():
            dummy = torch.zeros(1, 3, self.imgsz, self.imgsz, device=self.device, dtype=self._dtype)
            feats = self.torch_model.forward_neck(dummy)
            self.torch_model.head.prepare_export(list(feats))
            self.torch_model.head.export = True

    def _unpack_raw(self, raw):
        mc = proto = None
        if isinstance(raw, dict):
            pred = torch.from_numpy(np.asarray(raw["detections"])) if not torch.is_tensor(raw["detections"]) else raw["detections"]
            if "mask_coeff" in raw and "proto" in raw:
                mc = raw["mask_coeff"]
                proto = raw["proto"]
                if not torch.is_tensor(mc):
                    mc = torch.from_numpy(np.asarray(mc))
                if not torch.is_tensor(proto):
                    proto = torch.from_numpy(np.asarray(proto))
            return pred, mc, proto
        if isinstance(raw, (tuple, list)) and not torch.is_tensor(raw):
            if len(raw) >= 3:
                return raw[0], raw[1], raw[2]
            return raw[0], None, None
        return raw, None, None

    def _postprocess(
        self,
        pred,
        mc,
        proto,
        orig_wh: tuple[int, int],
        pad: tuple[float, float],
        ratio: float,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if isinstance(pred, np.ndarray):
            pred = torch.from_numpy(pred)
        if pred.ndim == 2:
            pred = pred.unsqueeze(0)
        if mc is not None and not torch.is_tensor(mc):
            mc = torch.from_numpy(np.asarray(mc))
        if proto is not None and not torch.is_tensor(proto):
            proto = torch.from_numpy(np.asarray(proto))
        if mc is not None:
            mc = mc.to(device=pred.device, dtype=pred.dtype)
        if proto is not None:
            proto = proto.to(device=pred.device, dtype=pred.dtype)
        masks_t = None
        if pred.ndim == 3 and pred.shape[-1] == 6:
            det = pred[0]
            keep = det[:, 4] > self.conf
            if self.classes is not None:
                allow = torch.zeros_like(keep)
                for c in self.classes:
                    allow |= det[:, 5].long() == c
                keep = keep & allow
            det = det[keep]
            if mc is not None and proto is not None and det.numel():
                coeff = mc[0][keep] if mc.ndim == 3 else mc[keep]
                proto_i = proto[0] if proto.ndim == 4 else proto
                masks_t = instance_masks(proto_i, coeff, det[:, :4], self.imgsz, orig_wh, pad, ratio)
        elif mc is not None and proto is not None:
            dets, coeffs = non_max_suppression(pred, self.conf, self.iou, extra=mc, classes=self.classes)
            det = dets[0]
            proto_i = proto[0] if proto.ndim == 4 else proto
            if det.numel():
                masks_t = instance_masks(proto_i, coeffs[0], det[:, :4], self.imgsz, orig_wh, pad, ratio)
        else:
            det = non_max_suppression(pred, self.conf, self.iou, classes=self.classes)[0]
        if det.numel():
            det = scale_boxes(det, (self.imgsz, self.imgsz), orig_wh, pad, ratio)
        masks_np = None if masks_t is None else masks_t.detach().cpu().numpy()
        return det.detach().cpu().numpy(), masks_np

    def _forward_rgb(self, rgb: np.ndarray) -> Any:
        tensor = torch.from_numpy(np.array(rgb, copy=True, order="C"))
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self._dtype).div_(255)
        return self.torch_model(tensor)

    @torch.inference_mode()
    def predict_full(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
        orig = image.convert("RGB")
        canvas, ratio, pad = letterbox(orig, self.imgsz, scaleup=False)
        if self.backend == "coreml":
            assert self.coreml_engine is not None
            raw = self.coreml_engine.predict_letterboxed(canvas)
        else:
            raw = self._forward_rgb(np.asarray(canvas))
        pred, mc, proto = self._unpack_raw(raw)
        return self._postprocess(pred, mc, proto, orig.size, pad, ratio)

    @torch.inference_mode()
    def predict_bgr(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
        """OpenCV BGR frame → detections and masks in original pixel space."""
        import cv2

        canvas, ratio, pad = _letterbox_bgr(frame, self.imgsz)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        if self.backend == "coreml":
            assert self.coreml_engine is not None
            raw = self.coreml_engine.predict_letterboxed(Image.fromarray(rgb))
        else:
            raw = self._forward_rgb(rgb)
        pred, mc, proto = self._unpack_raw(raw)
        return self._postprocess(pred, mc, proto, (int(frame.shape[1]), int(frame.shape[0])), pad, ratio)

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> np.ndarray:
        det, _ = self.predict_full(image)
        return det

    def predict_path(self, source: str | Path, save_dir: str | Path | None = None) -> list[dict]:
        source = Path(source)
        if source.is_dir():
            files = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        else:
            files = [source]
        results = []
        out_dir = Path(save_dir) if save_dir else None
        for path in files:
            image = Image.open(path).convert("RGB")
            det, masks = self.predict_full(image)
            item = {"path": str(path), "detections": det, "masks": masks, "names": self.names}
            if out_dir is not None:
                vis = annotate(image, det, self.names, masks=masks)
                item["saved"] = str(save_annotated(vis, out_dir / path.name))
            results.append(item)
        return results
