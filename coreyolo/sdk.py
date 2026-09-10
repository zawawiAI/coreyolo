"""Python SDK: ``Detector`` session for train, val, predict, and Core ML export.

Typical use::

    from coreyolo import Detector

    model = Detector("n")
    model.train(data="data.yaml", epochs=100, imgsz=640)
    model.save("weights/app.coreyolo")
    results = model.predict("photo.jpg", classes="person")
    model.export()

``YOLO`` remains a compatibility alias. YOLO is a trademark of its owners.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image

from coreyolo.infer.predictor import Predictor, resolve_class_filter
from coreyolo.nn.model import CoreYOLO, build_model, is_e2e_family, normalize_family
from coreyolo.nn.modules import normalize_act
from coreyolo.results import Result
from coreyolo.utils import (
    AGPL_WEIGHT_LICENSE,
    IMAGE_EXTS,
    MIT_WEIGHT_LICENSE,
    load_checkpoint,
    origin_metadata,
    save_checkpoint,
)

SCALES = ("n", "s", "m", "l", "x")
_Source = str | Path | Image.Image | np.ndarray


class Detector:
    """High-level CoreYOLO handle. The CLI is a thin wrapper around this API.

    Parameters
    ----------
    model:
        Scale ``n``/``s``/``m``/``l``/``x`` for a new network, or a path to a
        CoreYOLO ``.coreyolo`` / ``.pt`` / Core ML ``.mlpackage``.
        Do not pass Ultralytics files such as ``yolov9t.pt``.
    """

    def __init__(
        self,
        model: str | Path = "n",
        *,
        task: str = "detect",
        family: str = "gelan",
        act: str = "relu",
        nc: int = 80,
        names: list[str] | None = None,
        device: str = "gpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.45,
        classes: str | int | Sequence[str | int] | None = None,
        nm: int = 32,
    ) -> None:
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.nm = nm
        self.last_train: Path | None = None
        self._predictor: Predictor | None = None
        self._nn: CoreYOLO | None = None
        self._origin: dict[str, Any] = {}

        raw = str(model).strip()
        path = Path(raw)
        if raw.lower() in SCALES:
            self.weights: Path | None = None
            self.scale = raw.lower()
            self.family = normalize_family(family)
            self.task = str(task).lower()
            self.act = normalize_act(act)
            self.nc = int(nc)
            self.names = names or [f"class_{i}" for i in range(self.nc)]
            self.end2end = is_e2e_family(self.family)
        elif path.exists() or path.suffix in {".pt", ".coreyolo", ".mlpackage", ".mlmodel"} or path.is_dir():
            if not path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
            self.weights = path
            self.scale = "n"
            self.family = normalize_family(family)
            self.task = str(task).lower()
            self.act = act
            self.nc = int(nc)
            self.names = names or []
            self.end2end = is_e2e_family(self.family)
            self._hydrate_from_weights()
        else:
            raise FileNotFoundError(
                f"Unknown model {model!r}. Pass a scale {list(SCALES)} or a CoreYOLO .coreyolo / .pt / .mlpackage path."
            )
        self.classes = classes

    def _hydrate_from_weights(self) -> None:
        assert self.weights is not None
        if self.weights.suffix in {".mlpackage", ".mlmodel"} or self.weights.is_dir():
            pred = Predictor(self.weights, device=self.device, imgsz=self.imgsz, conf=self.conf, iou=self.iou)
            self.names = pred.names or self.names
            self.task = pred.task
            self.end2end = pred.end2end
            self.imgsz = pred.imgsz
            self.nc = len(self.names) or self.nc
            self._predictor = pred
            return
        ckpt = load_checkpoint(self.weights, map_location="cpu")
        self.nc = int(ckpt.get("nc", self.nc))
        self.scale = str(ckpt.get("scale", self.scale))
        self.act = normalize_act(str(ckpt.get("act", self.act)))
        self.family = normalize_family(ckpt.get("family", self.family))
        self.task = str(ckpt.get("task", self.task))
        self.nm = int(ckpt.get("nm", self.nm))
        self.imgsz = int(ckpt.get("imgsz", self.imgsz))
        self.end2end = bool(ckpt.get("end2end", is_e2e_family(self.family)))
        self.names = ckpt.get("names") or self.names or [f"class_{i}" for i in range(self.nc)]
        self._origin = origin_metadata(ckpt)

    def _get_predictor(self, classes: Any = None) -> Predictor:
        filt = self.classes if classes is None else classes
        if self._predictor is None:
            if self.weights is not None:
                self._predictor = Predictor(
                    self.weights,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    names=self.names,
                    classes=filt,
                )
            else:
                nn = self.model
                self._predictor = Predictor(
                    None,
                    device=self.device,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    names=self.names,
                    classes=filt,
                    module=nn,
                    task=self.task,
                    end2end=self.end2end,
                )
            self.names = self._predictor.names or self.names
            self.task = self._predictor.task
        else:
            self._predictor.conf = self.conf
            self._predictor.iou = self.iou
            if filt is None:
                self._predictor.classes = None
            else:
                seq = [filt] if isinstance(filt, (int, str)) else list(filt)
                self._predictor.classes = resolve_class_filter(self._predictor.names, [str(c) for c in seq])
        return self._predictor

    @property
    def model(self) -> CoreYOLO:
        """Underlying PyTorch module (not used for Core ML packages)."""
        if self._nn is not None:
            return self._nn
        if self.weights is not None and self.weights.suffix in {".mlpackage", ".mlmodel"}:
            raise TypeError("Core ML packages have no PyTorch module; load a .coreyolo / .pt file")
        if self.weights is not None and self.weights.suffix in {".pt", ".coreyolo"}:
            self._nn = build_model(
                nc=self.nc,
                scale=self.scale,
                act=self.act,
                family=self.family,
                task=self.task,
                nm=self.nm,
                weights=str(self.weights),
            )
            return self._nn
        self._nn = build_model(nc=self.nc, scale=self.scale, act=self.act, family=self.family, task=self.task, nm=self.nm)
        return self._nn

    def info(self) -> dict[str, Any]:
        data = {
            "weights": str(self.weights) if self.weights else None,
            "scale": self.scale,
            "family": self.family,
            "task": self.task,
            "act": self.act,
            "nc": self.nc,
            "names": self.names,
            "imgsz": self.imgsz,
            "device": self.device,
            "weights_license": (
                str(self._origin.get("weights_license") or AGPL_WEIGHT_LICENSE)
                if self._origin
                else MIT_WEIGHT_LICENSE
            ),
        }
        if self.weights is None or self.weights.suffix in {".pt", ".coreyolo"}:
            data.update(self.model.info())
        return data

    def __repr__(self) -> str:
        src = self.weights or f"family={self.family} {self.scale} {self.task}"
        return f"Detector({src})"

    def __call__(self, source: _Source | Sequence[_Source], **kwargs: Any) -> list[Result]:
        return self.predict(source, **kwargs)

    def train(self, data: str | Path, **kwargs: Any) -> Path:
        """Fit on a Roboflow YOLO ``data.yaml``. Returns the run directory."""
        from coreyolo.train.trainer import TrainConfig, train

        allowed = {f.name for f in fields(TrainConfig)}
        payload = {k: v for k, v in kwargs.items() if k in allowed}
        payload.pop("data", None)
        payload.setdefault("model", self.scale)
        payload.setdefault("family", self.family)
        payload.setdefault("task", self.task)
        payload.setdefault("act", self.act)
        payload.setdefault("device", self.device)
        payload.setdefault("imgsz", self.imgsz)
        if self._origin and self.weights is not None and payload.get("resume") is None:
            payload["resume"] = str(self.weights)
        cfg = TrainConfig(data=str(data), **payload)
        save_dir = train(cfg)
        self.last_train = save_dir
        best = save_dir / "weights" / "best.pt"
        if best.is_file():
            self.weights = best
            self._predictor = None
            self._nn = None
            self._hydrate_from_weights()
        return save_dir

    def val(self, data: str | Path, batch: int = 16, workers: int = 0, **kwargs: Any) -> dict[str, float]:
        """Validation mAP on a Roboflow split."""
        from torch.utils.data import DataLoader

        from coreyolo.data.dataset import YOLODetectionDataset, collate_fn
        from coreyolo.data.yaml import YOLODatasetYAML
        from coreyolo.train.trainer import validate
        from coreyolo.utils import place_module, select_device

        spec = YOLODatasetYAML(data)
        device = select_device(kwargs.get("device", self.device))
        nn, device = place_module(self.model, device)
        nn.eval()
        imgsz = int(kwargs.get("imgsz", self.imgsz))
        ds = YOLODetectionDataset(spec, "val", imgsz=imgsz, augment=False, mosaic=0, task=self.task)
        loader = DataLoader(
            ds,
            batch_size=min(batch, len(ds)),
            shuffle=False,
            num_workers=workers,
            collate_fn=collate_fn,
        )
        return validate(nn, loader, device, spec.nc)

    def predict(
        self,
        source: _Source | Sequence[_Source],
        *,
        save: bool = False,
        save_dir: str | Path = "runs/predict",
        conf: float | None = None,
        iou: float | None = None,
        device: str | None = None,
        classes: str | int | Sequence[str | int] | None = None,
        stream: bool = False,
    ) -> list[Result]:
        """Run detection or segmentation.

        ``source`` may be a path, directory, PIL image, numpy HWC array, webcam
        index / ``\"webcam\"``, or a list of those. Set ``stream=True`` (or pass
        a camera index) for the OpenCV live window.
        """
        if conf is not None:
            self.conf = conf
        if iou is not None:
            self.iou = iou
        if device is not None:
            self.device = device
            self._predictor = None

        from coreyolo.infer.stream import parse_camera_index, run_webcam

        if isinstance(source, (str, int)) and parse_camera_index(str(source)) is not None:
            cam = parse_camera_index(str(source))
            assert cam is not None
            run_webcam(self._get_predictor(classes), camera=cam, track=True)
            return []
        if stream:
            run_webcam(self._get_predictor(classes), camera=0, track=True)
            return []

        items = list(self._expand_source(source))
        pred = self._get_predictor(classes)
        out_dir = Path(save_dir) if save else None
        results: list[Result] = []
        for path, image in items:
            det, masks = pred.predict_full(image)
            saved = None
            item = Result(det, pred.names, masks=masks, path=path, orig=image)
            if out_dir is not None:
                name = Path(path).name if path else "result.jpg"
                saved = str(item.save(out_dir / name))
                item.saved = saved
            results.append(item)
        return results

    def export(
        self,
        out: str | Path | None = None,
        imgsz: int | None = None,
        fp16: bool = True,
        int8: bool = False,
        palettize: bool | None = None,
        tensor_input: bool = False,
    ) -> Path:
        """Write a Core ML ``.mlpackage`` (Apple GPU / Neural Engine)."""
        from coreyolo.export.coreml import export_coreml

        weights = self._ensure_checkpoint()
        return export_coreml(
            weights,
            out=out,
            imgsz=int(imgsz or self.imgsz),
            fp16=fp16,
            quantize_8bit=int8,
            palettize=palettize,
            image_input=not tensor_input,
        )

    def save(self, path: str | Path | None = None) -> Path:
        """Write a native CoreYOLO checkpoint (``.coreyolo`` by default)."""
        dest = Path(path) if path else Path("weights") / f"coreyolo-{self.scale}-{self.task}.coreyolo"
        if dest.suffix not in {".pt", ".coreyolo"}:
            dest = dest.with_suffix(".coreyolo")
        nn = self.model
        payload: dict[str, Any] = {
            "model": nn.state_dict(),
            "nc": self.nc,
            "names": self.names,
            "scale": self.scale,
            "act": self.act,
            "family": self.family,
            "task": self.task,
            "nm": self.nm,
            "end2end": self.end2end,
            "reg_max": int(getattr(nn, "reg_max", 16)),
            "imgsz": self.imgsz,
            "weights_license": (
                str(self._origin.get("weights_license") or AGPL_WEIGHT_LICENSE)
                if self._origin
                else MIT_WEIGHT_LICENSE
            ),
        }
        if self._origin:
            payload.update(self._origin)
        save_checkpoint(dest, payload)
        self.weights = dest
        self._predictor = None
        self._nn = None
        return dest

    def _ensure_checkpoint(self) -> Path:
        if self.weights is not None and self.weights.suffix in {".pt", ".coreyolo"} and self.weights.is_file():
            return self.weights
        if self.weights is not None and (
            self.weights.suffix in {".mlpackage", ".mlmodel"} or self.weights.is_dir()
        ):
            raise TypeError("Already a Core ML package; export needs a CoreYOLO .coreyolo / .pt")
        return self.save()

    @classmethod
    def convert(cls, weights: str | Path, out: str | Path | None = None, scale: str | None = None) -> "Detector":
        """One-time bootstrap: Ultralytics YOLOv9 ``.pt`` → CoreYOLO ``.coreyolo``.

        Remaps tensor names only. Ultralytics weights stay AGPL-3.0. Do not call this
        from app inference. Convert once, then ``Detector(coreyolo_path)``. For an
        MIT weight path, train CoreYOLO on your labels instead.
        """
        from coreyolo.export.convert import convert_ultralytics

        path = convert_ultralytics(weights, out=out, scale=scale)
        return cls(path)

    @staticmethod
    def dummy_data(
        root: str | Path = "datasets/dummy",
        n_train: int = 32,
        n_val: int = 8,
        size: int = 320,
        segment: bool = False,
    ) -> Path:
        """Write a tiny Roboflow-layout dataset and return ``data.yaml``."""
        from coreyolo.data.dummy import write_dummy_dataset

        return write_dummy_dataset(root, n_train=n_train, n_val=n_val, size=size, segment=segment)

    @staticmethod
    def _as_image(source: _Source) -> tuple[str | None, Image.Image]:
        if isinstance(source, Image.Image):
            return None, source.convert("RGB")
        if isinstance(source, np.ndarray):
            arr = source
            if arr.ndim == 3 and arr.shape[0] in {1, 3} and arr.shape[0] < arr.shape[-1]:
                arr = np.transpose(arr, (1, 2, 0))
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8) if arr.max() > 1 else (arr * 255).astype(np.uint8)
            return None, Image.fromarray(arr).convert("RGB")
        path = Path(source)
        return str(path), Image.open(path).convert("RGB")

    def _expand_source(self, source: _Source | Sequence[_Source]) -> Iterable[tuple[str | None, Image.Image]]:
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
                for f in files:
                    yield str(f), Image.open(f).convert("RGB")
                return
            yield self._as_image(source)
            return
        if isinstance(source, (Image.Image, np.ndarray)):
            yield self._as_image(source)
            return
        for item in source:
            yield from self._expand_source(item)


YOLO = Detector  # compatibility alias; YOLO is a trademark of its owners
