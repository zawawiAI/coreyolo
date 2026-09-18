"""Parse Roboflow-style YOLO ``data.yaml`` files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _as_names(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, dict):
        items = sorted(raw.items(), key=lambda kv: int(kv[0]))
        return [str(v) for _, v in items]
    if isinstance(raw, (list, tuple)):
        return [str(n) for n in raw]
    raise TypeError(f"Unsupported names field: {type(raw)}")


def _resolve_split(root: Path, yaml_value: Any, *fallback_dirs: str) -> Path | None:
    """Resolve a split path from yaml, then common Roboflow folder names."""
    candidates: list[Path] = []
    if yaml_value:
        p = Path(str(yaml_value))
        candidates.append(p if p.is_absolute() else (root / p).resolve())
        # Roboflow often writes ``../train/images`` relative to a nested yaml.
        candidates.append((root / p).resolve())
    for name in fallback_dirs:
        candidates.append(root / name)
        candidates.append(root / name / "images")
    seen: set[Path] = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if c.is_dir():
            return c
    return None


def _labels_for_images(images_dir: Path) -> Path | None:
    """YOLO layout: ``.../images`` ↔ ``.../labels``; also sibling ``labels``."""
    if images_dir.name == "images":
        labels = images_dir.parent / "labels"
        if labels.is_dir():
            return labels
    sibling = images_dir.parent / "labels"
    if sibling.is_dir():
        return sibling
    nested = images_dir / "labels"
    if nested.is_dir():
        return nested
    return None


class YOLODatasetYAML:
    """Roboflow YOLO export layout.

    Expected tree (Roboflow YOLO export)::

        dataset/
          data.yaml
          train/images  train/labels
          valid/images  valid/labels   # Roboflow uses ``valid``, not ``val``
          test/images   test/labels    # optional

    ``data.yaml`` may use list or dict ``names``, and ``train``/``val``/``test``
    paths that are relative or leftover Colab absolute paths.
    """

    def __init__(self, path: str | Path) -> None:
        self.yaml_path = Path(path).expanduser().resolve()
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"data.yaml not found: {self.yaml_path}")
        self.root = self.yaml_path.parent
        with self.yaml_path.open() as f:
            cfg = yaml.safe_load(f) or {}
        self.raw = cfg
        self.names = _as_names(cfg.get("names"))
        self.nc = int(cfg.get("nc", len(self.names)))
        if self.names and self.nc != len(self.names):
            self.nc = len(self.names)
        if not self.names:
            self.names = [f"class_{i}" for i in range(self.nc)]

        self.train_images = _resolve_split(self.root, cfg.get("train"), "train/images", "train")
        self.val_images = _resolve_split(
            self.root,
            cfg.get("val") or cfg.get("valid"),
            "valid/images",
            "val/images",
            "valid",
            "val",
        )
        self.test_images = _resolve_split(self.root, cfg.get("test"), "test/images", "test")

        self.train_labels = _labels_for_images(self.train_images) if self.train_images else None
        self.val_labels = _labels_for_images(self.val_images) if self.val_images else None
        self.test_labels = _labels_for_images(self.test_images) if self.test_images else None

    def split(self, name: str) -> tuple[Path, Path]:
        mapping = {
            "train": (self.train_images, self.train_labels),
            "val": (self.val_images, self.val_labels),
            "valid": (self.val_images, self.val_labels),
            "test": (self.test_images, self.test_labels),
        }
        if name not in mapping:
            raise KeyError(name)
        images, labels = mapping[name]
        if images is None or labels is None:
            raise FileNotFoundError(
                f"Split '{name}' is missing images/ or labels/ under {self.root}. "
                "Export from Roboflow as YOLO detect txt (or YOLO-Seg) and unzip so "
                "data.yaml sits next to train/ and valid/."
            )
        return images, labels

    def summary(self) -> dict[str, Any]:
        return {
            "yaml": str(self.yaml_path),
            "nc": self.nc,
            "names": self.names,
            "train": str(self.train_images) if self.train_images else None,
            "val": str(self.val_images) if self.val_images else None,
            "test": str(self.test_images) if self.test_images else None,
        }
