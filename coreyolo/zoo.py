"""Catalog of public CoreYOLO checkpoints (GitHub Releases, not git)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from coreyolo.utils import __version__

__all__ = [
    "DEFAULT_MANIFEST",
    "assert_zoo_license_labels",
    "listed_release_files",
    "load_manifest",
    "record_metrics",
    "write_metrics_json",
]

DEFAULT_MANIFEST = Path("weights") / "manifest.json"


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path or DEFAULT_MANIFEST)
    return json.loads(path.read_text())


def write_metrics_json(path: str | Path, payload: dict[str, Any]) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload)
    body.setdefault("coreyolo", __version__)
    body.setdefault("recorded_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    dest.write_text(json.dumps(body, indent=2) + "\n")
    return dest


def record_metrics(
    model_id: str,
    metrics: dict[str, Any],
    *,
    manifest: str | Path | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Fill ``mAP50`` / ``mAP50-95`` on an existing zoo entry."""
    path = Path(manifest or DEFAULT_MANIFEST)
    catalog = load_manifest(path)
    found = None
    for entry in catalog.get("models", []):
        if entry.get("id") == model_id:
            found = entry
            break
    if found is None:
        known = [m.get("id") for m in catalog.get("models", [])]
        raise KeyError(f"unknown zoo id {model_id!r}; known: {known}")
    slot = found.setdefault("metrics", {})
    slot["mAP50"] = metrics.get("mAP50")
    slot["mAP50-95"] = metrics.get("mAP50-95", metrics.get("mAP50_95"))
    slot["updated"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if extra:
        if extra.get("weights"):
            found["source_weights"] = extra["weights"]
        if extra.get("data"):
            slot["data"] = extra["data"]
        if extra.get("imgsz") is not None:
            found["imgsz"] = int(extra["imgsz"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(catalog, indent=2) + "\n")
    return path


def listed_release_files(manifest: str | Path | None = None, root: str | Path | None = None) -> list[Path]:
    """Local files named by the catalog, if they exist (for ``gh release upload``)."""
    catalog = load_manifest(manifest)
    base = Path(root or Path(manifest or DEFAULT_MANIFEST).parent)
    found: list[Path] = []
    for entry in catalog.get("models", []):
        files = entry.get("files") or {}
        for name in files.values():
            path = base / name
            if path.is_file() or path.is_dir():
                found.append(path)
    return found


def assert_zoo_license_labels(path: str | Path | None = None) -> None:
    """Refuse to publish converted Ultralytics tensors under an MIT label."""
    catalog = load_manifest(path)
    errors: list[str] = []
    for entry in catalog.get("models", []):
        model_id = str(entry.get("id") or "?")
        how = str(entry.get("how") or "")
        origin = str(entry.get("weights_origin") or "")
        license_id = str(entry.get("weights_license") or "")
        converted = how.startswith("coreyolo convert") or origin.lower().startswith("ultralytics")
        if converted and not license_id.upper().startswith("AGPL"):
            errors.append(f"{model_id}: converted / Ultralytics tensors must be AGPL-3.0, not {license_id!r}")
        if converted and license_id.upper() == "MIT":
            errors.append(f"{model_id}: do not rehost converted tensors as MIT")
        if how.startswith("coreyolo train") and license_id.upper() != "MIT":
            errors.append(f"{model_id}: trained-from-scratch rows should be MIT, got {license_id!r}")
    if errors:
        raise ValueError("zoo license labels are wrong:\n  " + "\n  ".join(errors))
