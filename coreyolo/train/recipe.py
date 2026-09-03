"""Load train recipe YAML files (COCO n/s/m/… hypers)."""

from __future__ import annotations

from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import yaml

from coreyolo.train.trainer import TrainConfig

_RECIPE_ALIASES = {"coco-n-26": "coco-n-e2e"}


def _recipe_dirs() -> list[Path]:
    dirs: list[Path] = []
    packaged = Path(__file__).resolve().parent.parent / "recipes"
    if packaged.is_dir():
        dirs.append(packaged)
    repo = Path(__file__).resolve().parents[2] / "configs" / "recipes"
    if repo.is_dir() and repo.resolve() not in {p.resolve() for p in dirs}:
        dirs.append(repo)
    return dirs


def recipe_path(name: str) -> Path:
    """``coco-n`` or a filesystem path to a recipe yaml."""
    raw = Path(name)
    if raw.suffix in {".yaml", ".yml"} and raw.is_file():
        return raw.resolve()
    stem = _RECIPE_ALIASES.get(name, name)
    looked: list[Path] = []
    for folder in _recipe_dirs():
        for candidate in (folder / f"{stem}.yaml", folder / f"{name}.yaml"):
            looked.append(candidate)
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"Recipe not found: {name} (looked at {looked[0] if looked else 'no recipe dirs'})")


def load_recipe(name: str) -> dict[str, Any]:
    path = recipe_path(name)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"Recipe {path} must be a mapping")
    data["_recipe"] = str(path)
    return data


def train_config_from_recipe(recipe: dict[str, Any], overrides: dict[str, Any] | None = None) -> TrainConfig:
    allowed = {f.name for f in fields(TrainConfig)}
    payload = {k: v for k, v in recipe.items() if k in allowed}
    if overrides:
        payload.update({k: v for k, v in overrides.items() if k in allowed and v is not None})
    if "data" not in payload or not payload["data"]:
        raise ValueError("Recipe must set data: path/to/data.yaml")
    cfg = TrainConfig(**payload)
    return cfg


def dump_defaults() -> dict[str, Any]:
    return asdict(TrainConfig(data="datasets/coco/data.yaml"))
