import json
from pathlib import Path

from coreyolo.data.coco import (
    COCO_ID_TO_IDX,
    COCO_NAMES,
    coco_bbox_to_yolo,
    convert_coco_json,
    write_data_yaml,
)
from coreyolo.data.yaml import YOLODatasetYAML
from coreyolo.train.recipe import load_recipe, train_config_from_recipe


def test_coco_80_mapping() -> None:
    assert len(COCO_NAMES) == 80
    assert COCO_NAMES[0] == "person"
    assert COCO_NAMES[2] == "car"
    assert COCO_ID_TO_IDX[1] == 0
    assert COCO_ID_TO_IDX[90] == 79
    assert 12 not in COCO_ID_TO_IDX


def test_coco_bbox_to_yolo() -> None:
    box = coco_bbox_to_yolo([50, 40, 100, 80], 200, 160)
    assert box is not None
    xc, yc, w, h = box
    assert abs(xc - 0.5) < 1e-6
    assert abs(yc - 0.5) < 1e-6
    assert abs(w - 0.5) < 1e-6
    assert abs(h - 0.5) < 1e-6
    assert coco_bbox_to_yolo([0, 0, 0, 10], 100, 100) is None


def test_convert_coco_json(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    coco = {
        "images": [
            {"id": 7, "file_name": "000000000007.jpg", "width": 100, "height": 50},
            {"id": 8, "file_name": "000000000008.jpg", "width": 80, "height": 80},
        ],
        "annotations": [
            {"image_id": 7, "category_id": 3, "bbox": [10, 5, 20, 10], "iscrowd": 0},
            {"image_id": 7, "category_id": 1, "bbox": [0, 0, 10, 10], "iscrowd": 1},
            {"image_id": 8, "category_id": 12, "bbox": [1, 1, 10, 10], "iscrowd": 0},
        ],
        "categories": [{"id": 3, "name": "car"}],
    }
    js = tmp_path / "instances.json"
    js.write_text(json.dumps(coco))
    labels = tmp_path / "labels"
    stats = convert_coco_json(js, tmp_path / "images", labels)
    assert stats["images"] == 2
    assert stats["boxes"] == 1
    lines = (labels / "000000000007.txt").read_text().strip().splitlines()
    assert len(lines) == 1
    cls = int(lines[0].split()[0])
    assert cls == COCO_ID_TO_IDX[3]
    assert (labels / "000000000008.txt").read_text() == ""


def test_coco_data_yaml_roundtrip(tmp_path: Path) -> None:
    (tmp_path / "train" / "images").mkdir(parents=True)
    (tmp_path / "train" / "labels").mkdir(parents=True)
    (tmp_path / "valid" / "images").mkdir(parents=True)
    (tmp_path / "valid" / "labels").mkdir(parents=True)
    (tmp_path / "train" / "images" / "a.jpg").write_bytes(b"")
    yaml_path = write_data_yaml(tmp_path)
    spec = YOLODatasetYAML(yaml_path)
    assert spec.nc == 80
    assert spec.names[0] == "person"
    assert spec.names[79] == "toothbrush"


def test_coco_n_recipe() -> None:
    recipe = load_recipe("coco-n")
    cfg = train_config_from_recipe(recipe, {"epochs": 2, "batch": 4})
    assert cfg.model == "n"
    assert cfg.epochs == 2
    assert cfg.batch == 4
    assert cfg.imgsz == 640
    assert cfg.data.endswith("coco/data.yaml")
    assert cfg.val_period == 5
    assert cfg.close_mosaic == 10
    assert cfg.family == "gelan"


def test_coco_n_e2e_recipe() -> None:
    recipe = load_recipe("coco-n-e2e")
    cfg = train_config_from_recipe(recipe)
    assert cfg.family == "e2e"
    assert cfg.dfl == 0.0
    alias = train_config_from_recipe(load_recipe("coco-n-26"))
    assert alias.family == "e2e"


def test_recipes_ship_inside_package() -> None:
    from coreyolo.train.recipe import recipe_path

    path = recipe_path("coco-n")
    assert path.is_file()
    assert "coreyolo" in path.parts
    assert path.name == "coco-n.yaml"


def test_native_recipes_use_relu() -> None:
    assert train_config_from_recipe(load_recipe("coco-n")).act == "relu"
    assert train_config_from_recipe(load_recipe("coco-s")).act == "relu"
    assert train_config_from_recipe(load_recipe("coco-m")).act == "relu"
    assert train_config_from_recipe(load_recipe("coco-l")).act == "relu"
