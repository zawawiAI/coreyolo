from pathlib import Path

from coreyolo.cli import build_parser
from coreyolo.zoo import listed_release_files, load_manifest, record_metrics, write_metrics_json


def test_train_cli_family_choices() -> None:
    args = build_parser().parse_args(["train", "--data", "data.yaml", "--family", "e2e"])
    assert args.family == "e2e"
    args = build_parser().parse_args(["train", "--data", "data.yaml", "--family", "8"])
    assert args.family == "8"
    args = build_parser().parse_args(["train", "--data", "data.yaml", "--family", "dfl"])
    assert args.family == "dfl"


def test_coco_cli_max_images() -> None:
    args = build_parser().parse_args(
        ["coco", "--out", "datasets/coco", "--no-download", "--max-images", "512", "--splits", "val"]
    )
    assert args.cmd == "coco"
    assert args.max_images == 512
    assert args.download is False
    assert args.splits == "val"


def test_coco_cli_max_images_default_none() -> None:
    args = build_parser().parse_args(["coco", "--no-download"])
    assert args.max_images is None


def test_zoo_record_metrics(tmp_path: Path) -> None:
    src = Path("weights/manifest.json")
    dest = tmp_path / "manifest.json"
    dest.write_text(src.read_text())
    metrics = {"mAP50": 0.51, "mAP50-95": 0.37}
    record_metrics("coreyolo-n-coco", metrics, manifest=dest)
    catalog = load_manifest(dest)
    entry = next(m for m in catalog["models"] if m["id"] == "coreyolo-n-coco")
    assert entry["metrics"]["mAP50"] == 0.51
    assert entry["metrics"]["mAP50-95"] == 0.37
    assert entry["metrics"]["updated"]


def test_write_metrics_json(tmp_path: Path) -> None:
    path = write_metrics_json(tmp_path / "m.json", {"metrics": {"mAP50": 0.1}})
    assert path.is_file()
    assert "mAP50" in path.read_text()


def test_listed_release_files_empty_when_missing() -> None:
    files = listed_release_files("weights/manifest.json")
    assert isinstance(files, list)
