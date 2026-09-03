from pathlib import Path

from coreyolo.data.dummy import write_dummy_dataset
from coreyolo.data.yaml import YOLODatasetYAML
from coreyolo.data.dataset import load_yolo_labels, load_yolo_instances


def test_roboflow_yaml_layout(tmp_path: Path) -> None:
    yaml_path = write_dummy_dataset(tmp_path, n_train=4, n_val=2, size=64)
    spec = YOLODatasetYAML(yaml_path)
    assert spec.nc == 2
    assert spec.names == ["red_box", "blue_box"]
    train_img, train_lab = spec.split("train")
    val_img, val_lab = spec.split("val")
    assert (train_img / "train_0000.jpg").is_file()
    assert (train_lab / "train_0000.txt").is_file()
    assert (val_img / "valid_0000.jpg").is_file()
    labels = load_yolo_labels(train_lab / "train_0000.txt")
    assert labels.shape[1] == 5
    assert labels[:, 1:].min() >= 0
    assert labels[:, 1:].max() <= 1


def test_roboflow_dict_names(tmp_path: Path) -> None:
    yaml_path = write_dummy_dataset(tmp_path, n_train=1, n_val=1, size=32)
    text = yaml_path.read_text()
    yaml_path.write_text(
        text.replace("names: ['red_box', 'blue_box']", "names:\n  0: red_box\n  1: blue_box")
        if "names: ['red_box'" in text
        else text.replace('names: ["red_box", "blue_box"]', "names:\n  0: red_box\n  1: blue_box")
    )
    spec = YOLODatasetYAML(yaml_path)
    assert spec.names == ["red_box", "blue_box"]


def test_seg_label_falls_back_to_box(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "a.txt"
    p.write_text("0 0.5 0.5 0.2 0.2 0.4 0.4 0.6 0.4 0.6 0.6 0.4 0.6\n")
    labels = load_yolo_labels(p)
    assert list(labels[0]) == [0, 0.5, 0.5, 0.2, 0.2]


def test_seg_polygon_instances(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("0 0.2 0.2 0.8 0.2 0.8 0.8 0.2 0.8\n")
    labels, polys = load_yolo_instances(p)
    assert labels.shape == (1, 5)
    assert abs(float(labels[0, 1]) - 0.5) < 1e-5
    assert abs(float(labels[0, 3]) - 0.6) < 1e-5
    assert polys[0].shape == (4, 2)
