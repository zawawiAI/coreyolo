"""COCO 2017 → Roboflow YOLO layout (data.yaml + train|valid images/labels)."""

from __future__ import annotations

import json
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

COCO_80: list[tuple[int, str]] = [
    (1, "person"),
    (2, "bicycle"),
    (3, "car"),
    (4, "motorcycle"),
    (5, "airplane"),
    (6, "bus"),
    (7, "train"),
    (8, "truck"),
    (9, "boat"),
    (10, "traffic light"),
    (11, "fire hydrant"),
    (13, "stop sign"),
    (14, "parking meter"),
    (15, "bench"),
    (16, "bird"),
    (17, "cat"),
    (18, "dog"),
    (19, "horse"),
    (20, "sheep"),
    (21, "cow"),
    (22, "elephant"),
    (23, "bear"),
    (24, "zebra"),
    (25, "giraffe"),
    (27, "backpack"),
    (28, "umbrella"),
    (31, "handbag"),
    (32, "tie"),
    (33, "suitcase"),
    (34, "frisbee"),
    (35, "skis"),
    (36, "snowboard"),
    (37, "sports ball"),
    (38, "kite"),
    (39, "baseball bat"),
    (40, "baseball glove"),
    (41, "skateboard"),
    (42, "surfboard"),
    (43, "tennis racket"),
    (44, "bottle"),
    (46, "wine glass"),
    (47, "cup"),
    (48, "fork"),
    (49, "knife"),
    (50, "spoon"),
    (51, "bowl"),
    (52, "banana"),
    (53, "apple"),
    (54, "sandwich"),
    (55, "orange"),
    (56, "broccoli"),
    (57, "carrot"),
    (58, "hot dog"),
    (59, "pizza"),
    (60, "donut"),
    (61, "cake"),
    (62, "chair"),
    (63, "couch"),
    (64, "potted plant"),
    (65, "bed"),
    (67, "dining table"),
    (70, "toilet"),
    (72, "tv"),
    (73, "laptop"),
    (74, "mouse"),
    (75, "remote"),
    (76, "keyboard"),
    (77, "cell phone"),
    (78, "microwave"),
    (79, "oven"),
    (80, "toaster"),
    (81, "sink"),
    (82, "refrigerator"),
    (84, "book"),
    (85, "clock"),
    (86, "vase"),
    (87, "scissors"),
    (88, "teddy bear"),
    (89, "hair drier"),
    (90, "toothbrush"),
]

COCO_NAMES = [name for _, name in COCO_80]
COCO_ID_TO_IDX = {cid: i for i, (cid, _) in enumerate(COCO_80)}

URLS = {
    "train": "http://images.cocodataset.org/zips/train2017.zip",
    "val": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

SPLIT_FILES = {
    "train": "instances_train2017.json",
    "val": "instances_val2017.json",
}

YOLO_SPLIT = {"train": "train", "val": "valid"}


def coco_bbox_to_yolo(bbox: list[float], width: int, height: int) -> tuple[float, float, float, float] | None:
    """COCO ``xywh`` pixels → YOLO normalized center ``xywh``. Skip degenerate boxes."""
    x, y, w, h = (float(v) for v in bbox)
    if w <= 1 or h <= 1 or width <= 0 or height <= 0:
        return None
    xc = (x + w / 2) / width
    yc = (y + h / 2) / height
    nw, nh = w / width, h / height
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    nw = min(max(nw, 0.0), 1.0)
    nh = min(max(nh, 0.0), 1.0)
    if nw < 1e-4 or nh < 1e-4:
        return None
    return xc, yc, nw, nh


def write_data_yaml(root: Path) -> Path:
    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(COCO_NAMES))
    text = (
        f"path: {root.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(COCO_NAMES)}\n"
        "names:\n"
        f"{names}\n"
    )
    path = root / "data.yaml"
    path.write_text(text)
    return path


def convert_coco_json(
    json_path: Path,
    images_dir: Path,
    labels_dir: Path,
    max_images: int | None = None,
    skip_crowd: bool = True,
) -> dict[str, int]:
    """Write YOLO ``.txt`` labels next to a COCO instances JSON."""
    with json_path.open() as f:
        coco = json.load(f)
    id_to_img = {im["id"]: im for im in coco["images"]}
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco.get("annotations", []):
        if skip_crowd and ann.get("iscrowd", 0):
            continue
        if ann.get("category_id") not in COCO_ID_TO_IDX:
            continue
        anns_by_img[ann["image_id"]].append(ann)

    labels_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = n_skip = 0
    image_ids = sorted(id_to_img)
    if max_images:
        image_ids = image_ids[:max_images]
    for image_id in image_ids:
        im = id_to_img[image_id]
        stem = Path(im["file_name"]).stem
        w, h = int(im["width"]), int(im["height"])
        lines = []
        for ann in anns_by_img.get(image_id, []):
            yolo = coco_bbox_to_yolo(ann["bbox"], w, h)
            if yolo is None:
                n_skip += 1
                continue
            cls = COCO_ID_TO_IDX[ann["category_id"]]
            xc, yc, bw, bh = yolo
            lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1
        (labels_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_img += 1
        _ = images_dir  # layout caller is responsible for image files / symlinks
    return {"images": n_img, "boxes": n_box, "skipped_boxes": n_skip}


def _download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 0:
        print(f"already have {dest}")
        return dest
    print(f"downloading {url} -> {dest}")
    tmp = dest.with_suffix(dest.suffix + ".part")

    def _progress(block: int, block_size: int, total: int) -> None:
        if total <= 0:
            return
        done = min(block * block_size, total)
        pct = 100 * done / total
        print(f"\r  {pct:5.1f}%  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    tmp.replace(dest)
    return dest


def _unzip(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"unzipping {zip_path} -> {out_dir}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)


def _link_or_copy_images(src: Path, dst: Path, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copytree(src, dst)
        return
    dst.symlink_to(src.resolve(), target_is_directory=True)


def prepare_coco(
    root: str | Path = "datasets/coco",
    *,
    download: bool = True,
    splits: tuple[str, ...] = ("train", "val"),
    copy_images: bool = False,
    max_images: int | None = None,
) -> Path:
    """Download COCO 2017 and emit a Roboflow-compatible YOLO tree.

    Result::

        datasets/coco/
          data.yaml
          train/images  train/labels
          valid/images  valid/labels
          raw/ ...
    """
    root = Path(root).expanduser().resolve()
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    if download:
        if "train" in splits:
            print("train2017.zip is ~18 GB")
            _download(URLS["train"], raw / "train2017.zip")
        if "val" in splits:
            print("val2017.zip is ~1 GB")
            _download(URLS["val"], raw / "val2017.zip")
        _download(URLS["annotations"], raw / "annotations_trainval2017.zip")
        if "train" in splits and not (raw / "train2017").is_dir():
            _unzip(raw / "train2017.zip", raw)
        if "val" in splits and not (raw / "val2017").is_dir():
            _unzip(raw / "val2017.zip", raw)
        if not (raw / "annotations").is_dir():
            _unzip(raw / "annotations_trainval2017.zip", raw)

    for split in splits:
        yolo_split = YOLO_SPLIT[split]
        json_name = SPLIT_FILES[split]
        json_path = raw / "annotations" / json_name
        if not json_path.is_file():
            raise FileNotFoundError(
                f"Missing {json_path}. Pass --download or place COCO annotations under {raw / 'annotations'}"
            )
        img_src = raw / f"{'train' if split == 'train' else 'val'}2017"
        img_dst = root / yolo_split / "images"
        lab_dst = root / yolo_split / "labels"
        if img_src.is_dir():
            _link_or_copy_images(img_src, img_dst, copy=copy_images)
        else:
            img_dst.mkdir(parents=True, exist_ok=True)
        stats = convert_coco_json(json_path, img_dst, lab_dst, max_images=max_images)
        print(f"{split}: {stats}")

    yaml_path = write_data_yaml(root)
    print(f"wrote {yaml_path}")
    return yaml_path
