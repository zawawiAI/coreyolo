"""Build a tiny Roboflow-layout dataset for pipeline smoke tests."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw


CLASS_NAMES = ["red_box", "blue_box"]
COLORS = [(220, 40, 40), (40, 80, 220)]


def write_dummy_dataset(
    root: str | Path, n_train: int = 32, n_val: int = 8, size: int = 320, segment: bool = False
) -> Path:
    """Create ``data.yaml`` + ``train|valid/{images,labels}`` with colored rectangles."""
    root = Path(root)
    for split, n in (("train", n_train), ("valid", n_val)):
        (root / split / "images").mkdir(parents=True, exist_ok=True)
        (root / split / "labels").mkdir(parents=True, exist_ok=True)
        for i in range(n):
            img = Image.new("RGB", (size, size), (240, 240, 240))
            draw = ImageDraw.Draw(img)
            labels = []
            for cls, color in enumerate(COLORS):
                x0 = 20 + (i * (17 + cls * 11) + cls * 40) % (size - 90)
                y0 = 20 + (i * (13 + cls * 9) + cls * 30) % (size - 90)
                w, h = 50 + (i * 3 + cls * 7) % 40, 40 + (i * 5 + cls * 5) % 40
                x1, y1 = x0 + w, y0 + h
                draw.rectangle([x0, y0, x1, y1], fill=color)
                if segment:
                    labels.append(
                        f"{cls} {x0 / size:.6f} {y0 / size:.6f} {x1 / size:.6f} {y0 / size:.6f} "
                        f"{x1 / size:.6f} {y1 / size:.6f} {x0 / size:.6f} {y1 / size:.6f}"
                    )
                else:
                    xc = (x0 + x1) / 2 / size
                    yc = (y0 + y1) / 2 / size
                    labels.append(f"{cls} {xc:.6f} {yc:.6f} {w / size:.6f} {h / size:.6f}")
            img.save(root / split / "images" / f"{split}_{i:04d}.jpg", quality=95)
            (root / split / "labels" / f"{split}_{i:04d}.txt").write_text("\n".join(labels) + "\n")

    yaml = (
        f"path: {root.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    (root / "data.yaml").write_text(yaml)
    return root / "data.yaml"
