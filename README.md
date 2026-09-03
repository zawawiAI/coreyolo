# CoreYOLO

Open-source YOLO-style object detection, trained in PyTorch and **deployed with Core ML on Apple silicon**.

CoreYOLO is original code under Apache-2.0. It is inspired by the YOLO detector family (CSP backbone, PAN-FPN, decoupled DFL head) and by the Roboflow YOLO label layout. It is not a fork of Ultralytics and is not affiliated with Ultralytics or Apple.

## Why CoreYOLO

- **Fully open source** (Apache-2.0), including train, export, and inference.
- **Core ML first**: fused Conv-BN, ReLU by default, static `imgsz`, FP16 ML Program. Inference defaults to the **GPU** (`CPU_AND_GPU`); pass `--device all` for GPU+ANE or `--device ane` for Neural Engine only.
- **Two detect graphs**: `--family dfl` is C2f + Distribution Focal Loss (host NMS). `--family e2e` is **C3k2 + C2PSA + NMS-free** top-300.
- **Instance segmentation**: `--task segment` adds a proto mask branch (YOLOv8-seg style). Roboflow **YOLO-Seg** polygon labels work; detect stays the default.
- **Roboflow YOLO labels**: drop in a Roboflow **YOLOv5 / YOLOv8 / YOLOv11** export (`data.yaml` + `train|valid/{images,labels}`).

## Python SDK

App code loads **CoreYOLO** files only (``.coreyolo``, trained ``best.pt``, or ``.mlpackage``). Do not pass Ultralytics ``yolov8n.pt`` into ``YOLO()`` — that layout is rejected.

```python
from coreyolo import YOLO

model = YOLO("n")
model.train(data="datasets/dummy/data.yaml", epochs=20, batch=8, imgsz=320)
model.save("weights/app.coreyolo")

model = YOLO("weights/app.coreyolo")  # or a Core ML .mlpackage
results = model.predict("photo.jpg", classes="person", save=True)
results[0].save("out.jpg")
model.export(imgsz=320)
```

`model.val(data="data.yaml")` returns mAP. `model.predict(0)` opens the webcam. Optional one-time bootstrap (not in the app): `coreyolo convert --weights yolov8n.pt --out weights/coreyolo-n-coco.coreyolo`.

## Install

```bash
pip install coreyolo
```

From a clone, for development:

```bash
cd CoreYOLO
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Apple silicon: PyTorch **train / val / predict** use **MPS (GPU)** whenever it actually works, then CUDA, then CPU. Core ML `.mlpackage` inference defaults to **GPU** (`CPU_AND_GPU`); `--device all` adds the Neural Engine, `--device ane` is ANE-only. Pass `--device cpu` to force CPU.

## Roboflow labelling

In Roboflow, label bounding boxes as usual, then **Export → YOLOv8** (YOLOv5 / YOLOv11 YOLO txt is the same detect format). Unzip so the tree looks like this:

```
dataset/
  data.yaml
  train/images/   train/labels/
  valid/images/   valid/labels/
  test/images/    test/labels/      # optional
```

Each label file is one image stem, one object per line, **normalized**:

```
class_id  x_center  y_center  width  height
```

`data.yaml` may use list or dict `names`, `val` or Roboflow's `valid`, and relative or leftover Colab paths — CoreYOLO resolves all of those against the yaml file.

```bash
coreyolo train --data /path/to/dataset/data.yaml --model n --epochs 100 --imgsz 640 --device gpu
```

### Instance segmentation

Export **YOLO-Seg** from Roboflow (or write `cls x1 y1 x2 y2 ... xn yn` polygons, normalized). Train with `--task segment`. Mosaic is off (polygons are rasterized after letterbox). Checkpoints store `task: segment`; `predict` overlays masks on the boxes.

```bash
coreyolo dummy-data --out datasets/dummy-seg --segment
coreyolo train --data datasets/dummy-seg/data.yaml --model n --task segment --epochs 20 --batch 8 --imgsz 320
coreyolo predict --weights runs/segment/train/weights/best.pt --source datasets/dummy-seg/valid/images
```

DFL is the supported seg graph. E2E can train a proto head too; Core ML mask export is three outputs (`detections`, `mask_coeff`, `proto`) with NMS / mask assembly on the host.

## Train → Core ML → predict

```bash
# optional: synthetic Roboflow-layout set to verify the pipeline
coreyolo dummy-data --out datasets/dummy

coreyolo train --data datasets/dummy/data.yaml --model n --epochs 20 --batch 8 --imgsz 320
coreyolo train --data datasets/dummy/data.yaml --model n --family e2e --epochs 20 --batch 8 --imgsz 320

coreyolo val --data datasets/dummy/data.yaml --weights runs/detect/train/weights/best.pt

coreyolo export --weights runs/detect/train/weights/best.pt --imgsz 320

coreyolo predict --weights runs/detect/train/weights/best.mlpackage --source datasets/dummy/valid/images
```

`export` writes an `.mlpackage` you can also drop into Xcode. DFL outputs decoded `xywh` plus class scores and **NMS runs on the host**. E2E exports NMS-free `(1, 300, 6)` `xyxy, conf, cls`.

### Swift (iOS / macOS)

Use the camera sample in [`examples/ios`](examples/ios): letterbox → Core ML → host NMS, same pad/ratio as Python. Drop an exported `CoreYOLO.mlpackage` into the Xcode target.

```swift
import CoreML

let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU            // GPU; .all = GPU+ANE; .cpuAndNeuralEngine = ANE
let model = try MLModel(contentsOf: url, configuration: config)
// Letterbox the frame to imgsz×imgsz RGB (pad 114), predict,
// then class-aware NMS on the (4+nc, N) output for DFL.
```

Do not use `VNRecognizedObjectObservation` with this graph. Outputs are raw tensors; NMS stays in Swift (`examples/ios/CoreYOLODemo/HostNMS.swift`).

## Model

| Scale | Depth | Width | Typical use        |
|-------|-------|-------|--------------------|
| `n`   | 0.33  | 0.25  | phones, real-time  |
| `s`   | 0.33  | 0.50  | Mac / iPad         |
| `m`   | 0.67  | 0.75  | accuracy           |
| `l`   | 1.00  | 1.00  | offline            |
| `x`   | 1.00  | 1.25  | max accuracy       |

Default activation is **ReLU** so fused Conv-BN-ReLU maps onto the Neural Engine. Pass `--act silu` if you care more about train-time accuracy than ANE mapping.

`--family dfl` (default): C2f backbone, DFL `reg_max=16`, host NMS. `--family e2e`: C3k2 backbone, residual SPPF, C2PSA, dual one-to-many / one-to-one head, DFL removed, NMS-free top-300 at inference.

Loss: task-aligned assignment, CIoU, Distribution Focal Loss (DFL graph), BCE classification. E2E trains both heads (TAL top-k and TAL top-1). Segmentation adds cropped mask BCE against proto coefficients.

`--task segment` swaps the detect head for a Segment head (32 proto maps on P3, per-anchor coefficients). Detect checkpoints and YOLOv8n conversion are unchanged.

## Train on COCO

Official COCO 2017 is converted into the same Roboflow YOLO layout (`data.yaml` + `train|valid/{images,labels}`), 80 contiguous classes, crowd boxes dropped.

```bash
# val2017 is ~1 GB; train2017 is ~18 GB. Images are symlinked, not copied.
coreyolo coco --out datasets/coco --download

# 100-epoch sanity pass (CoreYOLO-n, 640, batch 16 on 24 GB Apple silicon)
coreyolo train --recipe coco-n-fast

# full 300-epoch recipe
coreyolo train --recipe coco-n

# then Core ML
coreyolo val --data datasets/coco/data.yaml --weights runs/detect/coco-n/weights/best.pt
coreyolo export --weights runs/detect/coco-n/weights/best.pt --imgsz 640
```

Recipes live in `configs/recipes/`:

| Recipe | Scale | Epochs | Batch | Val every | Notes |
|--------|-------|--------|-------|-----------|--------|
| `coco-n-fast` | n | 100 | 16 | 5 | first COCO run |
| `coco-n` | n / DFL | 300 | 16 | 5 | default pretrained |
| `coco-n-e2e` | n / E2E | 300 | 16 | 5 | C3k2 + C2PSA, NMS-free |
| `coco-s` | s | 300 | 8 | 5 | drop batch if RAM-bound |
| `coco-m` | m | 300 | 6 | 10 | |

SGD + cosine LR (`lr0=0.01`, `lrf=0.01`), 3-epoch warmup, mosaic until the last 10 epochs, weight decay on conv weights only. Override any field from the CLI: `--recipe coco-n --batch 8 --device gpu --epochs 50`.

`--max-images 512` on `coreyolo coco` builds a tiny split for pipeline debugging without the full 118k train set.

```bash
coreyolo coco --out datasets/coco --download --splits val --max-images 512
```

### Zoo (GitHub Releases)

Checkpoints and `.mlpackage` files are **not** in git. The catalog is [`weights/manifest.json`](weights/manifest.json). After you convert or train, record COCO val (do not invent mAP):

```bash
coreyolo val --data datasets/coco/data.yaml --weights weights/coreyolo-n-coco.coreyolo \
  --json weights/metrics-coreyolo-n-coco.json --zoo-id coreyolo-n-coco
```

Publish with the **Release zoo** workflow, or see [`weights/README.md`](weights/README.md).

### COCO weights from Ultralytics

YOLO26n / YOLO11n **weights** cannot be copied (different tensors, AGPL checkpoint). CoreYOLO **E2E** is an original C3k2 + C2PSA + NMS-free graph you train yourself (`--family e2e` or `--recipe coco-n-e2e`). YOLOv8n **can** map onto DFL — same C2f + DFL graph, 355/355 tensors:

```bash
coreyolo convert --weights yolov8n.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo predict --weights weights/coreyolo-n-coco.coreyolo --source photo.jpg --device gpu
coreyolo export --weights weights/coreyolo-n-coco.coreyolo --imgsz 640
```

That convert step is optional and one-shot. The file you ship is CoreYOLO (``format: coreyolo``, ``stem.*`` tensors), not Ultralytics. The converted checkpoint uses **SiLU** (how YOLOv8 was trained).

## CLI

```
coreyolo coco        --out datasets/coco --download [--max-images 512]
coreyolo convert     --weights yolov8n.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo train       --recipe coco-n-fast
coreyolo train       --data data.yaml --model n --family e2e --epochs 100 --device gpu
coreyolo val         --data data.yaml --weights best.pt --device gpu [--json metrics.json --zoo-id coreyolo-n-coco]
coreyolo predict     --weights best.mlpackage --source photo.jpg --device gpu
coreyolo predict     --weights weights/coreyolo-n-coco.mlpackage --source 0 --device gpu
coreyolo export      --weights best.pt --imgsz 640 [--int8]
coreyolo train       --data data.yaml --model n --task segment --epochs 100 --device gpu
coreyolo dummy-data  --out datasets/dummy --segment
```

## License

Apache License 2.0. See [LICENSE](LICENSE).
