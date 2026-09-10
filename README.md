# CoreYOLO

Independent object detection, trained in PyTorch and **deployed with Core ML on Apple silicon**.

CoreYOLO is original code under MIT. It reimplements published detector methods (CSP backbone, PAN-FPN, decoupled DFL head) and the Roboflow YOLO label layout. It is **not** a fork of Ultralytics, **not** an Ultralytics product, and **not** affiliated with Ultralytics or Apple. YOLO, YOLOv9, and Ultralytics are trademarks of their owners — used here only to name third-party files and license duties.

The Python type is `from coreyolo import Detector`. `YOLO` is a compatibility alias.

This is not legal advice, and these notices do not mean nobody can sue. Details: [docs/licenses.md](docs/licenses.md).

## Why CoreYOLO

- **Fully open source** (MIT), including train, export, and inference.
- **Core ML first**: fused Conv-BN, ReLU by default, static `imgsz`, FP16 ML Program with 8-bit palettized weights. Converted SiLU graphs prefer GPU (`--device gpu`); ReLU prefers ANE (`--device ane` / iOS `.cpuAndNeuralEngine`).
- **Three detect graphs**: `--family gelan` is **GELAN** as in the YOLOv9 paper (default; scale `n` matches the public Ultralytics `yolov9t` file). `--family dfl` is C2f + Distribution Focal Loss (host NMS). `--family e2e` is **C3k2 + C2PSA + NMS-free** top-300.
- **Instance segmentation**: `--task segment` adds a proto mask branch. Roboflow **YOLO-Seg** polygon labels work; detect stays the default.
- **Roboflow YOLO labels**: drop in a Roboflow **YOLO** export (`data.yaml` + `train|valid/{images,labels}`).

## Python SDK

App code loads **CoreYOLO** files only (``.coreyolo``, trained ``best.pt``, or ``.mlpackage``). Do not pass Ultralytics ``yolov9t.pt`` into ``Detector()`` — that layout is rejected (AGPL-3.0 pickle, not a CoreYOLO checkpoint).

```python
from coreyolo import Detector

model = Detector("n")
model.train(data="datasets/dummy/data.yaml", epochs=20, batch=8, imgsz=320)
model.save("weights/app.coreyolo")

model = Detector("weights/app.coreyolo")  # or a Core ML .mlpackage
results = model.predict("photo.jpg", classes="person", save=True)
results[0].save("out.jpg")
model.export(imgsz=320)
```

`model.val(data="data.yaml")` returns mAP. `model.predict(0)` opens the webcam. Optional one-time bootstrap (not in the app): `coreyolo convert --weights yolov9t.pt --out weights/coreyolo-n-coco.coreyolo`.

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

In Roboflow, label bounding boxes as usual, then **Export → YOLO** (detect txt). Unzip so the tree looks like this:

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

Default activation is **ReLU** so fused Conv-BN-ReLU maps onto the Neural Engine. Native recipes (`coco-n` / `coco-s` / `coco-m` / `coco-l`) all train ReLU. **SiLU** is only for converted YOLOv9 tensors — do not swap it on those checkpoints. **StarReLU** (`--act star`) is a 2024 ReLU-family option that still maps to the Neural Engine. **GELU** remains available via `--act gelu`.

`--family gelan` (default, aliases `9` / `v9`): YOLOv9 GELAN, DFL head; scale `n` is Ultralytics `yolov9t` (there is no `yolov9n.yaml`). `--family dfl`: C2f backbone, DFL `reg_max=16`, host NMS. `--family e2e`: C3k2 backbone, residual SPPF, C2PSA, dual one-to-many / one-to-one head, DFL removed, NMS-free top-300 at inference.

Loss: task-aligned assignment, CIoU, Distribution Focal Loss (DFL graph), BCE classification. E2E trains both heads (TAL top-k and TAL top-1). Segmentation adds cropped mask BCE against proto coefficients.

`--task segment` swaps the detect head for a Segment head (32 proto maps on P3, per-anchor coefficients). Detect checkpoints and YOLOv9 conversion are unchanged.

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
| `coco-n` | n / GELAN | 300 | 16 | 5 | default pretrained |
| `coco-n-e2e` | n / E2E | 300 | 16 | 5 | C3k2 + C2PSA, NMS-free |
| `coco-s` | s | 300 | 8 | 5 | ReLU; drop batch if RAM-bound |
| `coco-m` | m | 300 | 6 | 10 | ReLU |
| `coco-l` | l | 300 | 4 | 10 | ReLU; ~YOLOv9-c size; drop to batch 2 if RAM-bound |

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

YOLO26n / YOLO11n **weights** cannot be copied (different tensors, AGPL checkpoint). CoreYOLO **E2E** is an original C3k2 + C2PSA + NMS-free graph you train yourself (`--family e2e` or `--recipe coco-n-e2e`). Compact **YOLOv9** (`yolov9t` / s / m / c) maps onto GELAN — nano is Ultralytics `yolov9t.pt` (no `yolov9n.yaml`):

```bash
coreyolo convert --weights yolov9t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights yolov9c.pt --out weights/coreyolo-l-coco.coreyolo
coreyolo predict --weights weights/coreyolo-n-coco.coreyolo --source photo.jpg --device gpu
coreyolo export --weights weights/coreyolo-n-coco.coreyolo --imgsz 640
```

That convert step is optional and one-shot. It remaps **names** onto a CoreYOLO file (`format: coreyolo`, `stem.*` tensors, SiLU). It does **not** relicense the Ultralytics tensors (AGPL-3.0). See [Licenses](docs/licenses.md).

## CLI

```
coreyolo coco        --out datasets/coco --download [--max-images 512]
coreyolo convert     --weights yolov9t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo train       --recipe coco-n-fast
coreyolo train       --data data.yaml --model n --family e2e --epochs 100 --device gpu
coreyolo val         --data data.yaml --weights best.pt --device gpu [--json metrics.json --zoo-id coreyolo-n-coco]
coreyolo predict     --weights best.mlpackage --source photo.jpg --device gpu
coreyolo predict     --weights weights/coreyolo-n-coco.mlpackage --source 0 --device gpu
coreyolo export      --weights best.pt --imgsz 640 [--int8] [--no-palette]
coreyolo train       --data data.yaml --model n --task segment --epochs 100 --device gpu
coreyolo dummy-data  --out datasets/dummy --segment
```

## License

**CoreYOLO code** is [MIT](LICENSE): you may view, share, modify, and distribute it, including in a closed app, subject to the MIT copyright and permission notice. Train **from scratch** on your own labels for an MIT **weight** path. This repo is not a fork of Ultralytics and does not include Ultralytics source.

**Ultralytics YOLOv9** is a separate project. Its source is open under **AGPL-3.0** (view, share, modify, distribute). The AGPL catch: if you **modify** YOLOv9 or offer it as a **network service (SaaS)** so users interact with it over a network, you typically must release **your whole application** under AGPL-3.0. For a **commercial product** or a **closed-source internal enterprise tool** without publishing that source, Ultralytics sells a [commercial license](https://www.ultralytics.com/license). `coreyolo convert` remaps names only — converted tensors, fine-tunes, and Core ML exports of those files stay AGPL-3.0. Do not rehost them as MIT. YOLO / YOLOv9 / Ultralytics are their trademarks. Prefer `Detector` in new code. Details: [docs/licenses.md](docs/licenses.md). This is not legal advice.
