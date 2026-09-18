<p align="center">
  <strong>CoreYOLO</strong><br />
  <sub>Object detection. Built for Apple.</sub>
</p>

<p align="center">
  <a href="https://pypi.org/project/coreyolo/"><img alt="PyPI" src="https://img.shields.io/pypi/v/coreyolo"></a>
  <a href="https://pypi.org/project/coreyolo/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/coreyolo"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/github/license/zawawiAI/coreyolo"></a>
  <a href="https://github.com/zawawiAI/coreyolo/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/zawawiAI/coreyolo/ci.yml?branch=main&label=CI"></a>
</p>

<p align="center">
  Train an independent detector in Python. Ship a palettized Core ML package to Apple silicon.<br />
  Code is MIT. Weights carry their own license, stated per checkpoint.
</p>

<p align="center">
  <img src="website/img/scene.jpg" alt="Street scene used on the CoreYOLO site" width="720" />
</p>

<p align="center">
  <a href="https://zawawiai.github.io/coreyolo/">Website</a>
  · <a href="https://zawawiai.github.io/coreyolo/docs.html">Docs</a>
  · <a href="https://zawawiai.github.io/coreyolo/models.html">Models</a>
  · <a href="https://zawawiai.github.io/coreyolo/deploy.html">Deploy</a>
  · <a href="examples/ios">iOS sample</a>
  · <a href="docs/licenses.md">Licenses</a>
</p>

```text
Roboflow data.yaml  →  train  →  .coreyolo  →  export  →  Core ML  →  iPhone / Mac
```

The public type is `Detector` (`YOLO` is a compatibility alias). App code loads CoreYOLO files only (`.coreyolo`, trained `best.pt`, or `.mlpackage`).

## Install

```bash
pip install coreyolo
```

```python
from coreyolo import Detector

model = Detector("n")              # GELAN n; also "s" / "m" / "l", or family="e2e"
model.train(data="data.yaml", epochs=100, imgsz=640)
model.save("weights/app.coreyolo")
model.export(imgsz=640)            # FP16 Core ML + 8-bit palettes, ANE for ReLU
```

## Designed for Apple silicon

| | |
| --- | --- |
| **Core ML first** | Fused Conv–BN, inference branch only, cached DFL grids (no `meshgrid` / `arange`), host NMS. FP16 ML Program with 8-bit palettes. ReLU converts for the Neural Engine. iOS 16 and macOS 13. |
| **MIT** | Code is MIT. Weights are per checkpoint. GELAN files from MultimediaTechLab `v9-*.pt` are MIT (keep the Wong/Tsui copyright). Ultralytics `yolov9*.pt` stays AGPL-3.0; converting the file does not change its applicable terms. Choosing a model means choosing its license. |
| **Your labels** | Drop in a Roboflow YOLO or YOLO-Seg export. Same `data.yaml` you already have. |

ReLU prefers ANE (`--device ane`). Converted SiLU graphs prefer GPU.

## Three graphs. One trainer.

| Graph | What it is | Ship path |
| --- | --- | --- |
| **GELAN** (default) | YOLOv9 GELAN. Scales n / s / m / l match public `v9-t` / `s` / `m` / `c`. Export is the main branch only (no PGI aux). | Train ReLU for ANE, or convert MultimediaTechLab `v9-*.pt` (SiLU, GPU, MIT). |
| **DFL** | C2f + Distribution Focal Loss. Host NMS. Detect and instance segmentation. | `--family dfl` |
| **E2E** | Original C3k2 + C2PSA. NMS-free top-300 at inference. | Train from scratch (`--family e2e` or `--recipe coco-n-e2e`). |

## n, s, m, and l

Same GELAN trainer. Pick the scale for the device.

| Scale | Params (detect, nc=80) | Typical use | Convert (MIT) |
| --- | --- | --- | --- |
| **n** | 2.13M GELAN · 3.16M DFL · 2.73M E2E | iPhone, real-time camera | `v9-t.pt` |
| **s** | 7.32M GELAN · 11.17M DFL · 10.65M E2E | Mac / iPad | `v9-s.pt` |
| **m** | 20.22M GELAN · 25.90M DFL · 22.63M E2E | Accuracy-biased | `v9-m.pt` |
| **l** | 25.59M GELAN · 43.69M DFL · 27.17M E2E | Offline / max accuracy | `v9-c.pt` |

```bash
coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights v9-s.pt --out weights/coreyolo-s-coco.coreyolo
coreyolo convert --weights v9-m.pt --out weights/coreyolo-m-coco.coreyolo
coreyolo convert --weights v9-c.pt --out weights/coreyolo-l-coco.coreyolo
```

Source: [MultimediaTechLab/YOLO v1.0-alpha](https://github.com/MultimediaTechLab/YOLO/releases/tag/v1.0-alpha) (copyright Kin-Yiu Wong and Hao-Tang Tsui). Same MIT files LibreYOLO converts. Ultralytics `yolov9t.pt` is a different dump (AGPL-3.0). Do not pass those pickles into `Detector()`.

## Zoo

Binaries live on GitHub Releases, not in git. mAP stays blank until you record COCO val. See [`weights/manifest.json`](weights/manifest.json) and [`weights/LICENSE_NOTICE.txt`](weights/LICENSE_NOTICE.txt). Choosing a model means choosing its license.

| Id | Graph | Act | Compute | How | Licenses |
| --- | --- | --- | --- | --- | --- |
| `coreyolo-n-coco` | gelan | SiLU | GPU | `convert --weights v9-t.pt` | Code MIT, weights MIT |
| `coreyolo-s-coco` | gelan | SiLU | GPU | `convert --weights v9-s.pt` | Code MIT, weights MIT |
| `coreyolo-m-coco` | gelan | SiLU | GPU | `convert --weights v9-m.pt` | Code MIT, weights MIT |
| `coreyolo-l-coco` | gelan | SiLU | GPU | `convert --weights v9-c.pt` | Code MIT, weights MIT |
| `coreyolo-n-coco-seg` | gelan | ReLU | ANE | `train --task segment` | Code MIT, weights MIT |
| `coreyolo-n-coco-relu` | gelan | ReLU | ANE | `train --recipe coco-n` | Code MIT, weights MIT |
| `coreyolo-e2e-n-coco` | e2e | ReLU | ANE | `train --recipe coco-n-e2e` | Code MIT, weights MIT |

## Python SDK

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

`model.val(data="data.yaml")` returns mAP. `model.predict(0)` opens the webcam.

From a clone:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Apple silicon: PyTorch train / val / predict use **MPS (GPU)** when it works, then CUDA, then CPU. Core ML `.mlpackage` inference defaults to **GPU** (`CPU_AND_GPU`); `--device all` adds the Neural Engine, `--device ane` is ANE-only.

## Labels

Roboflow **Export → YOLO** (detect) or **YOLO-Seg** (polygons):

```
dataset/
  data.yaml
  train/images/   train/labels/
  valid/images/   valid/labels/
```

Detect lines are normalized `class xc yc w h`. `data.yaml` may use list or dict `names`, `val` or `valid`.

```bash
coreyolo dummy-data --out datasets/dummy
coreyolo train --data datasets/dummy/data.yaml --model n --epochs 20 --imgsz 320
coreyolo export --weights runs/detect/train/weights/best.pt --imgsz 320
```

Segmentation: `--task segment`. Mosaic is off. Core ML writes three outputs; NMS and mask assembly stay on the host. iOS sample: [`examples/ios`](examples/ios).

## Train on COCO

```bash
coreyolo coco --out datasets/coco --download
coreyolo train --recipe coco-n-fast    # 100 epochs
coreyolo train --recipe coco-n         # 300 epochs, ReLU (ANE)
coreyolo train --recipe coco-n-e2e
```

| Recipe | Scale | Notes |
| --- | --- | --- |
| `coco-n-fast` | n | First COCO run |
| `coco-n` / `coco-s` / `coco-m` / `coco-l` | GELAN ReLU | Neural Engine path |
| `coco-n-e2e` | n E2E | NMS-free |

## CLI

```
coreyolo coco        --out datasets/coco --download [--max-images 512]
coreyolo convert     --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo train       --recipe coco-n-fast
coreyolo train       --data data.yaml --model n --family e2e --epochs 100 --device gpu
coreyolo val         --data data.yaml --weights best.pt --device gpu
coreyolo predict     --weights best.mlpackage --source photo.jpg --device gpu
coreyolo export      --weights best.pt --imgsz 640 [--int8] [--no-palette]
coreyolo dummy-data  --out datasets/dummy --segment
```

## License

CoreYOLO's code is [MIT](LICENSE). It does not require you to open source your application, and it does not change if you sell what you build.

Pretrained weights are separate: each one carries the license of whoever trained it, stated per checkpoint. GELAN COCO files converted from MultimediaTechLab `v9-*.pt` are MIT (copyright Kin-Yiu Wong and Hao-Tang Tsui). Ultralytics `yolov9*.pt` converts stay AGPL-3.0. A model you train yourself is yours. Converting a file does not change its applicable terms.

Choosing a model means choosing its license. See [docs/licenses.md](docs/licenses.md), [weights/LICENSE_NOTICE.txt](weights/LICENSE_NOTICE.txt), and [NOTICE](NOTICE). This is a description of the licenses involved, not legal advice. YOLO is a trademark of its owners.
