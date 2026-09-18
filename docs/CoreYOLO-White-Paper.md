---
title: "CoreYOLO"
subtitle: "An MIT YOLO-style detector for Core ML on Apple silicon"
author: "CoreYOLO contributors"
date: "2 September 2026"
version: "0.1.0"
---

# CoreYOLO: An Open Detector Built for Apple Deployment

**White paper** · Version 0.1.0 (alpha) · 2 September 2026

CoreYOLO is original software under the MIT License. It is inspired by the YOLO family of one-stage detectors and by the Roboflow YOLO label layout. Independent MIT code.

---

## Abstract

Teams that want real-time object detection on iPhone and Mac often hit two constraints at once: they need a graph that maps cleanly onto Core ML and the Apple Neural Engine, and they need a license that allows a closed App Store binary. Widely used YOLO toolkits solve the first problem as one export among many, and they typically ship under AGPL-3.0, which is a poor fit for proprietary mobile apps.

CoreYOLO is a small, original PyTorch trainer whose product path is Core ML. It implements two detector graphs (GELAN: YOLOv9 with host NMS, and E2E: C3k2 + C2PSA, NMS-free), optional instance segmentation, Roboflow YOLO labels, a Python SDK, and a native checkpoint format. Application code loads CoreYOLO files (`.coreyolo`, trained `.pt`, or `.mlpackage`)—not a sequential `yolov9t.pt` pickle. This paper describes the architecture, the train-to-device workflow, measured latency on Apple silicon, and when CoreYOLO is the right choice versus a general YOLO toolkit.

**Keywords:** object detection, instance segmentation, Core ML, Apple Neural Engine, MIT, YOLO, iOS

---

## 1. Introduction

YOLO-style detectors remain the default for on-device bounding boxes: one forward pass, a feature pyramid, and decoded boxes at several strides. The research lineage is public. The *product* problem is narrower. An iOS camera app does not need a 50-task training zoo. It needs (1) a static, fused convolution graph, (2) an activation the Neural Engine actually maps, (3) a host or in-graph NMS story that Vision.framework can consume, and (4) a license the legal team will sign.

A typical YOLO toolkit is a strong general trainer. Public YOLOv9 weights are usually AGPL-3.0. Shipping those weights inside a closed iPhone app usually requires a commercial license from the copyright holders. Core ML export exists there, but the trainer is SiLU-first and Core ML is one backend among ONNX, TensorRT, OpenVINO, and TFLite.

CoreYOLO inverts that priority. The default activation is ReLU so fused Conv–BN–ReLU maps onto the Neural Engine. Native training uses ReLU; StarReLU (`s * ReLU(x)^2 + b`, 2024) is the ReLU-family option that still maps to the Neural Engine. Converted YOLOv9 tensors keep SiLU so they match the checkpoint they came from. Export is an FP16 ML Program with a fixed square `imgsz`. Compute units are first-class (`gpu`, `ane`, `all`, `cpu`). The Python package and the CLI are the same internals. The iPhone artifact is a `.mlpackage` dropped into Xcode—not a Python process on the phone.

This white paper is five short chapters: problem and thesis (this page), network graphs, software and data, Apple deployment, and licensing plus conclusions.

---

## 2. Detector graphs

CoreYOLO is an **anchor-free, one-stage** network. An RGB letterboxed tensor of shape `(1, 3, imgsz, imgsz)` goes through a CSP backbone, a PAN-FPN neck, and a decoupled head at strides 8, 16, and 32 (P3 / P4 / P5). At `imgsz = 640` that is \(80^2 + 40^2 + 20^2 = 8400\) prediction cells.

GELAN (the default) uses a fixed channel table; scale `n` matches public `yolov9t`. DFL still uses the usual `n` / `s` / `m` / `l` / `x` compound ladder (nano: depth 0.33, width 0.25). E2E uses depth 0.50 on n/s/m. Typical use: `n` real-time on phone, `s` on iPad/Mac, `m`+ when accuracy dominates and the model can run offline or on a desk.

### 2.1 GELAN (default, YOLOv9)

`--family gelan` (aliases `9` / `v9`) is the compact GELAN detector: ELAN1 / RepNCSPELAN4, AConv (or ADown on `l`), SPPELAN, PAN-FPN, and a DFL head (`reg_max = 16`). Native train still defaults to ReLU; converted `v9-t.pt` stays SiLU.

Because the tensors match compact YOLOv9, a **one-time** remap can copy COCO-pretrained MultimediaTechLab `v9-t.pt` onto CoreYOLO GELAN-n (MIT). The output is a CoreYOLO checkpoint (`format: coreyolo`, `stem.*` names, SiLU). Application code never loads the raw pickle. YOLO11n and YOLO26n **cannot** be copied: different blocks and different heads.

### 2.2 DFL

The DFL graph is the older C2f + SPPF detector: PAN-FPN, decoupled box and class branches, and Distribution Focal Loss with `reg_max = 16`. Boxes are decoded from a 16-bin distribution per side, converted to `xywh` in pixels, concatenated with sigmoid class scores, and filtered with **class-aware NMS on the host**. That layout is what Core ML exports as `(1, 4+nc, N)`.

### 2.3 E2E

The E2E graph is an **original** C3k2 + residual SPPF + C2PSA network with a dual head: one-to-many (TAL top-k) and one-to-one (TAL top-1) in train, DFL removed (`reg_max = 1`), and **NMS-free top-300** at inference. Eval layout is `(B, 300, 6)` as `xyxy, conf, cls`. You train E2E from scratch (`--family e2e` or recipe `coco-n-e2e`).

### 2.4 Segmentation

`--task segment` adds a proto mask branch on P3 (32 prototypes, width-scaled proto channels) and per-anchor coefficients. Training adds cropped mask BCE on top of TAL + CIoU + DFL + classification BCE. Labels are Roboflow YOLO-Seg polygons (`cls x1 y1 … xn yn`). Detect-only five-value lines become filled rectangles. Mosaic is off so polygons stay valid. Core ML seg export is three outputs (`detections`, `mask_coeff`, `proto`); NMS and `sigmoid(coeff @ proto)` stay on the host.

Assignment is Task-Aligned: \(\mathrm{score}^{\alpha} \times \mathrm{IoU}^{\beta}\), top-k positives (k = 10 in the CoreYOLO loss; the YOLOv9 paper often uses 13). Optimizer is SGD with Nesterov momentum, cosine LR, 3-epoch warmup, mosaic until the last 10 epochs (detect).

---

## 3. Software, data, and the SDK

The package `coreyolo` (Python ≥ 3.10) is both the library and the CLI (`coreyolo train|val|predict|export|convert|coco|dummy-data`). The inference path does not import a third-party YOLO package.

### 3.1 Labels

The dataset is the Roboflow YOLO export: `data.yaml` plus `train|valid/{images,labels}`. Each label line is normalized `class xc yc w h`, or a polygon for seg. The loader accepts list or dict `names`, `val` or `valid`, and leftover Colab paths. Official COCO 2017 can be downloaded and rewritten into that layout (`coreyolo coco`), 80 contiguous classes, crowd boxes dropped.

### 3.2 Native checkpoints

CoreYOLO does **not** use a sequential `model.N.*` pickle (`model.22.cv2…`). A native file is a dict tagged `format: "coreyolo"` with a `state_dict` under `stem.*`, `stage*`, `head.*`, plus `nc`, `names`, `scale`, `act`, `family`, `task`. The suffix may be `.coreyolo` or `.pt` (PyTorch’s usual extension). `YOLO()` and `load_checkpoint()` **reject** raw `v9-t.pt` / `yolov9t.pt` so app code cannot accidentally load an upstream pickle.

Optional bootstrap, once, not in the app:

```text
coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo
```

### 3.3 Python SDK

```python
from coreyolo import Detector

model = Detector("n")
model.train(data="data.yaml", epochs=100, imgsz=640)
model.save("weights/app.coreyolo")
results = model.predict("photo.jpg", classes="person")
model.export(imgsz=640)   # Core ML for iPhone / Mac
```

`Detector("n")` builds an untrained nano graph. `Detector("weights/app.coreyolo")` or a `.mlpackage` loads a trained artifact. `predict(0)` is a Mac webcam helper; it is not the iPhone runtime. `YOLO` is a compatibility alias.

### 3.4 Maturity

Version 0.1.0 is alpha. Detect and instance segmentation are implemented. Pose, OBB, and classify are not. Training recipes are simpler than a large commercial zoo (no mixup/copy-paste by default). Converted YOLOv9t is the practical COCO starting point for GELAN; E2E is trained by the user.

---

## 4. Core ML, iPhone, and latency

Training happens on a Mac (PyTorch MPS when it works, else CUDA, else CPU). Deployment on iPhone is **Core ML**, not PyTorch.

### 4.1 Export

`export` traces a fused **inference-only** graph (Conv–BN merged). GELAN never had YOLOv9 PGI auxiliary nodes; E2E drops the one-to-many train head. DFL grids are cached as static buffers so Core ML never sees `meshgrid`, `arange`, or `-1` views. Input shape is fixed `(1, 3, imgsz, imgsz)`. The result is an ML Program for macOS 13 / **iOS 16+**. Default precision is FP16. INT8 weight quant is optional. Image input is RGB `imgsz×imgsz` with 1/255 scale.

DFL output: decoded `xywh` plus class scores; **NMS on device** (Swift or Python host). E2E output: top-300 `xyxy, conf, cls`; no NMS. Segmentation: three tensors; assemble masks after NMS.

On device the camera frame must be **letterboxed** to `imgsz`—the same pad/ratio the trainer used—then class-aware NMS for DFL, then boxes mapped back to the original pixel frame.

Swift sketch:

```swift
let config = MLModelConfiguration()
config.computeUnits = .cpuAndGPU    // or .cpuAndNeuralEngine / .all
let model = try VNCoreMLModel(for: CoreYOLO(configuration: config).model)
```

ReLU checkpoints map better on ANE. Converted YOLOv9 weights are SiLU; prefer GPU for those, or train ReLU from scratch for Neural Engine.

You do **not** need a third-party convert to run on iPhone. Train CoreYOLO → `export()` → Xcode. Convert is only if you want YOLOv9 COCO init.

### 4.2 Performance is not portable as a single FPS number

The `.mlpackage` is the same graph on Mac and iPhone. The **chip, thermals, and camera loop are not**. Burst Core ML times on an M4 Pro are not iPhone camera FPS.

Local burst medians (Core ML, isolated process, CoreYOLO-n FP16 vs official YOLO26n INT8, 1 September 2026):

| Config | CoreYOLO-n | YOLO26n |
|---|---|---|
| ANE 640×640 | 2.03 ms (~493 FPS burst) | 1.84 ms (~545 FPS burst) |
| GPU-only 640×640 | 3.30 ms (~303 FPS) | crash (MLIR pass manager) |
| PyTorch MPS 640 | ~8 ms fused | — |

A live camera app stays near display rate (often ~30 FPS) once letterbox, NMS, tracking, and UI are included. iPhone ANE/GPU is smaller than M4 Pro and will throttle. Phone recipe: scale `n`, start at 640, drop to 320 if the pipeline stutters, ReLU + FP16, measure with Instruments on device—not Mac burst tables.

DFL GPU Core ML ran on this Mac; official YOLO26 GPU-only Core ML aborted. ANE remains the preferred phone compute unit when the graph is ReLU-fused.

---

## 5. Licensing, comparison, and conclusions

### 5.1 Why MIT matters

CoreYOLO’s train, export, and inference code is MIT. You may embed it in a proprietary iOS or macOS binary, subject to the MIT copyright and permission notice.

Public YOLOv9 weight files are **not** one license. MultimediaTechLab/YOLO `v9-t.pt` / `s` / `m` / `c` are **MIT** (copyright Kin-Yiu Wong and Hao-Tang Tsui) — the same source LibreYOLO converts. Ultralytics `yolov9t.pt` is typically **AGPL-3.0**. AGPL-3.0 is copyleft, including over a network: if you modify those AGPL tensors or offer them as a SaaS so users interact with them over a network, you typically must release the corresponding source of your whole application under AGPL-3.0. A commercial product or a closed-source internal enterprise tool that uses **Ultralytics** YOLOv9 weights without publishing that source usually requires a **commercial license from the copyright holders**.

CoreYOLO is a separate implementation. It does not vendor third-party YOLO trainer source. Optional convert copies **tensors** into CoreYOLO names; the checkpoint format is CoreYOLO, but the **license follows the source file**. MultimediaTechLab converts stay MIT (keep the copyright notice). Ultralytics converts stay AGPL-3.0. YOLO26/YOLO11 weights are not copied. Train CoreYOLO from scratch on your labels for an original MIT + ReLU path. See `docs/licenses.md`. This is not legal advice.

### 5.2 Comparison (honest)

| | CoreYOLO | Typical YOLO toolkit |
|---|---|---|
| License | MIT (code + MultimediaTechLab convert + weights you train here) | AGPL-3.0 on typical Ultralytics weights; commercial license for closed products |
| Deploy priority | Core ML / ANE / iPhone | Many backends |
| Default activation | ReLU (ANE + native train) | SiLU |
| Graphs | GELAN (YOLOv9, host NMS), E2E (C3k2, trained here) | Official v9 / 11 / 26 weights |
| Tasks (v0.1) | Detect, instance seg | Detect, seg, pose, OBB, cls, … |
| App checkpoint | `.coreyolo` / CoreYOLO `.pt` / `.mlpackage` | `yolov9t.pt` and variants |
| Maturity | Alpha 0.1.0 | Large ecosystem |

A large YOLO toolkit is the better **general** trainer: more pretrained models, more tasks, more export formats, more community. CoreYOLO is the better **Apple product path** when AGPL is unacceptable and the ship target is Core ML.

### 5.3 Recommended path to iPhone

1. Label in Roboflow; export YOLO (or YOLO-Seg).
2. `Detector("n").train(data="data.yaml", epochs=…)` or a CoreYOLO COCO recipe—not `yolov9t.pt` in the app.
3. `model.save("weights/app.coreyolo")` then `model.export(imgsz=640)`.
4. Add the `.mlpackage` to Xcode; letterbox; NMS on host for DFL; set `computeUnits` to GPU or Neural Engine.
5. Measure on a physical iPhone.

### 5.4 Conclusion

CoreYOLO exists so a YOLO-style detector can be trained in the open, stored in a native checkpoint, and shipped as Core ML without taking an AGPL runtime into a closed Apple app. It does not try to replace every YOLO toolkit. It tries to be the small stack you actually put on an iPhone: C2f or C3k2, ReLU, fused convolutions, host or NMS-free decode, MIT.

Further work includes richer training augs, pose/OBB if needed, tighter ANE INT8, and on-device NMS options. The public interface today is `from coreyolo import Detector` and `coreyolo export`.

---

*CoreYOLO v0.1.0 · MIT License · Independent MIT code. Latency figures are single-machine burst measurements on Apple M4 Pro (1 Sep 2026), not product SLAs.*
