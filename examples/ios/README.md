# CoreYOLO iPhone sample

SwiftUI camera app for **iOS 16+**. Letterbox → Core ML → host NMS (GELAN/DFL) or top-300 (E2E). Matches `coreyolo.data.augment.letterbox` and `coreyolo.infer.nms`.

## 1. Export a model

```bash
coreyolo export --weights weights/coreyolo-n-coco.coreyolo --imgsz 640 --out weights/coreyolo-n-coco.mlpackage
```

Rename or copy to `CoreYOLO.mlpackage` (the name `Detector.swift` looks for in the app bundle).

## 2. Add it to Xcode

1. Open `CoreYOLODemo.xcodeproj`.
2. Drag `CoreYOLO.mlpackage` into the `CoreYOLODemo` group.
3. Check **Copy items if needed** and the **CoreYOLODemo** target.
4. Set your **Team** under Signing.
5. Run on a **device** (camera + Neural Engine / GPU). The simulator has no camera and a weak Core ML path.

Without the package the app still builds; the overlay tells you to add it.

Use a package from **weights you trained** (`coreyolo train` then `export`) for an MIT path. A package exported from converted `yolov9*.pt` still contains Ultralytics YOLOv9 tensors (AGPL-3.0). See `docs/licenses.md`.

## 3. What the host does

| Graph | Core ML output | Host |
|--------|----------------|------|
| DFL | `(1, 4+nc, N)` xywh + scores | class-aware NMS, then un-letterbox |
| E2E | `(1, 300, 6)` xyxy, conf, cls | conf filter, then un-letterbox |

Pad color is `(114, 114, 114)`. Do not use Vision `VNRecognizedObjectObservation` — this graph is raw tensors, not a Vision detector with baked-in NMS.

`computeUnits` defaults to `.cpuAndGPU`. Switch to `.cpuAndNeuralEngine` in `Detector.swift` for ReLU graphs.
