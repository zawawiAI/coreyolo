# CoreYOLO zoo

Binaries are **not** stored in git (`*.coreyolo` and `*.mlpackage/` are ignored). The catalog is [`manifest.json`](manifest.json). Published files belong on a GitHub Release.

## Converted COCO (DFL, SiLU)

```bash
coreyolo convert --weights yolov8n.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo export --weights weights/coreyolo-n-coco.coreyolo --imgsz 640 --out weights/coreyolo-n-coco.mlpackage

coreyolo convert --weights yolov8n-seg.pt --out weights/coreyolo-n-coco-seg.coreyolo
coreyolo export --weights weights/coreyolo-n-coco-seg.coreyolo --imgsz 640 --out weights/coreyolo-n-coco-seg.mlpackage
```

The file you ship is CoreYOLO. Converted weights keep **SiLU**. For Neural Engine, train ReLU (`--recipe coco-n`) instead of converting.

## Record mAP

```bash
coreyolo val --data datasets/coco/data.yaml --weights weights/coreyolo-n-coco.coreyolo \
  --json weights/metrics-coreyolo-n-coco.json --zoo-id coreyolo-n-coco
```

`--zoo-id` writes `mAP50` / `mAP50-95` into `manifest.json`. Until that command is run on COCO val, those fields stay `null`. Do not invent numbers.

## GitHub Release

Place the checkpoint and `.mlpackage` next to this README, then run the **Release zoo** workflow (`workflow_dispatch`) or:

```bash
gh release create v0.1.0-zoo --title "CoreYOLO zoo" --notes "See weights/manifest.json" \
  weights/coreyolo-n-coco.coreyolo
# zip the package: ditto -c -k --sequesterRsrc --keepParent weights/coreyolo-n-coco.mlpackage \
#   weights/coreyolo-n-coco.mlpackage.zip
```
