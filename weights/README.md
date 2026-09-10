# CoreYOLO zoo

Binaries are **not** stored in git (`*.coreyolo` and `*.mlpackage/` are ignored). The catalog is [`manifest.json`](manifest.json). Published files belong on a GitHub Release.

## Converted COCO (SiLU)

```bash
coreyolo convert --weights yolov9t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights yolov9c.pt --out weights/coreyolo-l-coco.coreyolo
coreyolo export --weights weights/coreyolo-n-coco.coreyolo --imgsz 640 --out weights/coreyolo-n-coco.mlpackage
```

The file you ship is CoreYOLO format (`format: coreyolo`), not the Ultralytics pickle. Nano YOLOv9 is Ultralytics `yolov9t.pt` (no `yolov9n.yaml`). Converted weights keep **SiLU** and remain **Ultralytics tensors (AGPL-3.0)** — convert remaps names, it does not relicense. GitHub Releases of those files must stay labeled AGPL-3.0, never MIT. For Neural Engine and an MIT weight path, train ReLU (`--recipe coco-n`) instead of converting. See [docs/licenses.md](../docs/licenses.md).

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
