# CoreYOLO zoo

Binaries are **not** stored in git (`*.coreyolo` and `*.mlpackage/` are ignored). The catalog is [`manifest.json`](manifest.json). Published files belong on a GitHub Release. Per-source terms: [`LICENSE_NOTICE.txt`](LICENSE_NOTICE.txt). Choosing a model means choosing its license.

## Converted COCO (SiLU, MIT)

Code MIT. Remap [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) `v1.0-alpha` files. Keep the MIT copyright notice (Kin-Yiu Wong and Hao-Tang Tsui). Converting the file does not change its applicable terms.

```bash
# https://github.com/MultimediaTechLab/YOLO/releases/tag/v1.0-alpha
coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights v9-s.pt --out weights/coreyolo-s-coco.coreyolo
coreyolo convert --weights v9-m.pt --out weights/coreyolo-m-coco.coreyolo
coreyolo convert --weights v9-c.pt --out weights/coreyolo-l-coco.coreyolo
coreyolo export --weights weights/coreyolo-n-coco.coreyolo --imgsz 640 --out weights/coreyolo-n-coco.mlpackage
```

The file you ship is CoreYOLO format (`format: coreyolo`), not a numbered `0.conv.*` pickle. Converted weights keep **SiLU** (GPU). Ultralytics `yolov9t.pt` is a **different** file (AGPL-3.0) — convert will stamp AGPL and must not be rehosted as MIT. For Neural Engine, train ReLU (`--recipe coco-n`). See [docs/licenses.md](../docs/licenses.md).

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
