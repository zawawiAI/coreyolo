# YOLOSilicon zoo

Binaries are **not** stored in git (`*.yolosilicon` and `*.mlpackage/` are ignored). The catalog is [`manifest.json`](manifest.json). Published files belong on a GitHub Release. Per-source terms: [`LICENSE_NOTICE.txt`](LICENSE_NOTICE.txt). Choosing a model means choosing its license.

## Converted COCO (SiLU, MIT)

Code MIT. Remap YOLOv9 `v1.0-alpha` files `v9-t.pt` / `s` / `m` / `c`. Keep the MIT copyright notice (Kin-Yiu Wong and Hao-Tang Tsui). Converting the file does not change its applicable terms.

```bash
yolosilicon convert --weights v9-t.pt --out weights/yolosilicon-n-coco.yolosilicon
yolosilicon convert --weights v9-s.pt --out weights/yolosilicon-s-coco.yolosilicon
yolosilicon convert --weights v9-m.pt --out weights/yolosilicon-m-coco.yolosilicon
yolosilicon convert --weights v9-c.pt --out weights/yolosilicon-l-coco.yolosilicon
yolosilicon export --weights weights/yolosilicon-n-coco.yolosilicon --imgsz 640 --out weights/yolosilicon-n-coco.mlpackage
```

The file you ship is YOLOSilicon format (`format: yolosilicon`), not a numbered `0.conv.*` pickle. Converted weights keep **SiLU** (GPU). Other published YOLOv9 dumps are AGPL-3.0 — convert will stamp AGPL and must not be rehosted as MIT. For Neural Engine, train ReLU (`--recipe coco-n`). See [docs/licenses.md](../docs/licenses.md).

## Record mAP

```bash
yolosilicon val --data datasets/coco/data.yaml --weights weights/yolosilicon-n-coco.yolosilicon \
  --json weights/metrics-yolosilicon-n-coco.json --zoo-id yolosilicon-n-coco
```

`--zoo-id` writes `mAP50` / `mAP50-95` into `manifest.json`. Until that command is run on COCO val, those fields stay `null`. Do not invent numbers.

## GitHub Release

Place the checkpoint and `.mlpackage` next to this README, then run the **Release zoo** workflow (`workflow_dispatch`) or:

```bash
gh release create v0.1.0-zoo --title "YOLOSilicon zoo" --notes "See weights/manifest.json" \
  weights/yolosilicon-n-coco.yolosilicon
# zip the package: ditto -c -k --sequesterRsrc --keepParent weights/yolosilicon-n-coco.mlpackage \
#   weights/yolosilicon-n-coco.mlpackage.zip
```
