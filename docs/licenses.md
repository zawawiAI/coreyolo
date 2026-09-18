# Licenses

This page is a product summary, **not legal advice**, and **not a promise that nobody can sue**. Anyone can file a lawsuit. These notices reduce confusion; they do not waive anyone’s rights. Read the MIT text in [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE), and the [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) text.

## CoreYOLO (this project)

CoreYOLO’s **source** (train, val, predict, export, Core ML path) is original software under the **MIT License**. You may view, share, modify, and distribute it, including in a closed App Store or internal binary, subject to the MIT copyright and permission notice.

This repository does **not** vendor third-party detector source and does **not** depend on another YOLO Python package. The public Python type is `from coreyolo import Detector`. `YOLO` is a compatibility alias only.

## Independent implementation

Detectors in this family (GELAN, CSP backbone, PAN-FPN, decoupled head, Distribution Focal Loss, SiLU/Swish) are described in public papers and blogs. CoreYOLO reimplements a small Core ML-first stack of those published ideas. Similarity of architecture is not a license to copy someone else’s **source** or **pretrained weight files**.

## Trademarks

“YOLO”, “YOLOv9”, “YOLO11”, and “YOLO26” are trademarks of their respective owners.

They appear here only as **nominative** references: to name third-party files (`v9-t.pt`, `yolov9t.pt`) and to state license duties. CoreYOLO is an independent implementation.

The project name includes “YOLO” as a descriptive reference to the published detector family. The SDK class is `Detector`. Prefer that name in new code.

## Two different YOLOv9 weight files

Public “YOLOv9” checkpoints are **not** one license. Convert copies **numbers**; it does **not** relicense them. The source file decides the stamp.

### MIT — MultimediaTechLab/YOLO (the LibreYOLO path)

The YOLOv9 authors re-released an MIT implementation at [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (also [WongKinYiu/YOLO](https://github.com/WongKinYiu/YOLO)). Pretrained files from release `v1.0-alpha`:

- `v9-t.pt` → CoreYOLO scale `n`
- `v9-s.pt` → `s`
- `v9-m.pt` → `m`
- `v9-c.pt` → `l`

Copyright (c) 2024 Kin-Yiu Wong and Hao-Tang Tsui. Licensed under the MIT License.

[LibreYOLO](https://www.libreyolo.com/docs/licensing) converts those same files by remapping tensor names and keeps MIT plus that copyright notice. CoreYOLO does the same onto `format: coreyolo`. Keep the MIT copyright and permission notice with every copy, fine-tune, and Core ML export.

```bash
coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights v9-s.pt --out weights/coreyolo-s-coco.coreyolo
coreyolo convert --weights v9-m.pt --out weights/coreyolo-m-coco.coreyolo
coreyolo convert --weights v9-c.pt --out weights/coreyolo-l-coco.coreyolo
```

Published zoo rows that list those `v9-*.pt` files are **MIT**.

### AGPL-3.0 — Ultralytics `yolov9*.pt`

Files named `yolov9t.pt` / `yolov9s.pt` / `yolov9m.pt` / `yolov9c.pt` from Ultralytics are typically **AGPL-3.0**. They are a different training dump, not the MultimediaTechLab release. Convert still remaps them, but the checkpoint is stamped `weights_license: AGPL-3.0`. Do **not** rehost those files as MIT.

```bash
coreyolo convert --weights yolov9t.pt --out weights/coreyolo-n-coco.coreyolo
```

### The AGPL-3.0 catch

AGPL-3.0 is copyleft, including over a network (the “SaaS” clause):

- If you **modify** AGPL-covered YOLOv9 code or weights, or
- If you use them **inside a software service** and let users interact with it over a network,

you typically must make the **corresponding source of your whole application** available under AGPL-3.0 (or a compatible copyleft), not only the detector file.

Renaming tensors, wrapping the model in another runtime, fine-tuning, or exporting Core ML does not, by itself, remove that.

### Commercial and closed internal use

If you want **Ultralytics** YOLOv9 weights in a **commercial product** or a **closed-source internal enterprise tool** without releasing your application source, you usually need a **separate commercial license** from the copyright holders. MIT on CoreYOLO’s *code* does not replace that for *their* pretrained weights.

## Convert is not a relicensing

That command remaps tensor **names** onto CoreYOLO’s GELAN graph. The file format becomes `format: coreyolo` (`stem.*` / `head.*`). The **numbers still came from** the file you passed in. `Detector()` rejects the raw pickle so app code cannot load `v9-t.pt` or `yolov9t.pt` by accident; that is a file-layout guard, not a license wash.

| What you ship | Typical license shape |
|---|---|
| CoreYOLO code + weights you trained **from scratch** here | MIT |
| Converted MultimediaTechLab `v9-t.pt` / `s` / `m` / `c` | MIT (keep Wong/Tsui copyright) |
| Converted Ultralytics `yolov9t.pt` / `s` / `m` / `c` | Still AGPL-3.0 unless you have a commercial license for those weights |
| Fine-tune or Core ML export | Same license as the tensors you started from |

The other MIT **weight** path is to train CoreYOLO **from scratch** on your own labels (`coreyolo train` or `Detector("n").train(...)`) and ship those checkpoints / `.mlpackage` files. Native train uses ReLU (Neural Engine). Converted files keep SiLU (GPU).

YOLO11 and YOLO26 **weights** cannot be converted (different graph). Train CoreYOLO E2E yourself if you want that family without taking those checkpoints.
