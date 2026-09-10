# Licenses

This page is a product summary, **not legal advice**, and **not a promise that nobody can sue**. Anyone can file a lawsuit. These notices reduce confusion; they do not waive anyone’s rights. Read the MIT text in [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE), the [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) text, and [Ultralytics’ licensing](https://www.ultralytics.com/license).

## CoreYOLO (this project)

CoreYOLO’s **source** (train, val, predict, export, Core ML path) is original software under the **MIT License**. You may view, share, modify, and distribute it, including in a closed App Store or internal binary, subject to the MIT copyright and permission notice.

The clean MIT **weight** path is to train CoreYOLO **from scratch** on your own labels (`coreyolo train` or `Detector("n").train(...)`) and ship those checkpoints / `.mlpackage` files.

This repository does **not** vendor Ultralytics source, does **not** depend on the `ultralytics` Python package, and is **not** a fork of Ultralytics. CoreYOLO is not affiliated with, endorsed by, or sponsored by Ultralytics or Apple.

The public Python type is `from coreyolo import Detector`. `YOLO` is a compatibility alias only.

## Independent implementation

Detectors in this family (GELAN, CSP backbone, PAN-FPN, decoupled head, Distribution Focal Loss, SiLU/Swish) are described in public papers and blogs. CoreYOLO reimplements a small Core ML-first stack of those published ideas. Similarity of architecture is not a license to copy Ultralytics **source** or **pretrained weight files**.

## Trademarks

“YOLO”, “YOLOv9”, “YOLO11”, “YOLO26”, and “Ultralytics” are trademarks of their respective owners.

They appear here only as **nominative** references: to name third-party files (`yolov9t.pt`), to state license duties, and to say this project is **not** those products. CoreYOLO is an independent implementation. It is not an Ultralytics product.

The project name includes “YOLO” as a descriptive reference to the published detector family. The SDK class is `Detector`. Prefer that name in new code. Do not use Ultralytics logos, and do not imply endorsement.

## Ultralytics YOLOv9 (not this project)

YOLOv9 is developed by Ultralytics and builds on WongKinYiu’s GELAN paper. Its source is **open**: you may view, share, modify, and distribute it under **AGPL-3.0**. That is a real open-source license. The obligations below are why many product teams buy a commercial license instead of relying on AGPL.

### The AGPL-3.0 catch

AGPL-3.0 is copyleft, including over a network (the “SaaS” clause):

- If you **modify** YOLOv9, or
- If you use it **inside a software service** and let users interact with it over a network,

you typically must make the **corresponding source of your whole application** available under AGPL-3.0 (or a compatible copyleft), not only the detector file.

Renaming tensors, wrapping the model in another runtime, fine-tuning, or exporting Core ML does not, by itself, remove that.

### Commercial and closed internal use

If you want YOLOv9 in a **commercial product** or a **closed-source internal enterprise tool** without releasing your application source, Ultralytics requires a **separate commercial license**. MIT on CoreYOLO’s *code* does not replace that for *their* code or *their* pretrained weights.

## Convert is not a relicensing

```bash
coreyolo convert --weights yolov9t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights yolov9c.pt --out weights/coreyolo-l-coco.coreyolo
```

That command remaps YOLOv9 tensor **names** onto CoreYOLO’s GELAN graph. Ultralytics’ nano YOLOv9 file is `yolov9t.pt` — there is no `yolov9n.yaml`; CoreYOLO scale `n` is that graph. The file format becomes `format: coreyolo` (`stem.*` / `head.*`). The **numbers still came from** Ultralytics (AGPL-3.0). `Detector()` rejects the raw Ultralytics pickle so app code cannot load `yolov9t.pt` by accident; that is a file-layout guard, not a license wash.

Converted checkpoints are stamped `weights_license: AGPL-3.0`. Loading them prints a reminder. Fine-tunes (`--resume` from a converted file, or `Detector(converted).train(...)`) and Core ML exports **stay AGPL-3.0**. Do not rehost those files as MIT. This git tree does not contain converted binaries. GitHub Releases of converted files must stay labeled AGPL-3.0.

| What you ship | Typical license shape |
|---|---|
| CoreYOLO code + weights you trained **from scratch** here | MIT |
| Converted `yolov9t.pt` / `yolov9c.pt` tensors | Still Ultralytics YOLOv9 (AGPL-3.0) unless you have their commercial license |
| Fine-tune or Core ML export of converted tensors | Still AGPL-3.0 (derivative of those weights) |
| Ultralytics runtime, `yolo` CLI, or `from ultralytics import YOLO` | AGPL-3.0 (or Ultralytics commercial) |

YOLO11 and YOLO26 **weights** cannot be converted (different graph). Train CoreYOLO E2E yourself if you want that family without taking those checkpoints.
