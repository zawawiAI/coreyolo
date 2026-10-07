# Licensing

CoreYOLO carries two separately licensed things: its own code, and pretrained checkpoints. They are often not the same license.

This page describes the licenses involved. It is a description, not legal advice, and it does not create any warranty. If the answer matters commercially, read the licenses yourself and take your own counsel.

Read the MIT text in [`LICENSE`](../LICENSE), [`NOTICE`](../NOTICE), [`weights/LICENSE_NOTICE.txt`](../weights/LICENSE_NOTICE.txt), and the [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) text.

## CoreYOLO's own code

The library is MIT. That covers the Python API, the CLI, the trainers, validators and exporters, the dataset loaders, and the conversion code under `coreyolo/export/convert.py`. Use it in a commercial or closed-source product, keep the copyright line and the license text with any copy you redistribute, and the obligation ends there.

The grant stops at the code. [`NOTICE`](../NOTICE) puts it plainly:

> Those licenses vary and are not all permissive: some published YOLOv9 dumps are AGPL-3.0, and this MIT License does not extend to them. Choosing a model means choosing its license.

This repository does **not** vendor third-party detector source. The public Python type is `from coreyolo import Detector`. `YOLO` is a compatibility alias only.

## Upstream code

CoreYOLO reimplements a small Core ML-first stack of published detector ideas (GELAN, CSP backbone, PAN-FPN, decoupled head, Distribution Focal Loss). Similarity of architecture is not a license to copy someone else’s **source** or **pretrained weight files**. MIT does not overwrite an upstream checkpoint, and CoreYOLO does not relicense anyone's work.

The GELAN convert path follows the authors' **MIT re-release** of YOLOv9 at [MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO) (copyright Kin-Yiu Wong and Hao-Tang Tsui), not the GPL-3.0 repository `WongKinYiu/yolov9` that carries the same model, and not Ultralytics `yolov9*.pt` (AGPL-3.0).

“YOLO”, “YOLOv9”, “YOLO11”, and “YOLO26” are trademarks of their respective owners. They appear only as nominative references.

## Weights, per checkpoint

No pretrained weight file ships inside the package. Published checkpoints live on GitHub Releases. The catalog is [`weights/manifest.json`](../weights/manifest.json). That row, and the file itself (`weights_license`), are the terms for that checkpoint.

Licenses differ between sources, and converting the file does not change its applicable terms. A Core ML or ONNX artifact built from a restricted checkpoint inherits the restriction.

| Checkpoint | Upstream | Weights |
|---|---|---|
| `coreyolo-n-coco` from `v9-t.pt` | MultimediaTechLab/YOLO v1.0-alpha | MIT |
| `coreyolo-s-coco` from `v9-s.pt` | MultimediaTechLab/YOLO v1.0-alpha | MIT |
| `coreyolo-m-coco` from `v9-m.pt` | MultimediaTechLab/YOLO v1.0-alpha | MIT |
| `coreyolo-l-coco` from `v9-c.pt` | MultimediaTechLab/YOLO v1.0-alpha | MIT |
| `coreyolo-n-coco-relu`, `coreyolo-n-coco-seg`, `coreyolo-e2e-n-coco`, `coreyolo-rfdetr-n` | trained here | MIT |
| Roboflow RF-DETR Nano / Small / Medium / Large | roboflow/rf-detr | Apache-2.0 (not remapped) |
| Roboflow RF-DETR XLarge / 2XLarge | roboflow/rf-detr | PML-1.0 (refused) |
| Converted Ultralytics `yolov9t.pt` / `s` / `m` / `c` | Ultralytics | AGPL-3.0 |

```bash
coreyolo convert --weights v9-t.pt --out weights/coreyolo-n-coco.coreyolo
coreyolo convert --weights v9-s.pt --out weights/coreyolo-s-coco.coreyolo
coreyolo convert --weights v9-m.pt --out weights/coreyolo-m-coco.coreyolo
coreyolo convert --weights v9-c.pt --out weights/coreyolo-l-coco.coreyolo
```

## Finding the terms for one model

`weights/LICENSE_NOTICE.txt` is the per-source summary. `NOTICE` names the upstream files convert can remap. The zoo table on [Models](https://zawawiai.github.io/coreyolo/models.html) has a Weights column, one row per published file.

Then check the GitHub Release of the exact file you are about to download. It is authoritative, and it can change without a docs page changing with it.

## GELAN COCO (MultimediaTechLab) — interpretation

Original work: YOLOv9, MultimediaTechLab.

Upstream license: MIT.

Upstream source: [github.com/MultimediaTechLab/YOLO](https://github.com/MultimediaTechLab/YOLO).

CoreYOLO code: MIT.

Weights: MIT. Convert remaps names onto `format: coreyolo`.

Interpretation: MIT is a permissive license, so these weights can be used in commercial and closed-source products. The one standing obligation is to keep the license text and the copyright notice, Kin-Yiu Wong and Hao-Tang Tsui, with any copy you redistribute. It places no condition on your own application code, and a model you train yourself on your own data is yours. The port follows the authors' MIT re-release of YOLOv9, not the GPL-3.0 repository that carries the same model, so the permissive terms come from the source CoreYOLO actually converts.

## Ultralytics `yolov9*.pt` — interpretation

Those files are a different dump, typically AGPL-3.0. Convert still remaps names; the checkpoint is stamped `weights_license: AGPL-3.0`. They are not covered by CoreYOLO's MIT License. Do not rehost them as MIT.

AGPL-3.0 is copyleft, including over a network: if you modify those tensors or use them inside a software service (SaaS) so users interact with it over a network, you typically must make the corresponding source of your whole application available under AGPL-3.0. A commercial product or a closed-source internal enterprise tool that uses those weights without publishing source usually needs a commercial license from the copyright holders.

## Commercial use

Code is rarely the problem. MIT permits commercial and closed-source use. It asks you to keep its license text and attribution notices with copies you redistribute, and it places no conditions on your own application code.

Checkpoints are where products get stuck. A restricted checkpoint stays restricted however permissive the surrounding code is, and converting the file does not change its applicable terms, which is what `weights/LICENSE_NOTICE.txt` states directly. A Core ML package exported from those tensors inherits the restriction.

Where a license carries its restriction into derivative works, fine-tuning does not escape it either. Training the same architecture from scratch on data you have the right to use does: the code is permissive, so a model you train yourself is yours, and the pretrained checkpoint's terms never enter it.

Decide the license question when you pick the model rather than when you ship, and read the terms on the file you actually downloaded.

YOLO11 and YOLO26 **weights** cannot be converted (different graph). Train CoreYOLO E2E yourself if you want that family without taking those checkpoints.

## RF-DETR (Roboflow) — interpretation

Original work: RF-DETR, Roboflow.

Upstream license: Apache-2.0 for Nano, Small, Medium, and Large. Platform Model License 1.0 for XLarge and 2XLarge.

Upstream source: [github.com/roboflow/rf-detr](https://github.com/roboflow/rf-detr). The backbone lineage is DINOv2 (Apache-2.0, Meta).

CoreYOLO code: MIT. `family=rfdetr` follows LibreYOLO's Core ML contract for this family (fixed RGB image, ImageNet norm inside the graph, set prediction, no NMS) and pins FP16 to the Neural Engine. The graph is still CoreYOLO's own (windowed self-attention, dense query cross-attention). Similarity of that idea is not a license to copy LibreYOLO's or Roboflow's source or weight files. This repository does not vendor either project and does not remap `rf-detr-*.pth`.

Weights you train with `--family rfdetr` or `--recipe coco-n-rfdetr`: MIT. The code is permissive, so a model you train yourself on data you have the right to use is yours.

Upstream Nano–Large checkpoints: Apache-2.0, on Roboflow's graph. Apache-2.0 is a permissive license, so those weights can be used in commercial and closed-source products. It asks you to keep its license text, attribution notices, and patent notice with any copy of the weights you redistribute. It places no obligation on your own application code. CoreYOLO does not relicense them as MIT. A fine-tune of those tensors keeps Apache-2.0.

Upstream XLarge and 2XLarge: Platform Model License 1.0. CoreYOLO refuses those files. Do not stamp them MIT or Apache-2.0, and do not put them in the zoo.

```
@inproceedings{robinson2026rfdetr,
  title     = {RF-DETR: Real-Time Detection Transformer},
  author    = {Robinson, Isaac and Robicheaux, Peter and Popov, Matvei and Ramanan, Deva and Peri, Neehar},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2511.09554}
}
```

"RF-DETR" is a name of its owner. It appears here only as a nominative reference.

## Not legal advice

This page describes the licenses involved. It is a description, not legal advice, and it does not create any warranty. If the answer matters commercially, read the licenses yourself and take your own counsel.
