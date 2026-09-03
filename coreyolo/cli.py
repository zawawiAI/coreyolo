"""Command-line interface: train, val, predict, export, dummy-data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from coreyolo.utils import __version__, load_checkpoint, place_module, select_device


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--imgsz", type=int, default=640, help="square inference / train size")
    p.add_argument(
        "--device",
        default="gpu",
        help="gpu (default: MPS/CUDA if available) | all | ane | cpu | cuda",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coreyolo",
        description="Open-source YOLO-style detector with a Core ML / Apple silicon path.",
    )
    parser.add_argument("--version", action="version", version=f"CoreYOLO {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Train on a Roboflow YOLO dataset or a COCO recipe")
    t.add_argument("--data", default=None, help="path to data.yaml (optional if --recipe sets it)")
    t.add_argument("--recipe", default=None, help="coco-n | coco-n-fast | coco-n-e2e | coco-s | coco-m | path to yaml")
    t.add_argument("--model", default=None, choices=["n", "s", "m", "l", "x"])
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch", type=int, default=None)
    t.add_argument("--lr0", type=float, default=None)
    t.add_argument("--workers", type=int, default=None)
    t.add_argument("--project", default=None)
    t.add_argument("--name", default=None)
    t.add_argument("--act", default=None, help="relu (ANE-friendly) or silu")
    t.add_argument(
        "--family",
        default=None,
        type=lambda s: str(s).strip().lower(),
        choices=["dfl", "e2e", "8", "26", "v8", "v26", "c2f", "c3k2"],
        help="dfl = C2f+DFL host NMS (default); e2e = C3k2+C2PSA NMS-free. 8/26 still accepted.",
    )
    t.add_argument("--task", default=None, choices=["detect", "segment"], help="detect (default) or instance segment")
    t.add_argument("--mosaic", type=float, default=None)
    t.add_argument("--patience", type=int, default=None)
    t.add_argument("--resume", default=None)
    t.add_argument("--seed", type=int, default=None)
    _add_common(t)

    v = sub.add_parser("val", help="Run validation mAP")
    v.add_argument("--data", required=True)
    v.add_argument("--weights", required=True)
    v.add_argument("--batch", type=int, default=16)
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--json", default=None, help="write metrics JSON (for the zoo catalog)")
    v.add_argument("--zoo-id", default=None, help="update weights/manifest.json entry id with these metrics")
    _add_common(v)

    p = sub.add_parser("predict", help="Run PyTorch or Core ML inference")
    p.add_argument("--weights", required=True, help=".pt or .mlpackage")
    p.add_argument("--source", required=True, help="image, directory, webcam, or camera index (0)")
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--iou", type=float, default=0.45)
    p.add_argument("--save-dir", default="runs/predict")
    p.add_argument("--track", action=argparse.BooleanOptionalAction, default=True, help="stable IDs on webcam")
    p.add_argument("--classes", nargs="+", default=None, help="keep only these classes, e.g. person")
    _add_common(p)

    e = sub.add_parser("export", help="Export a checkpoint to Core ML")
    e.add_argument("--weights", required=True)
    e.add_argument("--out", default=None)
    e.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--int8", action="store_true", help="symmetric 8-bit weight quant")
    e.add_argument("--tensor-input", action="store_true", help="float CHW input instead of ImageType")
    _add_common(e)

    d = sub.add_parser("dummy-data", help="Write a tiny Roboflow-layout dataset")
    d.add_argument("--out", default="datasets/dummy")
    d.add_argument("--n-train", type=int, default=32)
    d.add_argument("--n-val", type=int, default=8)
    d.add_argument("--segment", action="store_true", help="write YOLO-seg polygon labels")

    c = sub.add_parser("coco", help="Download COCO 2017 and convert to YOLO/Roboflow layout")
    c.add_argument("--out", default="datasets/coco")
    c.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    c.add_argument("--splits", default="train,val", help="comma-separated: train,val")
    c.add_argument("--copy-images", action="store_true", help="copy images instead of symlink")
    c.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="cap each split (e.g. 512) for a tiny debug set without the full 118k train images",
    )
    w = sub.add_parser("convert", help="Convert Ultralytics YOLOv8 COCO .pt into CoreYOLO")
    w.add_argument("--weights", required=True, help="yolov8n.pt or yolov8n-seg.pt (YOLO26n/YOLO11n cannot map)")
    w.add_argument("--out", default="weights/coreyolo-n-coco.coreyolo")
    w.add_argument("--model", default=None, choices=["n", "s", "m", "l", "x"], help="scale; inferred from stem if omitted")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.cmd == "train":
        from coreyolo.train.recipe import load_recipe, train_config_from_recipe
        from coreyolo.train.trainer import train

        if args.recipe:
            recipe = load_recipe(args.recipe)
        elif args.data:
            recipe = {"data": args.data}
        else:
            print("train requires --data or --recipe", file=sys.stderr)
            return 2
        overrides = {
            "data": args.data,
            "model": args.model,
            "epochs": args.epochs,
            "batch": args.batch,
            "imgsz": args.imgsz if "--imgsz" in sys.argv else None,
            "device": args.device if "--device" in sys.argv else None,
            "lr0": args.lr0,
            "workers": args.workers,
            "project": args.project,
            "name": args.name,
            "act": args.act,
            "family": args.family,
            "task": args.task,
            "mosaic": args.mosaic,
            "patience": args.patience,
            "resume": args.resume,
            "seed": args.seed,
        }
        cfg = train_config_from_recipe(recipe, overrides)
        print(
            f"training task={cfg.task} family={cfg.family} {cfg.model} on {cfg.data}  "
            f"epochs={cfg.epochs} batch={cfg.batch} imgsz={cfg.imgsz}"
        )
        save_dir = train(cfg)
        print(f"saved to {save_dir}")
        return 0

    if args.cmd == "val":
        from torch.utils.data import DataLoader

        from coreyolo.data.dataset import YOLODetectionDataset, collate_fn
        from coreyolo.data.yaml import YOLODatasetYAML
        from coreyolo.nn.model import build_model
        from coreyolo.train.trainer import validate

        spec = YOLODatasetYAML(args.data)
        ckpt = load_checkpoint(args.weights, map_location="cpu")
        device = select_device(args.device)
        model = build_model(
            nc=ckpt.get("nc", spec.nc),
            scale=ckpt.get("scale", "n"),
            act=ckpt.get("act", "relu"),
            family=ckpt.get("family", "dfl"),
            task=str(ckpt.get("task", "detect")),
            nm=int(ckpt.get("nm", 32)),
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model, device = place_module(model, device)
        model.eval()
        print(f"PyTorch device: {device}")
        ds = YOLODetectionDataset(
            spec, "val", imgsz=args.imgsz, augment=False, mosaic=0, task=str(ckpt.get("task", "detect"))
        )
        loader = DataLoader(ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
        metrics = validate(model, loader, device, spec.nc)
        print(json.dumps(metrics, indent=2))
        if args.json or args.zoo_id:
            from coreyolo.zoo import record_metrics, write_metrics_json

            payload = {
                "weights": str(Path(args.weights).resolve()),
                "data": str(Path(args.data).resolve()),
                "imgsz": args.imgsz,
                "metrics": metrics,
            }
            if args.json:
                write_metrics_json(args.json, payload)
                print(f"wrote {args.json}")
            if args.zoo_id:
                path = record_metrics(args.zoo_id, metrics, extra=payload)
                print(f"updated zoo entry {args.zoo_id} in {path}")
        return 0

    if args.cmd == "predict":
        from coreyolo.infer.predictor import Predictor

        pred = Predictor(
            args.weights,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
        )
        if pred.backend == "coreml" and pred.coreml_engine is not None:
            print(f"Core ML compute: {pred.coreml_engine.device_name}")
        else:
            print(f"PyTorch device: {pred.device}")
        from coreyolo.infer.stream import parse_camera_index, run_webcam

        cam = parse_camera_index(args.source)
        if cam is not None:
            run_webcam(pred, camera=cam, track=args.track)
            return 0
        results = pred.predict_path(args.source, save_dir=args.save_dir)
        for r in results:
            n = 0 if r["detections"] is None else len(r["detections"])
            print(f"{r['path']}: {n} boxes -> {r.get('saved', '')}")
        return 0

    if args.cmd == "export":
        from coreyolo.export.coreml import export_coreml

        path = export_coreml(
            args.weights,
            out=args.out,
            imgsz=args.imgsz,
            fp16=args.fp16,
            quantize_8bit=args.int8,
            image_input=not args.tensor_input,
        )
        print(f"exported {path}")
        return 0

    if args.cmd == "dummy-data":
        from coreyolo.data.dummy import write_dummy_dataset

        yaml_path = write_dummy_dataset(
            args.out, n_train=args.n_train, n_val=args.n_val, segment=args.segment
        )
        print(f"wrote {yaml_path}")
        return 0

    if args.cmd == "coco":
        from coreyolo.data.coco import prepare_coco

        splits = tuple(s.strip() for s in args.splits.split(",") if s.strip())
        yaml_path = prepare_coco(
            args.out,
            download=args.download,
            splits=splits,
            copy_images=args.copy_images,
            max_images=args.max_images,
        )
        print(f"COCO YOLO dataset: {yaml_path}")
        return 0

    if args.cmd == "convert":
        from coreyolo.export.convert import convert_ultralytics

        try:
            path = convert_ultralytics(args.weights, out=args.out, scale=args.model)
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 2
        print(f"wrote {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
