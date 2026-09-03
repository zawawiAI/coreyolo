"""Live webcam / video stream with optional track IDs."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from coreyolo.infer.draw import annotate
from coreyolo.infer.predictor import Predictor
from coreyolo.infer.track import IOUTracker


def parse_camera_index(source: str) -> int | None:
    raw = str(source).strip().lower()
    if raw in {"webcam", "cam", "camera"}:
        return 0
    if raw.isdigit() and not Path(raw).exists():
        return int(raw)
    return None


def _overlay_hud(image: Image.Image, fps: float, n: int) -> Image.Image:
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"CoreYOLO  {fps:.1f} FPS  {n} tracks   Q quit"
    if font is not None:
        bbox = draw.textbbox((8, 8), text, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((8, 8), text, fill=(255, 255, 255), font=font)
    else:
        draw.text((8, 8), text, fill=(255, 255, 255))
    return image


def run_webcam(
    predictor: Predictor,
    camera: int = 0,
    track: bool = True,
    window: str = "CoreYOLO live",
) -> None:
    """Open the camera, run detection, and show a live window until Q."""
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Webcam tracking needs OpenCV: pip install opencv-python") from exc

    print(f"opening camera {camera}…", flush=True)
    cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera}. On macOS, allow Camera access for Terminal/Cursor "
            "in System Settings → Privacy & Security → Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame = None
    for _ in range(40):
        ok, frame = cap.read()
        if ok:
            break
        time.sleep(0.05)
    if frame is None or not ok:
        cap.release()
        raise SystemExit(
            f"Camera {camera} opened but produced no frames. Quit other camera apps, then retry. "
            "On macOS, allow Camera access for Cursor in System Settings → Privacy & Security → Camera."
        )

    tracker = IOUTracker() if track else None
    print(f"webcam {camera}  window={window!r}  press Q in the window to quit", flush=True)
    fps = 0.0
    drops = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            drops += 1
            if drops > 30:
                print("camera frame dropped; stopping")
                break
            continue
        drops = 0
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        t0 = time.perf_counter()
        det, masks = predictor.predict_full(image)
        if tracker is not None:
            det = tracker.update(det)
        vis = annotate(image, det, predictor.names, masks=masks)
        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        vis = _overlay_hud(vis, fps, 0 if det is None else len(det))
        bgr = cv2.cvtColor(np.asarray(vis), cv2.COLOR_RGB2BGR)
        cv2.imshow(window, bgr)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), ord("Q"), 27}:
            break
    cap.release()
    cv2.destroyAllWindows()
