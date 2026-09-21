"""Live webcam / video stream with optional track IDs."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

from coreyolo.infer.draw import annotate_bgr
from coreyolo.infer.predictor import Predictor
from coreyolo.infer.track import IOUTracker


def parse_camera_index(source: str) -> int | None:
    raw = str(source).strip().lower()
    if raw in {"webcam", "cam", "camera"}:
        return 0
    if raw.isdigit() and not Path(raw).exists():
        return int(raw)
    return None


def _overlay_hud_bgr(frame: np.ndarray, fps: float, n: int) -> np.ndarray:
    import cv2

    text = f"CoreYOLO  {fps:.0f} FPS  {n}  Q quit"
    cv2.rectangle(frame, (6, 6), (320, 32), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def _open_camera(camera: int):
    import cv2

    if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
        cap = cv2.VideoCapture(camera, cv2.CAP_AVFOUNDATION)
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(camera)


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
        raise SystemExit("Webcam needs OpenCV: pip install 'coreyolo[video]'") from exc

    print(f"opening camera {camera}…", flush=True)
    cap = _open_camera(camera)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {camera}. On macOS, allow Camera access for Terminal/Cursor "
            "in System Settings → Privacy & Security → Camera."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame = None
    ok = False
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
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1280, 720)
    try:
        cv2.setWindowProperty(window, cv2.WND_PROP_TOPMOST, 1)
    except cv2.error:
        pass
    print(f"webcam {camera}  window={window!r}  press Q in the window to quit", flush=True)
    fps = 0.0
    drops = 0
    while True:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        if not ok:
            drops += 1
            if drops > 30:
                print("camera frame dropped; stopping")
                break
            continue
        drops = 0
        det, masks = predictor.predict_bgr(frame)
        if tracker is not None:
            det = tracker.update(det)
        vis = annotate_bgr(frame, det, predictor.names, masks=masks)
        dt = time.perf_counter() - t0
        fps = 0.9 * fps + 0.1 * (1.0 / dt) if dt > 0 else fps
        vis = _overlay_hud_bgr(vis, fps, 0 if det is None else len(det))
        cv2.imshow(window, vis)
        key = cv2.waitKey(1) & 0xFF
        if key in {ord("q"), ord("Q"), 27}:
            break
    cap.release()
    cv2.destroyAllWindows()
