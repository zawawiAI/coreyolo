"""Greedy IoU tracker so live boxes keep a stable ID across frames."""

from __future__ import annotations

import numpy as np


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``a`` (n, 4) and ``b`` (m, 4) xyxy → (n, m) IoU."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:4], b[None, :, 2:4])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter + 1e-6
    return inter / union


class IOUTracker:
    """Class-aware greedy IoU association. Output is ``(k, 7)`` xyxy, conf, cls, id."""

    def __init__(self, iou_thresh: float = 0.3, max_age: int = 30) -> None:
        self.iou_thresh = iou_thresh
        self.max_age = max_age
        self._next_id = 1
        self._tracks: list[dict] = []

    def reset(self) -> None:
        self._next_id = 1
        self._tracks = []

    def update(self, detections: np.ndarray) -> np.ndarray:
        if detections is None or len(detections) == 0:
            self._age_unmatched(set())
            return np.zeros((0, 7), dtype=np.float32)

        det = np.asarray(detections, dtype=np.float32)
        if det.ndim != 2 or det.shape[1] < 6:
            return np.zeros((0, 7), dtype=np.float32)

        assigned: dict[int, int] = {}
        used_tracks: set[int] = set()
        if self._tracks:
            iou = _iou_matrix(det[:, :4], np.stack([t["xyxy"] for t in self._tracks]))
            order = np.dstack(np.unravel_index(np.argsort(iou.ravel())[::-1], iou.shape))[0]
            for di, ti in order:
                di, ti = int(di), int(ti)
                if di in assigned or ti in used_tracks:
                    continue
                if iou[di, ti] < self.iou_thresh:
                    break
                if int(det[di, 5]) != int(self._tracks[ti]["cls"]):
                    continue
                assigned[di] = ti
                used_tracks.add(ti)

        out = np.zeros((len(det), 7), dtype=np.float32)
        matched_tracks: set[int] = set()
        for di, row in enumerate(det):
            if di in assigned:
                t = self._tracks[assigned[di]]
                t["xyxy"] = row[:4].copy()
                t["conf"] = float(row[4])
                t["cls"] = float(row[5])
                t["age"] = 0
                tid = t["id"]
                matched_tracks.add(assigned[di])
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append(
                    {"id": tid, "xyxy": row[:4].copy(), "conf": float(row[4]), "cls": float(row[5]), "age": 0}
                )
                matched_tracks.add(len(self._tracks) - 1)
            out[di] = (*row[:6], tid)
        self._age_unmatched(matched_tracks)
        return out

    def _age_unmatched(self, matched: set[int]) -> None:
        keep = []
        for i, t in enumerate(self._tracks):
            if i not in matched:
                t["age"] += 1
            if t["age"] <= self.max_age:
                keep.append(t)
        self._tracks = keep
