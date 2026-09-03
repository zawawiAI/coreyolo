"""mAP@0.5 and mAP@0.5:0.95 from decoded detections vs YOLO labels."""

from __future__ import annotations

import numpy as np


def box_iou_numpy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between two xyxy arrays, shape (n,4) and (m,4) → (n,m)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(br - tl, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def ap_per_class(
    detections: list[np.ndarray],
    labels: list[np.ndarray],
    nc: int,
    iou_thresholds: np.ndarray | None = None,
) -> dict[str, float]:
    """detections[i]: (k, 6) xyxy, conf, cls. labels[i]: (m, 5) cls, xyxy."""
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    stats = {c: {"tp": [], "conf": [], "n_gt": 0} for c in range(nc)}
    for det, lab in zip(detections, labels):
        for c in range(nc):
            gt = lab[lab[:, 0] == c][:, 1:] if lab.size else np.zeros((0, 4))
            stats[c]["n_gt"] += len(gt)
            pd = det[det[:, 5] == c] if det.size else np.zeros((0, 6))
            if len(pd) == 0:
                continue
            order = np.argsort(-pd[:, 4])
            pd = pd[order]
            if len(gt) == 0:
                stats[c]["tp"].append(np.zeros((len(pd), len(iou_thresholds)), dtype=bool))
                stats[c]["conf"].append(pd[:, 4])
                continue
            iou = box_iou_numpy(pd[:, :4], gt)
            assigned = np.zeros(len(gt), dtype=bool)
            tp = np.zeros((len(pd), len(iou_thresholds)), dtype=bool)
            for i in range(len(pd)):
                j = int(np.argmax(iou[i]))
                for t, thr in enumerate(iou_thresholds):
                    if iou[i, j] >= thr and not assigned[j]:
                        tp[i, t] = True
                if iou[i, j] >= iou_thresholds[0]:
                    assigned[j] = True
            stats[c]["tp"].append(tp)
            stats[c]["conf"].append(pd[:, 4])

    ap50, ap = [], []
    for c in range(nc):
        n_gt = stats[c]["n_gt"]
        if n_gt == 0:
            continue
        if not stats[c]["conf"]:
            ap50.append(0.0)
            ap.append(0.0)
            continue
        conf = np.concatenate(stats[c]["conf"])
        tp = np.concatenate(stats[c]["tp"], 0)
        order = np.argsort(-conf)
        tp = tp[order]
        fp = 1 - tp.astype(np.float32)
        tpc = np.cumsum(tp, 0)
        fpc = np.cumsum(fp, 0)
        recall = tpc / max(n_gt, 1)
        precision = tpc / (tpc + fpc + 1e-9)
        class_ap = [_pr_ap(recall[:, t], precision[:, t]) for t in range(tp.shape[1])]
        ap50.append(class_ap[0])
        ap.append(float(np.mean(class_ap)))
    return {
        "mAP50": float(np.mean(ap50) if ap50 else 0.0),
        "mAP50-95": float(np.mean(ap) if ap else 0.0),
        "nc_eval": len(ap50),
    }


def _pr_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
