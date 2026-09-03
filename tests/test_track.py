import numpy as np

from coreyolo.infer.track import IOUTracker, _iou_matrix


def test_iou_identical_boxes() -> None:
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert float(_iou_matrix(a, a)[0, 0]) > 0.99


def test_tracker_keeps_id() -> None:
    tr = IOUTracker(iou_thresh=0.3, max_age=5)
    a = np.array([[10, 10, 40, 80, 0.9, 0]], dtype=np.float32)
    b = np.array([[12, 12, 42, 82, 0.8, 0]], dtype=np.float32)
    id0 = int(tr.update(a)[0, 6])
    id1 = int(tr.update(b)[0, 6])
    assert id0 == id1 == 1


def test_tracker_new_id_for_new_class() -> None:
    tr = IOUTracker()
    person = np.array([[10, 10, 40, 80, 0.9, 0]], dtype=np.float32)
    bus = np.array([[10, 10, 40, 80, 0.9, 5]], dtype=np.float32)
    id0 = int(tr.update(person)[0, 6])
    id1 = int(tr.update(bus)[0, 6])
    assert id0 != id1
