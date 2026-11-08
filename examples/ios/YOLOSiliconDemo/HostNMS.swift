import CoreML
import Foundation

/// Host post-process matching `coreyolo.infer.nms.non_max_suppression`.
enum HostNMS {
    struct Detection: Identifiable {
        var id = UUID()
        var xyxy: CGRect
        var conf: Float
        var cls: Int
    }

    static let classOffset: Float = 7680

    static func xywhToXyxy(cx: Float, cy: Float, w: Float, h: Float) -> (Float, Float, Float, Float) {
        (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    }

    static func iou(_ a: (Float, Float, Float, Float), _ b: (Float, Float, Float, Float)) -> Float {
        let x1 = max(a.0, b.0)
        let y1 = max(a.1, b.1)
        let x2 = min(a.2, b.2)
        let y2 = min(a.3, b.3)
        let inter = max(0, x2 - x1) * max(0, y2 - y1)
        let aa = max(0, a.2 - a.0) * max(0, a.3 - a.1)
        let ba = max(0, b.2 - b.0) * max(0, b.3 - b.1)
        return inter / (aa + ba - inter + 1e-9)
    }

    /// DFL: `pred` shape `(1, 4+nc, N)` xywh pixels + per-class scores.
    static func dfl(_ pred: MLMultiArray, confThres: Float = 0.25, iouThres: Float = 0.45, maxDet: Int = 300) -> [Detection] {
        let shape = pred.shape.map(\.intValue)
        precondition(shape.count == 3)
        let ch = shape[1]
        let n = shape[2]
        let nc = ch - 4
        var cands: [(box: (Float, Float, Float, Float), conf: Float, cls: Int)] = []
        for a in 0..<n {
            var best: Float = -Float.greatestFiniteMagnitude
            var cls = 0
            if nc == 1 {
                best = pred[[0, 4, a] as [NSNumber]].floatValue
                cls = 0
            } else {
                for c in 0..<nc {
                    let s = pred[[0, 4 + c, a] as [NSNumber]].floatValue
                    if s > best {
                        best = s
                        cls = c
                    }
                }
            }
            if best <= confThres { continue }
            let cx = pred[[0, 0, a] as [NSNumber]].floatValue
            let cy = pred[[0, 1, a] as [NSNumber]].floatValue
            let w = pred[[0, 2, a] as [NSNumber]].floatValue
            let h = pred[[0, 3, a] as [NSNumber]].floatValue
            cands.append((xywhToXyxy(cx: cx, cy: cy, w: w, h: h), best, cls))
        }
        return greedy(cands, iouThres: iouThres, maxDet: maxDet, classAware: true)
    }

    /// E2E: `(1, K, 6)` xyxy, conf, cls (already top-k, no NMS).
    static func e2e(_ pred: MLMultiArray, confThres: Float = 0.25) -> [Detection] {
        let shape = pred.shape.map(\.intValue)
        precondition(shape.count == 3)
        let k = shape[1]
        var out: [Detection] = []
        for i in 0..<k {
            let conf = pred[[0, i, 4] as [NSNumber]].floatValue
            if conf < confThres { continue }
            let x1 = CGFloat(pred[[0, i, 0] as [NSNumber]].floatValue)
            let y1 = CGFloat(pred[[0, i, 1] as [NSNumber]].floatValue)
            let x2 = CGFloat(pred[[0, i, 2] as [NSNumber]].floatValue)
            let y2 = CGFloat(pred[[0, i, 3] as [NSNumber]].floatValue)
            let cls = Int(pred[[0, i, 5] as [NSNumber]].floatValue)
            out.append(Detection(xyxy: CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1), conf: conf, cls: cls))
        }
        return out
    }

    private static func greedy(
        _ cands: [(box: (Float, Float, Float, Float), conf: Float, cls: Int)],
        iouThres: Float,
        maxDet: Int,
        classAware: Bool
    ) -> [Detection] {
        let order = cands.indices.sorted { cands[$0].conf > cands[$1].conf }
        var keep: [Int] = []
        for i in order {
            let a = cands[i]
            var ok = true
            for j in keep {
                let b = cands[j]
                if classAware && a.cls != b.cls { continue }
                if iou(a.box, b.box) > iouThres {
                    ok = false
                    break
                }
            }
            if ok {
                keep.append(i)
                if keep.count >= maxDet { break }
            }
        }
        return keep.map { i in
            let c = cands[i]
            return Detection(
                xyxy: CGRect(
                    x: CGFloat(c.box.0),
                    y: CGFloat(c.box.1),
                    width: CGFloat(c.box.2 - c.box.0),
                    height: CGFloat(c.box.3 - c.box.1)
                ),
                conf: c.conf,
                cls: c.cls
            )
        }
    }
}
