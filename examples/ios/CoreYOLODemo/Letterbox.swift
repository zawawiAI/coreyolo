import CoreGraphics
import CoreVideo
import UIKit

/// Matches `coreyolo.data.augment.letterbox`: aspect-preserving resize, gray pad 114.
enum Letterbox {
    static let padColor = UIColor(red: 114 / 255, green: 114 / 255, blue: 114 / 255, alpha: 1)

    struct Result {
        var image: UIImage
        var scale: CGFloat
        var padX: CGFloat
        var padY: CGFloat
        var imgsz: Int
    }

    static func apply(_ image: UIImage, imgsz: Int, scaleUp: Bool = true) -> Result {
        let w = image.size.width
        let h = image.size.height
        var scale = min(CGFloat(imgsz) / w, CGFloat(imgsz) / h)
        if !scaleUp {
            scale = min(scale, 1)
        }
        let nw = (w * scale).rounded()
        let nh = (h * scale).rounded()
        let padX = (CGFloat(imgsz) - nw) / 2
        let padY = (CGFloat(imgsz) - nh) / 2
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        format.opaque = true
        let renderer = UIGraphicsImageRenderer(size: CGSize(width: imgsz, height: imgsz), format: format)
        let canvas = renderer.image { ctx in
            padColor.setFill()
            ctx.fill(CGRect(x: 0, y: 0, width: imgsz, height: imgsz))
            image.draw(in: CGRect(x: padX.rounded(), y: padY.rounded(), width: nw, height: nh))
        }
        return Result(image: canvas, scale: scale, padX: padX, padY: padY, imgsz: imgsz)
    }

    static func pixelBuffer(_ image: UIImage) -> CVPixelBuffer? {
        let width = Int(image.size.width)
        let height = Int(image.size.height)
        var buffer: CVPixelBuffer?
        let attrs: [CFString: Any] = [
            kCVPixelBufferCGImageCompatibilityKey: true,
            kCVPixelBufferCGBitmapContextCompatibilityKey: true,
        ]
        let status = CVPixelBufferCreate(
            kCFAllocatorDefault,
            width,
            height,
            kCVPixelFormatType_32BGRA,
            attrs as CFDictionary,
            &buffer
        )
        guard status == kCVReturnSuccess, let pb = buffer else { return nil }
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let ctx = CGContext(
            data: CVPixelBufferGetBaseAddress(pb),
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(pb),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Little.rawValue | CGImageAlphaInfo.premultipliedFirst.rawValue
        ), let cg = image.cgImage else { return nil }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
        return pb
    }

    /// Map letterboxed xyxy back to the original frame — `scale_boxes` in Python.
    static func scaleBoxes(_ boxes: [CGRect], padX: CGFloat, padY: CGFloat, scale: CGFloat, dest: CGSize) -> [CGRect] {
        boxes.map { b in
            var x1 = (b.minX - padX) / scale
            var y1 = (b.minY - padY) / scale
            var x2 = (b.maxX - padX) / scale
            var y2 = (b.maxY - padY) / scale
            x1 = min(max(x1, 0), dest.width)
            y1 = min(max(y1, 0), dest.height)
            x2 = min(max(x2, 0), dest.width)
            y2 = min(max(y2, 0), dest.height)
            return CGRect(x: x1, y: y1, width: x2 - x1, height: y2 - y1)
        }
    }
}
