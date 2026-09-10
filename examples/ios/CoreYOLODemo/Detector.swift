import CoreML
import Foundation
import UIKit

/// Loads a bundled `CoreYOLO.mlpackage` and runs letterbox → predict → host NMS.
final class Detector {
    struct Output {
        var detections: [HostNMS.Detection]
        var names: [String]
        var letterbox: Letterbox.Result
        var originalSize: CGSize
    }

    enum LoadError: LocalizedError {
        case missingPackage
        var errorDescription: String? {
            "Add CoreYOLO.mlpackage to the app target (export with `coreyolo export`, then drag into Xcode)."
        }
    }

    let model: MLModel
    let imgsz: Int
    let names: [String]
    let end2end: Bool
    let inputName: String
    let outputName: String

    init(computeUnits: MLComputeUnits = .cpuAndNeuralEngine) throws {
        guard let url =
            Bundle.main.url(forResource: "CoreYOLO", withExtension: "mlpackage")
            ?? Bundle.main.url(forResource: "CoreYOLO", withExtension: "mlmodelc")
        else {
            throw LoadError.missingPackage
        }
        let config = MLModelConfiguration()
        config.computeUnits = computeUnits
        model = try MLModel(contentsOf: url, configuration: config)
        let meta = (model.modelDescription.metadata[.creatorDefinedKey] as? [String: String]) ?? [:]
        imgsz = Int(meta["imgsz"] ?? "640") ?? 640
        if let raw = meta["names"], let parsed = try? JSONDecoder().decode([String].self, from: Data(raw.utf8)) {
            names = parsed
        } else {
            names = (0..<(Int(meta["nc"] ?? "80") ?? 80)).map { "class_\($0)" }
        }
        end2end = ["end2end", "topk", "none"].contains(meta["nms"] ?? "host")
        inputName = model.modelDescription.inputDescriptionsByName.keys.first ?? "image"
        outputName = model.modelDescription.outputDescriptionsByName["detections"] != nil
            ? "detections"
            : (model.modelDescription.outputDescriptionsByName.keys.first ?? "detections")
    }

    func predict(image: UIImage, conf: Float = 0.25, iou: Float = 0.45) throws -> Output {
        let boxed = Letterbox.apply(image, imgsz: imgsz)
        guard let pb = Letterbox.pixelBuffer(boxed.image) else {
            throw NSError(domain: "CoreYOLO", code: 1, userInfo: [NSLocalizedDescriptionKey: "pixel buffer failed"])
        }
        let input = try MLDictionaryFeatureProvider(dictionary: [inputName: MLFeatureValue(pixelBuffer: pb)])
        let out = try model.prediction(from: input)
        guard let arr = out.featureValue(for: outputName)?.multiArrayValue else {
            throw NSError(domain: "CoreYOLO", code: 2, userInfo: [NSLocalizedDescriptionKey: "missing detections output"])
        }
        var dets = end2end ? HostNMS.e2e(arr, confThres: conf) : HostNMS.dfl(arr, confThres: conf, iouThres: iou)
        let scaled = Letterbox.scaleBoxes(
            dets.map(\.xyxy),
            padX: boxed.padX,
            padY: boxed.padY,
            scale: boxed.scale,
            dest: image.size
        )
        for i in dets.indices {
            dets[i].xyxy = scaled[i]
        }
        return Output(detections: dets, names: names, letterbox: boxed, originalSize: image.size)
    }
}
