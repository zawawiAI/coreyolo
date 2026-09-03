import AVFoundation
import SwiftUI
import UIKit

final class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var frame: UIImage?
    @Published var detections: [HostNMS.Detection] = []
    @Published var names: [String] = []
    @Published var status: String = "Starting camera…"
    @Published var fps: String = ""

    private let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "coreyolo.camera")
    private var detector: Detector?
    private var busy = false
    var conf: Float = 0.25

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            configure()
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
                DispatchQueue.main.async { ok ? self?.configure() : (self?.status = "Camera permission denied") }
            }
        default:
            status = "Camera permission denied"
        }
    }

    private func configure() {
        queue.async { [weak self] in
            guard let self else { return }
            self.session.beginConfiguration()
            self.session.sessionPreset = .hd1280x720
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: device),
                  self.session.canAddInput(input)
            else {
                DispatchQueue.main.async { self.status = "No camera" }
                return
            }
            self.session.addInput(input)
            let output = AVCaptureVideoDataOutput()
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: self.queue)
            guard self.session.canAddOutput(output) else { return }
            self.session.addOutput(output)
            output.connection(with: .video)?.videoOrientation = .portrait
            self.session.commitConfiguration()
            do {
                self.detector = try Detector()
                DispatchQueue.main.async { self.status = "CoreYOLO · \(self.detector!.end2end ? "E2E top-300" : "DFL host NMS")" }
            } catch {
                DispatchQueue.main.async { self.status = error.localizedDescription }
            }
            self.session.startRunning()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard !busy, let detector, let image = Self.image(from: sampleBuffer) else { return }
        busy = true
        let t0 = CACurrentMediaTime()
        do {
            let out = try detector.predict(image: image, conf: conf)
            let dt = CACurrentMediaTime() - t0
            DispatchQueue.main.async {
                self.frame = image
                self.detections = out.detections
                self.names = out.names
                self.fps = String(format: "%.0f ms", dt * 1000)
                self.busy = false
            }
        } catch {
            DispatchQueue.main.async {
                self.status = error.localizedDescription
                self.busy = false
            }
        }
    }

    private static func image(from sampleBuffer: CMSampleBuffer) -> UIImage? {
        guard let buf = CMSampleBufferGetImageBuffer(sampleBuffer) else { return nil }
        let ci = CIImage(cvPixelBuffer: buf)
        let ctx = CIContext()
        guard let cg = ctx.createCGImage(ci, from: ci.extent) else { return nil }
        return UIImage(cgImage: cg)
    }
}

struct CameraDetectView: View {
    @StateObject private var camera = CameraController()

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if let frame = camera.frame {
                GeometryReader { geo in
                    Image(uiImage: frame)
                        .resizable()
                        .scaledToFit()
                        .overlay {
                            Canvas { ctx, size in
                                let sx = size.width / frame.size.width
                                let sy = size.height / frame.size.height
                                for det in camera.detections {
                                    var r = det.xyxy
                                    r.origin.x *= sx
                                    r.origin.y *= sy
                                    r.size.width *= sx
                                    r.size.height *= sy
                                    ctx.stroke(Path(r), with: .color(.green), lineWidth: 2)
                                    let label = "\(name(det.cls)) \(Int(det.conf * 100))%"
                                    ctx.draw(Text(label).font(.caption2).foregroundColor(.white), at: CGPoint(x: r.minX, y: r.minY - 8), anchor: .bottomLeading)
                                }
                            }
                        }
                        .frame(width: geo.size.width, height: geo.size.height)
                }
            }
            VStack {
                HStack {
                    Text(camera.status)
                        .font(.caption.monospaced())
                        .padding(8)
                        .background(.black.opacity(0.55))
                        .foregroundStyle(.white)
                    Spacer()
                    Text(camera.fps)
                        .font(.caption.monospaced())
                        .padding(8)
                        .background(.black.opacity(0.55))
                        .foregroundStyle(.white)
                }
                Spacer()
            }
            .padding()
        }
        .onAppear { camera.start() }
    }

    private func name(_ cls: Int) -> String {
        if cls >= 0, cls < camera.names.count { return camera.names[cls] }
        return "\(cls)"
    }
}
