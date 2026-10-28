import AVFoundation
import CoreImage
import Foundation
import QuartzCore

/// Camera capture -> engine -> CGImage for the preview.
final class CameraController: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    @Published var frame: CGImage?
    @Published var fps: Double = 0
    @Published var swapMs: Double = 0
    @Published var faceFound = false
    @Published var devices: [AVCaptureDevice] = []
    @Published var error: String?
    /// Picker selection; switching it switches the capture device.
    @Published var selectedID = "" {
        didSet {
            guard selectedID != oldValue, !selectedID.isEmpty,
                  let d = devices.first(where: { $0.uniqueID == selectedID }) else { return }
            select(d)
        }
    }

    let session = AVCaptureSession()
    private let queue = DispatchQueue(label: "camera.frames", qos: .userInteractive)
    private let ciContext = CIContext(options: [.cacheIntermediates: false])
    private var input: AVCaptureDeviceInput?

    /// Set from the UI; read on the camera queue.
    var engine: FaceSwapEngine?
    var swapEnabled = true

    private var lastTime = CACurrentMediaTime()
    private var fpsAccum = 0.0, fpsFrames = 0

    func start() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] ok in
            guard let self else { return }
            guard ok else {
                DispatchQueue.main.async { self.error = L("Нет доступа к камере: Системные настройки → Конфиденциальность → Камера",
                                                            "No camera access: System Settings → Privacy & Security → Camera") }
                return
            }
            self.queue.async { self.configure() }
        }
    }

    private func configure() {
        let found = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video, position: .unspecified).devices
        DispatchQueue.main.async {
            self.devices = found
            if self.selectedID.isEmpty, let first = found.first { self.selectedID = first.uniqueID }
        }
        // same device the UI picker shows first
        guard let dev = found.first ?? AVCaptureDevice.default(for: .video) else {
            DispatchQueue.main.async { self.error = L("Камера не найдена", "No camera found") }
            return
        }
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720
        let out = AVCaptureVideoDataOutput()
        out.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        out.alwaysDiscardsLateVideoFrames = true
        out.setSampleBufferDelegate(self, queue: queue)
        if session.canAddOutput(out) { session.addOutput(out) }
        session.commitConfiguration()
        select(dev)
        session.startRunning()
    }

    func select(_ device: AVCaptureDevice) {
        queue.async {
            self.session.beginConfiguration()
            if let i = self.input { self.session.removeInput(i) }
            do {
                let i = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(i) { self.session.addInput(i); self.input = i }
            } catch {
                DispatchQueue.main.async { self.error = error.localizedDescription }
            }
            self.session.commitConfiguration()
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sb: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pb = CMSampleBufferGetImageBuffer(sb) else { return }
        var found = false
        let t0 = CACurrentMediaTime()
        if swapEnabled, let engine {
            do { found = try engine.process(pb) }
            catch { DispatchQueue.main.async { self.error = error.localizedDescription } }
        }
        let dt = CACurrentMediaTime() - t0
        let cg = ciContext.createCGImage(CIImage(cvPixelBuffer: pb),
                                         from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb),
                                                      height: CVPixelBufferGetHeight(pb)))
        let now = CACurrentMediaTime()
        fpsAccum += now - lastTime; fpsFrames += 1; lastTime = now
        var newFps: Double?
        if fpsAccum > 0.5 { newFps = Double(fpsFrames) / fpsAccum; fpsAccum = 0; fpsFrames = 0 }
        DispatchQueue.main.async {
            self.frame = cg
            self.faceFound = found
            self.swapMs = self.swapMs * 0.9 + dt * 1000 * 0.1
            if let newFps { self.fps = newFps }
        }
    }
}
