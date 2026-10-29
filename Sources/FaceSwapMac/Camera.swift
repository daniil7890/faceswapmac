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

    /// Stage 2 (inswapper + paste) runs here while the camera queue already prepares the next frame.
    private let swapQueue = DispatchQueue(label: "camera.swap", qos: .userInteractive)
    private let pipeLock = NSLock()
    private var swapBusy = false
    private var pending: PreparedFrame?   // newest prepared frame waiting for stage 2

    // main-thread only
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
        guard swapEnabled, let engine, engine.hasSource else {
            present(pb, found: false, swapMs: nil)
            return
        }
        // stage 1 on the camera queue
        let prepared: PreparedFrame?
        do { prepared = try engine.prepare(pb) } catch {
            report(error); present(pb, found: false, swapMs: nil); return
        }
        guard let prepared else {
            // no face: show the raw frame unless a swapped frame is about to come out
            pipeLock.lock(); let idle = !swapBusy; pipeLock.unlock()
            if idle { present(pb, found: false, swapMs: nil) }
            return
        }
        pipeLock.lock()
        if swapBusy {
            pending = prepared          // replace an older waiting frame; only the newest matters
            pipeLock.unlock()
            return
        }
        swapBusy = true
        pipeLock.unlock()
        swapQueue.async { self.runSwaps(engine, first: prepared) }
    }

    /// Stage 2 loop: process the frame, then whatever newest frame stage 1 left in `pending`.
    private func runSwaps(_ engine: FaceSwapEngine, first: PreparedFrame) {
        var job: PreparedFrame? = first
        while let j = job {
            let t0 = CACurrentMediaTime()
            do {
                let out = try engine.finish(j)
                present(out, found: true, swapMs: (CACurrentMediaTime() - t0) * 1000)
            } catch {
                report(error)
                present(j.pixelBuffer, found: false, swapMs: nil)
            }
            pipeLock.lock()
            job = pending
            pending = nil
            if job == nil { swapBusy = false }
            pipeLock.unlock()
        }
    }

    private func report(_ error: Error) {
        let msg = error.localizedDescription
        DispatchQueue.main.async { self.error = msg }
    }

    private func present(_ pb: CVPixelBuffer, found: Bool, swapMs: Double?) {
        let cg = ciContext.createCGImage(CIImage(cvPixelBuffer: pb),
                                         from: CGRect(x: 0, y: 0, width: CVPixelBufferGetWidth(pb),
                                                      height: CVPixelBufferGetHeight(pb)))
        DispatchQueue.main.async {
            let now = CACurrentMediaTime()
            self.fpsAccum += now - self.lastTime; self.fpsFrames += 1; self.lastTime = now
            if self.fpsAccum > 0.5 { self.fps = Double(self.fpsFrames) / self.fpsAccum; self.fpsAccum = 0; self.fpsFrames = 0 }
            self.frame = cg
            self.faceFound = found
            if let swapMs { self.swapMs = self.swapMs * 0.9 + swapMs * 0.1 }
        }
    }
}
