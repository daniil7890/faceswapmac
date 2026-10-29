import CoreGraphics
import CoreVideo
import Foundation

/// One camera frame after stage 1 (detect + align), ready for stage 2 (inswapper + paste).
struct PreparedFrame {
    let pixelBuffer: CVPixelBuffer
    let m: Affine          // frame -> 128x128 crop
    let crop: [Float]      // CHW RGB 0...1
    let latent: [Float]
}

/// Pipeline in two stages so they can run in parallel on different frames:
///   prepare (camera queue): detect -> smooth -> align -> crop       ~ a few ms
///   finish  (swap queue):   inswapper -> paste back on GPU           ~ inswapper time
final class FaceSwapEngine {
    private let liveDetector: LiveFaceDetector
    private let photoDetector: FaceDetector
    private let embedder: FaceEmbedder
    private let swapper: FaceSwapper
    private let compositor: MetalCompositor?
    private let mask = ImageOps.makeMask()
    private let template128 = arcfaceTemplate112.map { Pt(x: $0.x * 128 / 112, y: $0.y * 128 / 112) }

    private let lock = NSLock()
    private var latent: [Float]?
    private var smoothed: [Pt]?   // touched only by prepare()
    private var missed = 0

    /// 0 = no smoothing; higher = steadier but laggier landmarks.
    var smoothing: Float = 0.5
    /// Shown in the UI, e.g. "Core ML · Metal".
    let backendDescription: String

    init(modelsDir: URL) throws {
        let rt = try Runtime(modelsDir: modelsDir)
        let live: LiveFaceDetector
        if let cml = CoreMLFaceDetector(modelsDir: modelsDir) {
            live = cml
        } else {
            live = try FaceDetector(runtime: rt, file: "scrfd_2.5g.onnx", size: 320)
        }
        let comp = MetalCompositor(mask: ImageOps.makeMask())
        liveDetector = live
        compositor = comp
        photoDetector = try FaceDetector(runtime: rt, file: "scrfd_2.5g.onnx", size: 640)
        embedder = try FaceEmbedder(runtime: rt)
        swapper = try FaceSwapper(runtime: rt)
        backendDescription = "\(live.kind) · \(comp == nil ? "CPU paste" : "Metal")"
    }

    var hasSource: Bool { lock.lock(); defer { lock.unlock() }; return latent != nil }

    /// Takes the identity from a photo. Throws if no face is found.
    func setSource(_ image: CGImage) throws {
        let bmp = try Bitmap(image)
        let view = bmp.view
        guard let face = try photoDetector.detect(view, threshold: 0.4) else { throw EngineError.noFaceInPhoto }
        let emb = try embedder.embed(view, kps: face.kps)
        let l = swapper.latent(from: emb)
        lock.lock(); latent = l; lock.unlock()
    }

    /// Stage 1. Returns nil when there is no source face or no face in the frame.
    func prepare(_ pb: CVPixelBuffer) throws -> PreparedFrame? {
        lock.lock(); let l = latent; lock.unlock()
        guard let l else { return nil }

        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let view = BGRAView(base: base.assumingMemoryBound(to: UInt8.self),
                            width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                            rowBytes: CVPixelBufferGetBytesPerRow(pb))

        guard let face = try liveDetector.detect(view, threshold: 0.5) else {
            missed += 1
            if missed > 3 { smoothed = nil }   // face gone: drop the history so it snaps on return
            return nil
        }
        missed = 0
        let kps = smooth(face.kps)
        let m = Affine.similarity(from: kps, to: template128)
        let crop = ImageOps.warpTensor(view, m, size: 128) { $0 / 255 }
        return PreparedFrame(pixelBuffer: pb, m: m, crop: crop, latent: l)
    }

    /// Stage 2. Returns the frame with the face swapped (a new buffer on the Metal path).
    func finish(_ f: PreparedFrame) throws -> CVPixelBuffer {
        let out = try swapper.swap(crop: f.crop, latent: f.latent)
        if let compositor, let res = compositor.composite(src: f.pixelBuffer, face: out, m: f.m) {
            return res
        }
        // CPU fallback: paste in place
        let pb = f.pixelBuffer
        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        if let base = CVPixelBufferGetBaseAddress(pb) {
            let view = BGRAView(base: base.assumingMemoryBound(to: UInt8.self),
                                width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                                rowBytes: CVPixelBufferGetBytesPerRow(pb))
            ImageOps.pasteBack(into: view, face: out, mask: mask, m: f.m)
        }
        return pb
    }

    /// Exponential smoothing of landmarks; resets on big jumps so fast head turns don't lag.
    private func smooth(_ kps: [Pt]) -> [Pt] {
        guard let prev = smoothed, smoothing > 0 else { smoothed = kps; return kps }
        let eyeDist = hypotf(kps[1].x - kps[0].x, kps[1].y - kps[0].y)
        var jump: Float = 0
        for i in 0..<5 { jump = max(jump, hypotf(kps[i].x - prev[i].x, kps[i].y - prev[i].y)) }
        if jump > eyeDist * 0.35 { smoothed = kps; return kps }
        let a = smoothing
        let s = (0..<5).map { Pt(x: prev[$0].x * a + kps[$0].x * (1 - a), y: prev[$0].y * a + kps[$0].y * (1 - a)) }
        smoothed = s
        return s
    }
}

enum EngineError: LocalizedError {
    case noFaceInPhoto, badImage
    var errorDescription: String? {
        switch self {
        case .noFaceInPhoto: return L("На фото не найдено лицо. Нужен анфас, лицо хорошо видно.", "No face found in the photo. Use a frontal, clearly visible face.")
        case .badImage: return L("Не удалось прочитать изображение", "Could not read the image")
        }
    }
}

/// Owned BGRA8 bitmap made from a CGImage (for the source photo).
final class Bitmap {
    let width: Int, height: Int, rowBytes: Int
    private let data: UnsafeMutablePointer<UInt8>

    init(_ image: CGImage) throws {
        width = image.width; height = image.height; rowBytes = width * 4
        data = .allocate(capacity: rowBytes * height)
        data.initialize(repeating: 0, count: rowBytes * height)
        let info = CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: data, width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: rowBytes, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: info) else { data.deallocate(); throw EngineError.badImage }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
    }
    deinit { data.deallocate() }

    var view: BGRAView { BGRAView(base: data, width: width, height: height, rowBytes: rowBytes) }
}
