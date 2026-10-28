import CoreGraphics
import CoreVideo
import Foundation

/// Whole pipeline: detect -> align -> inswapper -> paste back. Runs on the camera queue.
final class FaceSwapEngine {
    private let liveDetector: FaceDetector
    private let photoDetector: FaceDetector
    private let embedder: FaceEmbedder
    private let swapper: FaceSwapper
    private let mask = ImageOps.makeMask()
    private let template128 = arcfaceTemplate112.map { Pt(x: $0.x * 128 / 112, y: $0.y * 128 / 112) }

    private let lock = NSLock()
    private var latent: [Float]?
    private var smoothed: [Pt]?
    private var missed = 0

    /// 0 = no smoothing; higher = steadier but laggier landmarks.
    var smoothing: Float = 0.5

    init(modelsDir: URL) throws {
        let rt = try Runtime(modelsDir: modelsDir)
        liveDetector = try FaceDetector(runtime: rt, file: "scrfd_2.5g.onnx", size: 320)
        photoDetector = try FaceDetector(runtime: rt, file: "scrfd_2.5g.onnx", size: 640)
        embedder = try FaceEmbedder(runtime: rt)
        swapper = try FaceSwapper(runtime: rt)
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

    /// Swaps the face in the frame in place. Returns true if a face was swapped.
    @discardableResult
    func process(_ pb: CVPixelBuffer) throws -> Bool {
        lock.lock(); let l = latent; lock.unlock()
        guard let l else { return false }

        CVPixelBufferLockBaseAddress(pb, [])
        defer { CVPixelBufferUnlockBaseAddress(pb, []) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return false }
        let view = BGRAView(base: base.assumingMemoryBound(to: UInt8.self),
                            width: CVPixelBufferGetWidth(pb), height: CVPixelBufferGetHeight(pb),
                            rowBytes: CVPixelBufferGetBytesPerRow(pb))

        guard let face = try liveDetector.detect(view) else {
            missed += 1
            if missed > 3 { smoothed = nil }   // face gone: drop the history so it snaps on return
            return false
        }
        missed = 0
        let kps = smooth(face.kps)

        let m = Affine.similarity(from: kps, to: template128)
        let crop = ImageOps.warpTensor(view, m, size: 128) { $0 / 255 }
        let out = try swapper.swap(crop: crop, latent: l)
        ImageOps.pasteBack(into: view, face: out, mask: mask, m: m)
        return true
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
