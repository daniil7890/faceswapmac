import CoreML
import Foundation

/// SCRFD-2.5g at 320x320 as a native Core ML model (runs on the Neural Engine, ~0.7 ms on M-series).
/// Expects models/coreml/scrfd_320.mlpackage made by scripts/convert_coreml.sh
/// (input "input" 1x3x320x320, outputs out0...out8 in ONNX order).
final class CoreMLFaceDetector: LiveFaceDetector {
    let kind = "Core ML"
    let size = 320
    private let model: MLModel
    private let input: MLMultiArray

    /// Returns nil when the model is not installed or fails to load; the caller falls back to ONNX.
    init?(modelsDir: URL) {
        let fm = FileManager.default
        let dir = modelsDir.appendingPathComponent("coreml")
        let pkg = dir.appendingPathComponent("scrfd_320.mlpackage")
        let compiled = dir.appendingPathComponent("scrfd_320.mlmodelc")
        func mtime(_ u: URL) -> Date {
            ((try? fm.attributesOfItem(atPath: u.path))?[.modificationDate] as? Date) ?? .distantPast
        }
        do {
            let hasPkg = fm.fileExists(atPath: pkg.path), hasCompiled = fm.fileExists(atPath: compiled.path)
            guard hasPkg || hasCompiled else { return nil }
            // compile once and cache next to the package; recompile if the package was replaced
            if hasPkg && (!hasCompiled || mtime(pkg) > mtime(compiled)) {
                let tmp = try MLModel.compileModel(at: pkg)
                try? fm.removeItem(at: compiled)
                try fm.copyItem(at: tmp, to: compiled)
                try? fm.removeItem(at: tmp)
            }
            let cfg = MLModelConfiguration()
            cfg.computeUnits = .all
            model = try MLModel(contentsOf: compiled, configuration: cfg)
            input = try MLMultiArray(shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32)
        } catch {
            NSLog("FaceSwapMac: Core ML detector unavailable, using ONNX: \(error)")
            return nil
        }
    }

    func detect(_ img: BGRAView, threshold: Float) throws -> (kps: [Pt], score: Float)? {
        let (t, scale) = ImageOps.letterboxTensor(img, size: size)
        let dst = input.dataPointer.assumingMemoryBound(to: Float.self)
        t.withUnsafeBufferPointer { dst.update(from: $0.baseAddress!, count: $0.count) }
        let feats = try MLDictionaryFeatureProvider(dictionary: ["input": MLFeatureValue(multiArray: input)])
        let out = try model.prediction(from: feats)
        func arr(_ i: Int) throws -> [Float] {
            guard let m = out.featureValue(for: "out\(i)")?.multiArrayValue else {
                throw ModelError.badOutput("scrfd_320 out\(i)")
            }
            return m.floatArray
        }
        return SCRFD.decode(scores: [try arr(0), try arr(1), try arr(2)],
                            kps: [try arr(6), try arr(7), try arr(8)],
                            size: size, scale: scale, threshold: threshold)
    }
}

extension MLMultiArray {
    /// Flattened values in logical (row-major) order, whatever the storage type and strides.
    var floatArray: [Float] {
        let n = count
        let shape = self.shape.map { $0.intValue }
        let strides = self.strides.map { $0.intValue }
        let p = UnsafeRawPointer(dataPointer)
        var contiguous = true, expect = 1
        for d in stride(from: shape.count - 1, through: 0, by: -1) {
            if shape[d] > 1 && strides[d] != expect { contiguous = false }
            expect *= shape[d]
        }
        if contiguous && dataType == .float32 {
            return Array(UnsafeBufferPointer(start: p.assumingMemoryBound(to: Float.self), count: n))
        }
        var out = [Float](repeating: 0, count: n)
        var idx = [Int](repeating: 0, count: shape.count)
        for flat in 0..<n {
            var off = 0
            for d in 0..<shape.count { off += idx[d] * strides[d] }
            switch dataType {
            case .float32: out[flat] = p.load(fromByteOffset: off * 4, as: Float.self)
            case .float16: out[flat] = Float(p.load(fromByteOffset: off * 2, as: Float16.self))
            case .double: out[flat] = Float(p.load(fromByteOffset: off * 8, as: Double.self))
            default: break
            }
            var d = shape.count - 1
            while d >= 0 { idx[d] += 1; if idx[d] < shape[d] { break }; idx[d] = 0; d -= 1 }
        }
        return out
    }
}
