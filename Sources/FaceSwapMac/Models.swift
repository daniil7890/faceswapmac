import Foundation
import OnnxRuntimeBindings

enum ModelError: LocalizedError {
    case missing(String), badOutput(String), noEmap
    var errorDescription: String? {
        switch self {
        case .missing(let f): return L("Нет файла модели: \(f). Запусти scripts/download_models.sh", "Model file missing: \(f). Run scripts/download_models.sh")
        case .badOutput(let s): return L("Неожиданный выход модели: \(s)", "Unexpected model output: \(s)")
        case .noEmap: return L("Не удалось достать emap из inswapper", "Could not extract emap from inswapper")
        }
    }
}

/// Shared ONNX Runtime env + session factory with the CoreML execution provider (ANE/GPU).
final class Runtime {
    let env: ORTEnv
    let modelsDir: URL

    init(modelsDir: URL) throws {
        env = try ORTEnv(loggingLevel: .warning)
        self.modelsDir = modelsDir
    }

    func session(_ file: String, coreML: Bool = true) throws -> ORTSession {
        let path = modelsDir.appendingPathComponent(file).path
        guard FileManager.default.fileExists(atPath: path) else { throw ModelError.missing(file) }
        let opts = try ORTSessionOptions()
        try opts.setGraphOptimizationLevel(.all)
        if coreML {
            // MLProgram + all compute units; nodes CoreML can't take fall back to CPU automatically.
            try opts.appendCoreMLExecutionProvider(withOptionsV2: [
                "ModelFormat": "MLProgram",
                "MLComputeUnits": "ALL",
                "ModelCacheDirectory": modelsDir.appendingPathComponent(".coreml-cache").path,
            ])
        }
        return try ORTSession(env: env, modelPath: path, sessionOptions: opts)
    }
}

@inline(__always) func tensor(_ data: [Float], _ shape: [Int]) throws -> ORTValue {
    let d = data.withUnsafeBufferPointer { NSMutableData(bytes: $0.baseAddress, length: $0.count * 4) }
    return try ORTValue(tensorData: d, elementType: .float, shape: shape.map { NSNumber(value: $0) })
}

@inline(__always) func floats(_ v: ORTValue) throws -> [Float] {
    let d = try v.tensorData() as Data
    return d.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
}

/// SCRFD / RetinaFace (insightface layout: 3 strides x [score, bbox, kps]). Returns the 5 landmarks
/// of the most confident face, in image pixels.
final class FaceDetector {
    private let session: ORTSession
    private let outputs: [String]
    let size: Int

    init(runtime: Runtime, file: String, size: Int) throws {
        // dynamic input dims: keep detector on CPU, it is tiny (~3 ms at 320 on Apple Silicon)
        session = try runtime.session(file, coreML: false)
        outputs = try session.outputNames()
        self.size = size
        guard outputs.count == 9 else { throw ModelError.badOutput("detector outputs \(outputs.count)") }
    }

    func detect(_ img: BGRAView, threshold: Float = 0.5) throws -> (kps: [Pt], score: Float)? {
        let (input, scale) = ImageOps.letterboxTensor(img, size: size)
        let res = try session.run(withInputs: ["input": try tensor(input, [1, 3, size, size])],
                                  outputNames: Set(outputs), runOptions: nil)
        var best: (kps: [Pt], score: Float)?
        for (k, stride) in [8, 16, 32].enumerated() {
            guard let s = res[outputs[k]], let p = res[outputs[k + 6]] else { continue }
            let scores = try floats(s), kps = try floats(p)
            let g = size / stride
            for i in 0..<scores.count where scores[i] > threshold && scores[i] > (best?.score ?? 0) {
                let cell = i / 2                       // 2 anchors per cell
                let ax = Float(cell % g * stride), ay = Float(cell / g * stride)
                var pts = [Pt]()
                for j in 0..<5 {
                    pts.append(Pt(x: (ax + kps[i * 10 + j * 2] * Float(stride)) / scale,
                                  y: (ay + kps[i * 10 + j * 2 + 1] * Float(stride)) / scale))
                }
                best = (pts, scores[i])
            }
        }
        return best
    }
}

/// ArcFace w600k_r50: 112x112 aligned crop -> L2-normalised 512-d identity embedding.
final class FaceEmbedder {
    private let session: ORTSession
    init(runtime: Runtime) throws { session = try runtime.session("arcface_w600k_r50.onnx") }

    func embed(_ img: BGRAView, kps: [Pt]) throws -> [Float] {
        let m = Affine.similarity(from: kps, to: arcfaceTemplate112)
        let x = ImageOps.warpTensor(img, m, size: 112) { ($0 - 127.5) / 127.5 }
        let res = try session.run(withInputs: ["input": try tensor(x, [1, 3, 112, 112])],
                                  outputNames: ["output"], runOptions: nil)
        guard let o = res["output"] else { throw ModelError.badOutput("arcface") }
        return normalize(try floats(o))
    }
}

@inline(__always) func normalize(_ v: [Float]) -> [Float] {
    let n = sqrtf(v.reduce(0) { $0 + $1 * $1 })
    return v.map { $0 / max(n, 1e-8) }
}

/// inswapper_128: aligned 128x128 target crop + source latent -> swapped face crop.
final class FaceSwapper {
    private let session: ORTSession
    private let emap: [Float]   // 512x512, row-major
    static let file = "inswapper_128_fp16.onnx"

    init(runtime: Runtime) throws {
        session = try runtime.session(FaceSwapper.file)
        emap = try EmapReader.read(runtime.modelsDir.appendingPathComponent(FaceSwapper.file))
    }

    /// latent = normalize(embedding @ emap)
    func latent(from embedding: [Float]) -> [Float] {
        var out = [Float](repeating: 0, count: 512)
        for i in 0..<512 where embedding[i] != 0 {
            let e = embedding[i], row = i * 512
            for j in 0..<512 { out[j] += e * emap[row + j] }
        }
        return normalize(out)
    }

    func swap(crop: [Float], latent: [Float]) throws -> [Float] {
        let res = try session.run(
            withInputs: ["target": try tensor(crop, [1, 3, 128, 128]), "source": try tensor(latent, [1, 512])],
            outputNames: ["output"], runOptions: nil)
        guard let o = res["output"] else { throw ModelError.badOutput("inswapper") }
        return try floats(o)
    }
}

/// Pulls the last 512x512 initializer (the "emap") straight out of the ONNX protobuf.
enum EmapReader {
    static func read(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) throws -> [Float] in
            let b = raw.bindMemory(to: UInt8.self)
            func varint(_ i: inout Int) -> Int {
                var r = 0, s = 0
                while true { let c = Int(b[i]); i += 1; r |= (c & 0x7F) << s; s += 7; if c < 0x80 { return r } }
            }
            /// Calls body(field, wire, value-or-start, end) for each field in [start, end).
            func walk(_ start: Int, _ end: Int, _ body: (Int, Int, Int, Int) -> Void) {
                var i = start
                while i < end {
                    let key = varint(&i), f = key >> 3, w = key & 7
                    switch w {
                    case 0: let v = varint(&i); body(f, w, v, 0)
                    case 1: i += 8
                    case 5: i += 4
                    case 2: let n = varint(&i); body(f, w, i, i + n); i += n
                    default: return
                    }
                }
            }
            var found: (dtype: Int, start: Int, end: Int)?
            walk(0, b.count) { f, w, s, e in
                guard f == 7, w == 2 else { return }            // ModelProto.graph
                walk(s, e) { gf, gw, gs, ge in
                    guard gf == 5, gw == 2 else { return }      // GraphProto.initializer
                    var dims = [Int](), dtype = 0, rawRange: (Int, Int)?
                    walk(gs, ge) { tf, tw, ts, te in
                        if tf == 1 {
                            if tw == 0 { dims.append(ts) }
                            else { var j = ts; while j < te { dims.append(varint(&j)) } }
                        } else if tf == 2 { dtype = ts }
                        else if tf == 9 { rawRange = (ts, te) }
                    }
                    if dims == [512, 512], let r = rawRange { found = (dtype, r.0, r.1) }
                }
            }
            guard let f = found else { throw ModelError.noEmap }
            let p = raw.baseAddress! + f.start
            switch f.dtype {
            // raw_data has arbitrary alignment inside the file -> unaligned loads
            case 1: return (0..<512 * 512).map { p.loadUnaligned(fromByteOffset: $0 * 4, as: Float.self) }
            case 10: return (0..<512 * 512).map { Float(p.loadUnaligned(fromByteOffset: $0 * 2, as: Float16.self)) }
            default: throw ModelError.noEmap
            }
        }
    }
}
