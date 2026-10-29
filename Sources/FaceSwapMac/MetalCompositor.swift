import CoreVideo
import Foundation
import Metal

/// Pastes the swapped face back into the frame on the GPU.
/// Reads the camera frame, writes a new frame from our own pool, so the camera buffer is never modified.
/// Same math as ImageOps.pasteBack: frame pixel -> crop coords via `m`, bilinear face + mask, alpha blend.
final class MetalCompositor {
    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let pipeline: MTLComputePipelineState
    private let cache: CVMetalTextureCache
    private let faceTex: MTLTexture
    private var pool: CVPixelBufferPool?
    private var poolSize = (0, 0)
    private var facePixels = [Float16](repeating: 1, count: 128 * 128 * 4)

    private struct Params { var a, b, tx, c, d, ty: Float }

    private static let shader = """
    #include <metal_stdlib>
    using namespace metal;
    struct Params { float a, b, tx, c, d, ty; };
    kernel void composite(texture2d<float, access::read>   src  [[texture(0)]],
                          texture2d<float, access::write>  dst  [[texture(1)]],
                          texture2d<float, access::sample> face [[texture(2)]],
                          texture2d<float, access::sample> mask [[texture(3)]],
                          constant Params& p [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
        if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
        float4 c = src.read(gid);
        float x = float(gid.x), y = float(gid.y);
        float cx = p.a * x + p.b * y + p.tx;
        float cy = p.c * x + p.d * y + p.ty;
        if (cx > -1.0 && cy > -1.0 && cx < 129.0 && cy < 129.0) {
            constexpr sampler s(coord::normalized, address::clamp_to_zero, filter::linear);
            float2 uv = (float2(cx, cy) + 0.5) / 128.0;
            float a = mask.sample(s, uv).r;
            if (a > 0.002) {
                float3 f = clamp(face.sample(s, uv).rgb, 0.0, 1.0);
                c.rgb = mix(c.rgb, f, a);
            }
        }
        dst.write(c, gid);
    }
    """

    /// `mask` is the 128x128 soft mask from ImageOps.makeMask(). Returns nil if Metal is unavailable.
    init?(mask: [Float]) {
        guard let dev = MTLCreateSystemDefaultDevice(), let q = dev.makeCommandQueue() else { return nil }
        do {
            let lib = try dev.makeLibrary(source: MetalCompositor.shader, options: nil)
            guard let fn = lib.makeFunction(name: "composite") else { return nil }
            pipeline = try dev.makeComputePipelineState(function: fn)
        } catch {
            NSLog("FaceSwapMac: Metal shader failed, using CPU paste: \(error)")
            return nil
        }
        device = dev; queue = q
        var c: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, dev, nil, &c) == kCVReturnSuccess, let c else { return nil }

        let fd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 128, height: 128, mipmapped: false)
        fd.usage = .shaderRead
        let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Float, width: 128, height: 128, mipmapped: false)
        md.usage = .shaderRead
        guard let ft = dev.makeTexture(descriptor: fd), let mt = dev.makeTexture(descriptor: md) else { return nil }
        faceTex = ft; maskTex = mt; cache = c
        let m16 = mask.map { Float16($0) }
        m16.withUnsafeBytes {
            mt.replace(region: MTLRegionMake2D(0, 0, 128, 128), mipmapLevel: 0,
                       withBytes: $0.baseAddress!, bytesPerRow: 128 * 2)
        }
    }
    private let maskTex: MTLTexture

    /// Returns a new BGRA frame = `src` with the face (CHW RGB 0...1, 128x128) blended in. `m`: frame -> crop.
    func composite(src: CVPixelBuffer, face: [Float], m: Affine) -> CVPixelBuffer? {
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        guard let out = makeOutput(w, h),
              let srcTex = texture(src, cache, w, h), let dstTex = texture(out, cache, w, h),
              let s = CVMetalTextureGetTexture(srcTex), let d = CVMetalTextureGetTexture(dstTex)
        else { return nil }

        // CHW float -> RGBA half for the face texture
        let plane = 128 * 128
        face.withUnsafeBufferPointer { f in
            for i in 0..<plane {
                facePixels[i * 4] = Float16(f[i])
                facePixels[i * 4 + 1] = Float16(f[plane + i])
                facePixels[i * 4 + 2] = Float16(f[2 * plane + i])
            }
        }
        facePixels.withUnsafeBytes {
            faceTex.replace(region: MTLRegionMake2D(0, 0, 128, 128), mipmapLevel: 0,
                            withBytes: $0.baseAddress!, bytesPerRow: 128 * 8)
        }

        var p = Params(a: m.a, b: m.b, tx: m.tx, c: m.c, d: m.d, ty: m.ty)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        enc.setComputePipelineState(pipeline)
        enc.setTexture(s, index: 0); enc.setTexture(d, index: 1)
        enc.setTexture(faceTex, index: 2); enc.setTexture(maskTex, index: 3)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        // keep the CVMetalTextures alive until the GPU is done
        withExtendedLifetime((srcTex, dstTex)) {}
        return cb.status == .completed ? out : nil
    }

    private func texture(_ pb: CVPixelBuffer, _ cache: CVMetalTextureCache, _ w: Int, _ h: Int) -> CVMetalTexture? {
        var t: CVMetalTexture?
        let r = CVMetalTextureCacheCreateTextureFromImage(nil, cache, pb, nil, .bgra8Unorm, w, h, 0, &t)
        return r == kCVReturnSuccess ? t : nil
    }

    private func makeOutput(_ w: Int, _ h: Int) -> CVPixelBuffer? {
        if pool == nil || poolSize != (w, h) {
            let attrs: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: w,
                kCVPixelBufferHeightKey as String: h,
                kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var p: CVPixelBufferPool?
            CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &p)
            pool = p; poolSize = (w, h)
        }
        guard let pool else { return nil }
        var pb: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &pb)
        return pb
    }
}
