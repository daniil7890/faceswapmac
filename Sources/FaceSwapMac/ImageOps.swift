import Foundation

struct Pt { var x: Float; var y: Float }

/// 2x3 affine: (x, y) -> (a*x + b*y + tx, c*x + d*y + ty). Same layout as cv2's M.
struct Affine {
    var a: Float, b: Float, tx: Float
    var c: Float, d: Float, ty: Float

    @inline(__always) func apply(_ x: Float, _ y: Float) -> (Float, Float) {
        (a * x + b * y + tx, c * x + d * y + ty)
    }

    var inverse: Affine {
        let det = a * d - b * c
        let ia = d / det, ib = -b / det, ic = -c / det, id = a / det
        return Affine(a: ia, b: ib, tx: -(ia * tx + ib * ty),
                      c: ic, d: id, ty: -(ic * tx + id * ty))
    }

    /// Least-squares similarity transform src -> dst (rotation + uniform scale + shift).
    /// Equals skimage/insightface Umeyama estimate for non-reflected point sets.
    static func similarity(from src: [Pt], to dst: [Pt]) -> Affine {
        let n = Float(src.count)
        var sx: Float = 0, sy: Float = 0, dx: Float = 0, dy: Float = 0
        for i in 0..<src.count { sx += src[i].x; sy += src[i].y; dx += dst[i].x; dy += dst[i].y }
        sx /= n; sy /= n; dx /= n; dy /= n
        var num1: Float = 0, num2: Float = 0, den: Float = 0
        for i in 0..<src.count {
            let ux = src[i].x - sx, uy = src[i].y - sy
            let vx = dst[i].x - dx, vy = dst[i].y - dy
            num1 += ux * vx + uy * vy
            num2 += ux * vy - uy * vx
            den += ux * ux + uy * uy
        }
        let a = num1 / den, b = num2 / den
        return Affine(a: a, b: -b, tx: dx - (a * sx - b * sy),
                      c: b, d: a, ty: dy - (b * sx + a * sy))
    }
}

/// ArcFace 5-point template for 112x112: left eye, right eye, nose, left mouth, right mouth.
let arcfaceTemplate112: [Pt] = [
    Pt(x: 38.2946, y: 51.6963), Pt(x: 73.5318, y: 51.5014), Pt(x: 56.0252, y: 71.7366),
    Pt(x: 41.5493, y: 92.3655), Pt(x: 70.7299, y: 92.2041),
]

/// Non-owning view over a BGRA8 buffer (CVPixelBuffer or our own bitmap).
struct BGRAView {
    let base: UnsafeMutablePointer<UInt8>
    let width: Int
    let height: Int
    let rowBytes: Int

    /// Bilinear sample, black outside the image (cv2 BORDER_CONSTANT 0). Returns R, G, B in 0...255.
    @inline(__always) func sample(_ x: Float, _ y: Float) -> (Float, Float, Float) {
        let x0f = floorf(x), y0f = floorf(y)
        let x0 = Int(x0f), y0 = Int(y0f)
        let fx = x - x0f, fy = y - y0f
        var r: Float = 0, g: Float = 0, b: Float = 0
        @inline(__always) func acc(_ px: Int, _ py: Int, _ w: Float) {
            guard px >= 0, py >= 0, px < width, py < height, w > 0 else { return }
            let p = base + py * rowBytes + px * 4
            b += Float(p[0]) * w; g += Float(p[1]) * w; r += Float(p[2]) * w
        }
        acc(x0, y0, (1 - fx) * (1 - fy)); acc(x0 + 1, y0, fx * (1 - fy))
        acc(x0, y0 + 1, (1 - fx) * fy);   acc(x0 + 1, y0 + 1, fx * fy)
        return (r, g, b)
    }
}

enum ImageOps {
    /// Letterbox (top-left aligned, like insightface) into size x size, CHW RGB, (v - 127.5) / 128.
    /// Returns tensor data and the scale used.
    static func letterboxTensor(_ img: BGRAView, size: Int) -> ([Float], Float) {
        let scale = min(Float(size) / Float(img.height), Float(size) / Float(img.width))
        let nw = Int(Float(img.width) * scale), nh = Int(Float(img.height) * scale)
        let plane = size * size
        var t = [Float](repeating: -127.5 / 128, count: 3 * plane)
        let sx = Float(img.width) / Float(nw), sy = Float(img.height) / Float(nh)
        t.withUnsafeMutableBufferPointer { out in
            for y in 0..<nh {
                let srcY = (Float(y) + 0.5) * sy - 0.5
                for x in 0..<nw {
                    let (r, g, b) = img.sample((Float(x) + 0.5) * sx - 0.5, srcY)
                    let i = y * size + x
                    out[i] = (r - 127.5) / 128
                    out[plane + i] = (g - 127.5) / 128
                    out[2 * plane + i] = (b - 127.5) / 128
                }
            }
        }
        return (t, scale)
    }

    /// cv2.warpAffine(img, M, (size, size)) -> CHW RGB tensor, each value mapped by `norm`.
    static func warpTensor(_ img: BGRAView, _ m: Affine, size: Int, norm: (Float) -> Float) -> [Float] {
        let inv = m.inverse
        let plane = size * size
        var t = [Float](repeating: 0, count: 3 * plane)
        t.withUnsafeMutableBufferPointer { out in
            for y in 0..<size {
                for x in 0..<size {
                    let (sx, sy) = inv.apply(Float(x), Float(y))
                    let (r, g, b) = img.sample(sx, sy)
                    let i = y * size + x
                    out[i] = norm(r); out[plane + i] = norm(g); out[2 * plane + i] = norm(b)
                }
            }
        }
        return t
    }

    /// Soft mask 128x128: inset box blurred by a Gaussian (matches reference_swap.py).
    static func makeMask(size: Int = 128, inset: Int = 10, sigma: Float = 6) -> [Float] {
        var m = [Float](repeating: 0, count: size * size)
        for y in inset..<(size - inset) { for x in inset..<(size - inset) { m[y * size + x] = 1 } }
        let r = Int(ceilf(sigma * 4))
        var k = (-r...r).map { expf(-Float($0 * $0) / (2 * sigma * sigma)) }
        let s = k.reduce(0, +); k = k.map { $0 / s }
        func reflect(_ i: Int) -> Int { // cv2 BORDER_REFLECT_101
            var i = i
            while i < 0 || i >= size { i = i < 0 ? -i : 2 * size - 2 - i }
            return i
        }
        var tmp = m
        for y in 0..<size { for x in 0..<size {
            var v: Float = 0
            for j in -r...r { v += m[y * size + reflect(x + j)] * k[j + r] }
            tmp[y * size + x] = v
        } }
        for y in 0..<size { for x in 0..<size {
            var v: Float = 0
            for j in -r...r { v += tmp[reflect(y + j) * size + x] * k[j + r] }
            m[y * size + x] = v
        } }
        return m
    }

    /// Paste swapped 128x128 face (CHW RGB 0...1) back into the frame in place, blended by mask.
    /// `m` maps frame -> crop coordinates.
    static func pasteBack(into img: BGRAView, face: [Float], mask: [Float], m: Affine, size: Int = 128) {
        let inv = m.inverse
        var minX = Float.greatestFiniteMagnitude, minY = minX, maxX = -minX, maxY = -minX
        for (cx, cy) in [(0, 0), (size, 0), (0, size), (size, size)] {
            let (x, y) = inv.apply(Float(cx), Float(cy))
            minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y)
        }
        let x0 = max(0, Int(floorf(minX))), x1 = min(img.width - 1, Int(ceilf(maxX)))
        let y0 = max(0, Int(floorf(minY))), y1 = min(img.height - 1, Int(ceilf(maxY)))
        guard x0 <= x1, y0 <= y1 else { return }
        let plane = size * size
        face.withUnsafeBufferPointer { f in mask.withUnsafeBufferPointer { mk in
            @inline(__always) func bil(_ buf: UnsafeBufferPointer<Float>, _ off: Int, _ fx: Float, _ fy: Float,
                                       _ ix: Int, _ iy: Int) -> Float {
                @inline(__always) func at(_ x: Int, _ y: Int) -> Float {
                    (x < 0 || y < 0 || x >= size || y >= size) ? 0 : buf[off + y * size + x]
                }
                return at(ix, iy) * (1 - fx) * (1 - fy) + at(ix + 1, iy) * fx * (1 - fy)
                     + at(ix, iy + 1) * (1 - fx) * fy + at(ix + 1, iy + 1) * fx * fy
            }
            for y in y0...y1 {
                let row = img.base + y * img.rowBytes
                for x in x0...x1 {
                    let (cx, cy) = m.apply(Float(x), Float(y))
                    if cx < -1 || cy < -1 || cx > Float(size) || cy > Float(size) { continue }
                    let ixf = floorf(cx), iyf = floorf(cy)
                    let ix = Int(ixf), iy = Int(iyf), fx = cx - ixf, fy = cy - iyf
                    let a = bil(mk, 0, fx, fy, ix, iy)
                    if a <= 0.002 { continue }
                    let r = min(max(bil(f, 0, fx, fy, ix, iy), 0), 1) * 255
                    let g = min(max(bil(f, plane, fx, fy, ix, iy), 0), 1) * 255
                    let b = min(max(bil(f, 2 * plane, fx, fy, ix, iy), 0), 1) * 255
                    let p = row + x * 4
                    p[0] = UInt8(min(255, max(0, b * a + Float(p[0]) * (1 - a) + 0.5)))
                    p[1] = UInt8(min(255, max(0, g * a + Float(p[1]) * (1 - a) + 0.5)))
                    p[2] = UInt8(min(255, max(0, r * a + Float(p[2]) * (1 - a) + 0.5)))
                }
            }
        } }
    }
}
