"""Line-by-line numpy port of ImageOps.swift / Engine.swift, run against the cv2 reference.
Checks that the Swift math (letterbox, closed-form similarity, warp, mask, paste) gives the same swap."""
import sys, os, numpy as np, cv2, onnxruntime as ort
sys.path.insert(0, os.path.dirname(__file__))
import reference_swap as R

def sample(img, xs, ys):  # BGRAView.sample, black border; img is RGB float
    h, w = img.shape[:2]
    x0, y0 = np.floor(xs).astype(int), np.floor(ys).astype(int)
    fx, fy = (xs - x0)[..., None], (ys - y0)[..., None]
    out = np.zeros(xs.shape + (3,), np.float32)
    for dx, dy, wt in [(0, 0, (1-fx)*(1-fy)), (1, 0, fx*(1-fy)), (0, 1, (1-fx)*fy), (1, 1, fx*fy)]:
        px, py = x0 + dx, y0 + dy
        ok = (px >= 0) & (py >= 0) & (px < w) & (py < h)
        v = np.zeros_like(out); v[ok] = img[py[ok], px[ok]]
        out += v * wt
    return out

def similarity(src, dst):  # Affine.similarity
    s, d = src.mean(0), dst.mean(0); u, v = src - s, dst - d
    den = (u ** 2).sum(); a = (u * v).sum() / den; b = (u[:, 0]*v[:, 1] - u[:, 1]*v[:, 0]).sum() / den
    return np.array([[a, -b, d[0] - (a*s[0] - b*s[1])], [b, a, d[1] - (b*s[0] + a*s[1])]], np.float32)

def inv(m): return cv2.invertAffineTransform(m)

def letterbox(img, size):
    h, w = img.shape[:2]; scale = min(size / h, size / w)
    nw, nh = int(w * scale), int(h * scale)
    t = np.full((size, size, 3), -127.5 / 128, np.float32)
    yy, xx = np.mgrid[:nh, :nw].astype(np.float32)
    t[:nh, :nw] = (sample(img, (xx + .5) * w / nw - .5, (yy + .5) * h / nh - .5) - 127.5) / 128
    return t.transpose(2, 0, 1)[None], scale

def detect(sess, img, size, thr=0.5):
    x, scale = letterbox(img, size)
    outs = sess.run(None, {'input': x}); best = None
    for k, st in enumerate([8, 16, 32]):
        sc, kp = outs[k][:, 0], outs[k + 6]; g = size // st
        for i in np.where(sc > thr)[0]:
            if best is None or sc[i] > best[1]:
                c = i // 2; ax, ay = c % g * st, c // g * st
                best = ((np.array([ax, ay]) + kp[i].reshape(5, 2) * st) / scale, sc[i])
    return best[0].astype(np.float32)

def warp(img, m, size, norm):
    yy, xx = np.mgrid[:size, :size].astype(np.float32); mi = inv(m)
    sx = mi[0, 0]*xx + mi[0, 1]*yy + mi[0, 2]; sy = mi[1, 0]*xx + mi[1, 1]*yy + mi[1, 2]
    return norm(sample(img, sx, sy)).transpose(2, 0, 1)[None].astype(np.float32)

def make_mask(size=128, inset=10, sigma=6):
    m = np.zeros((size, size), np.float32); m[inset:size-inset, inset:size-inset] = 1
    r = int(np.ceil(sigma * 4)); k = np.exp(-np.arange(-r, r+1)**2 / (2*sigma*sigma)); k /= k.sum()
    return cv2.sepFilter2D(m, -1, k.astype(np.float32), k.astype(np.float32), borderType=cv2.BORDER_REFLECT_101)

def paste(img, face, mask, m, size=128):
    h, w = img.shape[:2]; mi = inv(m)
    c = np.array([[0, 0], [size, 0], [0, size], [size, size]], np.float32) @ mi[:, :2].T + mi[:, 2]
    x0, x1 = max(0, int(np.floor(c[:, 0].min()))), min(w-1, int(np.ceil(c[:, 0].max())))
    y0, y1 = max(0, int(np.floor(c[:, 1].min()))), min(h-1, int(np.ceil(c[:, 1].max())))
    yy, xx = np.mgrid[y0:y1+1, x0:x1+1].astype(np.float32)
    cx = m[0, 0]*xx + m[0, 1]*yy + m[0, 2]; cy = m[1, 0]*xx + m[1, 1]*yy + m[1, 2]
    a = sample(np.repeat(mask[..., None], 3, 2), cx, cy)[..., :1]
    f = sample(face.clip(0, 1) * 255, cx, cy)
    out = img.copy(); reg = out[y0:y1+1, x0:x1+1]
    out[y0:y1+1, x0:x1+1] = np.where(a > 0.002, f * a + reg * (1 - a), reg)
    return out

def main(src_p, tgt_p, out_p, models):
    P = ['CPUExecutionProvider']
    det = ort.InferenceSession(f'{models}/scrfd_2.5g.onnx', providers=P)
    arc = ort.InferenceSession(f'{models}/arcface_w600k_r50.onnx', providers=P)
    swp = ort.InferenceSession(f'{models}/inswapper_128_fp16.onnx', providers=P)
    emap = np.fromfile(f'{models}/emap.bin', np.float32).reshape(512, 512)
    rgb = lambda p: cv2.imread(p)[:, :, ::-1].astype(np.float32)
    src, tgt = rgb(src_p), rgb(tgt_p)
    kps = detect(det, src, 640)
    e = arc.run(None, {'input': warp(src, similarity(kps, R.ARCFACE_112), 112, lambda v: (v-127.5)/127.5)})[0][0]
    e /= np.linalg.norm(e); lat = e @ emap; lat = (lat / np.linalg.norm(lat))[None].astype(np.float32)
    kps = detect(det, tgt, 320)
    m = similarity(kps, R.ARCFACE_112 * 128 / 112)
    y = swp.run(None, {'target': warp(tgt, m, 128, lambda v: v/255), 'source': lat})[0][0].transpose(1, 2, 0)
    out = paste(tgt, y, make_mask(), m)
    cv2.imwrite(out_p, out[:, :, ::-1].clip(0, 255).astype(np.uint8))
    # identity check with the independent cv2 reference pipeline
    rdet = ort.InferenceSession(f'{models}/retinaface_10g.onnx', providers=P)
    bgr = lambda p: cv2.imread(p)
    def emb(i): return R.embed(arc, i, R.detect(rdet, i))
    s0 = emb(bgr(src_p)); print(f'swift-port: cos(source,result)={float(s0 @ emb(bgr(out_p))):.3f}')

if __name__ == '__main__':
    main(*sys.argv[1:5])
