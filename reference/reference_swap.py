"""Reference pipeline in Python (ONNX Runtime) — the exact math the Swift app reproduces.
Usage: python3 reference_swap.py source.jpg target.jpg out.jpg --models ../models"""
import argparse, os, numpy as np, cv2, onnxruntime as ort

# ArcFace 5-point template for 112x112 (left eye, right eye, nose, left mouth, right mouth)
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)

def umeyama(src, dst):
    """Similarity transform src->dst (2x3), same as skimage SimilarityTransform."""
    n = src.shape[0]
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0: d[1] = -1
    R = U @ np.diag(d) @ Vt
    var = (sc ** 2).sum() / n
    s = (S * d).sum() / var
    t = mu_d - s * R @ mu_s
    return np.hstack([s * R, t[:, None]]).astype(np.float32)

def detect(sess, img, size=640, thr=0.5):
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    nh, nw = int(h * scale), int(w * scale)
    canvas = np.zeros((size, size, 3), np.uint8)
    canvas[:nh, :nw] = cv2.resize(img, (nw, nh))
    x = (canvas[:, :, ::-1].astype(np.float32) - 127.5) / 128.0
    outs = sess.run(None, {'input': x.transpose(2, 0, 1)[None]})
    best = None
    for k, stride in enumerate([8, 16, 32]):
        sc, bb, kp = outs[k][:, 0], outs[k + 3], outs[k + 6]
        g = size // stride
        yy, xx = np.mgrid[:g, :g]
        anchors = np.stack([xx, yy], -1).reshape(-1, 2).repeat(2, 0) * stride
        for i in np.where(sc > thr)[0]:
            if best is None or sc[i] > best[0]:
                kps = (anchors[i] + kp[i].reshape(5, 2) * stride) / scale
                best = (sc[i], kps)
    return None if best is None else best[1].astype(np.float32)

def embed(arc, img, kps):
    M = umeyama(kps, ARCFACE_112)
    crop = cv2.warpAffine(img, M, (112, 112), borderValue=0)
    x = (crop[:, :, ::-1].astype(np.float32) - 127.5) / 127.5
    e = arc.run(None, {'input': x.transpose(2, 0, 1)[None]})[0][0]
    return e / np.linalg.norm(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source'); ap.add_argument('target'); ap.add_argument('out')
    ap.add_argument('--models', default='../models')
    a = ap.parse_args()
    P = ['CPUExecutionProvider']
    det = ort.InferenceSession(os.path.join(a.models, 'retinaface_10g.onnx'), providers=P)
    arc = ort.InferenceSession(os.path.join(a.models, 'arcface_w600k_r50.onnx'), providers=P)
    swp = ort.InferenceSession(os.path.join(a.models, 'inswapper_128_fp16.onnx'), providers=P)
    emap = np.fromfile(os.path.join(a.models, 'emap.bin'), np.float32).reshape(512, 512)

    src, tgt = cv2.imread(a.source), cv2.imread(a.target)
    latent = embed(arc, src, detect(det, src)) @ emap
    latent = (latent / np.linalg.norm(latent))[None].astype(np.float32)

    kps = detect(det, tgt)
    M = umeyama(kps, ARCFACE_112 * (128 / 112))
    crop = cv2.warpAffine(tgt, M, (128, 128), borderValue=0)
    x = (crop[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    y = swp.run(None, {'target': x, 'source': latent})[0][0]
    face = (y.transpose(1, 2, 0)[:, :, ::-1].clip(0, 1) * 255).astype(np.float32)

    # soft mask: inset box with blurred edges (Swift app does the same on GPU)
    mask = np.zeros((128, 128), np.float32)
    mask[10:118, 10:118] = 1
    mask = cv2.GaussianBlur(mask, (0, 0), 6)
    Mi = cv2.invertAffineTransform(M)
    h, w = tgt.shape[:2]
    fw = cv2.warpAffine(face, Mi, (w, h))
    mw = cv2.warpAffine(mask, Mi, (w, h))[:, :, None]
    out = fw * mw + tgt.astype(np.float32) * (1 - mw)
    cv2.imwrite(a.out, out.clip(0, 255).astype(np.uint8))

    # sanity: identity of the result should be close to the source
    def sim(i1, i2): return float(embed(arc, i1, detect(det, i1)) @ embed(arc, i2, detect(det, i2)))
    res = cv2.imread(a.out)
    print(f'cos(source,target)={sim(src, tgt):.3f}  cos(source,result)={sim(src, res):.3f}')

if __name__ == '__main__':
    main()
