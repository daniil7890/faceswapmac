"""ONNX -> Core ML (.mlpackage) for FaceSwapMac, with accuracy + speed check against ONNX Runtime.

Runs on macOS (Apple Silicon). Called by scripts/convert_coreml.sh, which sets up the venv.
Output: <models>/coreml/*.mlpackage and a summary at the end of convert.log.
"""
import os, sys, time, traceback
import numpy as np

MODELS = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                            else "~/Library/Application Support/FaceSwapMac/models")
OUT = os.path.join(MODELS, "coreml")
os.makedirs(OUT, exist_ok=True)

import onnx, onnxruntime as ort, torch, coremltools as ct
from onnx2torch import convert as onnx_to_torch

print(f"torch {torch.__version__} | coremltools {ct.__version__} | onnx {onnx.__version__} | ort {ort.__version__}")
print(f"models: {MODELS}")

rng = np.random.default_rng(0)

# name, onnx file, {input: shape}, input generator
def face_crop(shape):          # inswapper target: 0..1 image
    return rng.random(shape, dtype=np.float32)
def norm_img(shape):           # arcface / scrfd: roughly -1..1
    return (rng.random(shape, dtype=np.float32) * 2 - 1)
def latent(shape):             # unit-norm 512 vector
    v = rng.standard_normal(shape).astype(np.float32)
    return v / np.linalg.norm(v)

JOBS = [
    ("scrfd_320",  "scrfd_2.5g.onnx",        {"input": (1, 3, 320, 320)}, {"input": norm_img}),
    ("scrfd_640",  "scrfd_2.5g.onnx",        {"input": (1, 3, 640, 640)}, {"input": norm_img}),
    ("arcface",    "arcface_w600k_r50.onnx", {"input": (1, 3, 112, 112)}, {"input": norm_img}),
    ("inswapper",  "inswapper_128.onnx",     {"target": (1, 3, 128, 128), "source": (1, 512)},
                                             {"target": face_crop, "source": latent}),
]

def bench(fn, n=30):
    fn(); fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000

results = []
for name, file, shapes, gens in JOBS:
    print(f"\n=== {name} ({file}) ===", flush=True)
    path = os.path.join(MODELS, file)
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"нет {path}")
        feeds = {k: gens[k](s) for k, s in shapes.items()}

        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out_names = [o.name for o in sess.get_outputs()]
        ref = sess.run(None, feeds)

        model = onnx_to_torch(onnx.load(path)).float().eval()
        names = list(shapes)
        with torch.no_grad():
            args = tuple(torch.from_numpy(feeds[k]) for k in names)
            traced = torch.jit.trace(model, args, check_trace=False)

        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name=k, shape=shapes[k], dtype=np.float32) for k in names],
            outputs=[ct.TensorType(name=f"out{i}" if len(out_names) > 1 else "output")
                     for i in range(len(out_names))],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS14,
            compute_units=ct.ComputeUnit.ALL,
        )
        dst = os.path.join(OUT, f"{name}.mlpackage")
        mlmodel.save(dst)
        print(f"saved {dst}")

        pred = mlmodel.predict(feeds)
        keys = [f"out{i}" for i in range(len(out_names))] if len(out_names) > 1 else ["output"]
        worst_cos, worst_abs = 1.0, 0.0
        for k, r in zip(keys, ref):
            p = np.asarray(pred[k], dtype=np.float32).reshape(r.shape)
            a, b = r.ravel().astype(np.float64), p.ravel().astype(np.float64)
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            worst_cos = min(worst_cos, cos)
            worst_abs = max(worst_abs, float(np.abs(a - b).max()))
        print(f"accuracy vs ONNX: cos={worst_cos:.5f} max_abs={worst_abs:.4f}")

        t_ort = bench(lambda: sess.run(None, feeds))
        ort_cml = ort.InferenceSession(path, providers=["CoreMLExecutionProvider", "CPUExecutionProvider"])
        t_ort_cml = bench(lambda: ort_cml.run(None, feeds))
        t_ml = bench(lambda: mlmodel.predict(feeds))
        print(f"speed: ORT-CPU {t_ort:.1f} ms | ORT+CoreML-EP {t_ort_cml:.1f} ms | native Core ML {t_ml:.1f} ms")
        results.append((name, "OK", worst_cos, t_ort_cml, t_ml))
    except Exception as e:
        traceback.print_exc()
        results.append((name, f"FAIL: {type(e).__name__}: {e}"[:200], None, None, None))

print("\n================ SUMMARY ================")
for name, status, cos, t1, t2 in results:
    if status == "OK":
        print(f"{name:10s} OK   cos={cos:.5f}  ORT+CoreML {t1:6.1f} ms -> native {t2:6.1f} ms  (x{t1 / t2:.2f})")
    else:
        print(f"{name:10s} {status}")
