"""Where should inswapper run? Benchmarks Core ML compute units and precisions vs ONNX Runtime.
Uses the venv and models made by convert_coreml.sh. Called by scripts/bench_coreml.sh."""
import os, sys, time, traceback
import numpy as np
import onnx, onnxruntime as ort, torch, coremltools as ct
from onnx2torch import convert as onnx_to_torch

MODELS = os.path.expanduser(sys.argv[1])
CML = os.path.join(MODELS, "coreml")
rng = np.random.default_rng(0)
lat = rng.standard_normal((1, 512)).astype(np.float32); lat /= np.linalg.norm(lat)
swap_feeds = {"target": rng.random((1, 3, 128, 128), dtype=np.float32), "source": lat}
det_feeds = {"input": (rng.random((1, 3, 320, 320), dtype=np.float32) * 2 - 1)}

def bench(fn, n=40):
    for _ in range(3): fn()
    t = time.perf_counter()
    for _ in range(n): fn()
    return (time.perf_counter() - t) / n * 1000

UNITS = [("ALL", ct.ComputeUnit.ALL), ("CPU+GPU", ct.ComputeUnit.CPU_AND_GPU),
         ("CPU+ANE", ct.ComputeUnit.CPU_AND_NE), ("CPU", ct.ComputeUnit.CPU_ONLY)]
rows = []

def run(label, fn):
    try:
        ms = bench(fn); rows.append((label, f"{ms:7.1f} ms")); print(f"{label:42s} {ms:7.1f} ms", flush=True)
    except Exception as e:
        rows.append((label, f"FAIL {type(e).__name__}")); traceback.print_exc()

print("== ONNX Runtime (как сейчас в приложении)")
for f in ["inswapper_128_fp16.onnx", "inswapper_128.onnx"]:
    p = os.path.join(MODELS, f)
    if os.path.exists(p):
        s = ort.InferenceSession(p, providers=[("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}),
                                               "CPUExecutionProvider"])
        run(f"ORT MLProgram EP  {f}", lambda s=s: s.run(None, swap_feeds))

print("== Core ML fp16 inswapper, разные вычислители")
for name, cu in UNITS:
    m = ct.models.MLModel(os.path.join(CML, "inswapper.mlpackage"), compute_units=cu)
    run(f"CoreML fp16 inswapper  {name}", lambda m=m: m.predict(swap_feeds))

print("== Core ML fp32 inswapper (конвертирую…)")
try:
    path = os.path.join(MODELS, "inswapper_128.onnx")
    model = onnx_to_torch(onnx.load(path)).float().eval()
    with torch.no_grad():
        tr = torch.jit.trace(model, (torch.from_numpy(swap_feeds["target"]), torch.from_numpy(lat)), check_trace=False)
    m32 = ct.convert(tr, inputs=[ct.TensorType(name="target", shape=(1, 3, 128, 128)), ct.TensorType(name="source", shape=(1, 512))],
                     outputs=[ct.TensorType(name="output")], convert_to="mlprogram",
                     compute_precision=ct.precision.FLOAT32, minimum_deployment_target=ct.target.macOS14)
    dst = os.path.join(CML, "inswapper_fp32.mlpackage"); m32.save(dst)
    for name, cu in UNITS[:2]:
        m = ct.models.MLModel(dst, compute_units=cu)
        run(f"CoreML fp32 inswapper  {name}", lambda m=m: m.predict(swap_feeds))
except Exception:
    traceback.print_exc()

print("== Детектор scrfd_320")
for name, cu in UNITS:
    m = ct.models.MLModel(os.path.join(CML, "scrfd_320.mlpackage"), compute_units=cu)
    run(f"CoreML scrfd_320  {name}", lambda m=m: m.predict(det_feeds))

print("\n================ BENCH SUMMARY ================")
for label, v in rows: print(f"{label:42s} {v}")
