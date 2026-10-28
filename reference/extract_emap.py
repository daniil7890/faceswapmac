"""Minimal protobuf walk of an ONNX file: pull the 512x512 'emap' initializer of inswapper.
No onnx package needed. Usage: python3 extract_emap.py inswapper.onnx out.bin"""
import sys, numpy as np

def varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80: return r, i

def fields(b, start, end):
    i = start
    while i < end:
        key, i = varint(b, i)
        f, w = key >> 3, key & 7
        if w == 0:
            v, i = varint(b, i); yield f, w, v, None
        elif w == 1: yield f, w, b[i:i+8], None; i += 8
        elif w == 5: yield f, w, b[i:i+4], None; i += 4
        elif w == 2:
            n, i = varint(b, i); yield f, w, i, i + n; i += n
        else: raise ValueError(w)

data = open(sys.argv[1], 'rb').read()
dtypes = {1: np.float32, 10: np.float16}
best = None
for f, w, s, e in fields(data, 0, len(data)):
    if f != 7 or w != 2: continue           # ModelProto.graph
    for gf, gw, gs, ge in fields(data, s, e):
        if gf != 5 or gw != 2: continue     # GraphProto.initializer
        dims, dt, name, raw = [], None, b'', None
        for tf, tw, ts, te in fields(data, gs, ge):
            if tf == 1:
                if tw == 0: dims.append(ts)
                else:
                    j = ts
                    while j < te: v, j = varint(data, j); dims.append(v)
            elif tf == 2: dt = ts
            elif tf == 8: name = data[ts:te]
            elif tf == 9: raw = (ts, te)
        if dims == [512, 512] and raw:
            best = (name, dt, raw)          # last 512x512 initializer = emap
name, dt, (s, e) = best
emap = np.frombuffer(data[s:e], dtype=dtypes[dt]).reshape(512, 512).astype(np.float32)
emap.tofile(sys.argv[2])
print('emap', name, dt, emap.shape, float(emap.mean()), float(emap.std()))
