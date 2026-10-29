#!/bin/bash
# Converts the ONNX models to native Core ML and compares accuracy/speed. Log: convert.log
set -uo pipefail
cd "$(dirname "$0")/.."
MODELS="${1:-$HOME/Library/Application Support/FaceSwapMac/models}"
exec > >(tee convert.log) 2>&1

echo "== Python"
PY=""
for c in /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "Нет python3. Установи: xcode-select --install"; exit 1; }
"$PY" --version

if [ ! -x .venv-coreml/bin/python ]; then
  echo "== Создаю окружение .venv-coreml (один раз, ~1-2 ГБ)"
  "$PY" -m venv .venv-coreml || exit 1
fi
V=.venv-coreml/bin/python
$V -m pip install -q --upgrade pip
echo "== Ставлю torch, coremltools, onnx2torch… (первый раз несколько минут)"
$V -m pip install -q "torch==2.5.1" "coremltools==8.3.0" "numpy<2" onnx onnx2torch onnxruntime || exit 1

# float32 inswapper converts cleanly; the app keeps using the fp16 one until we switch
BASE="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
mkdir -p "$MODELS"
for f in scrfd_2.5g.onnx arcface_w600k_r50.onnx inswapper_128.onnx; do
  if [ ! -s "$MODELS/$f" ]; then
    echo "== Скачиваю $f"
    curl -fL --progress-bar -o "$MODELS/$f.part" "$BASE/$f" && mv "$MODELS/$f.part" "$MODELS/$f"
  fi
done

echo "== Конвертация"
$V scripts/convert_coreml.py "$MODELS"
echo "== Готово. Лог: convert.log"
