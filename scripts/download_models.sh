#!/bin/bash
# Downloads the ONNX models (~460 MB) into ~/Library/Application Support/FaceSwapMac/models
set -euo pipefail
DIR="${1:-$HOME/Library/Application Support/FaceSwapMac/models}"
BASE="https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0"
mkdir -p "$DIR"
for f in scrfd_2.5g.onnx arcface_w600k_r50.onnx inswapper_128_fp16.onnx; do
  if [ -s "$DIR/$f" ]; then echo "✓ $f"; continue; fi
  echo "↓ $f"
  curl -fL --progress-bar -o "$DIR/$f.part" "$BASE/$f"
  mv "$DIR/$f.part" "$DIR/$f"
done
echo "Модели в: $DIR"
