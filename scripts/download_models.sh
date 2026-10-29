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

# Native Core ML face detector (optional: without it the app falls back to ONNX)
CML="https://github.com/daniil7890/faceswapmac/releases/download/models-coreml-v1/scrfd_320.mlpackage.zip"
if [ -d "$DIR/coreml/scrfd_320.mlpackage" ]; then
  echo "✓ coreml/scrfd_320.mlpackage"
else
  echo "↓ coreml/scrfd_320.mlpackage"
  mkdir -p "$DIR/coreml"
  if curl -fL --progress-bar -o "$DIR/coreml/scrfd_320.zip" "$CML"; then
    ditto -x -k "$DIR/coreml/scrfd_320.zip" "$DIR/coreml" && rm -f "$DIR/coreml/scrfd_320.zip"
  else
    echo "  (не удалось скачать Core ML детектор, приложение будет использовать ONNX)"
  fi
fi
echo "Модели в: $DIR"
