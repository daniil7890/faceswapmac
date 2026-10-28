#!/bin/bash
# Builds a release binary and wraps it into FaceSwap.app (needed for the camera permission prompt).
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Сборка… (первый раз качается ONNX Runtime, пару минут)"
set +e
swift build -c release --arch arm64 > build.log 2>&1
rc=$?
set -e
if [ $rc -ne 0 ]; then
  echo "Ошибки сборки (полный лог: build.log):"
  grep -E "error:" -A3 build.log | head -60
  exit 1
fi
APP="build/FaceSwap.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
BIN="$(swift build -c release --arch arm64 --show-bin-path)/FaceSwapMac"
[ -f "$BIN" ] || BIN="$(find .build -type f -perm -u+x -name FaceSwapMac -path '*elease*' | head -1)"
[ -f "$BIN" ] || { echo "Не нашла собранный бинарник FaceSwapMac в .build"; exit 1; }
cp "$BIN" "$APP/Contents/MacOS/FaceSwap"
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>FaceSwap</string>
  <key>CFBundleDisplayName</key><string>FaceSwap</string>
  <key>CFBundleIdentifier</key><string>local.faceswapmac</string>
  <key>CFBundleExecutable</key><string>FaceSwap</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>Камера нужна для подмены лица в реальном времени.</string>
</dict></plist>
PLIST
codesign --force --deep --sign - "$APP"
echo "Готово: $APP"
