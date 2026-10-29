<p align="center"><img src="assets/icon_256.png" width="128" alt="FaceSwapMac icon"></p>

# FaceSwapMac

[Русский](README.md) · **English**

A native macOS app that swaps the face in your camera feed in real time, using a single photo. No training needed: pick a photo and the face in the frame is replaced right away. It runs on Apple Silicon Macs, and everything is processed locally, so nothing leaves your computer.

## Features

- Real-time face swap from one photo: choose a file or drag it onto the window
- Face detection runs on native Core ML (Neural Engine), and the face is blended in on the GPU with Metal
- Two-stage pipeline: the next frame is prepared while the current one is being swapped
- Landmark smoothing, mirrored preview, and camera selection (built-in, USB, or an iPhone via Continuity Camera)
- Russian and English UI
- An "AI · face swap" label on the preview, on by default

## Installation

You'll need an Apple Silicon Mac (M1 or newer) running macOS 14 Sonoma or later, about 1 GB of free disk space, and a camera.

### Option 1. Prebuilt app (easiest)

1. **Download the app.** Go to the [Releases](https://github.com/daniil7890/faceswapmac/releases/latest) page and download `FaceSwap-vX.Y.Z-arm64.zip`.
2. **Unzip it and move it to Applications.** Double-click the zip, then drag `FaceSwap.app` into your Applications folder.
3. **Remove the quarantine flag.** The app isn't signed with an Apple certificate, so macOS won't open it with a double-click. Open Terminal and run:
   ```bash
   xattr -dr com.apple.quarantine /Applications/FaceSwap.app
   ```
   Or right-click the app → Open → Open. If macOS still blocks it, go to System Settings → Privacy & Security, scroll down, and click "Open Anyway".
4. **Download the models.** This is about 460 MB and only needed once. In Terminal, run:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/daniil7890/faceswapmac/main/scripts/download_models.sh | bash
   ```
   This downloads three ONNX models and the Core ML detector into `~/Library/Application Support/FaceSwapMac/models`. Files that are already there are skipped.
5. **Launch FaceSwap.** Allow camera access when macOS asks.
6. **Wait for the first load.** CoreML compiles the models for your Mac, which takes up to a minute. Later launches are fast.
7. **Pick a face.** Drop a photo onto the left panel or click "Choose photo…". A frontal, large, well-lit face with nothing covering the eyes works best.

### Option 2. Build from source

1. Install the developer tools. You don't need full Xcode.
   ```bash
   xcode-select --install
   ```
2. Clone the repo, download the models, and build:
   ```bash
   git clone https://github.com/daniil7890/faceswapmac.git
   cd faceswapmac
   bash scripts/download_models.sh
   bash scripts/make_app.sh
   open build/FaceSwap.app
   ```
   The first build downloads ONNX Runtime and takes a few minutes. If it fails, the full log is in `build.log`.

### Updating

Download the new version from [Releases](https://github.com/daniil7890/faceswapmac/releases/latest) and replace the old app in Applications. Then repeat steps 3 and 4 (the `xattr` command and the model download). The script only fetches what's new. For example, when upgrading from 0.1.0 it downloads the Core ML detector.

### Troubleshooting

| Problem | Fix |
|---|---|
| "App is damaged" or it won't open | Repeat step 3 (the `xattr` command) |
| "Model file missing …" | Repeat step 4. `~/Library/Application Support/FaceSwapMac/models` should contain three `.onnx` files and a `coreml` folder |
| The sidebar says "ONNX CPU" instead of "Core ML" | The Core ML detector isn't installed. Repeat step 4. The app still works without it, just a bit slower |
| Black preview or "No camera access" | System Settings → Privacy & Security → Camera → turn on FaceSwap |
| "No face found in the photo" | Use another photo: frontal, larger face, better lighting |
| Low fps | Plug in power and close heavy apps. MacBook Air models run slower than Pro and Max |

### Uninstall

Delete `FaceSwap.app` from Applications and remove the `~/Library/Application Support/FaceSwapMac` folder.

## Usage

- **Language.** The Русский / English switch is at the top of the sidebar. On first launch the app follows your system language.
- **Smoothing.** Higher values make the face steadier but make it lag more on fast head movements.
- **Status line** at the bottom of the sidebar shows fps, per-frame swap time, and whether a face was found. Below it is the active backend. "Core ML · Metal" is what you want to see.

### Discord, OBS, Zoom

There's no built-in virtual camera yet, because a Camera Extension needs a Developer ID signature. Workaround with OBS: add a Window Capture source for FaceSwap, then click Start Virtual Camera. In Discord or Zoom, pick "OBS Virtual Camera".

## How it works

```
camera 720p ──► stage 1: SCRFD detector (Core ML, ~0.7 ms) ─► 5 landmarks ─► alignment ─► 128×128 crop
                                                                                              │
screen ◄── stage 2: soft-mask paste-back (Metal) ◄── inswapper_128 (ONNX + CoreML) ◄──────────┘
                                                            ▲
photo ──► SCRFD 640 ─► ArcFace ─► identity (512 numbers) ───┘
```

- **Why inswapper runs on ONNX Runtime instead of native Core ML.** Measured on a MacBook Air, ONNX Runtime with the CoreML provider runs the model in about 60 ms, and the best native Core ML variant takes about 108 ms. The detector is the opposite: native Core ML brought it from about 7 ms to 0.7 ms. That's why the setup is hybrid.
- **Speed.** inswapper sets the ceiling at about 60 ms per frame. That's roughly up to 15 fps on a MacBook Air, and more on Pro and Max chips.
- **Accuracy.** The preprocessing (alignment, crop, mask, paste-back) is written in Swift and checked against a Python reference. The Core ML models are checked against the ONNX outputs.

## For developers

### Project layout

| Path | Contents |
|---|---|
| `Sources/FaceSwapMac/App.swift` | SwiftUI UI, photo loading |
| `Camera.swift` | camera capture and the two-stage pipeline |
| `Engine.swift` | `prepare` (detect and align) and `finish` (inswapper and paste-back) stages, landmark smoothing |
| `CoreMLDetector.swift` | native Core ML detector with ONNX fallback |
| `MetalCompositor.swift` | GPU paste-back (compute shader) |
| `Models.swift` | ONNX Runtime wrappers, SCRFD decoder, reads `emap` straight from inswapper.onnx |
| `ImageOps.swift` | similarity alignment, warp, mask, CPU paste-back (fallback) |
| `Localization.swift` | Russian/English strings: `L("…", "…")` |
| `assets/` | app icon: `icon.svg` (source) and `AppIcon.icns` |
| `reference/` | Python reference and a numpy port of the Swift math for checking |
| `scripts/` | model download, `.app` build, Core ML conversion and benchmarks |
| `.github/workflows/` | CI build, releases, Core ML model publishing |

### Core ML models

`bash scripts/convert_coreml.sh` converts the models to `.mlpackage`, checks them against ONNX, and compares speed. `bash scripts/bench_coreml.sh` benchmarks inswapper separately on the GPU, Neural Engine, and CPU. The `coreml-models` workflow builds the published Core ML detector and uploads it to the `models-coreml-v1` release.

### Releasing a version

1. Bump the version in `scripts/make_app.sh` (`CFBundleShortVersionString`) and update `.github/RELEASE_NOTES.md`.
2. Actions → `release` → Run workflow → enter the version, e.g. `v0.3.0`. It builds the app, creates the tag and the release, and attaches the zip.

## Ideas

- Face enhancer (GFPGAN/CodeFormer) applied every other frame
- Built-in virtual camera via a Camera Extension
- A "whole head" mode that includes hair (LivePortrait)
- Pairing with [RVCMac](https://github.com/daniil7890/rvcmac) to change voice and face in one app

## Responsible use and license

Don't impersonate real people without their consent. In most countries that's illegal (fraud, defamation, deepfake laws), and it's simply wrong. The "AI" label on the preview is on by default.

The code is released under the [MIT License](LICENSE). The license doesn't cover the models: InsightFace (inswapper, ArcFace, SCRFD) allows them for non-commercial use only. The models aren't part of this repository; the script downloads them separately.
