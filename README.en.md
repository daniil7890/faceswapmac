# FaceSwapMac

[Русский](README.md) · **English**

Real-time face swap on Apple Silicon Macs from a single photo. No training needed.

The camera feed (AVFoundation, 720p BGRA) goes to SCRFD-2.5g, which finds the face and 5 landmarks. ArcFace takes the identity from your photo, inswapper_128 renders the new face, and a soft mask blends it back into the frame. The models run on ONNX Runtime with the CoreML execution provider (Neural Engine/GPU). Preprocessing is written in Swift and checked against a Python reference.

## Installation

You'll need an Apple Silicon Mac (M1 or newer) running macOS 14 Sonoma or later, about 1 GB of free disk space, and a camera (built-in, USB, or an iPhone via Continuity Camera).

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
   The models are saved to `~/Library/Application Support/FaceSwapMac/models`.
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

### Troubleshooting

| Problem | Fix |
|---|---|
| "App is damaged" or it won't open | Repeat step 3 (the `xattr` command) |
| "Model file missing …" | Repeat step 4. There should be three `.onnx` files in `~/Library/Application Support/FaceSwapMac/models` |
| Black preview or "No camera access" | System Settings → Privacy & Security → Camera → turn on FaceSwap |
| "No face found in the photo" | Use another photo: frontal, larger face, better lighting |
| Low fps | Close heavy apps and plug in power. M1/M2 Air models run slower than the Pro models |

### Uninstall

Delete `FaceSwap.app` from Applications and remove the `~/Library/Application Support/FaceSwapMac` folder.

## Language

The UI is available in Russian and English. Use the switch at the top of the sidebar. The first launch follows your system language.

## Using it in Discord/OBS/Zoom

v0.1 has no virtual camera of its own, because a Camera Extension needs a Developer ID signature.
For now, add FaceSwap in OBS as a Window Capture source, then click Start Virtual Camera.

## Performance

Expect about 20–30 fps on M1 Pro and newer. Most of the time goes to inswapper.
If it's slow, open Activity Monitor and check GPU/ANE usage to confirm the CoreML EP is active.

## “AI” label

By default the preview shows an “AI · face swap” label in the corner. You can turn it off in settings.
Don't impersonate real people without their consent. In most countries that's illegal (fraud, defamation, deepfake laws), and it's simply wrong.

## Project layout

- `Sources/FaceSwapMac/ImageOps.swift`: similarity alignment, warp, mask, paste-back
- `Models.swift`: ONNX Runtime wrappers, reads `emap` directly from inswapper.onnx
- `Engine.swift`: the pipeline and landmark smoothing
- `Camera.swift`, `App.swift`: capture and SwiftUI
- `Localization.swift`: Russian/English strings (`L("…", "…")`)
- `reference/`: Python reference (`reference_swap.py`) and a numpy port of the Swift math used for checking it

## Ideas

- Native Core ML instead of ORT, for more fps
- Face enhancer (GFPGAN/CodeFormer) applied every other frame
- Camera Extension for a built-in virtual camera
- Pairing with [RVCMac](https://github.com/daniil7890/rvcmac) to change voice and face in one app

Models: InsightFace (inswapper, ArcFace, SCRFD). Their license allows non-commercial use only.
