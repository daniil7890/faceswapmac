# FaceSwapMac

[Русский](README.md) · **English**

Real-time face swap on Apple Silicon Macs from a single photo. No training needed.

The camera feed (AVFoundation, 720p BGRA) goes to SCRFD-2.5g, which finds the face and 5 landmarks. ArcFace takes the identity from your photo, inswapper_128 renders the new face, and a soft mask blends it back into the frame. The models run on ONNX Runtime with the CoreML execution provider (Neural Engine/GPU). Preprocessing is written in Swift and checked against a Python reference.

## Download

There's a prebuilt app on the [Releases](https://github.com/daniil7890/faceswapmac/releases) page. It's unsigned, so the first time you open it, right-click → Open, or run:

```bash
xattr -dr com.apple.quarantine FaceSwap.app
```

The models aren't bundled with the app. Get them with `scripts/download_models.sh` (see below).

## Build from source

```bash
bash scripts/download_models.sh     # ~460 MB into ~/Library/Application Support/FaceSwapMac/models
bash scripts/make_app.sh            # Command Line Tools are enough (xcode-select --install), macOS 14+
open build/FaceSwap.app
```

On first launch CoreML compiles the models, which takes up to a minute. After that they load from cache.
Drop a photo onto the left panel. A frontal, well-lit, large face works best.

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
