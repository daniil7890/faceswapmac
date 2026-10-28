# FaceSwapMac

Подмена лица в реальном времени на Mac (Apple Silicon) по одной фотографии. Обучать ничего не нужно.

Пайплайн: камера (AVFoundation, 720p BGRA), затем SCRFD-2.5g находит лицо и 5 точек. После этого ArcFace достаёт «личность» с фото, inswapper_128 рисует новое лицо, и оно вклеивается обратно с мягкой маской. Модели работают через ONNX Runtime с CoreML execution provider (Neural Engine/GPU). Предобработка написана на Swift и сверена с эталоном на Python.

## Запуск

```bash
bash scripts/download_models.sh     # ~460 МБ в ~/Library/Application Support/FaceSwapMac/models
bash scripts/make_app.sh            # хватает Command Line Tools (xcode-select --install), macOS 14+
open build/FaceSwap.app
```

При первом запуске CoreML компилирует модели, это до минуты. Потом они берутся из кэша.
Перетащи фото в левую панель: лучше всего анфас, лицо хорошо освещено и крупно.

## Язык

Интерфейс на русском и английском, переключатель вверху боковой панели.

## Чтобы картинка шла в Discord/OBS/Zoom

В v0.1 своей виртуальной камеры нет: Camera Extension требует подписи Developer ID.
Пока делай так: OBS → «Захват окна» → FaceSwap, затем «Запустить виртуальную камеру».

## Ожидаемая скорость

Около 20–30 fps на M1 Pro и новее. Бо́льшую часть времени занимает inswapper.
Если тормозит, проверь в Activity Monitor → GPU/ANE, что CoreML EP подхватился.

## Метка «AI»

По умолчанию в углу превью показана метка «AI · подмена лица», её можно выключить в настройках.
Не выдавай себя за реального человека без его согласия: в большинстве стран это незаконно
(мошенничество, клевета, deepfake-законы), да и просто подло.

## Структура

- `Sources/FaceSwapMac/ImageOps.swift` — similarity-выравнивание, warp, маска, вклейка
- `Models.swift` — обёртки ONNX Runtime, чтение `emap` прямо из inswapper.onnx
- `Engine.swift` — пайплайн и сглаживание точек
- `Camera.swift`, `App.swift` — захват и SwiftUI
- `reference/` — эталон на Python (`reference_swap.py`) и порт Swift-математики для сверки

## Идеи на потом

- Нативный Core ML вместо ORT (coremltools через onnx2torch) даст ещё больше fps
- Улучшайзер лица (GFPGAN/CodeFormer) через кадр
- Camera Extension, чтобы была своя виртуальная камера
- Связка с RVCMac: голос и лицо в одном приложении

Модели: InsightFace (inswapper, ArcFace, SCRFD), лицензия только для некоммерческого использования.
