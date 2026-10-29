**Русский** · English ниже

## 0.2.1

- У приложения появилась иконка. В сборку 0.2.0 она не попала.

## Что нового в 0.2.0: быстрее

- **Детектор лица на нативном Core ML (Neural Engine).** На каждом кадре он работает около 0,7 мс вместо ~7 мс. Если модели Core ML нет, приложение само переключается на ONNX.
- **Конвейер из двух потоков.** Пока inswapper обрабатывает кадр, следующий кадр уже проходит поиск лица и выравнивание.
- **Вклейка лица на GPU (Metal).** Её делает шейдер, кадр камеры больше не меняется на месте. Если Metal недоступен, вклейка идёт на CPU, как раньше.
- Под счётчиком fps показано, на чём сейчас работает приложение, например «Core ML · Metal».

## Установка

1. Скачай `FaceSwap-v0.2.1-arm64.zip`, распакуй и перенеси `FaceSwap.app` в «Программы».
2. Сними карантин (приложение не подписано):
   `xattr -dr com.apple.quarantine /Applications/FaceSwap.app`
3. Скачай модели (~460 МБ, плюс Core ML-детектор). Если ставил 0.1.0, запусти ещё раз: скрипт докачает только новое.
   `curl -fsSL https://raw.githubusercontent.com/daniil7890/faceswapmac/main/scripts/download_models.sh | bash`

Требования: macOS 14+, Apple Silicon. Подробная инструкция в README.

---

## 0.2.1

- The app now has an icon. It was missing from the 0.2.0 build.

## What's new in 0.2.0: faster

- **Native Core ML face detector (Neural Engine).** Detection takes about 0.7 ms per frame instead of ~7 ms, with automatic ONNX fallback.
- **Two-stage pipeline.** The next frame is detected and aligned while inswapper processes the current one.
- **GPU paste-back (Metal).** The face is blended in by a compute shader, and the camera frame is no longer modified in place.
- The sidebar shows the active backend, e.g. "Core ML · Metal".

**Upgrading from 0.1.0:** run the model script again. It only downloads the new Core ML detector:
`curl -fsSL https://raw.githubusercontent.com/daniil7890/faceswapmac/main/scripts/download_models.sh | bash`

Requires macOS 14+ and Apple Silicon. The app is unsigned, so run the `xattr` command above or right-click → Open the first time.

Модели InsightFace можно использовать только в некоммерческих целях. Не выдавай себя за других людей без их согласия.
