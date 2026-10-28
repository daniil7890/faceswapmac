Первый релиз FaceSwapMac: подмена лица в реальном времени по одной фотографии на Mac с Apple Silicon.

**Что умеет**
- Подмена лица с камеры: SCRFD → ArcFace → inswapper_128 через ONNX Runtime + CoreML (Neural Engine/GPU)
- Фото-источник можно выбрать или перетащить, работают сглаживание точек, зеркальное превью и выбор камеры
- Интерфейс на русском и английском
- Метка «AI · подмена лица» на превью, включена по умолчанию

**Установка**
1. Скачай `FaceSwap-v0.1.0-arm64.zip` и распакуй.
2. Модели (~460 МБ) в комплект не входят: склонируй репозиторий и запусти `bash scripts/download_models.sh`.
3. Приложение не подписано, поэтому при первом запуске открой его через правый клик → «Открыть»
   или выполни `xattr -dr com.apple.quarantine FaceSwap.app`.

Требования: macOS 14+, Apple Silicon.

---
First release of FaceSwapMac: real-time one-photo face swap for Apple Silicon Macs. Russian/English UI.
Models are downloaded separately with `scripts/download_models.sh`. The app is unsigned, so open it via right-click → Open the first time.

Модели InsightFace разрешено использовать только в некоммерческих целях. Не выдавай себя за других людей без их согласия.
