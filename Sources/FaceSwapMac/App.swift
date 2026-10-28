import AppKit
import AVFoundation
import ImageIO
import SwiftUI
import UniformTypeIdentifiers

@main
struct FaceSwapMacApp: App {
    init() {
        // Launched as a bare SwiftPM binary: make it a regular foreground app with a Dock icon.
        NSApplication.shared.setActivationPolicy(.regular)
        NSApplication.shared.activate(ignoringOtherApps: true)
    }

    var body: some Scene {
        WindowGroup("FaceSwap") { ContentView() }
            .defaultSize(width: 1180, height: 640)
    }
}

/// Where the .onnx files live: next to the app bundle (Resources/models) or Application Support.
enum ModelLocation {
    static var dir: URL {
        if let r = Bundle.main.resourceURL?.appendingPathComponent("models"),
           FileManager.default.fileExists(atPath: r.appendingPathComponent(FaceSwapper.file).path) { return r }
        return FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("FaceSwapMac/models")
    }
}

final class AppState: ObservableObject {
    @Published var status = Loc("Готовлю модели… (первый запуск CoreML компилирует их, до минуты)",
                                 "Preparing models… (first launch compiles them for CoreML, up to a minute)")
    /// Changing it redraws the whole UI in the new language.
    @Published var lang = Lang.current { didSet { Lang.current = lang } }
    @Published var ready = false
    @Published var sourceThumb: NSImage?
    @Published var swapOn = true { didSet { camera.swapEnabled = swapOn } }
    @Published var mirror = true
    @Published var showLabel = true
    @Published var dropTarget = false
    @Published var smoothing: Double = 0.5 { didSet { engine?.smoothing = Float(smoothing) } }

    let camera = CameraController()
    private(set) var engine: FaceSwapEngine?

    private var booted = false
    func boot() {
        guard !booted else { return }
        booted = true
        camera.start()
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                let e = try FaceSwapEngine(modelsDir: ModelLocation.dir)
                DispatchQueue.main.async {
                    self.engine = e
                    self.camera.engine = e
                    self.ready = true
                    self.status = Loc("Выбери или перетащи фото с лицом", "Choose or drop a photo with a face")
                }
            } catch {
                DispatchQueue.main.async {
                    let msg = error.localizedDescription, dir = ModelLocation.dir.path
                    self.status = Loc("\(msg)\nПапка моделей: \(dir)", "\(msg)\nModels folder: \(dir)")
                }
            }
        }
    }

    func loadSource(_ url: URL) {
        guard let engine else { return }
        status = Loc("Считываю лицо…", "Reading the face…")
        DispatchQueue.global(qos: .userInitiated).async {
            do {
                guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
                      // thumbnail-with-transform applies EXIF orientation (phone photos)
                      let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                          kCGImageSourceCreateThumbnailFromImageAlways: true,
                          kCGImageSourceCreateThumbnailWithTransform: true,
                          kCGImageSourceThumbnailMaxPixelSize: 1600,
                      ] as CFDictionary)
                else { throw EngineError.badImage }
                try engine.setSource(cg)
                let thumb = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
                DispatchQueue.main.async { self.sourceThumb = thumb; self.status = Loc("Готово", "Ready") }
            } catch {
                DispatchQueue.main.async { self.status = Loc(raw: error.localizedDescription) }
            }
        }
    }

    func pickFile() {
        let p = NSOpenPanel()
        p.allowedContentTypes = [.image]
        p.allowsMultipleSelection = false
        if p.runModal() == .OK, let u = p.url { loadSource(u) }
    }
}

struct ContentView: View {
    // one shared state; avoids the @StateObject/@State macros missing in Command Line Tools-only setups
    private static let shared = AppState()
    @ObservedObject private var app = ContentView.shared
    var body: some View { MainView(app: app, cam: app.camera) }
}

struct MainView: View {
    @ObservedObject var app: AppState
    @ObservedObject var cam: CameraController

    var body: some View {
        HStack(spacing: 0) {
            sidebar.frame(width: 280)
            Divider()
            preview
        }
        .onAppear { app.boot() }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            Picker(L("Язык", "Language"), selection: $app.lang) {
                ForEach(Lang.allCases) { (l: Lang) in Text(l.title).tag(l) }
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            Text(L("Лицо-источник", "Source face")).font(.headline)
            ZStack {
                RoundedRectangle(cornerRadius: 12)
                    .strokeBorder(app.dropTarget ? Color.accentColor : .secondary.opacity(0.4),
                                  style: StrokeStyle(lineWidth: 2, dash: [6]))
                if let t = app.sourceThumb {
                    Image(nsImage: t).resizable().scaledToFill()
                        .frame(width: 244, height: 244).clipShape(RoundedRectangle(cornerRadius: 10))
                } else {
                    VStack(spacing: 6) {
                        Image(systemName: "person.crop.square").font(.system(size: 40))
                        Text(L("Перетащи фото сюда", "Drop a photo here")).font(.callout)
                    }.foregroundStyle(.secondary)
                }
            }
            .frame(width: 248, height: 248)
            .onDrop(of: [.fileURL], isTargeted: $app.dropTarget) { items in
                guard let item = items.first else { return false }
                _ = item.loadObject(ofClass: URL.self) { url, _ in
                    if let url { DispatchQueue.main.async { app.loadSource(url) } }
                }
                return true
            }

            Button(L("Выбрать фото…", "Choose photo…")) { app.pickFile() }.disabled(!app.ready)
            Text(app.status.text).font(.caption).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)

            Divider()
            Toggle(L("Подмена лица", "Face swap"), isOn: $app.swapOn)
            Toggle(L("Зеркалить превью", "Mirror preview"), isOn: $app.mirror)
            Toggle(L("Метка «AI» на картинке", "“AI” label on image"), isOn: $app.showLabel)
            VStack(alignment: .leading) {
                Text(L("Сглаживание", "Smoothing") + ": \(Int(app.smoothing * 100))%").font(.caption)
                Slider(value: $app.smoothing, in: 0...0.85)
            }
            CameraPicker(cam: cam)
            Spacer()
            Text(String(format: L("%.0f fps · подмена %.0f мс · %@", "%.0f fps · swap %.0f ms · %@"), cam.fps, cam.swapMs,
                        cam.faceFound ? L("лицо есть", "face found") : L("лица нет", "no face")))
                .font(.caption.monospacedDigit()).foregroundStyle(.secondary)
            if let e = cam.error { Text(e).font(.caption).foregroundStyle(.red) }
        }
        .padding(16)
    }

    private var preview: some View {
        ZStack(alignment: .bottomTrailing) {
            Color.black
            if let f = cam.frame {
                Image(decorative: f, scale: 1).resizable().scaledToFit()
                    .scaleEffect(x: app.mirror ? -1 : 1, y: 1)
            } else {
                ProgressView().controlSize(.large)
            }
            if app.showLabel && app.swapOn && app.sourceThumb != nil {
                Text(L("AI · подмена лица", "AI · face swap"))
                    .font(.system(size: 13, weight: .semibold))
                    .padding(.horizontal, 10).padding(.vertical, 5)
                    .background(.black.opacity(0.55), in: Capsule())
                    .foregroundStyle(.white)
                    .padding(14)
            }
        }
    }
}

/// Camera selector, kept separate so the sidebar stays simple for the type checker.
struct CameraPicker: View {
    @ObservedObject var cam: CameraController

    var body: some View {
        if cam.devices.isEmpty {
            EmptyView()
        } else {
            Picker(L("Камера", "Camera"), selection: $cam.selectedID) {
                ForEach(cam.devices, id: \.uniqueID) { (d: AVCaptureDevice) in
                    Text(d.localizedName).tag(d.uniqueID)
                }
            }
        }
    }
}
