// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FaceSwapMac",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.24.2"),
    ],
    targets: [
        .executableTarget(
            name: "FaceSwapMac",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Sources/FaceSwapMac",
            swiftSettings: [.unsafeFlags(["-Ounchecked"])]
        ),
    ]
)
