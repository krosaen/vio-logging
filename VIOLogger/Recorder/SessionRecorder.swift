import ARKit
import Combine
import CoreImage
import UIKit

/// Owns the ARSession and, while recording, fans each ARFrame out to:
///  - frames.mov  (video via AVAssetWriter)
///  - frames.jsonl (timestamp, exposure, per-frame intrinsics, ARKit pose, tracking state)
///  - depth/      (LiDAR sceneDepth every 6th frame, raw float32 + confidence)
///  - pointcloud.jsonl (ARKit raw feature points every 30th frame)
/// while MotionLogger writes the IMU streams. ARFrame.timestamp and CoreMotion
/// timestamps share the seconds-since-boot clock, so no cross-clock alignment
/// is needed offline.
final class SessionRecorder: NSObject, ObservableObject, ARSessionDelegate {

    // MARK: UI state (main thread)

    @Published var previewImage: UIImage?
    @Published var isRecording = false
    @Published var trackingLabel = "starting…"
    @Published var frameCount = 0
    @Published var droppedCount = 0
    @Published var imuCount = 0
    @Published var hfIMULabel = "–"
    @Published var formatLabel = "–"
    @Published var statusLine = ""
    @Published var lastError: String?
    @Published var recordingStartedAt: Date?
    @Published var codec: VideoWriter.Codec = .hevc
    /// When enabled, ARKit tracking is reset at the moment recording starts, so
    /// ARKit faces the same cold-initialization problem as offline estimators.
    /// The log then captures its convergence (tracking states in frames.jsonl).
    @Published var coldStart = false

    let session = ARSession()

    // MARK: recording state — touched only on sessionQueue

    private let sessionQueue = DispatchQueue(label: "vio.arsession", qos: .userInitiated)
    private let depthQueue = DispatchQueue(label: "vio.depth", qos: .utility)
    private let ciContext = CIContext()

    private var active = false
    private var videoWriter: VideoWriter?
    private var frameWriter: JSONLWriter?
    private var cloudWriter: JSONLWriter?
    private let motionLogger = MotionLogger()
    private var folder: URL?
    private var depthDir: URL?
    private var frameIdx = 0
    private var wallClockStart: Date?
    private var uptimeStart: TimeInterval?
    private var chosenFormat: ARConfiguration.VideoFormat?
    private var configuration: ARWorldTrackingConfiguration?
    private var coldStartUsed = false
    private var depthSize: (width: Int, height: Int)?

    private static let depthEvery = 6
    private static let cloudEvery = 30

    // MARK: session lifecycle

    func startSession() {
        let config = ARWorldTrackingConfiguration()
        let formats = ARWorldTrackingConfiguration.supportedVideoFormats
        let best = formats
            .filter { $0.framesPerSecond == 60 }
            .max { $0.imageResolution.width < $1.imageResolution.width }
            ?? formats.first
        if let best { config.videoFormat = best }
        chosenFormat = config.videoFormat
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
            config.frameSemantics.insert(.sceneDepth)
        }
        configuration = config
        session.delegateQueue = sessionQueue
        session.delegate = self
        session.run(config)

        let f = config.videoFormat
        let label = "\(Int(f.imageResolution.width))×\(Int(f.imageResolution.height)) @\(f.framesPerSecond)"
        DispatchQueue.main.async { self.formatLabel = label }
    }

    func toggleRecording() {
        sessionQueue.async {
            self.active ? self.endRecording() : self.beginRecording()
        }
    }

    // MARK: recording (sessionQueue)

    private func beginRecording() {
        guard !active else { return }
        do {
            let fm = FileManager.default
            let docs = try fm.url(
                for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            let df = DateFormatter()
            df.dateFormat = "yyyyMMdd_HHmmss"
            let name = "session_\(df.string(from: Date()))"
            let dir = docs.appendingPathComponent(name, isDirectory: true)
            try fm.createDirectory(at: dir, withIntermediateDirectories: true)
            let depth = dir.appendingPathComponent("depth", isDirectory: true)
            try fm.createDirectory(at: depth, withIntermediateDirectories: true)

            guard let format = chosenFormat else {
                throw RecorderError(message: "No AR video format available")
            }
            videoWriter = try VideoWriter(
                url: dir.appendingPathComponent("frames.mov"),
                width: Int(format.imageResolution.width),
                height: Int(format.imageResolution.height),
                fps: format.framesPerSecond,
                codec: codec)
            frameWriter = try JSONLWriter(url: dir.appendingPathComponent("frames.jsonl"))
            cloudWriter = try JSONLWriter(url: dir.appendingPathComponent("pointcloud.jsonl"))
            try motionLogger.start(in: dir)

            coldStartUsed = coldStart
            if coldStartUsed, let config = configuration {
                session.run(config, options: [.resetTracking, .removeExistingAnchors])
            }

            folder = dir
            depthDir = depth
            frameIdx = 0
            depthSize = nil
            wallClockStart = Date()
            uptimeStart = ProcessInfo.processInfo.systemUptime
            active = true

            let hfLabel = motionLogger.hfAccelActive
                ? "\(batchedAccelHz()) Hz accel / \(batchedDMHz()) Hz dm"
                : "100 Hz (no HF support)"
            DispatchQueue.main.async {
                self.isRecording = true
                self.recordingStartedAt = Date()
                self.frameCount = 0
                self.droppedCount = 0
                self.imuCount = 0
                self.hfIMULabel = hfLabel
                self.statusLine = "recording \(name)"
                self.lastError = nil
                UIApplication.shared.isIdleTimerDisabled = true
            }
        } catch {
            cleanupAfterFailure()
            DispatchQueue.main.async {
                self.lastError = "Failed to start: \(error.localizedDescription)"
                self.isRecording = false
            }
        }
    }

    private func batchedAccelHz() -> Int {
        // Populated once updates start; 0 until the first batch arrives.
        // Displayed value refreshes with the stats, so a momentary 0 is fine.
        motionLogger.hfAccelActive ? 800 : 0
    }

    private func batchedDMHz() -> Int {
        motionLogger.hfDeviceMotionActive ? 200 : 0
    }

    private func cleanupAfterFailure() {
        motionLogger.stop()
        frameWriter?.close()
        cloudWriter?.close()
        videoWriter = nil
        frameWriter = nil
        cloudWriter = nil
        folder = nil
        depthDir = nil
        active = false
    }

    private func endRecording() {
        guard active, let dir = folder, let video = videoWriter else { return }
        active = false
        let hfWasActive = motionLogger.hfAccelActive
        let imuCounts = motionLogger.stop()
        let frames = frameWriter
        let cloud = cloudWriter
        let meta = metaDictionary(imuCounts: imuCounts, hfAccelActive: hfWasActive)
        videoWriter = nil
        frameWriter = nil
        cloudWriter = nil
        folder = nil
        depthDir = nil

        DispatchQueue.main.async {
            self.isRecording = false
            self.recordingStartedAt = nil
            self.statusLine = "finalizing…"
            UIApplication.shared.isIdleTimerDisabled = false
        }

        video.finish { [weak self] in
            guard let self else { return }
            self.sessionQueue.async {
                frames?.close()
                cloud?.close()
                var meta = meta
                meta["frames_appended"] = video.framesAppended
                meta["frames_dropped"] = video.framesDropped
                if let err = video.error {
                    meta["video_error"] = err.localizedDescription
                }
                self.writeMeta(meta, to: dir)
                let note = video.error.map { "video error: \($0.localizedDescription)" }
                    ?? "saved \(dir.lastPathComponent)"
                DispatchQueue.main.async {
                    self.statusLine = note
                    if let err = video.error {
                        self.lastError = err.localizedDescription
                    }
                }
            }
        }
    }

    private func metaDictionary(
        imuCounts: [String: Int], hfAccelActive: Bool
    ) -> [String: Any] {
        var systemInfo = utsname()
        uname(&systemInfo)
        let model = withUnsafeBytes(of: &systemInfo.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        var meta: [String: Any] = [
            "device_model": model,
            "ios_version": UIDevice.current.systemVersion,
            "wall_clock_start": ISO8601DateFormatter().string(from: wallClockStart ?? Date()),
            "uptime_at_start": uptimeStart ?? 0,
            "codec": codec.rawValue,
            "imu_sample_counts": imuCounts,
            "hf_accel_active": hfAccelActive,
            "cold_start": coldStartUsed,
            "conventions": [
                "timebase": "all t values are seconds since boot; shared by ARKit and CoreMotion",
                "wall_clock_mapping": "unix_time(t) = wall_clock_start + (t - uptime_at_start)",
                "transform": "column-major 4x4 camera-to-world (simd), ARKit convention: +x right, +y up, camera looks along -z (landscape-right sensor frame); world is gravity-aligned",
                "intrinsics": "[fx, fy, cx, cy] in pixels, for the native sensor resolution, updated per frame (autofocus/OIS)",
                "accel_units": "g (multiply by 9.80665 for m/s^2), raw, includes gravity",
                "gyro_units": "rad/s, raw (not bias-corrected)",
                "video_pts": "frames.mov presentation times are (t - t_first_frame); authoritative per-frame t is in frames.jsonl",
                "depth": "depth_NNNNNN.f32 raw little-endian float32 meters, row-major; conf_NNNNNN.u8 ARConfidenceLevel per pixel",
            ],
        ]
        if let f = chosenFormat {
            meta["video_format"] = [
                "width": Int(f.imageResolution.width),
                "height": Int(f.imageResolution.height),
                "fps": f.framesPerSecond,
            ]
        }
        if let d = depthSize {
            meta["depth_size"] = ["width": d.width, "height": d.height]
            meta["depth_every_n_frames"] = Self.depthEvery
        }
        return meta
    }

    private func writeMeta(_ meta: [String: Any], to dir: URL) {
        if let data = try? JSONSerialization.data(
            withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
            try? data.write(to: dir.appendingPathComponent("meta.json"))
        }
    }

    // MARK: ARSessionDelegate (sessionQueue)

    private var callbackIdx = 0

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let camera = frame.camera
        callbackIdx += 1

        if active {
            recordFrame(frame)
        }

        // Preview + status at reduced rate; recording gets priority.
        let previewEvery = active ? 6 : 3
        if callbackIdx % previewEvery == 0 {
            updatePreview(frame)
            let tracking = Self.trackingLabel(camera.trackingState)
            let stats = active
                ? (videoWriter?.framesAppended ?? 0, videoWriter?.framesDropped ?? 0,
                   motionLogger.sampleCount)
                : (0, 0, 0)
            DispatchQueue.main.async {
                self.trackingLabel = tracking
                if self.isRecording {
                    self.frameCount = stats.0
                    self.droppedCount = stats.1
                    self.imuCount = stats.2
                }
            }
        }
    }

    private func recordFrame(_ frame: ARFrame) {
        let camera = frame.camera
        let t = frame.timestamp
        let idx = frameIdx
        frameIdx += 1

        videoWriter?.append(frame.capturedImage, timestamp: t)

        var wroteDepth = false
        if idx % Self.depthEvery == 0, let sceneDepth = frame.sceneDepth,
            let depthDir = depthDir {
            wroteDepth = writeDepth(sceneDepth, idx: idx, dir: depthDir)
        }

        let m = camera.transform
        let tf = [m.columns.0, m.columns.1, m.columns.2, m.columns.3]
            .flatMap { [$0.x, $0.y, $0.z, $0.w] }
            .map { String(format: "%.7f", $0) }
            .joined(separator: ",")
        let K = camera.intrinsics
        let featureCount = frame.rawFeaturePoints?.points.count ?? 0
        let line = String(
            format: "{\"idx\":%d,\"t\":%.9f,\"exposure_duration\":%.7f,"
                + "\"exposure_offset\":%.4f,\"intrinsics\":[%.4f,%.4f,%.4f,%.4f],"
                + "\"transform\":[%@],\"tracking\":\"%@\",\"features\":%d,\"depth\":%@}",
            idx, t, camera.exposureDuration, camera.exposureOffset,
            K.columns.0.x, K.columns.1.y, K.columns.2.x, K.columns.2.y,
            tf, Self.trackingKey(camera.trackingState), featureCount,
            wroteDepth ? "true" : "false")
        frameWriter?.write(line)

        if idx % Self.cloudEvery == 0, let pts = frame.rawFeaturePoints?.points, !pts.isEmpty {
            let ptsStr = pts
                .map { String(format: "[%.4f,%.4f,%.4f]", $0.x, $0.y, $0.z) }
                .joined(separator: ",")
            cloudWriter?.write(
                String(format: "{\"idx\":%d,\"t\":%.9f,\"points\":[%@]}", idx, t, ptsStr))
        }
    }

    private func writeDepth(_ sceneDepth: ARDepthData, idx: Int, dir: URL) -> Bool {
        let depthData = Self.tightData(sceneDepth.depthMap, bytesPerPixel: 4)
        guard !depthData.isEmpty else { return false }
        if depthSize == nil {
            depthSize = (
                CVPixelBufferGetWidth(sceneDepth.depthMap),
                CVPixelBufferGetHeight(sceneDepth.depthMap)
            )
        }
        let confData = sceneDepth.confidenceMap.map { Self.tightData($0, bytesPerPixel: 1) }
        depthQueue.async {
            try? depthData.write(
                to: dir.appendingPathComponent(String(format: "depth_%06d.f32", idx)))
            if let confData, !confData.isEmpty {
                try? confData.write(
                    to: dir.appendingPathComponent(String(format: "conf_%06d.u8", idx)))
            }
        }
        return true
    }

    /// Copies a pixel buffer into tightly-packed rows (drops row padding).
    private static func tightData(_ pb: CVPixelBuffer, bytesPerPixel: Int) -> Data {
        CVPixelBufferLockBaseAddress(pb, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return Data() }
        let width = CVPixelBufferGetWidth(pb)
        let height = CVPixelBufferGetHeight(pb)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pb)
        let rowBytes = width * bytesPerPixel
        var data = Data(capacity: rowBytes * height)
        for row in 0..<height {
            data.append(
                Data(bytes: base.advanced(by: row * bytesPerRow), count: rowBytes))
        }
        return data
    }

    private func updatePreview(_ frame: ARFrame) {
        let ci = CIImage(cvPixelBuffer: frame.capturedImage)
            .oriented(.right)
            .transformed(by: CGAffineTransform(scaleX: 0.4, y: 0.4))
        guard let cg = ciContext.createCGImage(ci, from: ci.extent) else { return }
        let img = UIImage(cgImage: cg)
        DispatchQueue.main.async { self.previewImage = img }
    }

    private static func trackingKey(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "normal"
        case .notAvailable: return "not_available"
        case .limited(let reason):
            switch reason {
            case .initializing: return "limited_initializing"
            case .excessiveMotion: return "limited_excessive_motion"
            case .insufficientFeatures: return "limited_insufficient_features"
            case .relocalizing: return "limited_relocalizing"
            @unknown default: return "limited_unknown"
            }
        }
    }

    private static func trackingLabel(_ state: ARCamera.TrackingState) -> String {
        switch state {
        case .normal: return "tracking: normal"
        case .notAvailable: return "tracking: unavailable"
        case .limited(let reason):
            switch reason {
            case .initializing: return "tracking: initializing"
            case .excessiveMotion: return "tracking: excessive motion"
            case .insufficientFeatures: return "tracking: low features"
            case .relocalizing: return "tracking: relocalizing"
            @unknown default: return "tracking: limited"
            }
        }
    }

    // MARK: interruptions / errors

    func session(_ session: ARSession, didFailWithError error: Error) {
        sessionQueue.async {
            if self.active { self.endRecording() }
        }
        DispatchQueue.main.async {
            self.lastError = "ARSession failed: \(error.localizedDescription)"
        }
    }

    func sessionWasInterrupted(_ session: ARSession) {
        sessionQueue.async {
            if self.active { self.endRecording() }
        }
        DispatchQueue.main.async {
            self.statusLine = "session interrupted — recording stopped"
        }
    }
}
