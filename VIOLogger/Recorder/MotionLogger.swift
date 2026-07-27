import CoreMotion
import Foundation

/// Logs raw IMU streams to JSONL files inside a session folder.
///
/// Always logged (CMMotionManager, ~100 Hz):
///   accel.jsonl        raw accelerometer, units of g
///   gyro.jsonl         raw gyroscope, rad/s (not bias-corrected)
///   devicemotion.jsonl fused attitude/gravity/userAccel/rotationRate (reference only)
///
/// Logged when the device supports CMBatchedSensorManager (iPhone 15 Pro+, iOS 17+):
///   accel_hf.jsonl        ~800 Hz raw accelerometer
///   devicemotion_hf.jsonl ~200 Hz fused device motion
///
/// All timestamps are seconds since boot — the same clock as ARFrame.timestamp.
final class MotionLogger {
    private let manager = CMMotionManager()
    private let batched = CMBatchedSensorManager()
    private let opQueue: OperationQueue = {
        let q = OperationQueue()
        q.maxConcurrentOperationCount = 1
        q.qualityOfService = .userInitiated
        return q
    }()

    private var writers: [JSONLWriter] = []
    private var accelWriter: JSONLWriter?
    private var gyroWriter: JSONLWriter?
    private var dmWriter: JSONLWriter?
    private var accelHFWriter: JSONLWriter?
    private var dmHFWriter: JSONLWriter?

    private(set) var hfAccelActive = false
    private(set) var hfDeviceMotionActive = false

    /// Total samples across all streams (thread-safe, for UI display).
    var sampleCount: Int { writers.reduce(0) { $0 + $1.count } }

    func start(in folder: URL) throws {
        func makeWriter(_ name: String) throws -> JSONLWriter {
            let w = try JSONLWriter(url: folder.appendingPathComponent(name))
            writers.append(w)
            return w
        }
        accelWriter = try makeWriter("accel.jsonl")
        gyroWriter = try makeWriter("gyro.jsonl")
        dmWriter = try makeWriter("devicemotion.jsonl")

        manager.accelerometerUpdateInterval = 1.0 / 100.0
        manager.gyroUpdateInterval = 1.0 / 100.0
        manager.deviceMotionUpdateInterval = 1.0 / 100.0

        manager.startAccelerometerUpdates(to: opQueue) { [weak self] data, _ in
            guard let d = data else { return }
            self?.accelWriter?.write(String(
                format: "{\"t\":%.9f,\"x\":%.8f,\"y\":%.8f,\"z\":%.8f}",
                d.timestamp, d.acceleration.x, d.acceleration.y, d.acceleration.z))
        }
        manager.startGyroUpdates(to: opQueue) { [weak self] data, _ in
            guard let d = data else { return }
            self?.gyroWriter?.write(String(
                format: "{\"t\":%.9f,\"x\":%.8f,\"y\":%.8f,\"z\":%.8f}",
                d.timestamp, d.rotationRate.x, d.rotationRate.y, d.rotationRate.z))
        }
        manager.startDeviceMotionUpdates(using: .xArbitraryZVertical, to: opQueue) { [weak self] dm, _ in
            guard let d = dm else { return }
            self?.dmWriter?.write(Self.deviceMotionLine(d))
        }

        if CMBatchedSensorManager.isAccelerometerSupported {
            let w = try makeWriter("accel_hf.jsonl")
            accelHFWriter = w
            batched.startAccelerometerUpdates { [weak self] batch, error in
                guard error == nil, let batch else { return }
                for d in batch {
                    self?.accelHFWriter?.write(String(
                        format: "{\"t\":%.9f,\"x\":%.8f,\"y\":%.8f,\"z\":%.8f}",
                        d.timestamp, d.acceleration.x, d.acceleration.y, d.acceleration.z))
                }
            }
            hfAccelActive = true
        }
        if CMBatchedSensorManager.isDeviceMotionSupported {
            let w = try makeWriter("devicemotion_hf.jsonl")
            dmHFWriter = w
            batched.startDeviceMotionUpdates { [weak self] batch, error in
                guard error == nil, let batch else { return }
                for d in batch {
                    self?.dmHFWriter?.write(Self.deviceMotionLine(d))
                }
            }
            hfDeviceMotionActive = true
        }
    }

    private static func deviceMotionLine(_ d: CMDeviceMotion) -> String {
        let q = d.attitude.quaternion
        return String(
            format: "{\"t\":%.9f,\"quat\":[%.8f,%.8f,%.8f,%.8f],"
                + "\"rot\":[%.8f,%.8f,%.8f],\"grav\":[%.8f,%.8f,%.8f],"
                + "\"uacc\":[%.8f,%.8f,%.8f]}",
            d.timestamp, q.w, q.x, q.y, q.z,
            d.rotationRate.x, d.rotationRate.y, d.rotationRate.z,
            d.gravity.x, d.gravity.y, d.gravity.z,
            d.userAcceleration.x, d.userAcceleration.y, d.userAcceleration.z)
    }

    /// Stops all streams and returns the final per-stream sample counts.
    @discardableResult
    func stop() -> [String: Int] {
        manager.stopAccelerometerUpdates()
        manager.stopGyroUpdates()
        manager.stopDeviceMotionUpdates()
        if hfAccelActive { batched.stopAccelerometerUpdates() }
        if hfDeviceMotionActive { batched.stopDeviceMotionUpdates() }
        hfAccelActive = false
        hfDeviceMotionActive = false
        opQueue.waitUntilAllOperationsAreFinished()

        var counts: [String: Int] = [:]
        if let w = accelWriter { counts["accel"] = w.count }
        if let w = gyroWriter { counts["gyro"] = w.count }
        if let w = dmWriter { counts["devicemotion"] = w.count }
        if let w = accelHFWriter { counts["accel_hf"] = w.count }
        if let w = dmHFWriter { counts["devicemotion_hf"] = w.count }

        writers.forEach { $0.close() }
        writers.removeAll()
        accelWriter = nil
        gyroWriter = nil
        dmWriter = nil
        accelHFWriter = nil
        dmHFWriter = nil
        return counts
    }
}
