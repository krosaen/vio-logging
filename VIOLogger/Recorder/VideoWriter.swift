import AVFoundation
import CoreVideo

struct RecorderError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

/// Wraps AVAssetWriter for appending ARKit's YUV pixel buffers in real time.
/// Presentation times are relative to the first appended frame; the absolute
/// boot-relative timestamp of every frame lives in frames.jsonl.
final class VideoWriter {
    enum Codec: String, CaseIterable, Identifiable {
        case hevc
        case proRes422

        var id: String { rawValue }
        var avCodec: AVVideoCodecType { self == .hevc ? .hevc : .proRes422 }
        var label: String { self == .hevc ? "HEVC" : "ProRes" }
    }

    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private var startTimestamp: TimeInterval?
    private(set) var framesAppended = 0
    private(set) var framesDropped = 0

    init(url: URL, width: Int, height: Int, fps: Int, codec: Codec) throws {
        writer = try AVAssetWriter(outputURL: url, fileType: .mov)
        var settings: [String: Any] = [
            AVVideoCodecKey: codec.avCodec,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ]
        if codec == .hevc {
            settings[AVVideoCompressionPropertiesKey] = [
                AVVideoAverageBitRateKey: 100_000_000,
                AVVideoExpectedSourceFrameRateKey: fps,
                AVVideoMaxKeyFrameIntervalKey: fps,
            ]
        }
        input = AVAssetWriterInput(mediaType: .video, outputSettings: settings)
        input.expectsMediaDataInRealTime = true
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: nil)
        guard writer.canAdd(input) else {
            throw RecorderError(message: "AVAssetWriter rejected video input (\(codec.label))")
        }
        writer.add(input)
    }

    /// Returns true if the frame was appended, false if it had to be dropped.
    @discardableResult
    func append(_ pixelBuffer: CVPixelBuffer, timestamp: TimeInterval) -> Bool {
        if startTimestamp == nil {
            startTimestamp = timestamp
            writer.startWriting()
            writer.startSession(atSourceTime: .zero)
        }
        guard writer.status == .writing, input.isReadyForMoreMediaData else {
            framesDropped += 1
            return false
        }
        let pts = CMTime(
            seconds: timestamp - startTimestamp!, preferredTimescale: 1_000_000_000)
        if adaptor.append(pixelBuffer, withPresentationTime: pts) {
            framesAppended += 1
            return true
        } else {
            framesDropped += 1
            return false
        }
    }

    var error: Error? { writer.error }

    func finish(completion: @escaping () -> Void) {
        guard writer.status == .writing else {
            completion()
            return
        }
        input.markAsFinished()
        writer.finishWriting(completionHandler: completion)
    }
}
