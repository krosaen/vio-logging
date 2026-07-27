import Foundation

/// Buffered append-only line writer. `write` may be called from any thread;
/// lines are flushed to disk in ~64 KB chunks and on close.
final class JSONLWriter {
    private let handle: FileHandle
    private let queue: DispatchQueue
    private var buffer = Data()
    private var closed = false
    private static let flushThreshold = 1 << 16

    private(set) var lineCount = 0
    private let countLock = NSLock()

    init(url: URL) throws {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        self.handle = try FileHandle(forWritingTo: url)
        self.queue = DispatchQueue(label: "jsonl.\(url.lastPathComponent)", qos: .utility)
    }

    func write(_ line: String) {
        countLock.lock()
        lineCount += 1
        countLock.unlock()
        queue.async {
            guard !self.closed else { return }
            self.buffer.append(contentsOf: line.utf8)
            self.buffer.append(0x0A)
            if self.buffer.count > Self.flushThreshold { self.flush() }
        }
    }

    var count: Int {
        countLock.lock()
        defer { countLock.unlock() }
        return lineCount
    }

    private func flush() {
        guard !buffer.isEmpty else { return }
        try? handle.write(contentsOf: buffer)
        buffer.removeAll(keepingCapacity: true)
    }

    func close() {
        queue.sync {
            guard !closed else { return }
            flush()
            try? handle.close()
            closed = true
        }
    }

    deinit {
        close()
    }
}
