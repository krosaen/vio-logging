import SwiftUI

struct SessionInfo: Identifiable {
    let url: URL
    let name: String
    let bytes: Int64
    var id: String { name }

    var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
    }
}

/// Lists recorded session folders in Documents. Sessions can be AirDropped via
/// the share button, pulled off with Finder (iPhone → Files → VIO Logger), or
/// browsed in the Files app.
struct SessionsView: View {
    @State private var sessions: [SessionInfo] = []

    var body: some View {
        List {
            if sessions.isEmpty {
                Text("No sessions yet.")
                    .foregroundStyle(.secondary)
            }
            ForEach(sessions) { session in
                HStack {
                    VStack(alignment: .leading) {
                        Text(session.name).font(.system(.body, design: .monospaced))
                        Text(session.sizeLabel)
                            .font(.footnote)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    ShareLink(item: session.url) {
                        Image(systemName: "square.and.arrow.up")
                    }
                }
            }
            .onDelete(perform: delete)

            Section {
                Text("To copy to a Mac: connect via cable, open Finder, select the iPhone, open the Files tab, expand VIO Logger, and drag session folders out. Or AirDrop with the share button.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Sessions")
        .onAppear(perform: load)
    }

    private func load() {
        let fm = FileManager.default
        guard let docs = try? fm.url(
            for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: false),
            let entries = try? fm.contentsOfDirectory(
                at: docs, includingPropertiesForKeys: [.isDirectoryKey])
        else {
            sessions = []
            return
        }
        sessions = entries
            .filter { $0.lastPathComponent.hasPrefix("session_") }
            .map { url in
                SessionInfo(
                    url: url, name: url.lastPathComponent, bytes: folderSize(url))
            }
            .sorted { $0.name > $1.name }
    }

    private func folderSize(_ url: URL) -> Int64 {
        let fm = FileManager.default
        guard let enumerator = fm.enumerator(
            at: url, includingPropertiesForKeys: [.fileSizeKey])
        else { return 0 }
        var total: Int64 = 0
        for case let file as URL in enumerator {
            total += Int64((try? file.resourceValues(forKeys: [.fileSizeKey]))?.fileSize ?? 0)
        }
        return total
    }

    private func delete(at offsets: IndexSet) {
        for index in offsets {
            try? FileManager.default.removeItem(at: sessions[index].url)
        }
        sessions.remove(atOffsets: offsets)
    }
}
