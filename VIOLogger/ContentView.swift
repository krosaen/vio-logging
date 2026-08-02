import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = SessionRecorder()

    var body: some View {
        NavigationStack {
            ZStack {
                Color.black.ignoresSafeArea()
                if let img = recorder.previewImage {
                    Image(uiImage: img)
                        .resizable()
                        .scaledToFit()
                        .ignoresSafeArea()
                }
                VStack {
                    statsOverlay
                    Spacer()
                    controls
                }
                .padding()
            }
            .navigationTitle("VIO Logger")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarBackground(.hidden, for: .navigationBar)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    NavigationLink("Sessions") { SessionsView() }
                        .disabled(recorder.isRecording)
                }
            }
        }
        .preferredColorScheme(.dark)
        .onAppear { recorder.startSession() }
        .alert(
            "Error", isPresented: .init(
                get: { recorder.lastError != nil },
                set: { if !$0 { recorder.lastError = nil } })
        ) {
            Button("OK", role: .cancel) {}
        } message: {
            Text(recorder.lastError ?? "")
        }
    }

    private var statsOverlay: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(recorder.trackingLabel)
            Text("format: \(recorder.formatLabel)")
            if recorder.isRecording {
                Text("frames: \(recorder.frameCount)  dropped: \(recorder.droppedCount)")
                Text("imu samples: \(recorder.imuCount)")
                Text("hf imu: \(recorder.hfIMULabel)")
                if let start = recorder.recordingStartedAt {
                    TimelineView(.periodic(from: start, by: 1)) { context in
                        Text(String(
                            format: "elapsed: %.0f s",
                            context.date.timeIntervalSince(start)))
                    }
                }
            }
            if !recorder.statusLine.isEmpty {
                Text(recorder.statusLine).foregroundStyle(.secondary)
            }
        }
        .font(.system(.footnote, design: .monospaced))
        .padding(8)
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 8))
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private var controls: some View {
        VStack(spacing: 16) {
            Picker("Codec", selection: $recorder.codec) {
                ForEach(VideoWriter.Codec.allCases) { codec in
                    Text(codec.label).tag(codec)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 220)
            .disabled(recorder.isRecording)

            Toggle(isOn: $recorder.coldStart) {
                Text("Cold-start ARKit at record")
                    .font(.footnote)
            }
            .frame(maxWidth: 260)
            .tint(.orange)
            .disabled(recorder.isRecording)

            Button(action: { recorder.toggleRecording() }) {
                ZStack {
                    Circle()
                        .strokeBorder(.white, lineWidth: 4)
                        .frame(width: 74, height: 74)
                    if recorder.isRecording {
                        RoundedRectangle(cornerRadius: 6)
                            .fill(.red)
                            .frame(width: 32, height: 32)
                    } else {
                        Circle()
                            .fill(.red)
                            .frame(width: 60, height: 60)
                    }
                }
            }
        }
        .padding(.bottom, 8)
    }
}
