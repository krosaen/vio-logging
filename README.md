# VIO Logger

Minimal iPhone app that records time-synchronized VIO research data: camera
frames (via ARKit), raw IMU (CoreMotion, incl. 800 Hz batched accel on
supported devices), ARKit's own 6-DoF pose estimates, LiDAR depth, and ARKit
feature points. See `PLAN.md` for the research plan this feeds.

## One-time setup (Mac)

1. **Install Xcode** from the Mac App Store (large download). Then point the
   command line tools at it and finish setup:

   ```sh
   sudo xcode-select -s /Applications/Xcode.app
   sudo xcodebuild -license accept
   xcodebuild -downloadPlatform iOS   # installs iOS platform support if prompted
   ```

2. **Open the project**: `open VIOLogger.xcodeproj`

3. **Signing**: Xcode → Settings → Accounts → add your Apple ID. Then select
   the VIOLogger target → Signing & Capabilities → check "Automatically manage
   signing" and pick your (personal) team. A free account works; the app just
   expires after 7 days and needs a re-install.

   Optional: put your team ID into `project.yml` (`DEVELOPMENT_TEAM`) so
   regenerating the project keeps signing configured.

4. **iPhone**: Settings → Privacy & Security → Developer Mode → on (reboots).
   Connect via cable, unlock, tap Trust.

5. In Xcode, select your iPhone as the run destination and press Run. First
   launch: Settings → General → VPN & Device Management → trust your developer
   certificate if iOS asks.

The project file is generated — edit `project.yml` and run `xcodegen generate`
rather than editing `VIOLogger.xcodeproj` directly (it's fine to let Xcode
change signing settings, but they'll be lost on regenerate unless mirrored in
`project.yml`).

## Recording

- Launch the app, let tracking reach "normal", pick HEVC (default, ~12 MB/s) or
  ProRes (higher fidelity, ~50 MB/s — keep clips short).
- Big red button starts/stops a session. Stats overlay shows frames, drops,
  IMU sample counts, tracking state.
- Keep sessions ≤ 60 s to be kind to thermals and file sizes.

## Getting data onto the Mac

- **Finder (recommended for big sessions)**: connect the iPhone via cable →
  Finder → select iPhone → Files tab → expand "VIO Logger" → drag
  `session_*` folders out.
- **AirDrop**: Sessions screen in the app → share button on a session.
- **Files app**: sessions are also visible under On My iPhone → VIO Logger.

## Session format

```
session_YYYYMMDD_HHMMSS/
  meta.json              device, video format, codec, sample counts, units/conventions
  frames.mov             video (HEVC 100 Mbps or ProRes 422); PTS relative to first frame
  frames.jsonl           per frame: idx, t, exposure_duration, exposure_offset,
                         intrinsics [fx,fy,cx,cy], ARKit camera-to-world transform
                         (16 floats, column-major), tracking state, feature count,
                         whether a depth frame was saved
  accel.jsonl            raw accelerometer ~100 Hz (units: g, includes gravity)
  gyro.jsonl             raw gyro ~100 Hz (rad/s, not bias-corrected)
  devicemotion.jsonl     fused attitude/gravity/userAccel/rotationRate ~100 Hz (reference)
  accel_hf.jsonl         ~800 Hz raw accelerometer (CMBatchedSensorManager; iPhone 15 Pro+)
  devicemotion_hf.jsonl  ~200 Hz fused device motion (same)
  pointcloud.jsonl       ARKit raw feature points every 30th frame
  depth/depth_NNNNNN.f32 LiDAR depth every 6th frame, raw float32 meters, row-major
  depth/conf_NNNNNN.u8   per-pixel ARConfidenceLevel
```

All `t` values are seconds since boot — ARKit and CoreMotion share this clock,
so streams align without cross-clock calibration. `meta.json` records the
wall-clock ↔ uptime mapping and all unit conventions.

## Desktop tooling

Python utilities live in `tools/`; dependencies are in `requirements.txt`:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python tools/validate_session.py captures/<session> [--plot out.png]
```

`validate_session.py` reports stream rates/gaps, the ARKit-vs-gyro time offset
(expect < 1 ms at high correlation), a Kabsch estimate of the device→camera
rotation, and gravity consistency.

## Data rates (1920×1440 @ 60 fps)

| Stream | Rate |
| --- | --- |
| frames.mov (HEVC) | ~12 MB/s (~750 MB/min) |
| frames.mov (ProRes) | ~50 MB/s — short clips only |
| depth (10 Hz) | ~1 MB/s |
| all JSONL | < 0.5 MB/s |
