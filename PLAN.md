# vio-logging: iPhone VIO data logger + offline estimation playground

Goal: build a minimal iPhone app that logs hardware-timestamped camera frames, raw IMU,
and ARKit's own pose estimates to disk, so that offline experiments (discrete-time
sliding-window VIO vs. continuous-time B-spline / rolling-shutter-aware models) can be
run against real data with meaningful metrics.

## 1. Architecture decision: log *through* ARKit, not beside it

The Gemini conversation suggests `AVCaptureDataOutputSynchronizer` for hardware-synced
capture. That's the right tool for a pure raw logger, **but ARKit and AVCaptureSession
cannot own the camera at the same time**. If we want ARKit poses in the same session as
the raw data (and we do — it's our comparison baseline), the logger must use `ARSession`
as the single camera source:

- `ARFrame.capturedImage` — full-resolution YUV pixel buffer (1920×1440 @ 60 fps
  typical; check `ARWorldTrackingConfiguration.supportedVideoFormats`).
- `ARFrame.timestamp` — capture time on the same clock (`systemUptime` /
  mach_absolute_time base) as CoreMotion timestamps. This is what makes the whole
  plan work: one shared time base, no cross-clock alignment needed.
- `ARCamera` — per-frame 4×4 `transform` (ARKit's pose estimate), `intrinsics`
  (per-frame, tracks the VCM autofocus/OIS lens shifts), `exposureDuration`,
  `exposureOffset`, `trackingState` + reason.
- CoreMotion runs happily alongside ARKit — raw `CMAccelerometerData` +
  `CMGyroData` at 100 Hz via `CMMotionManager`. On iOS 17+ devices,
  `CMBatchedSensorManager` reportedly provides ~800 Hz accelerometer / 200 Hz
  device-motion batches (verify availability on the target phone — if it works,
  use it; 100 Hz is workable but 200+ Hz is much better for aggressive motion).
- If the phone has LiDAR: `ARFrame.sceneDepth` (256×192 depth + confidence) —
  cheap to log, useful later as unary depth factors / scale anchors.

A second "raw mode" using `AVCaptureSession` directly (manual exposure lock, other
formats, no ARKit) can come later; it forfeits the ARKit baseline so it's not phase 1.

### Corrections to keep in mind from the Gemini conversation

- CoreMotion via `CMMotionManager` tops out around 100 Hz, not 500 Hz. The
  higher-rate path is `CMBatchedSensorManager` (iOS 17+, device-dependent).
- ARKit does **not** expose `AVCameraCalibrationData` / `lensDistortionLookupTable`
  — only the 3×3 intrinsics. Distortion gets calibrated offline (Kalibr). Fine:
  the wide camera's distortion is small and stable.
- Rolling-shutter readout time is not in any API; treat it as a per-device constant
  to be calibrated offline (or estimated as a parameter in the spline fit — itself
  a nice validation: does the estimate converge to a plausible ~14–16 ms?).
- Camera-to-IMU extrinsics are not exposed; Kalibr solves them from logged data.

## 2. What gets logged (session format)

One directory per recording session:

Implemented in the `VIOLogger` app in this repo — authoritative schema is in
`README.md` ("Session format"). Summary:

```
session_YYYYMMDD_HHMMSS/
  meta.json              device, video format, codec, counts, units/conventions
  frames.mov             HEVC 100 Mbps (default) or ProRes 422 for short clips
  frames.jsonl           per frame: idx, t, exposure, per-frame intrinsics,
                         ARKit camera-to-world transform (column-major),
                         tracking state, feature count
  accel.jsonl gyro.jsonl raw IMU ~100 Hz (g / rad/s — raw, not gravity-removed)
  devicemotion.jsonl     fused reference ~100 Hz
  accel_hf.jsonl         ~800 Hz raw accel (CMBatchedSensorManager)
  devicemotion_hf.jsonl  ~200 Hz fused (same)
  pointcloud.jsonl       ARKit rawFeaturePoints every 30th frame
  depth/                 LiDAR sceneDepth + confidence every 6th frame
```

JSONL everywhere to start (debuggable, versionable); switch IMU to binary only if
it ever matters. Export via Finder file sharing / AirDrop / Files app.

### Logger validation (before trusting any data)

1. **Time-sync check:** correlate |gyro| against the rotation-rate implied by
   ARKit pose deltas. Cross-correlation peak should sit at a stable, small offset
   (ARKit poses are mid-exposure-timestamped; expect ~half exposure + constant).
2. **Gravity check:** low-pass accelerometer direction vs. ARKit gravity (world −Z
   through the pose) should agree to a fraction of a degree when static.
3. **Intrinsics sanity:** project ARKit `rawFeaturePoints` through the logged
   per-frame pose + intrinsics; they should land on visually plausible corners.

## 3. Ground truth: the layered answer

"How would we know we beat ARKit during dynamic motion?" — no single ground truth;
use four layers, cheapest first:

**Tier 0 — ARKit agreement in benign motion.** Slow, feature-rich handheld capture:
ARKit is near-truth there. Metric: ATE/RPE (after Sim3 or SE3 + time alignment)
between our trajectory and ARKit's. This is the "are we in the ballpark" gate, not
a win condition.

**Tier 1 — Return-to-start drift.** 3D-print or tape a cradle; every session starts
and ends with the phone seated in it. Final-pose error is then measurable against
near-perfect physical ground truth (sub-mm repeatability), independent of ARKit.
Cheap, absolute, works for any motion profile in the middle.

**Tier 2 — AprilTag board + held-out rolling-shutter-aware reprojection.** This is
the key discriminator for the dynamic-motion hypothesis. Print a large AprilTag
grid on a wall. During fast motion, tag-corner PnP is itself corrupted by rolling
shutter, so don't use tag poses as truth directly. Instead:

- Fit each candidate trajectory model (discrete-time vs. continuous B-spline)
  **without** the tag observations (IMU + natural features only), or with tags in
  a held-out split.
- Then ask: how well does the model predict where each tag corner *actually
  landed on the sensor*, evaluating the pose at that corner's **row readout
  time**? Tag corners are known 3D points with sub-pixel detections — a model
  that represents intra-frame motion correctly explains the RS-warped corner
  positions; a global-shutter model can't.
- Metric: held-out corner reprojection RMSE, **binned by gyro angular velocity**.

**Tier 3 — Full-batch offline pseudo-ground-truth.** Global continuous-time
optimization over the entire capture using *everything* (all frames, tags, loop
closures, IMU) with unlimited compute. Use it as reference truth to score causal /
sliding-window variants — standard practice when there's no mocap. (Tier 4, if
ever: borrow time in a Vicon/OptiTrack room.)

**The killer plot:** reprojection error (or per-second drift) vs. angular velocity,
one curve per model. If the B-spline curve stays flat where the discrete-time curve
grows, that's the quantified win — "X px vs Y px held-out reprojection error above
200 °/s."

## 4. Roadmap

**Phase 1 — Logger app (Swift).** ARSession + CoreMotion + AVAssetWriter, session
format above, export. Keep clips ≤ 60 s (thermals, file size).

**Phase 2 — Desktop tooling (Python).** Session parser, validation notebook
(section 2 checks), Kalibr calibration: camera intrinsics + distortion,
camera-IMU extrinsics, time offset, RS readout time. Print the AprilTag board.

**Phase 3 — Baseline harness.** Run an existing open pipeline (OpenVINS or
VINS-Fusion are the easiest to feed custom data; ORB-SLAM3 as a second opinion)
on logged sessions. Build the metrics code (Tiers 0–2). Deliverable: a table of
{ARKit, OpenVINS} × {benign, aggressive} × {ATE vs ARKit, return-to-start drift,
tag reprojection}. This proves the whole loop before any novel estimation work.

**Phase 4 — Continuous-time experiment.** Implement a B-spline-on-SE(3) (or split
SO(3)×R³) trajectory fit — SymForce or Ceres, or adapt Basalt which has spline
machinery — with per-row timestamps in the projection model. A/B against the
discrete baseline on paired captures:

- (a) slow smooth pan of the tag wall, (b) fast shakes / whip-pans of the same wall.
- Prediction: near-identical on (a); spline wins on (b) on the Tier-2 metric.
- Bonus experiments: estimate RS readout time as a free parameter (does it
  converge to the physical constant?); ablate IMU rate (100 Hz vs 800 Hz) to see
  when the spline's smoothness prior earns its keep.

## 5. First session checklist (once logger exists)

1. Static on table 10 s (noise floors, gravity check).
2. Slow feature-rich walk with loop back to cradle (Tier 0 + Tier 1).
3. Tag wall, slow pan 30 s (calibration + Tier 2 baseline).
4. Tag wall, aggressive shake/whip 30 s (the money capture).
5. Kalibr calibration sequence (excited motion in front of the board, all axes).
