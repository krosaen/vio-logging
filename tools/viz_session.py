#!/usr/bin/env python3
"""Interactive session visualization with rerun (rerun.io).

Logs a VIOLogger session to a shared timeline: camera frustum + video moving
along the ARKit trajectory, LiDAR depth, ARKit feature points, IMU magnitudes,
and any number of estimated trajectories (SE3-aligned to ARKit), each drawn as
a line plus points colored by instantaneous position error (turbo colormap).

Usage:
  viz_session.py <session_dir> [--est name=path/to/traj.txt ...] [--spawn]

By default looks for openvins/traj_estimate.txt and includes it if present.
Writes <session>/session.rrd; open with `rerun <file>.rrd` or pass --spawn.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import rerun as rr

sys.path.insert(0, str(Path(__file__).parent))
from eval_traj import load_estimate, umeyama
from vio_session import Session

# ARKit camera convention (y-up/OpenGL) -> CV convention (y-down/z-forward)
FLIP_YZ = np.diag([1.0, -1.0, -1.0])

EST_COLORS = [
    (66, 133, 244),   # blue
    (219, 68, 55),    # red
    (244, 180, 0),    # yellow
    (15, 157, 88),    # green
    (171, 71, 188),   # purple
]


def error_colors(err: np.ndarray) -> np.ndarray:
    """Map error magnitudes to turbo RGB (uint8), normalized at p95."""
    from matplotlib import cm

    scale = max(np.percentile(err, 95), 1e-9)
    return (np.array(cm.turbo(np.clip(err / scale, 0, 1)))[:, :3] * 255).astype(
        np.uint8)


def log_estimate(name: str, path: Path, s: Session, color) -> None:
    t_est, p_est = load_estimate(path)
    keep = (t_est >= s.frames.t[0]) & (t_est <= s.frames.t[-1])
    t_est, p_est = t_est[keep], p_est[keep]
    p_ark = np.column_stack(
        [np.interp(t_est, s.frames.t, s.frames.p[:, i]) for i in range(3)])
    _, R, t = umeyama(p_est, p_ark, with_scale=False)
    p_al = (R @ p_est.T).T + t
    err = np.linalg.norm(p_al - p_ark, axis=1)

    rr.log(f"world/traj/{name}/line",
           rr.LineStrips3D([p_al], colors=[color], radii=0.002), static=True)
    rr.log(f"world/traj/{name}/error",
           rr.Points3D(p_al, colors=error_colors(err), radii=0.004), static=True)

    t0 = s.frames.t[0]
    for i in range(len(t_est)):
        rr.set_time("capture", duration=t_est[i] - t0)
        rr.log(f"plots/error/{name}", rr.Scalars(err[i]))
        rr.log(f"world/traj/{name}/current",
               rr.Points3D([p_al[i]], colors=[color], radii=0.012))
    print(f"  estimate '{name}': {len(t_est)} poses, "
          f"err median {np.median(err):.3f} m, p95 {np.percentile(err, 95):.3f} m")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--est", action="append", default=[],
                    help="name=path of a trajectory estimate (TUM-ish txt); "
                         "repeatable. Default: openvins/traj_estimate.txt")
    ap.add_argument("--spawn", action="store_true", help="open the viewer")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--frame-stride", type=int, default=4,
                    help="log every Nth video frame (default 4 = 15 Hz)")
    ap.add_argument("--image-scale", type=float, default=0.25)
    args = ap.parse_args()

    import cv2

    s = Session(args.session)
    out = args.out or (s.path / "session.rrd")

    rr.init("vio-logging", spawn=args.spawn)
    rr.save(str(out))
    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)

    t0 = s.frames.t[0]

    # --- ARKit reference trajectory ---
    rr.log("world/traj/arkit",
           rr.LineStrips3D([s.frames.p], colors=[(160, 160, 160)], radii=0.002),
           static=True)

    # --- estimates ---
    ests = list(args.est)
    default = s.path / "openvins" / "traj_estimate.txt"
    if not ests and default.exists():
        ests = [f"openvins={default}"]
    for i, spec in enumerate(ests):
        name, _, path = spec.partition("=")
        try:
            log_estimate(name, Path(path), s, EST_COLORS[i % len(EST_COLORS)])
        except RuntimeError as e:
            print(f"  skipping estimate '{name}': {e}")

    # --- IMU magnitude plots ---
    for i in range(len(s.gyro.t)):
        rr.set_time("capture", duration=s.gyro.t[i] - t0)
        rr.log("plots/gyro_mag", rr.Scalars(np.linalg.norm(s.gyro.xyz[i])))
    a_mag = np.linalg.norm(s.accel.xyz, axis=1) * 9.80665
    for i in range(len(s.accel.t)):
        rr.set_time("capture", duration=s.accel.t[i] - t0)
        rr.log("plots/accel_mag", rr.Scalars(a_mag[i]))

    # --- tracking state changes ---
    prev = None
    for i, state in enumerate(s.frames.tracking):
        if state != prev:
            rr.set_time("capture", duration=s.frames.t[i] - t0)
            level = rr.TextLogLevel.INFO if state == "normal" else rr.TextLogLevel.WARN
            rr.log("tracking", rr.TextLog(f"ARKit tracking: {state}", level=level))
            prev = state

    # --- ARKit feature points (sparse cloud snapshots) ---
    import json
    cloud_path = s.path / "pointcloud.jsonl"
    if cloud_path.exists():
        with open(cloud_path) as f:
            for line in f:
                row = json.loads(line)
                rr.set_time("capture", duration=row["t"] - t0)
                rr.log("world/features",
                       rr.Points3D(row["points"], colors=[(230, 190, 80)],
                                   radii=0.005))

    # --- camera pose (60 Hz) ---
    K = s.median_intrinsics()
    for i in range(len(s.frames.t)):
        rr.set_time("capture", duration=s.frames.t[i] - t0)
        R_cv = s.frames.R[i] @ FLIP_YZ
        rr.log("world/cam",
               rr.Transform3D(translation=s.frames.p[i], mat3x3=R_cv))

    # --- video frames + depth (strided) ---
    sc = args.image_scale
    w = int(round(s.meta["video_format"]["width"] * sc))
    h = int(round(s.meta["video_format"]["height"] * sc))
    rr.log("world/cam/image",
           rr.Pinhole(image_from_camera=[[K[0] * sc, 0, K[2] * sc],
                                         [0, K[1] * sc, K[3] * sc],
                                         [0, 0, 1]],
                      resolution=[w, h]),
           static=True)
    depth_meta = s.meta.get("depth_size")
    if depth_meta:
        dw, dh = depth_meta["width"], depth_meta["height"]
        dsc = dw / s.meta["video_format"]["width"]
        rr.log("world/cam/depth",
               rr.Pinhole(image_from_camera=[[K[0] * dsc, 0, K[2] * dsc],
                                             [0, K[1] * dsc, K[3] * dsc],
                                             [0, 0, 1]],
                          resolution=[dw, dh]),
               static=True)
    for idx, t, img in s.iter_video():
        if idx % args.frame_stride:
            continue
        rr.set_time("capture", duration=t - t0)
        rgb = cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA),
                           cv2.COLOR_BGR2RGB)
        rr.log("world/cam/image", rr.Image(rgb).compress(jpeg_quality=75))
        if depth_meta and s.frames.has_depth[idx]:
            dpath = s.path / "depth" / f"depth_{idx:06d}.f32"
            if dpath.exists():
                depth = np.fromfile(dpath, dtype=np.float32).reshape(
                    depth_meta["height"], depth_meta["width"])
                rr.log("world/cam/depth",
                       rr.DepthImage(depth, meter=1.0, colormap="viridis"))

    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")
    if not args.spawn:
        print(f"view with: .venv/bin/rerun {out}")


if __name__ == "__main__":
    main()
