#!/usr/bin/env python3
"""Camera intrinsics + radtan distortion via cv2.calibrateCamera on Aprilgrid
corner detections from a calibration session.

Selects sharp (low angular velocity), well-covered, temporally spread frames,
detects tag corners at full resolution, and calibrates. Writes a Kalibr-format
camchain yaml (scaled to the ROS-bag resolution) for kalibr_calibrate_imu_camera
and downstream tools.

Usage: calibrate_intrinsics.py <session_dir> [--target target/aprilgrid.yaml]
       [--max-frames 80] [--max-w 0.3]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detect_tags import corners_3d, detect_frame, load_target, make_detector
from vio_session import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--target", type=Path,
                    default=Path(__file__).parent.parent / "target/aprilgrid.yaml")
    ap.add_argument("--max-frames", type=int, default=80)
    ap.add_argument("--max-w", type=float, default=0.3,
                    help="max |gyro| rad/s for a frame to count as sharp")
    ap.add_argument("--min-tags", type=int, default=20)
    args = ap.parse_args()

    s = Session(args.session)
    target = load_target(args.target)
    board = corners_3d(int(target["tagRows"]), int(target["tagCols"]),
                       float(target["tagSize"]), float(target["tagSpacing"]))
    detector = make_detector()

    w_mag = np.interp(s.frames.t, s.gyro.t,
                      np.linalg.norm(s.gyro.xyz, axis=1))

    # candidate frames: sharp enough, spread evenly across the capture
    sharp_idx = [i for i in range(len(s.frames.t)) if w_mag[i] < args.max_w]
    stride = max(1, len(sharp_idx) // (args.max_frames * 2))
    wanted = set(sharp_idx[::stride])

    obj_pts, img_pts = [], []
    shape = None
    for idx, t, gray in s.iter_video(gray=True):
        if idx not in wanted or len(obj_pts) >= args.max_frames * 2:
            continue
        shape = gray.shape[::-1]
        dets = detect_frame(detector, gray)
        if len(dets) < args.min_tags:
            continue
        obj_pts.append(np.concatenate(
            [board[i] for i, _ in dets]).astype(np.float32))
        img_pts.append(np.concatenate(
            [c for _, c in dets]).astype(np.float32).reshape(-1, 1, 2))
    if len(obj_pts) > args.max_frames:
        keep = np.linspace(0, len(obj_pts) - 1, args.max_frames).astype(int)
        obj_pts = [obj_pts[i] for i in keep]
        img_pts = [img_pts[i] for i in keep]

    print(f"calibrating on {len(obj_pts)} views "
          f"({sum(len(o) for o in obj_pts)} corners) at {shape[0]}x{shape[1]}")
    if len(obj_pts) < 15:
        sys.exit("too few usable views — capture more sharp, tag-filled frames")

    flags = cv2.CALIB_FIX_K3
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_pts, img_pts, shape, None, None, flags=flags)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2 = dist.flatten()[:4]

    print(f"reprojection RMS: {rms:.3f} px (full res)")
    print(f"fx fy cx cy: {fx:.2f} {fy:.2f} {cx:.2f} {cy:.2f}")
    print(f"radtan k1 k2 p1 p2: {k1:.5f} {k2:.5f} {p1:.6f} {p2:.6f}")
    ark = s.median_intrinsics()
    print(f"ARKit median for comparison: fx {ark[0]:.2f} fy {ark[1]:.2f} "
          f"cx {ark[2]:.2f} cy {ark[3]:.2f}")

    # kalibr camchain at the exported bag resolution (default scale 0.5)
    import json
    scale = 0.5
    bag_info = s.path / "openvins" / "bag_info.json"
    if bag_info.exists():
        scale = json.loads(bag_info.read_text())["scale"]
    out = s.path / "kalibr"
    out.mkdir(exist_ok=True)
    chain = out / "camchain-opencv.yaml"
    chain.write_text(
        "cam0:\n"
        "  camera_model: pinhole\n"
        f"  intrinsics: [{fx * scale:.4f}, {fy * scale:.4f}, "
        f"{cx * scale:.4f}, {cy * scale:.4f}]\n"
        "  distortion_model: radtan\n"
        f"  distortion_coeffs: [{k1:.6f}, {k2:.6f}, {p1:.7f}, {p2:.7f}]\n"
        f"  resolution: [{int(shape[0] * scale)}, {int(shape[1] * scale)}]\n"
        "  rostopic: /cam0/image_raw\n")
    print(f"wrote {chain} (scaled x{scale} for the exported bag)")


if __name__ == "__main__":
    main()
