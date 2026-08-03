#!/usr/bin/env python3
"""Detect Aprilgrid (tag36h11) corners in a session's video.

Writes <session>/tags.jsonl, one line per frame with detections:
  {"idx": n, "t": <boot s>, "tags": [{"id": k, "corners": [[x,y]*4]}, ...],
   "pnp": {"rvec": [...], "tvec": [...], "reproj_rms": px}}   # when --pnp

Corners are sub-pixel refined, in full-resolution pixel coordinates, in
OpenCV order: marker top-left, top-right, bottom-right, bottom-left.

Board geometry follows tools/make_aprilgrid.py: IDs row-major with tag 0 at
the board's bottom-left, board frame x right / y up / z out of the board,
origin at tag 0's bottom-left corner. `corners_3d()` returns the matching
metric 3D corner positions for held-out reprojection / PnP / Kalibr checks.

Usage: detect_tags.py <session_dir> [--target target/aprilgrid.yaml] [--pnp]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vio_session import Session


def load_target(path: Path) -> dict:
    """Minimal parser for our aprilgrid.yaml (kalibr format, flat keys)."""
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            v = v.strip().strip("'\"")
            try:
                out[k.strip()] = float(v) if "." in v else int(v)
            except ValueError:
                out[k.strip()] = v
    return out


def corners_3d(rows: int, cols: int, tag_size: float, spacing: float) -> dict:
    """id -> (4,3) corner positions in board frame, OpenCV detection order
    (TL, TR, BR, BL as seen on an upright board)."""
    pitch = tag_size * (1.0 + spacing)
    out = {}
    for tag_id in range(rows * cols):
        r, c = divmod(tag_id, cols)
        x0, y0 = c * pitch, r * pitch          # tag bottom-left
        out[tag_id] = np.array([
            [x0, y0 + tag_size, 0.0],           # TL
            [x0 + tag_size, y0 + tag_size, 0.0],# TR
            [x0 + tag_size, y0, 0.0],           # BR
            [x0, y0, 0.0],                      # BL
        ])
    return out


def make_detector() -> "cv2.aruco.ArucoDetector":
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 5
    params.cornerRefinementMinAccuracy = 0.01
    params.cornerRefinementMaxIterations = 50
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    return cv2.aruco.ArucoDetector(dictionary, params)


def detect_frame(detector, gray: np.ndarray):
    """[(id, (4,2) corners)] for one grayscale image."""
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None:
        return []
    return [(int(i), c.reshape(4, 2)) for i, c in zip(ids.flatten(), corners)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--target", type=Path,
                    default=Path(__file__).parent.parent / "target/aprilgrid.yaml")
    ap.add_argument("--pnp", action="store_true",
                    help="also solve per-frame board pose and report reproj RMS")
    ap.add_argument("--calib", type=Path,
                    default=Path(__file__).parent.parent
                    / "calibration/intrinsics_fullres.yaml",
                    help="full-res intrinsics yaml (from calibrate_intrinsics); "
                         "falls back to ARKit median + zero distortion")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = Session(args.session)
    out_path = args.out or (s.path / "tags.jsonl")
    detector = make_detector()

    target = load_target(args.target)
    board = corners_3d(int(target["tagRows"]), int(target["tagCols"]),
                       float(target["tagSize"]), float(target["tagSpacing"]))

    K, dist = None, None
    if args.pnp:
        if args.calib.exists():
            vals = load_target(args.calib)  # flat key: value parser works here
            K = np.array([[vals["fx"], 0, vals["cx"]],
                          [0, vals["fy"], vals["cy"]], [0, 0, 1]])
            dist = np.array([vals["k1"], vals["k2"], vals["p1"], vals["p2"]])
            print(f"PnP using calibrated model from {args.calib}")
        else:
            fx, fy, cx, cy = s.median_intrinsics()
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            print("PnP using ARKit median intrinsics, zero distortion")

    n_frames_with = 0
    n_corners = 0
    rms_all = []
    with open(out_path, "w") as f:
        for idx, t, gray in s.iter_video(gray=True):
            dets = detect_frame(detector, gray)
            row = {"idx": idx, "t": round(t, 9),
                   "tags": [{"id": i, "corners": np.round(c, 3).tolist()}
                            for i, c in dets]}
            if dets:
                n_frames_with += 1
                n_corners += 4 * len(dets)
            if K is not None and len(dets) >= 2:
                obj = np.concatenate([board[i] for i, _ in dets])
                img = np.concatenate([c for _, c in dets])
                ok, rvec, tvec = cv2.solvePnP(
                    obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
                if ok:
                    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
                    rms = float(np.sqrt(
                        ((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
                    rms_all.append(rms)
                    row["pnp"] = {"rvec": rvec.flatten().round(6).tolist(),
                                  "tvec": tvec.flatten().round(5).tolist(),
                                  "reproj_rms": round(rms, 3)}
            f.write(json.dumps(row) + "\n")
            if idx % 120 == 0:
                print(f"  frame {idx}/{len(s.frames.t)}  "
                      f"({len(dets)} tags in view)")

    total = len(s.frames.t)
    print(f"wrote {out_path}")
    print(f"frames with detections: {n_frames_with}/{total} "
          f"({100 * n_frames_with / total:.0f}%), corners total: {n_corners}")
    if rms_all:
        rms_all = np.array(rms_all)
        print(f"PnP reproj RMS (zero-distortion pinhole): "
              f"median {np.median(rms_all):.2f} px, p95 "
              f"{np.percentile(rms_all, 95):.2f} px "
              f"(includes distortion + rolling shutter + intrinsics error)")


if __name__ == "__main__":
    main()
