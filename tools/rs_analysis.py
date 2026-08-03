#!/usr/bin/env python3
"""Rolling-shutter signature analysis on a board capture.

Reports, binned by gyro angular speed:
  1. tag detection yield (fraction of frames with detections)
  2. per-frame PnP reprojection RMS (pose-free: how well can ANY single
     global-shutter pose explain the corners of that frame)
  3. ARKit-pose corner residual: board corners projected through ARKit's
     pose for the frame (global-shutter assumption) vs detected corners.
     Board world pose comes from the nearest calm frame (PnP âŠ• ARKit pose),
     which also absorbs slow ARKit drift.

Requires tags.jsonl from detect_tags.py --pnp and calibration/.

Usage: rs_analysis.py <session_dir> [--plot out.png]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from detect_tags import corners_3d, load_target
from vio_session import Session

FLIP_YZ = np.diag([1.0, -1.0, -1.0])
BINS = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0, 25.0]


def pose_matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, float))[0]
    T[:3, 3] = np.asarray(tvec, float).flatten()
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--target", type=Path,
                    default=Path(__file__).parent.parent / "target/aprilgrid.yaml")
    ap.add_argument("--calib", type=Path,
                    default=Path(__file__).parent.parent
                    / "calibration/intrinsics_fullres.yaml")
    ap.add_argument("--calm-w", type=float, default=0.2)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    s = Session(args.session)
    rows = [json.loads(l) for l in open(s.path / "tags.jsonl")]
    target = load_target(args.target)
    board = corners_3d(int(target["tagRows"]), int(target["tagCols"]),
                       float(target["tagSize"]), float(target["tagSpacing"]))
    vals = load_target(args.calib)
    K = np.array([[vals["fx"], 0, vals["cx"]],
                  [0, vals["fy"], vals["cy"]], [0, 0, 1]])
    dist = np.array([vals["k1"], vals["k2"], vals["p1"], vals["p2"]])

    w_at = lambda t: np.interp(t, s.gyro.t, np.linalg.norm(s.gyro.xyz, axis=1))

    # ARKit camera-to-world in CV convention, per frame idx
    T_wc = np.tile(np.eye(4), (len(s.frames.t), 1, 1))
    T_wc[:, :3, :3] = s.frames.R @ FLIP_YZ
    T_wc[:, :3, 3] = s.frames.p

    # board world pose anchors from calm, well-fit frames
    anchors = []  # (t, T_wb)
    for r in rows:
        if "pnp" not in r or r["pnp"]["reproj_rms"] > 2.0:
            continue
        if w_at(r["t"]) > args.calm_w:
            continue
        T_cb = pose_matrix(r["pnp"]["rvec"], r["pnp"]["tvec"])
        anchors.append((r["t"], T_wc[r["idx"]] @ T_cb))
    if len(anchors) < 5:
        sys.exit("too few calm anchor frames to fix the board pose")
    anchor_ts = np.array([a[0] for a in anchors])
    board_pos = np.array([a[1][:3, 3] for a in anchors])
    spread = np.linalg.norm(board_pos - board_pos.mean(0), axis=1)
    print(f"board anchors: {len(anchors)} calm frames, position spread "
          f"std {spread.std() * 1000:.1f} mm (ARKit drift + PnP noise)")

    # per-frame stats
    recs = []  # (w, has_det, pnp_rms or nan, arkit_rms or nan)
    for r in rows:
        w = float(w_at(r["t"]))
        has = len(r["tags"]) > 0
        pnp_rms = r["pnp"]["reproj_rms"] if "pnp" in r else np.nan
        ark_rms = np.nan
        if len(r["tags"]) >= 2:
            near = int(np.argmin(np.abs(anchor_ts - r["t"])))
            T_wb = anchors[near][1]
            T_cb = np.linalg.inv(T_wc[r["idx"]]) @ T_wb
            obj = np.concatenate([board[t["id"]] for t in r["tags"]])
            img = np.concatenate([np.array(t["corners"]) for t in r["tags"]])
            rvec = cv2.Rodrigues(T_cb[:3, :3])[0]
            proj, _ = cv2.projectPoints(obj, rvec, T_cb[:3, 3], K, dist)
            ark_rms = float(np.sqrt(
                ((proj.reshape(-1, 2) - img) ** 2).sum(1).mean()))
        recs.append((w, has, pnp_rms, ark_rms))
    arr = np.array(recs, dtype=float)

    print(f"\n{'|w| bin rad/s':>14} {'frames':>7} {'det%':>5} "
          f"{'PnP px':>8} {'ARKit px':>9}")
    stats = []
    for lo, hi in zip(BINS, BINS[1:]):
        m = (arr[:, 0] >= lo) & (arr[:, 0] < hi)
        if m.sum() == 0:
            continue
        det = arr[m, 1].mean() * 100
        pnp = np.nanmedian(arr[m, 2]) if np.isfinite(arr[m, 2]).any() else np.nan
        ark = np.nanmedian(arr[m, 3]) if np.isfinite(arr[m, 3]).any() else np.nan
        stats.append((lo, hi, m.sum(), det, pnp, ark))
        print(f"{f'{lo:.1f} - {hi:.1f}':>14} {m.sum():>7} {det:>5.0f} "
              f"{pnp:>8.2f} {ark:>9.2f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centers = [0.5 * (a + b) for a, b, *_ in stats]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(centers, [x[4] for x in stats], "o-", label="per-frame PnP RMS")
        ax.plot(centers, [x[5] for x in stats], "s-",
                label="ARKit-pose corner RMS")
        ax.set_xscale("log")
        ax.set_xlabel("|angular velocity| (rad/s)")
        ax.set_ylabel("corner reprojection RMS (px, full res)")
        ax.set_title(f"{s.name} — rolling-shutter signature")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"plot written to {args.plot}")


if __name__ == "__main__":
    main()
