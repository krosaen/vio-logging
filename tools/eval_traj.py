#!/usr/bin/env python3
"""Compare an OpenVINS trajectory against ARKit's poses for the same session.

Both trajectories share the session's seconds-since-boot clock, so association
is direct interpolation. Alignment is SE(3) Umeyama (rotation free, no scale):
OpenVINS and ARKit each pick an arbitrary world yaw/origin, and their up-axis
conventions differ (ARKit: y-up, OpenVINS: z-up), all of which the fitted
rotation absorbs.

Caveat: OpenVINS reports the IMU pose, ARKit the camera pose — the constant
few-cm lever arm between them is not corrected here and slightly inflates ATE.

Usage: eval_traj.py <session_dir> [--est openvins/traj_estimate.txt] [--plot out.png]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vio_session import Session


def load_estimate(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            vals = line.split()
            rows.append([float(v) for v in vals[:8]])
    arr = np.array(rows)
    if len(arr) < 10:
        raise RuntimeError(f"only {len(arr)} poses in {path}")
    return arr[:, 0], arr[:, 1:4]  # t, position


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = False):
    """Similarity/rigid transform aligning src -> dst (Nx3 each)."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    scale = (np.trace(np.diag(D) @ S) / sc.var(0).sum()) if with_scale else 1.0
    t = mu_d - scale * R @ mu_s
    return scale, R, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--est", type=Path, default=None)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()

    s = Session(args.session)
    est_path = args.est or (s.path / "openvins" / "traj_estimate.txt")
    t_est, p_est = load_estimate(est_path)

    # ARKit positions interpolated onto estimate timestamps
    t0, t1 = s.frames.t[0], s.frames.t[-1]
    keep = (t_est >= t0) & (t_est <= t1)
    t_est, p_est = t_est[keep], p_est[keep]
    p_ark = np.column_stack(
        [np.interp(t_est, s.frames.t, s.frames.p[:, i]) for i in range(3)])

    coverage = (t_est[-1] - t_est[0]) / (t1 - t0) * 100
    print(f"session: {s.name}")
    print(f"estimate: {len(t_est)} poses covering {coverage:.0f}% of capture "
          f"({t_est[0] - t0:.2f}s .. {t_est[-1] - t0:.2f}s rel)")

    for with_scale in (False, True):
        scale, R, t = umeyama(p_est, p_ark, with_scale)
        p_al = (scale * (R @ p_est.T)).T + t
        err = np.linalg.norm(p_al - p_ark, axis=1)
        label = "sim3 (scale free)" if with_scale else "se3  (metric)    "
        extra = f", scale {scale:.4f}" if with_scale else ""
        print(f"  ATE {label}: rmse {np.sqrt((err**2).mean()):.4f} m, "
              f"median {np.median(err):.4f} m, max {err.max():.4f} m{extra}")

    # drift: error at end relative to path length, using metric alignment
    scale, R, t = umeyama(p_est, p_ark, False)
    p_al = (R @ p_est.T).T + t
    path_len = np.linalg.norm(np.diff(p_ark, axis=0), axis=1).sum()
    end_err = np.linalg.norm(p_al[-1] - p_ark[-1])
    print(f"  path length (ARKit): {path_len:.2f} m")
    print(f"  final-pose divergence: {end_err:.4f} m "
          f"({100 * end_err / max(path_len, 1e-9):.2f}% of path)")

    # RPE over 1 s windows (metric alignment, relative displacements)
    dt = 1.0
    idx2 = np.searchsorted(t_est, t_est + dt)
    ok = idx2 < len(t_est)
    d_est = p_al[idx2[ok]] - p_al[np.where(ok)[0]]
    d_ark = p_ark[idx2[ok]] - p_ark[np.where(ok)[0]]
    rpe = np.linalg.norm(d_est - d_ark, axis=1)
    print(f"  RPE @1s: rmse {np.sqrt((rpe**2).mean()):.4f} m, "
          f"median {np.median(rpe):.4f} m, max {rpe.max():.4f} m")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(p_ark[:, 0], p_ark[:, 2], label="ARKit (x/z)", lw=1.2)
        axes[0].plot(p_al[:, 0], p_al[:, 2], label="OpenVINS aligned", lw=1.2)
        axes[0].set_aspect("equal")
        axes[0].legend()
        axes[0].set_title(f"{s.name} — top-down")
        err = np.linalg.norm(p_al - p_ark, axis=1)
        axes[1].plot(t_est - t0, err, lw=1)
        axes[1].set_xlabel("t (s)")
        axes[1].set_ylabel("position error (m)")
        axes[1].set_title("ATE over time (SE3 aligned)")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"  plot written to {args.plot}")


if __name__ == "__main__":
    main()
