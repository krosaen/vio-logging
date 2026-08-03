#!/usr/bin/env python3
"""Spline milestone (a): fit a continuous-time B-spline to ARKit's poses.

Pure regression — validates the spline machinery against trusted data before
IMU and vision factors enter. Reports fit residuals, cross-checks the spline's
analytic-ish derivatives against the raw gyro (a derivative the fit never saw),
and exports a sampled trajectory for the rerun visualizer plus the control
points for later milestones.

Usage: fit_spline_arkit.py <session_dir> [--knot-dt 0.1]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).parent))
from bspline import fit_pos_spline, fit_so3_spline
from validate_session import body_angular_velocity, estimate_time_offset, kabsch
from vio_session import Session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--knot-dt", type=float, default=0.1,
                    help="knot spacing in seconds (default 0.1 = 10 Hz)")
    args = ap.parse_args()

    s = Session(args.session)
    t = s.frames.t
    R = Rotation.from_matrix(s.frames.R)

    print(f"fitting splines: {len(t)} poses, knot dt {args.knot_dt}s "
          f"({int((t[-1]-t[0])/args.knot_dt)+3} ctrl points)")
    pos = fit_pos_spline(t, s.frames.p, args.knot_dt)
    rot = fit_so3_spline(t, R, args.knot_dt)

    # --- fit residuals ---
    ep = np.linalg.norm(pos.pos(t) - s.frames.p, axis=1)
    er = np.linalg.norm((rot.rot(t).inv() * R).as_rotvec(), axis=1)
    print(f"position residual: rms {np.sqrt((ep**2).mean())*1000:.2f} mm, "
          f"max {ep.max()*1000:.2f} mm")
    print(f"rotation residual: rms {np.degrees(np.sqrt((er**2).mean())):.4f} deg, "
          f"max {np.degrees(er.max()):.4f} deg")

    # --- derivative cross-check vs raw gyro (data the fit never saw) ---
    # device->camera rotation + time offset via the usual Kabsch pipeline
    ts_a, w_a = body_angular_velocity(t, s.frames.R)
    off, peak, _ = estimate_time_offset(
        ts_a, np.linalg.norm(w_a, axis=1),
        s.gyro.t, np.linalg.norm(s.gyro.xyz, axis=1))
    grid = np.arange(t[0] + 0.05, t[-1] - 0.05, 0.01)
    A = np.column_stack([np.interp(grid + off, s.gyro.t, s.gyro.xyz[:, k])
                         for k in range(3)])
    B = np.column_stack([np.interp(grid, ts_a, w_a[:, k]) for k in range(3)])
    keep = (np.linalg.norm(A, axis=1) > 0.1) & (np.linalg.norm(B, axis=1) > 0.1)
    R_dc = kabsch(A[keep], B[keep])  # device -> camera(ARKit) frame

    m = (s.gyro.t - off >= t[0] + 0.05) & (s.gyro.t - off <= t[-1] - 0.05)
    w_spline = rot.angvel_body(s.gyro.t[m] - off)     # camera frame
    w_gyro_cam = (R_dc @ s.gyro.xyz[m].T).T           # gyro mapped to camera
    dw = np.linalg.norm(w_spline - w_gyro_cam, axis=1)
    corr = np.corrcoef(np.linalg.norm(w_spline, axis=1),
                       np.linalg.norm(w_gyro_cam, axis=1))[0, 1]
    print(f"spline angvel vs gyro ({m.sum()} samples): "
          f"|dw| rms {np.sqrt((dw**2).mean()):.4f} rad/s, "
          f"magnitude corr {corr:.4f}")
    print(f"  (gyro |w| median {np.median(np.linalg.norm(w_gyro_cam,axis=1)):.3f}"
          f" rad/s; residual includes gyro noise + bias)")

    # --- exports ---
    out = s.path / "spline"
    out.mkdir(exist_ok=True)
    tt = np.arange(t[0], t[-1] - 1e-6, 0.01)
    q = rot.rot(tt).as_quat()  # xyzw
    p = pos.pos(tt)
    with open(out / "arkit_fit.txt", "w") as f:
        f.write("# t x y z qx qy qz qw (ARKit world frame, spline fit)\n")
        for k in range(len(tt)):
            f.write(f"{tt[k]:.6f} {p[k,0]:.6f} {p[k,1]:.6f} {p[k,2]:.6f} "
                    f"{q[k,0]:.7f} {q[k,1]:.7f} {q[k,2]:.7f} {q[k,3]:.7f}\n")
    np.savez(out / "spline_arkit.npz",
             pos_ctrl=pos.ctrl, pos_t0=pos.t0, pos_dt=pos.dt,
             rot_ctrl_quat=rot.ctrl.as_quat(), rot_t0=rot.t0, rot_dt=rot.dt,
             R_dc=R_dc, time_offset=off)
    print(f"wrote {out / 'arkit_fit.txt'} and spline_arkit.npz")


if __name__ == "__main__":
    main()
