#!/usr/bin/env python3
"""Validate a VIOLogger session: stream rates/gaps, ARKit-vs-gyro time sync,
camera-IMU rotation estimate, and gravity consistency.

Usage: validate_session.py <session_dir> [--plot out.png]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def rotations_from_frames(frames):
    """ARKit camera-to-world rotation matrices from column-major 4x4 transforms."""
    ts = np.array([f["t"] for f in frames])
    Rs = np.empty((len(frames), 3, 3))
    for i, f in enumerate(frames):
        m = np.array(f["transform"]).reshape(4, 4).T  # stored column-major
        Rs[i] = m[:3, :3]
    return ts, Rs


def body_angular_velocity(ts, Rs):
    """Body-frame (camera-frame) angular velocity vectors from consecutive poses."""
    dts = np.diff(ts)
    mid = 0.5 * (ts[:-1] + ts[1:])
    w = np.empty((len(dts), 3))
    for i in range(len(dts)):
        R_rel = Rs[i].T @ Rs[i + 1]
        # log map of SO(3)
        cos = np.clip((np.trace(R_rel) - 1) / 2, -1, 1)
        angle = np.arccos(cos)
        if angle < 1e-8:
            w[i] = 0
            continue
        axis = (
            np.array([
                R_rel[2, 1] - R_rel[1, 2],
                R_rel[0, 2] - R_rel[2, 0],
                R_rel[1, 0] - R_rel[0, 1],
            ])
            / (2 * np.sin(angle))
        )
        w[i] = axis * angle / dts[i]
    return mid, w


def rate_report(name, ts):
    dts = np.diff(ts)
    if len(dts) == 0:
        return f"  {name}: <2 samples>"
    return (
        f"  {name}: {len(ts)} samples, {1 / np.median(dts):7.1f} Hz median, "
        f"dt p99 {np.percentile(dts, 99) * 1e3:6.2f} ms, max gap {dts.max() * 1e3:6.2f} ms"
    )


def estimate_time_offset(t_a, mag_a, t_b, mag_b, fs=200.0, max_lag_s=0.25):
    """Offset that best aligns |w| signals: positive means stream A lags B."""
    t0 = max(t_a[0], t_b[0])
    t1 = min(t_a[-1], t_b[-1])
    grid = np.arange(t0, t1, 1 / fs)
    a = np.interp(grid, t_a, mag_a)
    b = np.interp(grid, t_b, mag_b)
    a = a - a.mean()
    b = b - b.mean()
    max_lag = int(max_lag_s * fs)
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.array([
        np.dot(a[max(0, -k):len(a) - max(0, k)], b[max(0, k):len(b) - max(0, -k)])
        for k in lags
    ])
    norm = np.sqrt(np.sum(a**2) * np.sum(b**2))
    corr = corr / norm
    k = int(np.argmax(corr))
    # quadratic sub-sample interpolation around the peak
    if 0 < k < len(corr) - 1:
        y0, y1, y2 = corr[k - 1], corr[k], corr[k + 1]
        denom = y0 - 2 * y1 + y2
        frac = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-12 else 0.0
    else:
        frac = 0.0
    offset = (lags[k] + frac) / fs
    return offset, corr[k], (lags / fs, corr)


def kabsch(A, B):
    """Rotation R minimizing ||R@A - B|| (A, B: Nx3, rows are paired vectors)."""
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--plot", type=Path, default=None)
    args = ap.parse_args()
    s = args.session

    meta = json.loads((s / "meta.json").read_text())
    frames = load_jsonl(s / "frames.jsonl")
    accel = load_jsonl(s / "accel.jsonl")
    gyro = load_jsonl(s / "gyro.jsonl")
    dm = load_jsonl(s / "devicemotion.jsonl")

    print(f"session: {s.name}")
    print(f"device: {meta['device_model']} iOS {meta['ios_version']}, "
          f"codec {meta['codec']}, "
          f"video {meta['video_format']['width']}x{meta['video_format']['height']}"
          f"@{meta['video_format']['fps']}")

    # --- rates and gaps ---
    t_f = np.array([f["t"] for f in frames])
    t_a = np.array([x["t"] for x in accel])
    t_g = np.array([x["t"] for x in gyro])
    t_d = np.array([x["t"] for x in dm])
    print("\nstream rates:")
    print(rate_report("frames", t_f))
    print(rate_report("accel ", t_a))
    print(rate_report("gyro  ", t_g))
    print(rate_report("devmo ", t_d))

    n_meta = meta.get("frames_appended")
    if n_meta is not None and n_meta != len(frames):
        print(f"  WARNING frames.jsonl has {len(frames)} lines, meta says {n_meta}")
    dropped = meta.get("frames_dropped", 0)
    print(f"  video frames appended: {n_meta}, dropped: {dropped}")

    tracking = {}
    for f in frames:
        tracking[f["tracking"]] = tracking.get(f["tracking"], 0) + 1
    print(f"  tracking states: {tracking}")

    # --- time sync: |w| gyro vs |w| from ARKit poses ---
    ts_arkit, w_arkit = body_angular_velocity(*rotations_from_frames(frames))
    w_gyro = np.array([[x["x"], x["y"], x["z"]] for x in gyro])
    mag_arkit = np.linalg.norm(w_arkit, axis=1)
    mag_gyro = np.linalg.norm(w_gyro, axis=1)

    offset, peak, (lag_axis, corr) = estimate_time_offset(
        ts_arkit, mag_arkit, t_g, mag_gyro)
    print("\ntime sync (ARKit pose-derived |w| vs raw gyro |w|):")
    print(f"  motion level: gyro |w| mean {mag_gyro.mean():.3f} rad/s, "
          f"max {mag_gyro.max():.3f} rad/s")
    print(f"  best offset: {offset * 1e3:+.2f} ms (ARKit relative to gyro), "
          f"correlation {peak:.3f}")
    if peak < 0.7:
        print("  WARNING low correlation — capture may lack rotation, or sync is off")
    exp = np.median([f["exposure_duration"] for f in frames])
    print(f"  (median exposure {exp * 1e3:.2f} ms; a constant offset of roughly "
          f"±half-exposure/rolling-shutter readout is expected)")

    # --- camera-IMU rotation from angular-velocity vectors (Kabsch) ---
    grid = np.arange(max(ts_arkit[0], t_g[0]) + 0.05,
                     min(ts_arkit[-1], t_g[-1]) - 0.05, 1 / 100.0)
    A = np.column_stack([np.interp(grid + offset, t_g, w_gyro[:, i]) for i in range(3)])
    B = np.column_stack([np.interp(grid, ts_arkit, w_arkit[:, i]) for i in range(3)])
    keep = (np.linalg.norm(A, axis=1) > 0.1) & (np.linalg.norm(B, axis=1) > 0.1)
    print(f"\ncamera-IMU rotation (Kabsch over {keep.sum()} paired w vectors):")
    if keep.sum() > 50:
        R_dc = kabsch(A[keep], B[keep])
        resid = np.linalg.norm((R_dc @ A[keep].T).T - B[keep], axis=1)
        angle = np.degrees(np.arccos(np.clip((np.trace(R_dc) - 1) / 2, -1, 1)))
        np.set_printoptions(precision=4, suppress=True)
        print(f"  R (device->camera):\n{np.array2string(R_dc, prefix='  ')}")
        print(f"  rotation angle {angle:.2f} deg, residual |dw| RMS "
              f"{np.sqrt((resid**2).mean()):.4f} rad/s")
    else:
        R_dc = None
        print("  not enough rotational motion to estimate — wave the phone more")

    # --- gravity consistency ---
    a_vec = np.array([[x["x"], x["y"], x["z"]] for x in accel])
    g_dm = np.array([[*x["grav"]] for x in dm])
    # low-pass accel by simple moving average (1 s window)
    k = min(101, len(a_vec) // 2 * 2 + 1)
    kern = np.ones(k) / k
    a_lp = np.column_stack([np.convolve(a_vec[:, i], kern, mode="same") for i in range(3)])
    g_interp = np.column_stack([np.interp(t_a, t_d, g_dm[:, i]) for i in range(3)])
    # Apple reports raw accel in the same sign convention as deviceMotion.gravity
    # (flat face-up: z ~ -1 g in both), so the vectors agree directly.
    cosang = np.sum(a_lp * g_interp, axis=1) / (
        np.linalg.norm(a_lp, axis=1) * np.linalg.norm(g_interp, axis=1) + 1e-12)
    ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
    print("\ngravity consistency:")
    print(f"  |accel| mean {np.linalg.norm(a_vec, axis=1).mean():.4f} g")
    print(f"  angle(low-passed accel, devicemotion gravity): "
          f"median {np.median(ang):.2f} deg, p95 {np.percentile(ang, 95):.2f} deg")

    # --- optional plot ---
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(11, 7))
        axes[0].plot(ts_arkit - t_f[0], mag_arkit, label="|w| from ARKit poses", lw=1)
        axes[0].plot(t_g - t_f[0] + offset, mag_gyro,
                     label=f"|w| gyro (shifted {offset * 1e3:+.1f} ms)", lw=0.8, alpha=0.8)
        axes[0].set_xlabel("t (s)")
        axes[0].set_ylabel("rad/s")
        axes[0].legend()
        axes[0].set_title(s.name)
        axes[1].plot(lag_axis * 1e3, corr, lw=1)
        axes[1].axvline(offset * 1e3, color="r", ls="--", lw=0.8)
        axes[1].set_xlabel("lag (ms)")
        axes[1].set_ylabel("normalized correlation")
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nplot written to {args.plot}")


if __name__ == "__main__":
    sys.exit(main())
