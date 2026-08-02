#!/usr/bin/env python3
"""Generate OpenVINS config files for a VIOLogger session.

Writes captures/<session>/openvins/{estimator_config,kalibr_imu_chain,
kalibr_imucam_chain}.yaml. Camera intrinsics come from the session log
(median over frames, scaled to the exported bag resolution); the camera-IMU
rotation comes from a Kabsch fit of gyro vs ARKit angular velocities.
Unknowns (distortion, camera-IMU translation, time offset) start at zero and
are refined online by OpenVINS.

Run tools/export_rosbag.py first (bag_info.json records the export scale).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from validate_session import body_angular_velocity, estimate_time_offset, kabsch
from vio_session import Session


def estimate_R_CtoI(s: Session) -> np.ndarray:
    """Camera-to-IMU rotation from paired angular velocities."""
    ts_arkit, w_arkit = body_angular_velocity(s.frames.t, s.frames.R)
    mag_arkit = np.linalg.norm(w_arkit, axis=1)
    mag_gyro = np.linalg.norm(s.gyro.xyz, axis=1)
    offset, peak, _ = estimate_time_offset(ts_arkit, mag_arkit, s.gyro.t, mag_gyro)
    if peak < 0.7:
        raise RuntimeError(
            f"gyro/ARKit correlation too low ({peak:.2f}) to trust Kabsch fit")
    grid = np.arange(max(ts_arkit[0], s.gyro.t[0]) + 0.05,
                     min(ts_arkit[-1], s.gyro.t[-1]) - 0.05, 0.01)
    A = np.column_stack(
        [np.interp(grid + offset, s.gyro.t, s.gyro.xyz[:, i]) for i in range(3)])
    B = np.column_stack(
        [np.interp(grid, ts_arkit, w_arkit[:, i]) for i in range(3)])
    keep = (np.linalg.norm(A, axis=1) > 0.1) & (np.linalg.norm(B, axis=1) > 0.1)
    if keep.sum() < 50:
        raise RuntimeError("not enough rotational motion for Kabsch fit")
    R_ItoC_arkit = kabsch(A[keep], B[keep])  # IMU/device -> ARKit camera frame
    # ARKit's camera frame is y-up/z-toward-viewer (OpenGL); OpenVINS expects
    # the CV convention y-down/z-forward-through-lens: 180 deg flip about x.
    flip = np.diag([1.0, -1.0, -1.0])
    R_CtoI_cv = R_ItoC_arkit.T @ flip
    # sanity: device +z (out of screen) must oppose the optical axis (cam +z)
    assert R_CtoI_cv[2, 2] < -0.9, f"unexpected camera-IMU geometry:\n{R_CtoI_cv}"
    return R_CtoI_cv


def yaml_matrix(m: np.ndarray, indent: str = "    ") -> str:
    return "\n".join(
        indent + "- [" + ", ".join(f"{v:.8f}" for v in row) + "]" for row in m)


ESTIMATOR_CONFIG = """%YAML:1.0

verbosity: "INFO"

use_fej: true
integration: "rk4"
use_stereo: false
max_cameras: 1

calib_cam_extrinsics: true    # refine R_ItoC and p_CinI (we only seed the rotation)
calib_cam_intrinsics: true    # refine focal/center/distortion (we seed zero distortion)
calib_cam_timeoffset: true    # clocks are shared, expect ~ms exposure-anchor offset
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false

max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1

gravity_mag: 9.80665

feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"

try_zupt: false
zupt_chi2_multipler: 0
zupt_max_velocity: 0.1
zupt_noise_multiplier: 10
zupt_max_disparity: 0.5
zupt_only_at_beginning: false

init_window_time: 3.0
init_imu_thresh: 0.5          # handheld start is gentler than a drone takeoff
init_max_disparity: 10.0
init_max_features: 75

init_dyn_use: true            # allow initialization while already moving
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 100
init_dyn_mle_max_time: 0.5   # serial mode: we can afford a slow, careful init
init_dyn_mle_max_threads: 4
init_dyn_num_pose: 10
init_dyn_min_deg: 10.0
init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-12
init_dyn_bias_g: [0.0, 0.0, 0.0]
init_dyn_bias_a: [0.0, 0.0, 0.0]

record_timing_information: false
record_timing_filepath: "/data/openvins/traj_timing.txt"

save_total_state: true
filepath_est: "/data/openvins/state_estimate.txt"
filepath_std: "/data/openvins/state_deviation.txt"
filepath_gt: "/data/openvins/state_groundtruth.txt"

use_klt: true
num_pts: 200
fast_threshold: 20
grid_x: 5
grid_y: 5
min_px_dist: 10
knn_ratio: 0.70
track_frequency: 31.0
downsample_cameras: false
num_opencv_threads: 4
histogram_method: "HISTOGRAM"

use_aruco: false
num_aruco: 1024
downsize_aruco: true

# sigma inflated vs EuRoC: zero-seeded distortion + unmodeled rolling shutter
up_msckf_sigma_px: 2
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 2
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1

use_mask: false

relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
"""

# Phone-grade MEMS noise, inflated to absorb unmodeled effects.
IMU_CHAIN = """%YAML:1.0

imu0:
  T_i_b:
    - [1.0, 0.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0, 0.0]
    - [0.0, 0.0, 1.0, 0.0]
    - [0.0, 0.0, 0.0, 1.0]
  accelerometer_noise_density: 2.0e-2
  accelerometer_random_walk: 2.0e-3
  gyroscope_noise_density: 2.0e-3
  gyroscope_random_walk: 1.0e-4
  rostopic: /imu0
  time_offset: 0.0
  update_rate: 100.0
  model: "kalibr"
  Tw:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoGYRO:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Ta:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  R_IMUtoACC:
    - [1.0, 0.0, 0.0]
    - [0.0, 1.0, 0.0]
    - [0.0, 0.0, 1.0]
  Tg:
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
    - [0.0, 0.0, 0.0]
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    args = ap.parse_args()

    s = Session(args.session)
    out = s.path / "openvins"
    out.mkdir(exist_ok=True)

    info = json.loads((out / "bag_info.json").read_text())
    scale = info["scale"]
    fx, fy, cx, cy = s.median_intrinsics() * scale

    R_CtoI = estimate_R_CtoI(s)
    T_imu_cam = np.eye(4)
    T_imu_cam[:3, :3] = R_CtoI

    cam_chain = f"""%YAML:1.0

cam0:
  T_imu_cam:  # R_CtoI (from gyro/ARKit Kabsch fit), p_CinI seeded at zero
{yaml_matrix(T_imu_cam)}
  cam_overlaps: []
  camera_model: pinhole
  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]
  distortion_model: radtan
  intrinsics: [{fx:.4f}, {fy:.4f}, {cx:.4f}, {cy:.4f}]
  resolution: [{info["width"]}, {info["height"]}]
  rostopic: /cam0/image_raw
"""

    (out / "estimator_config.yaml").write_text(ESTIMATOR_CONFIG)
    (out / "kalibr_imu_chain.yaml").write_text(IMU_CHAIN)
    (out / "kalibr_imucam_chain.yaml").write_text(cam_chain)
    np.set_printoptions(precision=4, suppress=True)
    print(f"wrote configs to {out}")
    print(f"intrinsics @ {info['width']}x{info['height']}: "
          f"[{fx:.1f}, {fy:.1f}, {cx:.1f}, {cy:.1f}]")
    print(f"R_CtoI:\n{R_CtoI}")


if __name__ == "__main__":
    main()
