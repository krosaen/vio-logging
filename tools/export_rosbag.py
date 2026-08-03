#!/usr/bin/env python3
"""Export a VIOLogger session to a ROS1 bag for OpenVINS.

Writes /cam0/image_raw (mono8) and /imu0 (gyro rad/s + accel m/s^2, accel
interpolated onto gyro timestamps). Bag timestamps are the session's
seconds-since-boot clock, so estimator output aligns with frames.jsonl
timestamps directly.

Usage: export_rosbag.py <session_dir> [--scale 0.5] [--out <path.bag>]
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from vio_session import Session

from rosbags.rosbag1 import Writer
from rosbags.typesys import Stores, get_typestore


def make_header(typestore, seq: int, t: float, frame_id: str):
    """Build a std_msgs Header regardless of whether this store has 'seq'."""
    Header = typestore.types["std_msgs/msg/Header"]
    Time = typestore.types["builtin_interfaces/msg/Time"]
    stamp = Time(sec=int(t), nanosec=int(round((t - int(t)) * 1e9)))
    kwargs = {"stamp": stamp, "frame_id": frame_id}
    if "seq" in Header.__dataclass_fields__:
        kwargs["seq"] = seq
    return Header(**kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session", type=Path)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="image downscale factor (default 0.5 -> 960x720)")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="write every Nth camera frame (e.g. 15 -> 4 Hz for "
                         "kalibr_calibrate_cameras); IMU is always full rate")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = Session(args.session)
    out = args.out or (s.path / "cam_imu.bag")
    out.unlink(missing_ok=True)

    typestore = get_typestore(Stores.ROS1_NOETIC)
    Imu = typestore.types["sensor_msgs/msg/Imu"]
    Image = typestore.types["sensor_msgs/msg/Image"]
    Quaternion = typestore.types["geometry_msgs/msg/Quaternion"]
    Vector3 = typestore.types["geometry_msgs/msg/Vector3"]

    t_imu, gyro, accel = s.imu_merged()
    zeros9 = np.zeros(9)
    quat0 = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

    with Writer(out) as writer:
        conn_imu = writer.add_connection(
            "/imu0", Imu.__msgtype__, typestore=typestore)
        conn_cam = writer.add_connection(
            "/cam0/image_raw", Image.__msgtype__, typestore=typestore)

        for i in range(len(t_imu)):
            msg = Imu(
                header=make_header(typestore, i, t_imu[i], "imu0"),
                orientation=quat0,
                orientation_covariance=zeros9,
                angular_velocity=Vector3(*gyro[i]),
                angular_velocity_covariance=zeros9,
                linear_acceleration=Vector3(*accel[i]),
                linear_acceleration_covariance=zeros9,
            )
            writer.write(conn_imu, int(t_imu[i] * 1e9),
                         typestore.serialize_ros1(msg, Imu.__msgtype__))

        width = height = None
        for idx, t, img in s.iter_video(gray=True):
            if idx % args.frame_stride:
                continue
            if args.scale != 1.0:
                img = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                 interpolation=cv2.INTER_AREA)
            height, width = img.shape
            msg = Image(
                header=make_header(typestore, idx, t, "cam0"),
                height=height,
                width=width,
                encoding="mono8",
                is_bigendian=0,
                step=width,
                data=np.ascontiguousarray(img).reshape(-1),
            )
            writer.write(conn_cam, int(t * 1e9),
                         typestore.serialize_ros1(msg, Image.__msgtype__))
            if idx % 120 == 0:
                print(f"  frame {idx}/{len(s.frames.t)}")

    if args.frame_stride != 1:
        # calibration side-bag: don't overwrite the main bag's info
        print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), strided x{args.frame_stride}")
        return

    info = {
        "scale": args.scale,
        "width": width,
        "height": height,
        "imu_samples": len(t_imu),
        "frames": len(s.frames.t),
        "bag": out.name,
    }
    info_path = s.path / "openvins"
    info_path.mkdir(exist_ok=True)
    (info_path / "bag_info.json").write_text(json.dumps(info, indent=2))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB), "
          f"{info['frames']} frames @ {width}x{height}, {info['imu_samples']} imu")


if __name__ == "__main__":
    main()
