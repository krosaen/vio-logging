"""Parser for VIOLogger session folders.

A Session gives numpy views of every logged stream plus a lazy frame iterator
over the video. See README "Session format" for the on-disk schema; unit and
frame conventions are recorded in each session's meta.json.

Example:
    from vio_session import Session
    s = Session("captures/session_20260726_212538")
    s.frames.t              # (N,) ARKit frame timestamps (seconds since boot)
    s.frames.transform      # (N, 4, 4) camera-to-world poses
    s.gyro.t, s.gyro.xyz    # (M,), (M, 3)
    for idx, t, image in s.iter_video():   # BGR uint8 frames, in log order
        ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class ImuStream:
    """Timestamped 3-vector stream (accel in g, gyro in rad/s)."""

    t: np.ndarray    # (N,)
    xyz: np.ndarray  # (N, 3)

    @classmethod
    def from_jsonl(cls, path: Path) -> "ImuStream":
        rows = _load_jsonl(path)
        return cls(
            t=np.array([r["t"] for r in rows]),
            xyz=np.array([[r["x"], r["y"], r["z"]] for r in rows]),
        )


@dataclass
class DeviceMotionStream:
    t: np.ndarray      # (N,)
    quat: np.ndarray   # (N, 4) attitude wxyz
    rot: np.ndarray    # (N, 3) rotation rate rad/s (bias-corrected by CoreMotion)
    grav: np.ndarray   # (N, 3) gravity direction, g units
    uacc: np.ndarray   # (N, 3) user acceleration (gravity-removed), g units

    @classmethod
    def from_jsonl(cls, path: Path) -> "DeviceMotionStream":
        rows = _load_jsonl(path)
        return cls(
            t=np.array([r["t"] for r in rows]),
            quat=np.array([r["quat"] for r in rows]),
            rot=np.array([r["rot"] for r in rows]),
            grav=np.array([r["grav"] for r in rows]),
            uacc=np.array([r["uacc"] for r in rows]),
        )


@dataclass
class FrameStream:
    idx: np.ndarray               # (N,) frame index == video frame order
    t: np.ndarray                 # (N,) capture timestamp
    exposure_duration: np.ndarray # (N,)
    exposure_offset: np.ndarray   # (N,)
    intrinsics: np.ndarray        # (N, 4) fx, fy, cx, cy
    transform: np.ndarray         # (N, 4, 4) ARKit camera-to-world
    tracking: list[str]
    features: np.ndarray          # (N,) ARKit feature count
    has_depth: np.ndarray         # (N,) bool

    @classmethod
    def from_jsonl(cls, path: Path) -> "FrameStream":
        rows = _load_jsonl(path)
        # transforms are stored column-major (simd): reshape then transpose
        tf = np.array([r["transform"] for r in rows]).reshape(-1, 4, 4)
        tf = np.transpose(tf, (0, 2, 1))
        return cls(
            idx=np.array([r["idx"] for r in rows]),
            t=np.array([r["t"] for r in rows]),
            exposure_duration=np.array([r["exposure_duration"] for r in rows]),
            exposure_offset=np.array([r["exposure_offset"] for r in rows]),
            intrinsics=np.array([r["intrinsics"] for r in rows]),
            transform=tf,
            tracking=[r["tracking"] for r in rows],
            features=np.array([r["features"] for r in rows]),
            has_depth=np.array([r["depth"] for r in rows], dtype=bool),
        )

    @property
    def R(self) -> np.ndarray:
        """(N, 3, 3) camera-to-world rotations."""
        return self.transform[:, :3, :3]

    @property
    def p(self) -> np.ndarray:
        """(N, 3) camera positions in world frame (meters)."""
        return self.transform[:, :3, 3]


class Session:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not (self.path / "meta.json").exists():
            raise FileNotFoundError(f"not a session dir (no meta.json): {self.path}")
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.frames = FrameStream.from_jsonl(self.path / "frames.jsonl")
        self.accel = ImuStream.from_jsonl(self.path / "accel.jsonl")
        self.gyro = ImuStream.from_jsonl(self.path / "gyro.jsonl")
        self.devicemotion = DeviceMotionStream.from_jsonl(
            self.path / "devicemotion.jsonl")

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def video_path(self) -> Path:
        return self.path / "frames.mov"

    def median_intrinsics(self) -> np.ndarray:
        """[fx, fy, cx, cy] — per-frame values vary with autofocus/OIS."""
        return np.median(self.frames.intrinsics, axis=0)

    def imu_merged(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(t, gyro_xyz, accel_xyz) with accel interpolated onto gyro timestamps
        and converted to m/s^2 specific force — the convention IMU-driven
        estimators (ROS/EuRoC) expect.

        Sign note: CoreMotion reports acceleration in Apple's convention where
        a device at rest face-up reads z = -1 g (the gravity direction). The
        standard specific-force convention is its negation (+9.81 up at rest),
        hence the sign flip here. Gyro is right-handed in both conventions.
        """
        t = self.gyro.t
        acc = np.column_stack([
            np.interp(t, self.accel.t, self.accel.xyz[:, i]) for i in range(3)
        ]) * -9.80665
        return t, self.gyro.xyz.copy(), acc

    def iter_video(self, gray: bool = False):
        """Yield (idx, t, image) for each video frame, in log order.

        Video frame order matches frames.jsonl order (both are append order).
        Images are BGR uint8 (or single-channel if gray=True). Requires
        opencv-python.
        """
        import cv2

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.video_path}")
        try:
            for i, t in enumerate(self.frames.t):
                ok, img = cap.read()
                if not ok:
                    raise RuntimeError(
                        f"video ended at frame {i}, expected {len(self.frames.t)}")
                if gray:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                yield int(self.frames.idx[i]), float(t), img
        finally:
            cap.release()
