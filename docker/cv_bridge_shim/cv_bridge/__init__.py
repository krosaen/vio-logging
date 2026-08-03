"""Pure-python cv_bridge replacement (subset).

The apt/source cv_bridge boost extension fails to initialize on arm64 noetic
(SystemError at module init). Kalibr only uses CvBridge.imgmsg_to_cv2 and
compressed_imgmsg_to_cv2, both trivially implementable with numpy/cv2.
This module is prepended to PYTHONPATH by the container entrypoint.
"""

import numpy as np

__all__ = ["CvBridge", "CvBridgeError"]


class CvBridgeError(Exception):
    pass


_ENCODINGS = {
    # encoding -> (dtype, channels)
    "mono8": (np.uint8, 1),
    "8UC1": (np.uint8, 1),
    "mono16": (np.uint16, 1),
    "16UC1": (np.uint16, 1),
    "bgr8": (np.uint8, 3),
    "rgb8": (np.uint8, 3),
    "8UC3": (np.uint8, 3),
    "bgra8": (np.uint8, 4),
    "rgba8": (np.uint8, 4),
}


class CvBridge:
    def imgmsg_to_cv2(self, msg, desired_encoding="passthrough"):
        if msg.encoding not in _ENCODINGS:
            raise CvBridgeError(f"unsupported encoding {msg.encoding!r}")
        dtype, ch = _ENCODINGS[msg.encoding]
        itemsize = np.dtype(dtype).itemsize
        buf = np.frombuffer(msg.data, dtype=dtype)
        row_elems = msg.step // itemsize
        img = buf.reshape(msg.height, row_elems)
        img = img[:, : msg.width * ch]
        if ch > 1:
            img = img.reshape(msg.height, msg.width, ch)
        if msg.is_bigendian and itemsize > 1:
            img = img.byteswap()
        return self._convert(img, msg.encoding, desired_encoding)

    def compressed_imgmsg_to_cv2(self, msg, desired_encoding="passthrough"):
        import cv2

        img = cv2.imdecode(np.frombuffer(msg.data, np.uint8),
                           cv2.IMREAD_UNCHANGED)
        if img is None:
            raise CvBridgeError("cv2.imdecode failed")
        src = "mono8" if img.ndim == 2 else "bgr8"
        return self._convert(img, src, desired_encoding)

    @staticmethod
    def _convert(img, src, dst):
        if dst in ("passthrough", src):
            return img
        import cv2

        pairs = {
            ("bgr8", "rgb8"): cv2.COLOR_BGR2RGB,
            ("rgb8", "bgr8"): cv2.COLOR_RGB2BGR,
            ("bgr8", "mono8"): cv2.COLOR_BGR2GRAY,
            ("rgb8", "mono8"): cv2.COLOR_RGB2GRAY,
            ("mono8", "bgr8"): cv2.COLOR_GRAY2BGR,
            ("mono8", "rgb8"): cv2.COLOR_GRAY2RGB,
            ("bgra8", "bgr8"): cv2.COLOR_BGRA2BGR,
            ("rgba8", "rgb8"): cv2.COLOR_RGBA2RGB,
        }
        key = (src, dst)
        if key not in pairs:
            raise CvBridgeError(f"unsupported conversion {src} -> {dst}")
        return cv2.cvtColor(img, pairs[key])
