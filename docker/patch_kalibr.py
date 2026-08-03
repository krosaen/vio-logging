#!/usr/bin/env python3
"""Build-time patch: allow overriding Kalibr's focal-length initialization.

Kalibr's vanishing-point auto-init is unreliable on small low-distortion
boards (it converged to f~280 vs true ~672 on clean data and the optimizer
diverged). With KALIBR_FOCAL_INIT set, the value is written into the
projection parameters right after initializeIntrinsics().
"""

PATH = ("/catkin_ws/src/kalibr/aslam_offline_calibration/kalibr/python/"
        "kalibr_camera_calibration/CameraCalibrator.py")

OLD = """        #obtain focal length guess
        success = self.geometry.initializeIntrinsics(observations)
"""

NEW = OLD + """        import os as _os
        print("[focal-init-patch] env KALIBR_FOCAL_INIT =",
              _os.environ.get("KALIBR_FOCAL_INIT"), flush=True)
        if _os.environ.get("KALIBR_FOCAL_INIT"):
            _f = float(_os.environ["KALIBR_FOCAL_INIT"])
            _p = self.geometry.projection().getParameters()
            _p[0, 0] = _f
            _p[1, 0] = _f
            self.geometry.projection().setParameters(_p)
            _chk = self.geometry.projection().getParameters().flatten()
            print("[focal-init-patch] after setParameters:", _chk, flush=True)
            success = True
"""

src = open(PATH).read()
assert OLD in src, "patch anchor not found in CameraCalibrator.py"
open(PATH, "w").write(src.replace(OLD, NEW))
print("patched CameraCalibrator.py (KALIBR_FOCAL_INIT)")
