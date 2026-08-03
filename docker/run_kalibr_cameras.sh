#!/bin/bash
# Camera intrinsics + distortion calibration on a session's 4 Hz bag.
# Usage: docker/run_kalibr_cameras.sh captures/<session>
# Expects cam_imu_4hz.bag (tools/export_rosbag.py <s> --frame-stride 15
#   --out <s>/cam_imu_4hz.bag) and ../target/aprilgrid.yaml.
set -euo pipefail
SESSION_DIR=$(cd "$1" && pwd)
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$SESSION_DIR/kalibr"

docker run --rm \
    -v "$SESSION_DIR":/data \
    -v "$REPO_DIR/target":/target:ro \
    -e KALIBR_FOCAL_INIT="${KALIBR_FOCAL_INIT:-}" \
    -w /data/kalibr \
    kalibr \
    rosrun kalibr kalibr_calibrate_cameras \
        --bag /data/cam_imu_4hz.bag \
        --topics /cam0/image_raw \
        --models pinhole-radtan \
        --target /target/aprilgrid.yaml \
        --dont-show-report

echo "results in $SESSION_DIR/kalibr/ (camchain-*.yaml)"
