#!/bin/bash
# Camera-IMU spatial/temporal calibration on a session's full-rate bag.
# Usage: docker/run_kalibr_imucam.sh captures/<session>
# Expects cam_imu.bag, kalibr/camchain-*.yaml (from run_kalibr_cameras.sh),
# and docker/kalibr_imu.yaml.
set -euo pipefail
SESSION_DIR=$(cd "$1" && pwd)
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
CAMCHAIN=$(ls "$SESSION_DIR"/kalibr/camchain-*.yaml | head -1)
[ -f "$CAMCHAIN" ] || { echo "no camchain yaml — run run_kalibr_cameras.sh first"; exit 1; }

docker run --rm \
    -v "$SESSION_DIR":/data \
    -v "$REPO_DIR/target":/target:ro \
    -v "$REPO_DIR/docker/kalibr_imu.yaml":/imu.yaml:ro \
    -w /data/kalibr \
    kalibr \
    rosrun kalibr kalibr_calibrate_imu_camera \
        --bag /data/cam_imu.bag \
        --cam "/data/kalibr/$(basename "$CAMCHAIN")" \
        --imu /imu.yaml \
        --target /target/aprilgrid.yaml \
        --dont-show-report

echo "results in $SESSION_DIR/kalibr/ (camchain-imucam-*.yaml)"
