#!/bin/bash
# Run OpenVINS on a session: docker/run_openvins.sh captures/<session>
# Expects cam_imu.bag (tools/export_rosbag.py) and openvins/ configs
# (tools/make_openvins_config.py) inside the session dir.
set -euo pipefail
SESSION_DIR=$(cd "$1" && pwd)

docker run --rm \
    -v "$SESSION_DIR":/data \
    openvins \
    roslaunch /ov_serial.launch \
        config_path:=/data/openvins/estimator_config.yaml \
        bag:=/data/cam_imu.bag \
        path_est:=/data/openvins/traj_estimate.txt

echo "trajectory written to $SESSION_DIR/openvins/traj_estimate.txt"
