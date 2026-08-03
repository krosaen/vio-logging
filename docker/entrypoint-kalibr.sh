#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
# allow manual focal-length init if auto-init fails (kalibr PR #346)
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
exec "$@"
