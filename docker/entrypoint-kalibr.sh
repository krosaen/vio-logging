#!/bin/bash
set -e
# catkin setup scripts consume "$@" when sourced — clear args first
ARGS=("$@")
set --
source /opt/ros/noetic/setup.bash
source /catkin_ws/devel/setup.bash
# allow manual focal-length init if auto-init fails (kalibr PR #346)
export KALIBR_MANUAL_FOCAL_LENGTH_INIT=1
# pure-python cv_bridge shim shadows the broken arm64 boost extension
export PYTHONPATH="/shim:$PYTHONPATH"
exec "${ARGS[@]}"
