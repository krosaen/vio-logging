# OpenVINS runtime for VIOLogger sessions (arm64/amd64).
# Build:  docker build -t openvins docker/
# Run via docker/run_openvins.sh
FROM ros:noetic-ros-base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential cmake \
    libeigen3-dev libboost-all-dev libceres-dev \
    python3-catkin-tools python3-osrf-pycommon \
    ros-noetic-cv-bridge ros-noetic-image-transport ros-noetic-tf \
    && rm -rf /var/lib/apt/lists/*

# Pin OpenVINS (master as of 2025-11-30)
ARG OPENVINS_COMMIT=69488123ed9362dd44b6f28e7f4680abbff1442b
RUN mkdir -p /catkin_ws/src && cd /catkin_ws/src \
    && git clone https://github.com/rpng/open_vins.git \
    && cd open_vins && git checkout ${OPENVINS_COMMIT}

RUN . /opt/ros/noetic/setup.sh && cd /catkin_ws \
    && catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release \
    && catkin build -j2 ov_msckf ov_eval  # -j2: Eigen-heavy TUs OOM at higher parallelism in an 8 GB VM

COPY ov_serial.launch /ov_serial.launch
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
