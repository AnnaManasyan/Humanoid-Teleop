#!/bin/bash
# Starts the RealSense image server on the robot (Unitree G1 Jetson NX).
# Streams RGB / IR / depth over ZMQ on 192.168.123.164:5556 to the host worker.
#
# Usage (on the robot, after `ssh unitree@192.168.1.36`):
#   cd deployment
#   bash start_image_server.sh        # prompts for sudo password (123) to free the camera
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Free the RealSense if another process is holding the /dev/video* devices.
# (Needs sudo -> this is the password prompt; the camera won't open otherwise.)
sudo fuser -k /dev/video* 2>/dev/null || true

# uv-managed virtualenv with system pyrealsense2/cv2/numpy + pyzmq.
export PATH="$HOME/.local/bin:$PATH"
source "$SCRIPT_DIR/.venv/bin/activate"

exec python realsense_server.py
