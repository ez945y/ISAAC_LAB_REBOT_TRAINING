#!/bin/bash

# Demo 10: Twin-Lane Jetbot DAM Comparison (ROS 2 bridge, native isaacsim)
# Launches the DAM guard node (background) + the simulator with WebRTC streaming.

set -e

# Repo root = parent of this script's dir (this launcher lives in scripts/).
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Setup Isaac Lab environment
echo "[demo10] Setting up Isaac Lab environment..."
conda deactivate 2>/dev/null || true
source ~/IsaacLab/env_isaaclab/bin/activate

# Navigate to workspace
cd "$REPO_DIR"

# Setup ROS 2
echo "[demo10] Sourcing ROS 2..."
source /opt/ros/jazzy/setup.bash

# Export LD_PRELOAD if on aarch64 (ARM)
if [[ $(uname -m) == "aarch64" ]]; then
    export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
fi

echo ""
echo "=========================================================================="
echo "  Demo 10: Twin-Lane Jetbot DAM Comparison (ROS 2 bridge, WebRTC)"
echo "=========================================================================="
echo ""

source tools/ros/free_isaac_stream.sh
free_isaac_stream

# Launch the DAM guard node as a BACKGROUND process. This is a headless server —
# gnome-terminal needs an X display and dies with "Cannot open display", so the
# guard never starts and the DAM car never moves. Background it instead; logs to a file.
GUARD_LOG=/tmp/dam_guard.log
if pgrep -f "dam_jetbot_guard_node.py" > /dev/null; then
    echo "✓ DAM guard node is already running."
    GUARD_PID=""
else
    echo "  Launching DAM guard node in background (logs: $GUARD_LOG)..."
    nohup python tools/ros/dam_jetbot_guard_node.py > "$GUARD_LOG" 2>&1 &
    GUARD_PID=$!
    echo "  Guard node PID: $GUARD_PID  (tail -f $GUARD_LOG to watch its decisions)"
    # Stop the guard node when this script exits.
    trap '[ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null || true' EXIT
    sleep 3
fi

echo ""
echo "  Launching simulator with WebRTC streaming..."
echo "  (DAM car starts moving once the guard connects over ROS — a few seconds.)"
echo ""

# Launch demo with WebRTC livestream (foreground; Ctrl+C stops it and the guard).
# Extra args are forwarded, e.g. tune the camera:
#   scripts/demo10_ros_webrtc.sh --cam-eye "-3,0,2.5" --cam-target "0.7,0,0.1"
python scripts/10_dam_car_ros_comparison_demo.py --livestream 2 "$@"
