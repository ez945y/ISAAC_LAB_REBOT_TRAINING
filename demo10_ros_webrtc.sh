#!/bin/bash

# Demo 10: Twin-Lane Jetbot DAM Comparison (ROS 2 bridge, native isaacsim)
# Launches the DAM guard node (background) + the simulator with WebRTC streaming.

set -e

# Setup Isaac Lab environment
echo "[demo10] Setting up Isaac Lab environment..."
conda deactivate 2>/dev/null || true
cd ~/IsaacLab
source env_isaaclab/bin/activate

# Navigate to workspace
cd ~/ISAAC_LAB_REBOT_TRAINING

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

# Hard-clear any prior Isaac livestream before launch. The native app ignores
# Ctrl+C, so a previous run can leak the NVENC encoder + port 49100 -> the new
# stream fails with NVST_R_BUSY. Only one livestream NVENC session exists, so also
# clear demo 11's sim + stale guard. SIGKILL + wait for the port/encoder to free.
echo "  Clearing any previous Isaac livestream / nodes..."
pkill -9 -f "10_dam_car_ros_comparison_demo.py" 2>/dev/null || true
pkill -9 -f "11_go2_squad_dispatch.py"          2>/dev/null || true
pkill -9 -f "dam_jetbot_guard_node.py"          2>/dev/null || true
for _ in $(seq 1 20); do
    ss -tlnp 2>/dev/null | grep -q ":49100 " || break
    sleep 1
done
sleep 2  # let the GPU reclaim the NVENC session

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
#   ./demo10_ros_webrtc.sh --cam-eye "-3,0,2.5" --cam-target "0.7,0,0.1"
python scripts/10_dam_car_ros_comparison_demo.py --livestream 2 "$@"
