#!/bin/bash

# Demo 10: Twin-Lane Jetbot DAM Comparison (DIRECT, native isaacsim + WebRTC)
# Single process — DAM driven in-process like demo 09, no ROS, no guard node.
# (Legacy ROS-bridge variant: scripts/demo10_legacy_ros_webrtc.sh)
#
# Extra args are forwarded, e.g.:
#   scripts/demo10_webrtc.sh --worker
#   scripts/demo10_webrtc.sh --cam-eye "-3,0,2.5" --cam-target "0.7,0,0.1"

set -e

# Repo root = parent of this script's dir (this launcher lives in scripts/).
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "[demo10] Setting up Isaac Lab environment..."
conda deactivate 2>/dev/null || true
source ~/IsaacLab/env_isaaclab/bin/activate
cd "$REPO_DIR"

echo ""
echo "=========================================================================="
echo "  Demo 10: Twin-Lane Jetbot DAM Comparison (DIRECT, WebRTC)"
echo "=========================================================================="
echo ""

# Native app ignores Ctrl+C and can leak the NVENC encoder + port 49100; clear any
# prior stream before starting a new one.
source tools/ros/free_isaac_stream.sh
free_isaac_stream

echo "  Launching simulator with WebRTC streaming (single process, no guard node)..."
echo ""

# Foreground; Ctrl+C stops it. Extra args forwarded (e.g. --worker).
python scripts/10_dam_car_direct_comparison_demo.py --livestream 2 "$@"
