#!/bin/bash

# Demo 11: Go2 squad dispatch (native isaacsim) over ROS 2 + WebRTC.
# Launches the simulator (background, --ros-control + WebRTC stream) and the
# interactive dispatch client (foreground — you type squad commands here).
#
# Three pieces:
#   sim          scripts/11_go2_squad_dispatch.py        (SquadRosBridge: subscribes
#                                                          /squad/dispatch_cmd)
#   dispatch_cmd tools/ros/go2_squad_dispatch_client.py  (publishes /squad/dispatch_cmd)
#   操控/control  this client's --interactive mode        (keyboard -> dispatch)

set -e

echo "[demo11] Setting up Isaac Lab + ROS 2..."
conda deactivate 2>/dev/null || true
source ~/IsaacLab/env_isaaclab/bin/activate
cd ~/ISAAC_LAB_REBOT_TRAINING
source /opt/ros/jazzy/setup.bash
if [[ $(uname -m) == "aarch64" ]]; then
    export LD_PRELOAD="/lib/aarch64-linux-gnu/libgomp.so.1"
fi

CMD_TOPIC=/squad/dispatch_cmd
SIM_LOG=/tmp/go2_squad_sim.log

echo ""
echo "=========================================================================="
echo "  Demo 11: Go2 Squad Dispatch (ROS 2 control + WebRTC)"
echo "=========================================================================="

# Hard-clear any prior Isaac livestream (and stale clients) before launch. The
# native app ignores Ctrl+C, so a previous run can leak the NVENC encoder + port
# 49100 -> the new stream fails with NVST_R_BUSY. Only one livestream NVENC session
# exists, so also clear demo 10's sim. SIGKILL + wait for the port/encoder to free.
echo "  Clearing any previous Isaac livestream / clients..."
pkill -9 -f "11_go2_squad_dispatch.py"        2>/dev/null || true
pkill -9 -f "10_dam_car_ros_comparison_demo.py" 2>/dev/null || true
pkill -9 -f "go2_squad_dispatch_client.py"    2>/dev/null || true
# Wait until the WebRTC signaling port (49100) is released (or give up after 20s).
for _ in $(seq 1 20); do
    ss -tlnp 2>/dev/null | grep -q ":49100 " || break
    sleep 1
done
sleep 2  # let the GPU reclaim the NVENC session

echo "  Launching simulator in background (logs: $SIM_LOG)..."
echo "  (first boot pulls the Go2 + warehouse assets — can take a few minutes)"
nohup python scripts/11_go2_squad_dispatch.py \
    --ros-control --ros-node-name go2_squad_demo --livestream 2 "$@" \
    > "$SIM_LOG" 2>&1 &
SIM_PID=$!
# Kill the sim when this script exits (Ctrl+C in the client).
trap 'kill "$SIM_PID" 2>/dev/null || true' EXIT

# Wait until the sim's ROS control interface is listening (or it died).
echo -n "  Waiting for the squad ROS interface to come up"
for _ in $(seq 1 600); do
    if grep -q "waiting for ROS dispatch commands" "$SIM_LOG" 2>/dev/null; then
        echo " — ready."
        break
    fi
    if ! kill -0 "$SIM_PID" 2>/dev/null; then
        echo ""
        echo "[demo11][ERROR] simulator exited during startup. Last log lines:"
        tail -n 20 "$SIM_LOG"
        exit 1
    fi
    echo -n "."
    sleep 2
done

echo ""
echo "  Connect the WebRTC client to 192.168.90.162 (signal 49100 / stream 47998)."
echo "  Now driving the squad — type dispatch commands below (Ctrl+C to quit both)."
echo ""

# Interactive dispatch client in the FOREGROUND (needs the keyboard / TTY).
python tools/ros/go2_squad_dispatch_client.py --interactive --topic "$CMD_TOPIC"
