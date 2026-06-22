# Shared by the demo launchers. Hard-clear any prior Isaac livestream before a new
# one starts: the native app ignores Ctrl+C, so a killed run can leak the NVENC
# encoder + port 49100 and the next stream fails with NVST_R_BUSY. Only one
# livestream NVENC session exists, so clear BOTH demo sims + stale ROS nodes,
# SIGKILL, then wait for the signaling port to free.
free_isaac_stream() {
    echo "  Clearing any previous Isaac livestream / nodes..."
    pkill -9 -f "10_dam_car_.*_comparison_demo.*\.py" 2>/dev/null || true
    pkill -9 -f "11_go2_squad_dispatch.py"          2>/dev/null || true
    pkill -9 -f "dam_jetbot_guard_node.py"          2>/dev/null || true
    pkill -9 -f "go2_squad_dispatch_client.py"      2>/dev/null || true
    for _ in $(seq 1 20); do
        ss -tlnp 2>/dev/null | grep -q ":49100 " || break
        sleep 1
    done
    sleep 2  # let the GPU reclaim the NVENC session
}
