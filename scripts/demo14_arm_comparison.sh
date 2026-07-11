#!/bin/bash

# Demo 14: Scripted SO-ARM-101 DAM Comparison (Twin-Arm, self-contained)
# Self-contained demonstration showing DAM safety layer on dual arms.

set -e

# Repo root = parent of this script's dir (this launcher lives in scripts/).
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Setup Isaac Lab environment
echo "[demo14] Setting up Isaac Lab environment..."
conda deactivate 2>/dev/null || true
source ~/IsaacLab/env_isaaclab/bin/activate

# Navigate to workspace
cd "$REPO_DIR"

echo ""
echo "=========================================================================="
echo "  Demo 14: Scripted SO-ARM-101 DAM Comparison (Twin-Arm, self-contained)"
echo "=========================================================================="
echo ""
echo "  Left arm:  DAM safety layer enabled"
echo "  Right arm: RAW (no safety) — demonstrates unsafe behavior"
echo ""
echo "  Usage:"
echo "    --mode compare  : Side-by-side comparison (default)"
echo "    --mode dam      : Both arms with DAM"
echo "    --mode raw      : Both arms without DAM (unsafe)"
echo "    --unsafe-scale  : Aggressiveness factor (default: 1.25)"
echo ""
echo "=========================================================================="
echo ""

# Launch demo14
python scripts/14_dam_scripted_comparison_demo.py "$@"
