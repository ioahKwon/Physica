#!/bin/bash

#######################################
# Test Foot Orientation Optimization on P020_split2
#
# Compares:
# - v3_enhanced: MPJPE 74.62mm (position weighting only)
# - v3_foot_orientation: MPJPE = ? (position + orientation optimization)
#
# Expected improvements:
# - Foot orientation correct (no "pointing backward")
# - MPJPE maintains or improves
#######################################

set -e

PYTHON=/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python
SCRIPT_DIR=/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

# Test subject
B3D_FILE=/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Carter2023_Formatted_With_Arm/P020_split2/P020_split2.b3d
OUT_DIR=/tmp/test_foot_orientation_p020_split2
SMPL_MODEL=models/smpl_model.pkl
NUM_FRAMES=200
GPU_ID=3

echo "=============================================="
echo "Testing Foot Orientation Optimization"
echo "=============================================="
echo "Subject: P020_split2"
echo "Previous MPJPE (v3_enhanced): 74.62mm"
echo "Testing: v3_foot_orientation"
echo "GPU: $GPU_ID"
echo "=============================================="
echo ""

# Clean previous results
rm -rf "$OUT_DIR"

# Run optimization
echo "Running foot orientation optimization..."
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON $SCRIPT_DIR/addbiomechanics_to_smpl_v3_foot_orientation.py \
    --b3d "$B3D_FILE" \
    --out_dir "$OUT_DIR" \
    --smpl_model "$SMPL_MODEL" \
    --num_frames $NUM_FRAMES \
    --device cuda

echo ""
echo "=============================================="
echo "Test Complete!"
echo "=============================================="

# Show results
META_FILE="$OUT_DIR/foot_orient_carter2023_p020/meta.json"
if [ -f "$META_FILE" ]; then
    echo "Results:"
    $PYTHON << EOF
import json
with open('$META_FILE') as f:
    meta = json.load(f)
mpjpe = meta['metrics']['MPJPE']
print(f"  NEW MPJPE: {mpjpe:.2f} mm")
print(f"  Previous MPJPE: 74.62 mm")
improvement = 74.62 - mpjpe
print(f"  Improvement: {improvement:.2f} mm ({improvement/74.62*100:.1f}%)")
EOF
    echo ""
    echo "Full results saved to:"
    echo "  $OUT_DIR/foot_orient_carter2023_p020/"
else
    echo "[ERROR] meta.json not found!"
    exit 1
fi

echo ""
echo "Next steps:"
echo "1. Generate visualization video to check foot orientation"
echo "2. If successful, run on all subjects with run_foot_orientation_batch.sh"
