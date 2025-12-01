#!/bin/bash

# Run all 4 v5 versions on P020 with 200 frames for comparison
# Each version addresses the upper body lean issue differently

PYTHON=/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python
B3D_FILE=/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Carter2023_Formatted_With_Arm/P020_split0/P020_split0.b3d
SMPL_MODEL=/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl
NUM_FRAMES=200
DEVICE=cpu

BASE_OUT_DIR=/tmp/v5_comparison

echo "=========================================="
echo "FIX 5 Comparison: 4 Different Approaches"
echo "=========================================="
echo "Subject: P020 (Carter2023)"
echo "Frames: $NUM_FRAMES"
echo "Device: $DEVICE"
echo ""

# Create base output directory
mkdir -p $BASE_OUT_DIR

# Option 1: Pelvis Upright Constraint + Foot Smoothing
echo "[1/4] Running Option 1: Pelvis Upright Constraint..."
OUT_DIR_1=${BASE_OUT_DIR}/v5_option1_pelvis_constraint
CUDA_VISIBLE_DEVICES="" $PYTHON addbiomechanics_to_smpl_v5_pelvis_constraint.py \
    --b3d $B3D_FILE \
    --smpl_model $SMPL_MODEL \
    --out_dir $OUT_DIR_1 \
    --num_frames $NUM_FRAMES \
    --device $DEVICE \
    > ${OUT_DIR_1}_log.txt 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Option 1 complete"
    grep "MPJPE" ${OUT_DIR_1}_log.txt | tail -1
else
    echo "  ✗ Option 1 failed"
fi
echo ""

# Option 2: Spine2 Mapping + Spine Vertical Loss + Foot Smoothing
echo "[2/4] Running Option 2: Spine2 Mapping + Vertical Loss..."
OUT_DIR_2=${BASE_OUT_DIR}/v5_option2_spine_mapping
CUDA_VISIBLE_DEVICES="" $PYTHON addbiomechanics_to_smpl_v5_spine_mapping.py \
    --b3d $B3D_FILE \
    --smpl_model $SMPL_MODEL \
    --out_dir $OUT_DIR_2 \
    --num_frames $NUM_FRAMES \
    --device $DEVICE \
    > ${OUT_DIR_2}_log.txt 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Option 2 complete"
    grep "MPJPE" ${OUT_DIR_2}_log.txt | tail -1
else
    echo "  ✗ Option 2 failed"
fi
echo ""

# Option 3: 180° Coordinate Rotation + Foot Smoothing
echo "[3/4] Running Option 3: 180° Coordinate Rotation..."
OUT_DIR_3=${BASE_OUT_DIR}/v5_option3_coord_fix
CUDA_VISIBLE_DEVICES="" $PYTHON addbiomechanics_to_smpl_v5_coord_fix.py \
    --b3d $B3D_FILE \
    --smpl_model $SMPL_MODEL \
    --out_dir $OUT_DIR_3 \
    --num_frames $NUM_FRAMES \
    --device $DEVICE \
    > ${OUT_DIR_3}_log.txt 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Option 3 complete"
    grep "MPJPE" ${OUT_DIR_3}_log.txt | tail -1
else
    echo "  ✗ Option 3 failed"
fi
echo ""

# Option 4: ALL FIXES Combined
echo "[4/4] Running Option 4: ALL FIXES (Coord + Spine + Pelvis)..."
OUT_DIR_4=${BASE_OUT_DIR}/v5_option4_all_fixes
CUDA_VISIBLE_DEVICES="" $PYTHON addbiomechanics_to_smpl_v5_all_fixes.py \
    --b3d $B3D_FILE \
    --smpl_model $SMPL_MODEL \
    --out_dir $OUT_DIR_4 \
    --num_frames $NUM_FRAMES \
    --device $DEVICE \
    > ${OUT_DIR_4}_log.txt 2>&1

if [ $? -eq 0 ]; then
    echo "  ✓ Option 4 complete"
    grep "MPJPE" ${OUT_DIR_4}_log.txt | tail -1
else
    echo "  ✗ Option 4 failed"
fi
echo ""

echo "=========================================="
echo "All runs complete!"
echo "=========================================="
echo ""
echo "Results saved to: $BASE_OUT_DIR"
echo ""
echo "To compare results:"
echo "  - Option 1: ${OUT_DIR_1}"
echo "  - Option 2: ${OUT_DIR_2}"
echo "  - Option 3: ${OUT_DIR_3}"
echo "  - Option 4: ${OUT_DIR_4}"
echo ""
echo "Logs:"
ls -lh ${BASE_OUT_DIR}/*_log.txt
