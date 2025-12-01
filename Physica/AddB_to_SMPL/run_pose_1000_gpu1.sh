#!/bin/bash
# Run pose_1000 on GPU 1

BASE_DIR="/egr/research-zijunlab/kwonjoon"
SCRIPT="$BASE_DIR/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py"
DATA_DIR="$BASE_DIR/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm"
PYTHON="$BASE_DIR/Miscellaneous/miniforge3/envs/physpt/bin/python"
SMPL_MODEL="$BASE_DIR/Code/Physica/models/SMPL_NEUTRAL.pkl"
OUTPUT_BASE="$BASE_DIR/Output/test_pose_iters_gpu"
LOG_FILE="$OUTPUT_BASE/pose_1000.log"

# Create output directory
mkdir -p "$OUTPUT_BASE/pose_1000"

echo "Starting pose_1000 on GPU 1" >> "$LOG_FILE"

for subject in Subject7 Subject11 Subject19; do
    echo "[GPU 1] pose_1000 - $subject starting..." >> "$LOG_FILE"

    CUDA_VISIBLE_DEVICES=1 "$PYTHON" "$SCRIPT" \
        --b3d "$DATA_DIR/$subject/$subject.b3d" \
        --out_dir "$OUTPUT_BASE/pose_1000/$subject" \
        --smpl_model "$SMPL_MODEL" \
        --shape_iters 100 \
        --pose_iters 1000 \
        --max_passes 1 \
        --max_frames_optimize 500 \
        >> "$LOG_FILE" 2>&1

    echo "[GPU 1] pose_1000 - $subject completed!" >> "$LOG_FILE"
done

echo "pose_1000 on GPU 1 finished!" >> "$LOG_FILE"
