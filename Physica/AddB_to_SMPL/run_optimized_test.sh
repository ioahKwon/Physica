#!/bin/bash

# Run optimized test with shape_iters=300, pose_iters=500, max_passes=1 on GPU 1
# This configuration should reduce jittering with increased temporal smoothness weights

PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
SCRIPT_PATH="/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py"
SMPL_MODEL="/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
OUTPUT_BASE="/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"

CONFIG_NAME="shape_300_pose_500_optimized"
GPU_ID=1
SHAPE_ITERS=300
POSE_ITERS=500
MAX_PASSES=1

# Test subjects
declare -a SUBJECTS=(
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject7/Subject7.b3d"
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d"
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject19/Subject19.b3d"
)

echo "=========================================="
echo "LAUNCHING OPTIMIZED TEST"
echo "=========================================="
echo "Config: $CONFIG_NAME"
echo "GPU: $GPU_ID"
echo "Shape Iters: $SHAPE_ITERS"
echo "Pose Iters: $POSE_ITERS"
echo "Max Passes: $MAX_PASSES"
echo "Output: $OUTPUT_BASE/$CONFIG_NAME"
echo ""
echo "Fixes Applied:"
echo "  1. Coordinate conversion: AddB (X,Y,Z) → SMPL (Z,Y,X)"
echo "  2. temporal_smooth_weight: 0.1 → 0.5 (Line 594, 2211)"
echo "  3. velocity_smooth_weight: 0.15 → 0.5 (Line 613)"
echo "=========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE/$CONFIG_NAME"

# Log file
LOG_FILE="$OUTPUT_BASE/${CONFIG_NAME}_gpu${GPU_ID}.log"

# Run all subjects
export CUDA_VISIBLE_DEVICES=$GPU_ID

{
    echo "=========================================="
    echo "RUNNING: $CONFIG_NAME (GPU $GPU_ID)"
    echo "=========================================="
    echo "Config: shape_iters=$SHAPE_ITERS, pose_iters=$POSE_ITERS, max_passes=$MAX_PASSES"
    echo "Output: $OUTPUT_BASE/$CONFIG_NAME"
    echo "=========================================="
    echo ""

    for B3D_PATH in "${SUBJECTS[@]}"; do
        SUBJECT_NAME=$(basename "$B3D_PATH" .b3d)
        OUTPUT_DIR="$OUTPUT_BASE/$CONFIG_NAME/$SUBJECT_NAME"

        echo "=========================================="
        echo "Processing: $SUBJECT_NAME"
        echo "=========================================="
        echo ""

        mkdir -p "$OUTPUT_DIR"

        $PYTHON "$SCRIPT_PATH" \
            --b3d "$B3D_PATH" \
            --out_dir "$OUTPUT_DIR" \
            --smpl_model "$SMPL_MODEL" \
            --shape_iters "$SHAPE_ITERS" \
            --pose_iters "$POSE_ITERS" \
            --max_passes "$MAX_PASSES" \
            --num_frames 500

        # Check if meta.json exists and extract MPJPE
        META_FILE="$OUTPUT_DIR/meta.json"
        if [ -f "$META_FILE" ]; then
            MPJPE=$($PYTHON -c "import json; data=json.load(open('$META_FILE')); print(f\"{data['metrics']['MPJPE']:.2f}\")" 2>/dev/null || echo "N/A")
            echo ""
            echo "✓ $SUBJECT_NAME COMPLETE: MPJPE = ${MPJPE} mm"
        else
            echo ""
            echo "✗ $SUBJECT_NAME FAILED: No result"
        fi
        echo ""
    done

    echo "=========================================="
    echo "COMPLETED: $CONFIG_NAME"
    echo "=========================================="
    echo ""
} > "$LOG_FILE" 2>&1 &

PID=$!
echo "Started background process: PID=$PID"
echo "Log file: $LOG_FILE"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check results:"
echo "  bash /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/check_results.sh $CONFIG_NAME"
