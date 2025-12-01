#!/bin/bash
#
# PILOT TEST: Re-fit worst 4 subjects from scratch with 2× iterations
# NO warm-start, fresh random initialization
# Strategy: 2× iterations for stronger optimization
#

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

echo "================================================================================"
echo "PILOT TEST - Fresh Fitting with 2× Iterations"
echo "================================================================================"
echo ""
echo "Strategy: Re-fit from scratch (NO warm-start)"
echo "  - shape_iters: 100 → 200 (2×)"
echo "  - pose_iters: 40 → 80 (2×)"
echo "  - Fresh random initialization for better local minimum"
echo ""
echo "Testing on 4 worst subjects:"
echo "  1. Subject12 (180.42mm) → GPU 0"
echo "  2. Subject46 (174.90mm) → GPU 1"
echo "  3. Subject47 (131.89mm) → GPU 2"
echo "  4. s002_split0 (91.58mm) → GPU 3"
echo ""
echo "Output: /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/Pilot_2xIters"
echo "================================================================================"
echo ""

PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
SCRIPT="addbiomechanics_to_smpl_v3_enhanced.py"
SMPL_MODEL="models/smpl_model.pkl"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/Pilot_2xIters"

mkdir -p "$OUTPUT_DIR"

# Worst 4 subjects with their B3D paths
declare -a B3D_PATHS=(
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject12/Subject12.b3d"
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject46/Subject46.b3d"
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject47/Subject47.b3d"
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Han2023_Formatted_With_Arm/s002_split0/s002_split0.b3d"
)

declare -a OLD_MPJPES=(180.42 174.90 131.89 91.58)
declare -a SUBJECT_NAMES=("Subject12" "Subject46" "Subject47" "s002_split0")

# Launch on 4 GPUs in parallel (NO warm-start, 2× iterations)
for GPU_ID in 0 1 2 3; do
    B3D_PATH="${B3D_PATHS[$GPU_ID]}"
    OLD_MPJPE="${OLD_MPJPES[$GPU_ID]}"
    SUBJECT_NAME="${SUBJECT_NAMES[$GPU_ID]}"
    LOG_FILE="/tmp/pilot_gpu${GPU_ID}_2xiters.log"

    echo "Launching GPU $GPU_ID: $SUBJECT_NAME (old: ${OLD_MPJPE}mm)..."

    # Run with 2× iterations (NO --init_from, fresh fitting)
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u $SCRIPT \
        --b3d \"$B3D_PATH\" \
        --smpl_model $SMPL_MODEL \
        --out_dir $OUTPUT_DIR \
        --num_frames -1 \
        --max_frames_optimize 200 \
        --include_head_neck \
        --device cuda \
        --shape_iters 200 \
        --pose_iters 80 \
        > $LOG_FILE 2>&1" &

    PID=$!
    echo "  ✓ GPU $GPU_ID launched (PID: $PID, Log: $LOG_FILE)"
done

echo ""
echo "================================================================================"
echo "ALL 4 GPUS LAUNCHED - PILOT TEST RUNNING"
echo "================================================================================"
echo ""
echo "Monitor progress:"
echo "  tail -f /tmp/pilot_gpu{0,1,2,3}_2xiters.log"
echo ""
echo "Check if running:"
echo "  ps aux | grep addbiomechanics_to_smpl_v3_enhanced"
echo ""
echo "Expected time: ~10-15 minutes per subject (2× longer due to 2× iterations)"
echo ""
echo "================================================================================"
