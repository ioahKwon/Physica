#!/bin/bash
# Run additional iteration tests on GPUs 0, 1, 3

BASE_DIR="/egr/research-zijunlab/kwonjoon"
SCRIPT="$BASE_DIR/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py"
DATA_DIR="$BASE_DIR/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm"
PYTHON="$BASE_DIR/Miscellaneous/miniforge3/envs/physpt/bin/python"
SMPL_MODEL="$BASE_DIR/Code/Physica/models/SMPL_NEUTRAL.pkl"
OUTPUT_BASE="$BASE_DIR/Output/test_pose_iters_gpu"

SUBJECTS=(
    "$DATA_DIR/Subject7/Subject7.b3d"
    "$DATA_DIR/Subject11/Subject11.b3d"
    "$DATA_DIR/Subject19/Subject19.b3d"
)

echo "=========================================="
echo "ADDITIONAL ITERATION TESTS"
echo "=========================================="
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "GPU 0: pose_2000 (shape=100, pose=2000)"
echo "GPU 1: shape_200_pose_1000 (shape=200, pose=1000)"
echo "GPU 3: shape_300_pose_1000 (shape=300, pose=1000)"
echo ""

# Function to run single config
run_config() {
    local config_name=$1
    local gpu=$2
    local shape_iters=$3
    local pose_iters=$4
    local log_file="$OUTPUT_BASE/${config_name}.log"

    mkdir -p "$OUTPUT_BASE/$config_name"
    echo "Starting $config_name on GPU $gpu" >> "$log_file"

    for b3d_file in "${SUBJECTS[@]}"; do
        subject=$(basename "$b3d_file" .b3d)
        output_dir="$OUTPUT_BASE/$config_name/$subject"
        mkdir -p "$output_dir"

        echo "[GPU $gpu] $config_name - $subject starting..." >> "$log_file"

        CUDA_VISIBLE_DEVICES=$gpu "$PYTHON" "$SCRIPT" \
            --b3d "$b3d_file" \
            --out_dir "$output_dir" \
            --smpl_model "$SMPL_MODEL" \
            --shape_iters "$shape_iters" \
            --pose_iters "$pose_iters" \
            --max_passes 1 \
            --max_frames_optimize 500 \
            >> "$log_file" 2>&1

        echo "[GPU $gpu] $config_name - $subject completed!" >> "$log_file"
    done

    echo "$config_name on GPU $gpu completed!" >> "$log_file"
}

# Launch all configs in parallel
run_config "pose_2000" 0 100 2000 &
PID_0=$!

run_config "shape_200_pose_1000" 1 200 1000 &
PID_1=$!

run_config "shape_300_pose_1000" 3 300 1000 &
PID_3=$!

echo "All processes launched:"
echo "  pose_2000             (GPU 0): PID $PID_0"
echo "  shape_200_pose_1000   (GPU 1): PID $PID_1"
echo "  shape_300_pose_1000   (GPU 3): PID $PID_3"
echo ""
echo "Monitor with: ./monitor_all_tests.sh"
echo ""

# Wait for all to complete
wait $PID_0
echo "✓ pose_2000 (GPU 0) finished"

wait $PID_1
echo "✓ shape_200_pose_1000 (GPU 1) finished"

wait $PID_3
echo "✓ shape_300_pose_1000 (GPU 3) finished"

echo ""
echo "=========================================="
echo "ALL TESTS COMPLETED!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
