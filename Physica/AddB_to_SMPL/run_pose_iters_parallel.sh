#!/bin/bash
# Run pose iteration tests in parallel on different GPUs

BASE_DIR="/egr/research-zijunlab/kwonjoon"
SCRIPT="$BASE_DIR/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py"
DATA_DIR="$BASE_DIR/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm"
OUTPUT_BASE="$BASE_DIR/Output/test_pose_iters_gpu"
PYTHON="$BASE_DIR/Miscellaneous/miniforge3/envs/physpt/bin/python"
SMPL_MODEL="$BASE_DIR/Code/Physica/models/SMPL_NEUTRAL.pkl"

# Test subjects
SUBJECTS=(
    "$DATA_DIR/Subject7/Subject7.b3d"
    "$DATA_DIR/Subject11/Subject11.b3d"
    "$DATA_DIR/Subject19/Subject19.b3d"
)

# Create output directories
mkdir -p "$OUTPUT_BASE"/{baseline_40,pose_100,pose_300,pose_500}

echo "========================================"
echo "POSE ITERATION TEST - PARALLEL EXECUTION"
echo "========================================"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Function to run single config
run_config() {
    local config_name=$1
    local gpu=$2
    local shape_iters=$3
    local pose_iters=$4
    local log_file="$OUTPUT_BASE/${config_name}.log"

    echo "Starting $config_name on GPU $gpu (shape=$shape_iters, pose=$pose_iters)"

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

    echo "$config_name on GPU $gpu completed!"
}

# Launch all configs in parallel
run_config "baseline_40" 0 100 40 &
PID_0=$!

run_config "pose_100" 1 100 100 &
PID_1=$!

run_config "pose_300" 2 100 300 &
PID_2=$!

run_config "pose_500" 3 100 500 &
PID_3=$!

echo ""
echo "All processes launched:"
echo "  baseline_40 (GPU 0): PID $PID_0"
echo "  pose_100    (GPU 1): PID $PID_1"
echo "  pose_300    (GPU 2): PID $PID_2"
echo "  pose_500    (GPU 3): PID $PID_3"
echo ""
echo "Monitor with: ./monitor_pose_iters.sh"
echo "Or check logs: tail -f $OUTPUT_BASE/*.log"
echo ""

# Wait for all to complete
wait $PID_0
echo "✓ baseline_40 (GPU 0) finished"

wait $PID_1
echo "✓ pose_100 (GPU 1) finished"

wait $PID_2
echo "✓ pose_300 (GPU 2) finished"

wait $PID_3
echo "✓ pose_500 (GPU 3) finished"

echo ""
echo "========================================"
echo "ALL TESTS COMPLETED!"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Generate summary
python3 << 'EOF'
import json
import os
from pathlib import Path

OUTPUT_BASE = "/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"
configs = {
    "baseline_40": {"shape_iters": 100, "pose_iters": 40, "gpu": 0},
    "pose_100": {"shape_iters": 100, "pose_iters": 100, "gpu": 1},
    "pose_300": {"shape_iters": 100, "pose_iters": 300, "gpu": 2},
    "pose_500": {"shape_iters": 100, "pose_iters": 500, "gpu": 3},
}

results = {}
for config_name, config in configs.items():
    config_results = []
    config_dir = Path(OUTPUT_BASE) / config_name

    for subject_dir in config_dir.iterdir():
        if subject_dir.is_dir():
            meta_file = subject_dir / "meta.json"
            if meta_file.exists():
                with open(meta_file) as f:
                    meta = json.load(f)
                    metrics = meta.get("metrics", {})
                    config_results.append({
                        "subject": subject_dir.name,
                        "mpjpe": metrics.get("MPJPE"),
                        "pck_5cm": metrics.get("PCK@0.05m"),
                        "pck_10cm": metrics.get("PCK@0.10m"),
                    })

    if config_results:
        avg_mpjpe = sum(r["mpjpe"] for r in config_results if r["mpjpe"]) / len([r for r in config_results if r["mpjpe"]])
        results[config_name] = {
            "config": config,
            "avg_mpjpe": avg_mpjpe,
            "results": config_results,
        }

# Save summary
with open(f"{OUTPUT_BASE}/summary.json", "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("\n========================================")
print("SUMMARY")
print("========================================\n")

if "baseline_40" in results:
    baseline_mpjpe = results["baseline_40"]["avg_mpjpe"]

    print(f"{'Config':<15} {'Pose Iters':<12} {'Avg MPJPE':<15} {'Improvement'}")
    print("-" * 60)

    for config_name in ["baseline_40", "pose_100", "pose_300", "pose_500"]:
        if config_name in results:
            r = results[config_name]
            pose_iters = r["config"]["pose_iters"]
            mpjpe = r["avg_mpjpe"]
            improvement = ((baseline_mpjpe - mpjpe) / baseline_mpjpe) * 100

            print(f"{config_name:<15} {pose_iters:<12} {mpjpe:>8.2f} mm    {improvement:>+7.2f}%")

print(f"\nDetailed results saved to: {OUTPUT_BASE}/summary.json")
EOF
