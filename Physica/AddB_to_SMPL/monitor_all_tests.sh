#!/bin/bash
# Real-time monitoring for all iteration tests

PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_BASE="/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"

while true; do
    clear
    echo "=========================================="
    echo "ITERATION TESTS - LIVE MONITOR"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # GPU Status
    echo "GPU STATUS:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | \
        awk -F', ' '{printf "  GPU %s: %3s%% util, %5s/%5s MB\n", $1, $3, $4, $5}'
    echo ""

    # Progress Summary
    echo "=========================================="
    echo "PROGRESS SUMMARY"
    echo "=========================================="

    all_configs=(
        "baseline_40:40:100"
        "pose_100:100:100"
        "pose_300:300:100"
        "pose_500:500:100"
        "pose_1000:1000:100"
        "pose_2000:2000:100"
        "shape_200_pose_1000:1000:200"
        "shape_300_pose_1000:1000:300"
    )

    for config_info in "${all_configs[@]}"; do
        IFS=':' read -r config_name pose_iters shape_iters <<< "$config_info"

        if [ -d "$OUTPUT_BASE/$config_name" ]; then
            count=$(find "$OUTPUT_BASE/$config_name" -name "meta.json" 2>/dev/null | wc -l)
            printf "  %-25s (s=%3d,p=%4d): %d/3 " "$config_name" "$shape_iters" "$pose_iters" "$count"

            if [ $count -eq 3 ]; then
                echo "✓ COMPLETE"
            elif [ $count -gt 0 ]; then
                echo "⏳ IN PROGRESS"
            else
                echo "⌛ WAITING"
            fi
        fi
    done
    echo ""

    # Current Results Table
    echo "=========================================="
    echo "CURRENT RESULTS"
    echo "=========================================="

    "$PYTHON" << 'PYEOF'
import json
import os
from pathlib import Path

OUTPUT_BASE = "/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"

configs = [
    ("baseline_40", 40, 100),
    ("pose_100", 100, 100),
    ("pose_300", 300, 100),
    ("pose_500", 500, 100),
    ("pose_1000", 1000, 100),
    ("pose_2000", 2000, 100),
    ("shape_200_pose_1000", 1000, 200),
    ("shape_300_pose_1000", 1000, 300),
]

results = {}
for config_name, pose_iters, shape_iters in configs:
    config_dir = Path(OUTPUT_BASE) / config_name
    if not config_dir.exists():
        continue

    mpjpes = []
    for subject_dir in config_dir.glob("Subject*"):
        meta_files = list(subject_dir.rglob("meta.json"))
        if meta_files:
            try:
                with open(meta_files[0]) as f:
                    data = json.load(f)
                    mpjpe = data.get("metrics", {}).get("MPJPE")
                    if mpjpe is not None:
                        mpjpes.append(mpjpe)
            except:
                pass

    if mpjpes:
        results[config_name] = {
            "pose_iters": pose_iters,
            "shape_iters": shape_iters,
            "avg_mpjpe": sum(mpjpes) / len(mpjpes),
            "count": len(mpjpes),
        }

if results:
    print(f"{'Config':<25} {'Shape':<7} {'Pose':<7} {'Count':<7} {'Avg MPJPE':<12} {'Improvement'}")
    print("-" * 85)

    baseline_mpjpe = results.get("baseline_40", {}).get("avg_mpjpe", 0)

    for config_name, pose_iters, shape_iters in configs:
        if config_name in results:
            r = results[config_name]
            mpjpe = r["avg_mpjpe"]
            count = r["count"]

            if baseline_mpjpe > 0:
                improvement = ((baseline_mpjpe - mpjpe) / baseline_mpjpe) * 100
                status = "✓" if count == 3 else "⏳"
                print(f"{config_name:<25} {shape_iters:<7} {pose_iters:<7} {count}/3     {mpjpe:>7.2f} mm   {improvement:>+6.2f}% {status}")
            else:
                print(f"{config_name:<25} {shape_iters:<7} {pose_iters:<7} {count}/3     {mpjpe:>7.2f} mm")
else:
    print("No results available yet")
PYEOF

    echo ""
    echo "=========================================="

    # Running processes
    running=$(ps aux | grep "addbiomechanics_to_smpl" | grep -v grep | wc -l)
    echo "Active processes: $running"

    echo ""
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
    echo "=========================================="

    sleep 5
done
