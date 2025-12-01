#!/bin/bash

PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_BASE="/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"

echo "=== CURRENT RESULTS ==="
echo ""

for config in baseline_40 pose_100 pose_300 pose_500 pose_1000; do
    echo "[$config]"
    total_mpjpe=0
    count=0

    for subject_dir in "$OUTPUT_BASE/$config"/Subject*; do
        if [ -d "$subject_dir" ]; then
            subject=$(basename "$subject_dir")
            meta=$(find "$subject_dir" -name "meta.json" 2>/dev/null | head -1)

            if [ -f "$meta" ]; then
                result=$("$PYTHON" -c "
import json
try:
    with open('$meta', 'r') as f:
        data = json.load(f)
        metrics = data.get('metrics', {})
        mpjpe = metrics.get('MPJPE', 0)
        print(f'{mpjpe:.2f}')
except:
    pass
" 2>/dev/null)

                if [ ! -z "$result" ]; then
                    echo "  $subject: $result mm"
                    total_mpjpe=$(echo "$total_mpjpe + $result" | bc)
                    count=$((count + 1))
                fi
            fi
        fi
    done

    if [ $count -gt 0 ]; then
        avg=$(echo "scale=2; $total_mpjpe / $count" | bc)
        echo "  Average ($count/3): $avg mm"
    else
        echo "  No results yet (0/3)"
    fi
    echo ""
done

# Summary comparison
echo "=== SUMMARY COMPARISON ==="
"$PYTHON" << 'EOF'
import json
import os
from pathlib import Path

OUTPUT_BASE = "/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"
configs = {
    "baseline_40": 40,
    "pose_100": 100,
    "pose_300": 300,
    "pose_500": 500,
    "pose_1000": 1000,
}

results = {}
for config_name, pose_iters in configs.items():
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
            "avg_mpjpe": sum(mpjpes) / len(mpjpes),
            "count": len(mpjpes),
        }

if results:
    print(f"\n{'Config':<15} {'Pose Iters':<12} {'Count':<8} {'Avg MPJPE':<15} {'Improvement'}")
    print("-" * 75)

    baseline_mpjpe = results.get("baseline_40", {}).get("avg_mpjpe", 0)

    for config_name in ["baseline_40", "pose_100", "pose_300", "pose_500", "pose_1000"]:
        if config_name in results:
            r = results[config_name]
            pose_iters = r["pose_iters"]
            mpjpe = r["avg_mpjpe"]
            count = r["count"]

            if baseline_mpjpe > 0:
                improvement = ((baseline_mpjpe - mpjpe) / baseline_mpjpe) * 100
                print(f"{config_name:<15} {pose_iters:<12} {count}/3      {mpjpe:>8.2f} mm    {improvement:>+7.2f}%")
            else:
                print(f"{config_name:<15} {pose_iters:<12} {count}/3      {mpjpe:>8.2f} mm")
else:
    print("No results available yet")
EOF
