#!/bin/bash
# Real-time monitoring for pose iteration test with GPU parallelization

OUTPUT_BASE="/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"

echo "=========================================="
echo "POSE ITERATION TEST - LIVE MONITOR"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "POSE ITERATION TEST - LIVE MONITOR"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Count completed runs per config
    echo "Progress (completed subjects):"
    for config in baseline_40 pose_100 pose_300 pose_500; do
        count=$(find "$OUTPUT_BASE/$config" -name "meta.json" 2>/dev/null | wc -l)
        echo "  $config: $count / 3"
    done
    echo ""

    # GPU usage
    echo "=========================================="
    echo "GPU USAGE"
    echo "=========================================="
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
        awk -F', ' '{printf "  GPU %s: %3s%% util, %5s/%5s MB\n", $1, $3, $4, $5}'
    echo ""

    # Show current results if available
    if [ -f "$OUTPUT_BASE/summary.json" ]; then
        echo "=========================================="
        echo "CURRENT RESULTS"
        echo "=========================================="
        python3 -c "
import json
import sys

try:
    with open('$OUTPUT_BASE/summary.json', 'r') as f:
        data = json.load(f)
        summary = data.get('summary', {})

    if summary:
        print(f\"{'Config':<15} {'Pose Iters':<12} {'Avg MPJPE':<15} {'Improvement':<15}\")
        print('-' * 60)

        baseline_mpjpe = summary.get('baseline_40', {}).get('avg_mpjpe', 0)

        for config in ['baseline_40', 'pose_100', 'pose_300', 'pose_500']:
            if config in summary:
                s = summary[config]
                mpjpe = s.get('avg_mpjpe', 0)
                pose_iters = s.get('pose_iters', 0)

                if baseline_mpjpe > 0:
                    improvement = ((baseline_mpjpe - mpjpe) / baseline_mpjpe) * 100
                    print(f\"{config:<15} {pose_iters:<12} {mpjpe:>8.2f} mm    {improvement:>+7.2f}%\")
                else:
                    print(f\"{config:<15} {pose_iters:<12} {mpjpe:>8.2f} mm    {'N/A':>7}\")
except Exception as e:
    print(f'Error reading summary: {e}')
" 2>/dev/null
        echo ""
    fi

    # Show individual results
    echo "=========================================="
    echo "INDIVIDUAL RESULTS (Latest)"
    echo "=========================================="

    for config in baseline_40 pose_100 pose_300 pose_500; do
        if [ -d "$OUTPUT_BASE/$config" ]; then
            echo ""
            echo "[$config]"
            for meta in "$OUTPUT_BASE/$config"/*/meta.json; do
                if [ -f "$meta" ]; then
                    subject=$(basename $(dirname "$meta"))
                    python3 -c "
import json
try:
    with open('$meta', 'r') as f:
        data = json.load(f)
        metrics = data.get('metrics', {})
        mpjpe = metrics.get('MPJPE', 0)
        print(f'  {subject:<15} MPJPE: {mpjpe:>8.2f} mm')
except:
    pass
" 2>/dev/null
                fi
            done
        fi
    done

    echo ""
    echo "=========================================="
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
    echo "=========================================="

    sleep 5
done
