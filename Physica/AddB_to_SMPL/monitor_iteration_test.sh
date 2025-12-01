#!/bin/bash
# Real-time monitoring for iteration increase test

OUTPUT_BASE="/egr/research-zijunlab/kwonjoon/Output/test_iteration_increase"

echo "=========================================="
echo "ITERATION INCREASE TEST - LIVE MONITOR"
echo "=========================================="
echo ""

while true; do
    clear
    echo "=========================================="
    echo "ITERATION INCREASE TEST - LIVE MONITOR"
    echo "$(date '+%Y-%m-%d %H:%M:%S')"
    echo "=========================================="
    echo ""

    # Count completed runs
    baseline_count=$(find "$OUTPUT_BASE/baseline" -name "meta.json" 2>/dev/null | wc -l)
    increased_count=$(find "$OUTPUT_BASE/increased" -name "meta.json" 2>/dev/null | wc -l)

    echo "Progress:"
    echo "  Baseline runs:  $baseline_count / 3"
    echo "  Increased runs: $increased_count / 3"
    echo ""

    # Show latest results
    if [ -f "$OUTPUT_BASE/summary.json" ]; then
        echo "=========================================="
        echo "CURRENT RESULTS"
        echo "=========================================="
        python3 -c "
import json
with open('$OUTPUT_BASE/summary.json', 'r') as f:
    data = json.load(f)
    summary = data.get('summary', {})
    print(f\"  Baseline MPJPE:  {summary.get('avg_baseline_mpjpe', 0):.2f} mm\")
    print(f\"  Increased MPJPE: {summary.get('avg_increased_mpjpe', 0):.2f} mm\")
    print(f\"  Improvement:     {summary.get('avg_improvement_pct', 0):+.2f}%\")
    print(f\"  Time increase:   {summary.get('avg_time_increase_sec', 0):+.1f}s\")
" 2>/dev/null
        echo ""
    fi

    # Show individual results
    echo "=========================================="
    echo "INDIVIDUAL RESULTS"
    echo "=========================================="

    for config in baseline increased; do
        echo ""
        echo "[$config]"
        for meta in "$OUTPUT_BASE/$config"/*/meta.json; do
            if [ -f "$meta" ]; then
                subject=$(basename $(dirname "$meta"))
                python3 -c "
import json
with open('$meta', 'r') as f:
    data = json.load(f)
    metrics = data.get('metrics', {})
    mpjpe = metrics.get('MPJPE', 0)
    print(f\"  {subject:<15} MPJPE: {mpjpe:>8.2f} mm\")
" 2>/dev/null
            fi
        done
    done

    echo ""
    echo "=========================================="
    echo "Refreshing in 5 seconds... (Ctrl+C to stop)"
    echo "=========================================="

    sleep 5
done
