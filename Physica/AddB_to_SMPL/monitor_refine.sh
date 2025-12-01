#!/bin/bash
#
# Real-time monitoring for refinement processing on GPU 0, 1, 2, 3
#

while true; do
    clear
    echo "================================================================================"
    echo "🔧 HIGH-MPJPE REFINEMENT - LIVE MONITOR (4 GPUs)"
    echo "================================================================================"
    echo ""
    date
    echo ""

    # Overall progress
    TOTAL=237
    COMPLETED=$(find /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Refined -name "meta.json" 2>/dev/null | wc -l)
    PERCENT=$((COMPLETED * 100 / TOTAL))
    REMAINING=$((TOTAL - COMPLETED))

    echo "📊 Overall Progress: $COMPLETED / $TOTAL subjects ($PERCENT%) | Remaining: $REMAINING"
    echo ""

    # Progress bar (더 직관적으로)
    FILLED=$((PERCENT / 2))
    printf "["
    for i in $(seq 1 50); do
        if [ $i -le $FILLED ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "] $PERCENT%%\n\n"

    # GPU status in table format (더 직관적)
    echo "🖥️  GPU Status:"
    echo "--------------------------------------------------------------------------------"
    printf "%-8s %-12s %-10s %-15s %-30s\n" "GPU" "Status" "Assigned" "Completed" "Current Subject"
    echo "--------------------------------------------------------------------------------"

    for i in 0 1 2 3; do
        # Get assigned count
        if [ -f /tmp/gpu${i}_refine.txt ]; then
            ASSIGNED=$(wc -l < /tmp/gpu${i}_refine.txt)
        else
            ASSIGNED="N/A"
        fi

        # Check if process is running
        if pgrep -f "CUDA_VISIBLE_DEVICES=$i.*batch_refine_high_mpjpe" > /dev/null; then
            STATUS="✅ RUNNING"

            # Get current subject from log
            if [ -f /tmp/gpu${i}_refine.log ]; then
                CURRENT=$(grep -E "^\[GPU $i\].*Refining" /tmp/gpu${i}_refine.log 2>/dev/null | tail -1 | awk '{print $NF}' | sed 's/\.\.\.//' || echo "Starting...")
                # Get completed count for this GPU (rough estimate from log)
                GPU_COMPLETED=$(grep -c "✓.*refined" /tmp/gpu${i}_refine.log 2>/dev/null || echo "0")
            else
                CURRENT="No log"
                GPU_COMPLETED="0"
            fi
        else
            STATUS="❌ STOPPED"
            CURRENT="-"
            GPU_COMPLETED="0"
        fi

        printf "%-8s %-12s %-10s %-15s %-30s\n" "GPU $i" "$STATUS" "$ASSIGNED" "$GPU_COMPLETED" "$CURRENT"
    done
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Recent completions (last 10, more detailed)
    echo "📈 Recent Completions (Last 10):"
    echo "--------------------------------------------------------------------------------"
    find /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Refined -name "meta.json" -type f 2>/dev/null | \
        xargs -I {} sh -c 'echo "$(basename $(dirname $(dirname {}))): $(grep "\"MPJPE\":" {} | awk "{print \$2}" | tr -d ",") mm"' | \
        tail -10 | nl -w3 -s". "
    echo ""

    # MPJPE statistics (more detailed)
    echo "📊 MPJPE Statistics (Refined Results):"
    echo "--------------------------------------------------------------------------------"
    python3 -c "
import json
from pathlib import Path
import sys

output_dir = Path('/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Refined')
meta_files = list(output_dir.glob('*/with_arm_*/meta.json'))

if not meta_files:
    print('  No results yet')
    sys.exit(0)

mpjpes = []
for f in meta_files:
    try:
        with open(f) as fh:
            meta = json.load(fh)
        mpjpe = meta.get('metrics', {}).get('MPJPE', None)
        if mpjpe is not None:
            mpjpes.append(mpjpe)
    except:
        pass

if mpjpes:
    print(f'  Completed: {len(mpjpes)} subjects')
    print(f'  Average MPJPE: {sum(mpjpes)/len(mpjpes):.2f}mm')
    print(f'  Best: {min(mpjpes):.2f}mm')
    print(f'  Worst: {max(mpjpes):.2f}mm')
    print(f'  Median: {sorted(mpjpes)[len(mpjpes)//2]:.2f}mm')
else:
    print('  No valid results yet')
" 2>/dev/null
    echo ""

    # Estimated completion time
    if [ $COMPLETED -gt 0 ] && [ $REMAINING -gt 0 ]; then
        echo "⏱️  Estimated Time:"
        echo "--------------------------------------------------------------------------------"
        # Rough estimate: ~90s per subject on 4 GPUs
        EST_MINUTES=$(( (REMAINING * 90) / 4 / 60 ))
        EST_HOURS=$(( EST_MINUTES / 60 ))
        EST_MINS=$(( EST_MINUTES % 60 ))
        echo "  Remaining time: ~${EST_HOURS}h ${EST_MINS}m (based on ~90s/subject × 4 GPUs)"
        echo ""
    fi

    echo "================================================================================"
    echo "🔄 Refreshing every 5 seconds... (Ctrl+C to stop)"
    echo "================================================================================"

    sleep 5
done
