#!/bin/bash
#
# Real-time monitoring for ultra refinement on 4 GPUs
#

OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Ultra_Refined"

while true; do
    clear
    echo "================================================================================"
    echo "🚀 ULTRA REFINEMENT - LIVE MONITOR (4 GPUs)"
    echo "================================================================================"
    echo ""
    date
    echo ""

    # Overall progress
    if [ -f /tmp/ultra_refine_subjects.txt ]; then
        TOTAL=$(wc -l < /tmp/ultra_refine_subjects.txt)
    else
        TOTAL=0
    fi

    COMPLETED=$(find $OUTPUT_DIR -name "meta.json" 2>/dev/null | wc -l)

    if [ $TOTAL -gt 0 ]; then
        PERCENT=$((COMPLETED * 100 / TOTAL))
        REMAINING=$((TOTAL - COMPLETED))
    else
        PERCENT=0
        REMAINING=0
    fi

    echo "📊 Overall Progress: $COMPLETED / $TOTAL subjects ($PERCENT%) | Remaining: $REMAINING"
    echo ""

    # Progress bar
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

    # GPU status
    echo "🖥️  GPU Status:"
    echo "--------------------------------------------------------------------------------"
    printf "%-8s %-12s %-10s %-30s\n" "GPU" "Status" "Assigned" "Current Subject"
    echo "--------------------------------------------------------------------------------"

    for i in 0 1 2 3; do
        # Get assigned count
        if [ -f /tmp/ultra_gpu${i}.txt ]; then
            ASSIGNED=$(wc -l < /tmp/ultra_gpu${i}.txt)
        else
            ASSIGNED="N/A"
        fi

        # Check if process is running
        if pgrep -f "CUDA_VISIBLE_DEVICES=$i.*batch_ultra_refine" > /dev/null; then
            STATUS="✅ RUNNING"

            # Get current subject from log
            if [ -f /tmp/ultra_gpu${i}.log ]; then
                CURRENT=$(grep -E "^\[GPU $i\].*Processing:" /tmp/ultra_gpu${i}.log 2>/dev/null | tail -1 | awk '{print $NF}' || echo "Starting...")
            else
                CURRENT="No log"
            fi
        else
            STATUS="❌ STOPPED"
            CURRENT="-"
        fi

        printf "%-8s %-12s %-10s %-30s\n" "GPU $i" "$STATUS" "$ASSIGNED" "$CURRENT"
    done
    echo "--------------------------------------------------------------------------------"
    echo ""

    # Recent completions (last 10)
    echo "📈 Recent Completions (Last 10):"
    echo "--------------------------------------------------------------------------------"
    if [ -d "$OUTPUT_DIR" ]; then
        find $OUTPUT_DIR -name "meta.json" -type f 2>/dev/null | \
            xargs -I {} sh -c 'meta={}; dir=$(dirname $(dirname $meta)); name=$(basename $dir); prev=$(grep "\"previous_MPJPE\":" $meta | awk "{print \$2}" | tr -d ","); final=$(grep "\"MPJPE\":" $meta | head -1 | awk "{print \$2}" | tr -d ","); impr=$(grep "\"improvement_mm\":" $meta | awk "{print \$2}" | tr -d ","); echo "$name: $prev → $final mm ($impr mm)"' | \
            tail -10 | nl -w3 -s". "
    fi
    echo ""

    # MPJPE statistics
    echo "📊 MPJPE Statistics (Ultra-Refined Results):"
    echo "--------------------------------------------------------------------------------"
    python3 << EOF
import json
from pathlib import Path

output_dir = Path('$OUTPUT_DIR')
meta_files = list(output_dir.glob('*/with_arm_*/meta.json'))

if not meta_files:
    print("  No completed subjects yet")
else:
    prev_mpjpes = []
    final_mpjpes = []
    improvements = []

    for meta_file in meta_files:
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            prev = meta.get('metrics', {}).get('previous_MPJPE', None)
            final = meta.get('metrics', {}).get('MPJPE', None)
            impr = meta.get('metrics', {}).get('improvement_mm', None)

            if prev is not None and final is not None:
                prev_mpjpes.append(prev)
                final_mpjpes.append(final)
                if impr is not None:
                    improvements.append(impr)
        except:
            pass

    if prev_mpjpes:
        print(f'  Completed: {len(prev_mpjpes)} subjects')
        print(f'  Average previous MPJPE: {sum(prev_mpjpes)/len(prev_mpjpes):.2f}mm')
        print(f'  Average final MPJPE: {sum(final_mpjpes)/len(final_mpjpes):.2f}mm')
        if improvements:
            print(f'  Average improvement: {sum(improvements)/len(improvements):.2f}mm')
            print(f'  Best improvement: {max(improvements):.2f}mm')
            print(f'  Worst improvement: {min(improvements):.2f}mm')
        print(f'  Below 50mm: {sum(1 for x in final_mpjpes if x < 50)} subjects ({sum(1 for x in final_mpjpes if x < 50)/len(final_mpjpes)*100:.1f}%)')
        print(f'  Below 100mm: {sum(1 for x in final_mpjpes if x < 100)} subjects ({sum(1 for x in final_mpjpes if x < 100)/len(final_mpjpes)*100:.1f}%)')
EOF
    echo ""

    # Estimated completion time
    if [ $COMPLETED -gt 0 ] && [ $REMAINING -gt 0 ]; then
        echo "⏱️  Estimated Time:"
        echo "--------------------------------------------------------------------------------"
        # Estimate: ~180s per subject on 4 GPUs (ultra refinement is slower)
        EST_MINUTES=$(( (REMAINING * 180) / 4 / 60 ))
        EST_HOURS=$(( EST_MINUTES / 60 ))
        EST_MINS=$(( EST_MINUTES % 60 ))
        echo "  Remaining time: ~${EST_HOURS}h ${EST_MINS}m (based on ~180s/subject × 4 GPUs)"
        echo ""
    fi

    echo "================================================================================"
    echo "🔄 Refreshing every 5 seconds... (Ctrl+C to stop)"
    echo "================================================================================"

    sleep 5
done
