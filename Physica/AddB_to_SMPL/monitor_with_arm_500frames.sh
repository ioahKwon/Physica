#!/bin/bash
#
# Real-time monitoring for With_Arm 500-frame processing
# Shows progress, recent MPJPE results, and ETA
#

OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"

while true; do
    clear
    echo "================================================================================"
    echo "WITH_ARM 500-FRAME PROCESSING - LIVE MONITOR"
    echo "================================================================================"
    date
    echo ""

    # Overall progress
    TOTAL_SUBJECTS=435
    COMPLETED=$(find "$OUTPUT_DIR" -name "meta.json" 2>/dev/null | wc -l)
    PERCENT=$((COMPLETED * 100 / TOTAL_SUBJECTS))

    echo "📊 Overall Progress: $COMPLETED / $TOTAL_SUBJECTS subjects ($PERCENT%)"
    echo ""

    # Progress bar
    BAR_WIDTH=50
    FILLED=$((COMPLETED * BAR_WIDTH / TOTAL_SUBJECTS))
    EMPTY=$((BAR_WIDTH - FILLED))
    printf "["
    printf "%${FILLED}s" | tr ' ' '█'
    printf "%${EMPTY}s" | tr ' ' '░'
    printf "]\n\n"

    # GPU-wise progress
    echo "🖥️  GPU Status:"
    echo "--------------------------------------------------------------------------------"
    for GPU_ID in 0 1 2 3; do
        LOG_FILE="/tmp/gpu${GPU_ID}_with_arm_500frames.log"

        if [ -f "$LOG_FILE" ]; then
            # Get last non-empty line
            LAST_LINE=$(grep -E "Processing|✓|✗|SKIP" "$LOG_FILE" | tail -1)

            # Count GPU's completed subjects from summary
            SUMMARY_FILE="$OUTPUT_DIR/gpu${GPU_ID}_summary.json"
            if [ -f "$SUMMARY_FILE" ]; then
                GPU_DONE=$(python3 -c "import json; print(sum(1 for r in json.load(open('$SUMMARY_FILE'))['results'] if r['status']=='success'))" 2>/dev/null || echo "0")
            else
                GPU_DONE="0"
            fi

            echo "GPU $GPU_ID: $GPU_DONE done | $LAST_LINE"
        else
            echo "GPU $GPU_ID: Not started"
        fi
    done

    echo ""
    echo "📈 Recent MPJPE Results (Last 10 subjects):"
    echo "--------------------------------------------------------------------------------"

    # Find 10 most recent meta.json files and extract MPJPE
    find "$OUTPUT_DIR" -name "meta.json" -type f -printf '%T+ %p\n' 2>/dev/null | \
        sort -r | head -10 | while read timestamp filepath; do
        SUBJECT=$(basename $(dirname "$filepath"))
        MPJPE=$(python3 -c "import json,sys; m=json.load(open('$filepath')); print(f\"{m.get('metrics',{}).get('MPJPE',0):.1f}\")" 2>/dev/null || echo "N/A")
        FRAMES=$(python3 -c "import json,sys; m=json.load(open('$filepath')); o=m.get('num_frames_original',0); n=m.get('num_frames',0); print(f\"{o}→{n}\" if o!=n else str(n))" 2>/dev/null || echo "N/A")
        printf "  %-50s MPJPE=%6s mm  Frames=%s\n" "$SUBJECT" "$MPJPE" "$FRAMES"
    done

    echo ""
    echo "📊 MPJPE Statistics (Completed subjects):"
    echo "--------------------------------------------------------------------------------"

    # Calculate statistics from all completed subjects
    STATS=$(python3 << 'EOF'
import json
from pathlib import Path
import numpy as np

output_dir = Path("/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames")
mpjpe_values = []

for meta_file in output_dir.glob("*/meta.json"):
    try:
        with open(meta_file) as f:
            meta = json.load(f)
            mpjpe = meta.get('metrics', {}).get('MPJPE')
            if mpjpe is not None:
                mpjpe_values.append(mpjpe)
    except:
        pass

if mpjpe_values:
    values = np.array(mpjpe_values)
    print(f"Count: {len(values)}")
    print(f"Mean: {np.mean(values):.2f} mm")
    print(f"Median: {np.median(values):.2f} mm")
    print(f"Std: {np.std(values):.2f} mm")
    print(f"Min: {np.min(values):.2f} mm")
    print(f"Max: {np.max(values):.2f} mm")
    print(f"<50mm: {sum(values < 50)} ({sum(values < 50)*100/len(values):.1f}%)")
    print(f"50-100mm: {sum((values >= 50) & (values < 100))} ({sum((values >= 50) & (values < 100))*100/len(values):.1f}%)")
    print(f">100mm: {sum(values >= 100)} ({sum(values >= 100)*100/len(values):.1f}%)")
else:
    print("No completed subjects yet")
EOF
)
    echo "$STATS"

    echo ""
    echo "⏱️  Estimated Time:"
    echo "--------------------------------------------------------------------------------"

    # Calculate ETA if some subjects are done
    if [ $COMPLETED -gt 0 ]; then
        # Get start time from oldest log file
        OLDEST_LOG=$(ls -t /tmp/gpu*_with_arm_500frames.log 2>/dev/null | tail -1)
        if [ -f "$OLDEST_LOG" ]; then
            START_TIME=$(stat -c %Y "$OLDEST_LOG")
            CURRENT_TIME=$(date +%s)
            ELAPSED=$((CURRENT_TIME - START_TIME))

            # Calculate average time per subject
            AVG_TIME_PER_SUBJECT=$((ELAPSED / COMPLETED))
            REMAINING=$((TOTAL_SUBJECTS - COMPLETED))
            ETA_SECONDS=$((REMAINING * AVG_TIME_PER_SUBJECT))

            ELAPSED_HOURS=$((ELAPSED / 3600))
            ELAPSED_MINS=$(((ELAPSED % 3600) / 60))
            ETA_HOURS=$((ETA_SECONDS / 3600))
            ETA_MINS=$(((ETA_SECONDS % 3600) / 60))

            echo "Elapsed: ${ELAPSED_HOURS}h ${ELAPSED_MINS}m"
            echo "ETA: ${ETA_HOURS}h ${ETA_MINS}m"
            echo "Avg per subject: ${AVG_TIME_PER_SUBJECT}s"
        fi
    else
        echo "Waiting for first subject to complete..."
    fi

    echo ""
    echo "================================================================================"
    echo "Press Ctrl+C to exit | Refreshing every 5 seconds..."
    echo "================================================================================"

    sleep 5
done
