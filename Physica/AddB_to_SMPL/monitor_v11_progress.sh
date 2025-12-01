#!/bin/bash
# Real-time monitoring script for FIX 11 batch processing

OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_v11_physics"
TOTAL_SUBJECTS=435

echo "========================================================================"
echo "FIX 11 Batch Processing - Real-time Monitor"
echo "========================================================================"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

while true; do
    clear
    echo "========================================================================"
    echo "FIX 11 Batch Processing - Real-time Monitor"
    echo "========================================================================"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Count completed subjects and get their meta.json for MPJPE
    COMPLETED=$(find "$OUTPUT_DIR" -name "meta.json" -type f 2>/dev/null | wc -l)
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED/$TOTAL_SUBJECTS)*100}")

    echo "========================================================================"
    echo "Overall Progress: $COMPLETED / $TOTAL_SUBJECTS subjects ($PERCENT%)"
    echo "========================================================================"
    echo ""

    # GPU status
    for GPU_ID in 0 1 2 3; do
        BATCH_FILE="/tmp/gpu${GPU_ID}_files.txt"

        echo "=== GPU $GPU_ID ==="

        if [ -f "$BATCH_FILE" ]; then
            ASSIGNED=$(wc -l < "$BATCH_FILE" 2>/dev/null || echo 0)
            echo "  Assigned: $ASSIGNED subjects"
        else
            echo "  Assigned: N/A"
        fi

        # Find directories being processed by this GPU (have files but no meta.json yet)
        PROCESSING_DIR=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "v11_*" -exec sh -c '
            dir="$1"
            if [ ! -f "$dir/meta.json" ] && [ "$(find "$dir" -type f 2>/dev/null | wc -l)" -gt 0 ]; then
                echo "$dir"
            fi
        ' _ {} \; 2>/dev/null | head -1)

        if [ -n "$PROCESSING_DIR" ]; then
            SUBJ_NAME=$(basename "$PROCESSING_DIR" | sed 's/v11_with_physics_//')
            echo "  Current: $SUBJ_NAME"
        else
            echo "  Current: (waiting...)"
        fi

        # Count completed by finding created directories
        GPU_COMPLETED=$(find "$OUTPUT_DIR" -name "meta.json" -type f 2>/dev/null | wc -l)
        echo "  Completed: $GPU_COMPLETED total"

        # Check if GPU process is still running
        if ps aux | grep -v grep | grep "CUDA_VISIBLE_DEVICES=$GPU_ID.*addbiomechanics_to_smpl_v11" > /dev/null 2>&1; then
            echo "  Status: RUNNING"
        else
            echo "  Status: IDLE or FINISHED"
        fi

        echo ""
    done

    echo "========================================================================"
    echo "Recent completions with MPJPE (last 5):"
    echo "------------------------------------------------------------------------"
    find "$OUTPUT_DIR" -name "meta.json" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | head -5 | while IFS= read -r line; do
        timestamp=$(echo "$line" | awk '{print $1}')
        path=$(echo "$line" | awk '{print $2}')
        subject=$(basename "$(dirname "$path")" | sed 's/v11_with_physics_//')
        time=$(date -d "@$timestamp" '+%H:%M:%S' 2>/dev/null || echo "??:??:??")

        # Try to extract MPJPE from meta.json
        if [ -f "$path" ]; then
            mpjpe=$(grep -o '"MPJPE": [0-9.]*' "$path" 2>/dev/null | awk '{print $2}')
            if [ -n "$mpjpe" ]; then
                printf "  [%s] %-40s MPJPE: %.2fmm\n" "$time" "$subject" "$mpjpe"
            else
                printf "  [%s] %-40s\n" "$time" "$subject"
            fi
        fi
    done

    echo ""
    echo "========================================================================"
    echo "Press Ctrl+C to exit monitoring | Updates every 5 seconds"
    echo "========================================================================"

    # Check if all processing is done
    RUNNING_PROCS=$(ps aux | grep -v grep | grep "addbiomechanics_to_smpl_v11" | wc -l)

    if [ "$RUNNING_PROCS" -eq 0 ] && [ "$COMPLETED" -ge "$TOTAL_SUBJECTS" ]; then
        echo ""
        echo "🎉 ALL PROCESSING COMPLETE! Total: $COMPLETED subjects"
        break
    fi

    # Wait 5 seconds before refresh
    sleep 5
done
