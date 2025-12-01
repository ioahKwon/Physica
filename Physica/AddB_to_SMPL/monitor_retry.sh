#!/bin/bash
#
# Real-time monitoring for retry processing on GPU 0, 1, 2
#

OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"

while true; do
    clear

    echo "================================================================================"
    echo "WITH_ARM 200-FRAME RETRY PROCESSING - LIVE MONITOR"
    echo "================================================================================"
    date
    echo ""

    # Count completed subjects
    COMPLETED=$(find "$OUTPUT_DIR" -name "meta.json" 2>/dev/null | wc -l)
    TOTAL=435
    REMAINING=$((TOTAL - COMPLETED))
    PERCENT=$((COMPLETED * 100 / TOTAL))

    echo "📊 Overall Progress: $COMPLETED / $TOTAL subjects ($PERCENT%)"
    echo ""

    # Progress bar
    BAR_WIDTH=50
    FILLED=$((PERCENT * BAR_WIDTH / 100))
    printf "["
    for ((i=0; i<BAR_WIDTH; i++)); do
        if [ $i -lt $FILLED ]; then
            printf "█"
        else
            printf "░"
        fi
    done
    printf "]\n\n"

    echo "🖥️  GPU Status (Retry):"
    echo "--------------------------------------------------------------------------------"

    for GPU_ID in 0 1 2; do
        LOG_FILE="/tmp/gpu${GPU_ID}_retry.log"
        SUBJECTS_FILE="/tmp/remaining_gpu${GPU_ID}.txt"

        if [ -f "$LOG_FILE" ]; then
            TOTAL_GPU=$(wc -l < "$SUBJECTS_FILE" 2>/dev/null || echo "?")
            CURRENT=$(grep -E "^\[GPU $GPU_ID\]" "$LOG_FILE" 2>/dev/null | tail -1 | grep -oE "\[[0-9]+/[0-9]+\]" || echo "[?/?]")
            LAST_SUBJECT=$(grep -E "^\[GPU $GPU_ID\].*Processing" "$LOG_FILE" 2>/dev/null | tail -1 | awk '{print $NF}' | tr -d '.')

            if [ -n "$LAST_SUBJECT" ]; then
                echo "GPU $GPU_ID: Processing $CURRENT | Last: $LAST_SUBJECT"
            else
                echo "GPU $GPU_ID: Starting... (Total: $TOTAL_GPU subjects)"
            fi
        else
            echo "GPU $GPU_ID: No log file"
        fi
    done

    echo ""
    echo "📈 Recent Completions (Last 10):"
    echo "--------------------------------------------------------------------------------"

    # Get last 10 completed subjects with MPJPE
    find "$OUTPUT_DIR" -name "meta.json" -type f -printf '%T@ %p\n' 2>/dev/null | \
        sort -rn | head -10 | while read timestamp path; do
        subj_dir=$(dirname "$path")
        subj_name=$(basename "$(dirname "$subj_dir")")
        mpjpe=$(grep '"MPJPE":' "$path" 2>/dev/null | awk '{print $2}' | sed 's/,//')
        num_frames=$(grep '"num_frames":' "$path" 2>/dev/null | head -1 | awk '{print $2}' | sed 's/,//')
        num_orig=$(grep '"num_frames_original":' "$path" 2>/dev/null | awk '{print $2}' | sed 's/,//')

        if [ -n "$mpjpe" ]; then
            if [ "$num_orig" != "$num_frames" ] && [ -n "$num_orig" ]; then
                printf "  %-50s MPJPE=%7.2f mm  Frames=%s→%s\n" "$subj_name" "$mpjpe" "$num_orig" "$num_frames"
            else
                printf "  %-50s MPJPE=%7.2f mm  Frames=%s\n" "$subj_name" "$mpjpe" "$num_frames"
            fi
        fi
    done

    echo ""
    echo "📊 MPJPE Statistics (All Completed):"
    echo "--------------------------------------------------------------------------------"

    # Calculate statistics
    find "$OUTPUT_DIR" -name "meta.json" -exec cat {} \; 2>/dev/null | \
        grep '"MPJPE":' | awk '{print $2}' | sed 's/,//g' | sort -n | awk '
        BEGIN {sum=0; count=0; bin1=0; bin2=0; bin3=0; bin4=0; bin5=0}
        {
            arr[NR]=$1
            sum+=$1
            count++
            if($1<30) bin1++
            else if($1<50) bin2++
            else if($1<75) bin3++
            else if($1<100) bin4++
            else bin5++
        }
        END {
            if(count>0) {
                print "Count:", count
                print "Mean:", sum/count, "mm"
                median = (count%2==1) ? arr[(count+1)/2] : (arr[count/2]+arr[count/2+1])/2
                print "Median:", median, "mm"
                print "Min:", arr[1], "mm"
                print "Max:", arr[count], "mm"
                print ""
                print "Distribution:"
                print "  <30mm:", bin1, "(" int(bin1/count*100) "%)"
                print "  30-50mm:", bin2, "(" int(bin2/count*100) "%)"
                print "  50-75mm:", bin3, "(" int(bin3/count*100) "%)"
                print "  75-100mm:", bin4, "(" int(bin4/count*100) "%)"
                print "  >100mm:", bin5, "(" int(bin5/count*100) "%)"
            } else {
                print "No completed subjects yet"
            }
        }'

    echo ""
    echo "⏱️  Estimated Completion:"
    echo "--------------------------------------------------------------------------------"
    echo "Remaining: $REMAINING subjects on 3 GPUs"
    echo "Estimated time: ~$(((REMAINING * 7) / 3)) minutes (~$((((REMAINING * 7) / 3 + 30) / 60)) hours)"
    echo ""
    echo "================================================================================>"
    echo "Refreshing in 5 seconds... (Ctrl+C to exit)"
    echo ""

    sleep 5
done
