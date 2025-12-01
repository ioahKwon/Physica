#!/bin/bash
# Monitor full body optimization progress across all 4 GPUs

echo "====================================================================="
echo "  With_Arm Full Body Optimization - Multi-GPU Monitor"
echo "====================================================================="
echo ""

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

while true; do
    clear
    echo "====================================================================="
    echo "  With_Arm Full Body Optimization - Progress Monitor"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "====================================================================="
    echo ""

    # Check if processes are running
    echo "Process Status:"
    for gpu_id in 0 1 2 3; do
        if pgrep -f "gpu_id ${gpu_id}.*batch_process_v3_fullbody_parallel.py" > /dev/null; then
            echo "  GPU ${gpu_id}: ✓ RUNNING"
        else
            echo "  GPU ${gpu_id}: ✗ NOT RUNNING"
        fi
    done
    echo ""

    # Show progress for each GPU
    for gpu_id in 0 1 2 3; do
        log_file="gpu${gpu_id}_fullbody.log"

        if [ ! -f "$log_file" ]; then
            echo "GPU ${gpu_id}: No log file yet"
            echo ""
            continue
        fi

        # Get assignment info
        assignment=$(grep "assigned" "$log_file" 2>/dev/null | tail -n 1)
        if [ -n "$assignment" ]; then
            echo "$assignment"
        else
            echo "GPU ${gpu_id}: Still initializing..."
        fi

        # Get latest progress line
        latest_progress=$(grep "^\[GPU${gpu_id}\]" "$log_file" 2>/dev/null | tail -n 1)
        if [ -n "$latest_progress" ]; then
            echo "$latest_progress"
        fi

        # Get latest stats line (with progress bar)
        latest_stats=$(grep -A 1 "^\[GPU${gpu_id}\]" "$log_file" 2>/dev/null | tail -n 1 | grep "█")
        if [ -n "$latest_stats" ]; then
            echo "$latest_stats"
        fi

        echo ""
    done

    echo "---------------------------------------------------------------------"

    # Count total completed subjects across all GPUs
    total_success=0
    total_failed=0
    total_skipped=0

    for gpu_id in 0 1 2 3; do
        log_file="gpu${gpu_id}_fullbody.log"
        if [ -f "$log_file" ]; then
            # Count checkmarks, X marks, and circles in the latest stats
            stats=$(grep -E "✓[0-9]+ ✗[0-9]+ ⊙[0-9]+" "$log_file" 2>/dev/null | tail -n 1)
            if [ -n "$stats" ]; then
                success=$(echo "$stats" | grep -oP '✓\K[0-9]+' || echo 0)
                failed=$(echo "$stats" | grep -oP '✗\K[0-9]+' || echo 0)
                skipped=$(echo "$stats" | grep -oP '⊙\K[0-9]+' || echo 0)
                total_success=$((total_success + success))
                total_failed=$((total_failed + failed))
                total_skipped=$((total_skipped + skipped))
            fi
        fi
    done

    total_completed=$((total_success + total_failed + total_skipped))
    echo "Overall Progress:"
    echo "  Total completed: ${total_completed}/463 subjects"
    echo "    ✓ Success: ${total_success}"
    echo "    ✗ Failed:  ${total_failed}"
    echo "    ⊙ Skipped: ${total_skipped}"

    if [ $total_success -gt 0 ]; then
        # Calculate average MPJPE across all GPUs
        echo ""
        echo "Average MPJPE per GPU:"
        for gpu_id in 0 1 2 3; do
            log_file="gpu${gpu_id}_fullbody.log"
            if [ -f "$log_file" ]; then
                avg_mpjpe=$(grep "Avg:" "$log_file" 2>/dev/null | tail -n 1 | grep -oP 'Avg: \K[0-9.]+')
                if [ -n "$avg_mpjpe" ]; then
                    echo "  GPU ${gpu_id}: ${avg_mpjpe}mm"
                fi
            fi
        done
    fi

    echo ""
    echo "====================================================================="
    echo "Press Ctrl+C to exit monitoring"
    echo "Refreshing in 10 seconds..."
    echo ""

    sleep 10
done
