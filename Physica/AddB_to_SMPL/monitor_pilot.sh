#!/bin/bash
# Monitor full-frame pilot test progress in real-time

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "================================================================================"
echo "FULL-FRAME PILOT TEST - REAL-TIME MONITORING"
echo "================================================================================"
echo ""
echo "Monitoring logs: gpu{0,1,2,3}_fullframes_pilot.log"
echo "Press Ctrl+C to exit"
echo ""
echo "================================================================================"

# Function to get last N lines from each log
show_progress() {
    for gpu in 0 1 2 3; do
        log_file="$SCRIPT_DIR/gpu${gpu}_fullframes_pilot.log"
        if [ -f "$log_file" ]; then
            echo ""
            echo "--- GPU $gpu (last 10 lines) ---"
            tail -n 10 "$log_file" | grep -E "(GPU|Progress|COMPLETE|Average|Total)" || echo "(no progress yet)"
        fi
    done
}

# Show initial status
show_progress

# Update every 10 seconds
while true; do
    sleep 10
    clear
    echo "================================================================================"
    echo "FULL-FRAME PILOT TEST - MONITORING (updates every 10s)"
    echo "================================================================================"
    show_progress
    echo ""
    echo "================================================================================"
    echo "Last update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "Press Ctrl+C to exit"
    echo "================================================================================"
done
