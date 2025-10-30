#!/bin/bash
# Run full-frame SMPL fitting pilot test on 4 GPUs in parallel

echo "================================================================================"
echo "FULL-FRAME SMPL FITTING - PILOT TEST"
echo "================================================================================"
echo ""
echo "Starting 4 GPU processes in parallel..."
echo "Logs will be saved to: gpu{0,1,2,3}_fullframes_pilot.log"
echo ""

# Output directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
LOG_DIR="$SCRIPT_DIR"

# Run 4 processes in background, one per GPU
CUDA_VISIBLE_DEVICES=0 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    "$SCRIPT_DIR/batch_process_fullframes_pilot.py" \
    --device cuda \
    --gpu_id 0 \
    --total_gpus 4 \
    > "$LOG_DIR/gpu0_fullframes_pilot.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    "$SCRIPT_DIR/batch_process_fullframes_pilot.py" \
    --device cuda \
    --gpu_id 1 \
    --total_gpus 4 \
    > "$LOG_DIR/gpu1_fullframes_pilot.log" 2>&1 &
PID1=$!

CUDA_VISIBLE_DEVICES=2 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    "$SCRIPT_DIR/batch_process_fullframes_pilot.py" \
    --device cuda \
    --gpu_id 2 \
    --total_gpus 4 \
    > "$LOG_DIR/gpu2_fullframes_pilot.log" 2>&1 &
PID2=$!

CUDA_VISIBLE_DEVICES=3 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    "$SCRIPT_DIR/batch_process_fullframes_pilot.py" \
    --device cuda \
    --gpu_id 3 \
    --total_gpus 4 \
    > "$LOG_DIR/gpu3_fullframes_pilot.log" 2>&1 &
PID3=$!

echo "✓ GPU 0 started (PID: $PID0) - log: gpu0_fullframes_pilot.log"
echo "✓ GPU 1 started (PID: $PID1) - log: gpu1_fullframes_pilot.log"
echo "✓ GPU 2 started (PID: $PID2) - log: gpu2_fullframes_pilot.log"
echo "✓ GPU 3 started (PID: $PID3) - log: gpu3_fullframes_pilot.log"
echo ""
echo "All 4 processes running in background!"
echo ""
echo "To monitor progress:"
echo "  tail -f gpu0_fullframes_pilot.log"
echo "  ./monitor_pilot.sh"
echo ""
echo "To stop all processes:"
echo "  kill $PID0 $PID1 $PID2 $PID3"
echo ""
echo "Waiting for all processes to complete..."
echo ""

# Wait for all background jobs
wait $PID0
wait $PID1
wait $PID2
wait $PID3

echo ""
echo "================================================================================"
echo "ALL PROCESSES COMPLETE!"
echo "================================================================================"
echo ""
echo "Check results in:"
echo "  /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/pilot_results/"
echo ""
echo "Check logs:"
echo "  gpu{0,1,2,3}_fullframes_pilot.log"
echo ""
