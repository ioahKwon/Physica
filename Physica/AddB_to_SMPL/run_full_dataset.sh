#!/bin/bash
# Full dataset processing (excluding 2000-frame subjects)
# Uses same config as pilot test: subsample_rate=48

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

echo "================================================================================"
echo "FULL DATASET SMPL FITTING"
echo "================================================================================"
echo
echo "Starting 4 GPU processes in parallel..."
echo

# GPU 0
CUDA_VISIBLE_DEVICES=0 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    batch_process_v3_fullbody_parallel.py \
    --device cuda \
    --gpu_id 0 \
    --total_gpus 4 \
    > gpu0_full_dataset.log 2>&1 &
PID0=$!
echo "✓ GPU 0 started (PID: $PID0) - log: gpu0_full_dataset.log"

# GPU 1  
CUDA_VISIBLE_DEVICES=1 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    batch_process_v3_fullbody_parallel.py \
    --device cuda \
    --gpu_id 1 \
    --total_gpus 4 \
    > gpu1_full_dataset.log 2>&1 &
PID1=$!
echo "✓ GPU 1 started (PID: $PID1) - log: gpu1_full_dataset.log"

# GPU 2
CUDA_VISIBLE_DEVICES=2 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    batch_process_v3_fullbody_parallel.py \
    --device cuda \
    --gpu_id 2 \
    --total_gpus 4 \
    > gpu2_full_dataset.log 2>&1 &
PID2=$!
echo "✓ GPU 2 started (PID: $PID2) - log: gpu2_full_dataset.log"

# GPU 3
CUDA_VISIBLE_DEVICES=3 /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
    batch_process_v3_fullbody_parallel.py \
    --device cuda \
    --gpu_id 3 \
    --total_gpus 4 \
    > gpu3_full_dataset.log 2>&1 &
PID3=$!
echo "✓ GPU 3 started (PID: $PID3) - log: gpu3_full_dataset.log"

echo
echo "All 4 processes running in background!"
echo
echo "To monitor progress:"
echo "  tail -f gpu0_full_dataset.log"
echo
echo "To stop all processes:"
echo "  kill $PID0 $PID1 $PID2 $PID3"
echo
echo "Waiting for all processes to complete..."

# Wait for all background jobs
wait

echo
echo "================================================================================"
echo "ALL PROCESSES COMPLETE!"
echo "================================================================================"
echo
echo "Check results in:"
echo "  /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/"
echo
echo "Check logs:"
echo "  gpu{0,1,2,3}_full_dataset.log"
