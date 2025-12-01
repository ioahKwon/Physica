#!/bin/bash
# Process FAST subjects (<500 frames, non-Carter/Han) first with 4 GPUs

echo "================================================================================"
echo "WITH_ARM FAST BATCH - 4 GPUs"
echo "================================================================================"
echo ""
echo "Processing: 91 subjects (excluding Carter2023, Han2023)"
echo "GPUs: 0, 1, 2, 3 (parallel processing)"
echo "Expected time: 3-5 hours"
echo ""
echo "================================================================================"
echo ""

DATASET_DIR="/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_AllFrames"
PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"

# Split fast subjects into 4 GPU lists
SUBJECTS_FILE="/tmp/with_arm_fast_subjects.txt"

if [ ! -f "$SUBJECTS_FILE" ]; then
    echo "ERROR: $SUBJECTS_FILE not found!"
    exit 1
fi

NUM_SUBJECTS=$(wc -l < "$SUBJECTS_FILE")
echo "Found $NUM_SUBJECTS subjects"
echo ""

# Split into 4 parts
split -n l/4 "$SUBJECTS_FILE" /tmp/fast_gpu_
mv /tmp/fast_gpu_aa /tmp/fast_gpu0_subjects.txt
mv /tmp/fast_gpu_ab /tmp/fast_gpu1_subjects.txt
mv /tmp/fast_gpu_ac /tmp/fast_gpu2_subjects.txt
mv /tmp/fast_gpu_ad /tmp/fast_gpu3_subjects.txt

GPU0_COUNT=$(wc -l < /tmp/fast_gpu0_subjects.txt)
GPU1_COUNT=$(wc -l < /tmp/fast_gpu1_subjects.txt)
GPU2_COUNT=$(wc -l < /tmp/fast_gpu2_subjects.txt)
GPU3_COUNT=$(wc -l < /tmp/fast_gpu3_subjects.txt)

echo "GPU 0: $GPU0_COUNT subjects"
echo "GPU 1: $GPU1_COUNT subjects"
echo "GPU 2: $GPU2_COUNT subjects"
echo "GPU 3: $GPU3_COUNT subjects"
echo ""
echo "Starting processing..."
echo ""

# Launch 4 GPU processes
for GPU_ID in 0 1 2 3; do
    LOG_FILE="/tmp/fast_gpu${GPU_ID}_with_arm.log"
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u batch_process_with_arm_gpu.py \
        --gpu_id=$GPU_ID \
        --subjects_file=/tmp/fast_gpu${GPU_ID}_subjects.txt \
        --output_dir=$OUTPUT_DIR \
        --log_file=$LOG_FILE \
        > $LOG_FILE 2>&1 &

    PID=$!
    echo "✓ GPU $GPU_ID launched (PID: $PID, Log: $LOG_FILE)"
done

echo ""
echo "================================================================================"
echo "ALL 4 GPUs LAUNCHED FOR FAST BATCH"
echo "================================================================================"
echo ""
echo "Monitor progress with:"
echo "  tail -f /tmp/fast_gpu0_with_arm.log"
echo "  tail -f /tmp/fast_gpu1_with_arm.log"
echo "  tail -f /tmp/fast_gpu2_with_arm.log"
echo "  tail -f /tmp/fast_gpu3_with_arm.log"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
echo "================================================================================"
