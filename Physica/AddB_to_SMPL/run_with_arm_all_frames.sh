#!/bin/bash
#
# Run With_Arm dataset processing with ALL frames on 4 GPUs
# 435 subjects, distributed evenly across GPUs 0,1,2,3
#

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

echo "================================================================================"
echo "WITH_ARM FULL FRAMES PROCESSING - 4 GPUs"
echo "================================================================================"
echo ""
echo "Dataset: 435 subjects from With_Arm"
echo "GPUs: 0, 1, 2, 3 (parallel processing)"
echo "Frames: ALL frames per subject (no subsampling)"
echo "Output: /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_AllFrames"
echo ""
echo "Estimated time: 12-20 hours (goal: <24 hours)"
echo ""
echo "================================================================================"
echo ""

# Get all .b3d files
find /egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm -name "*.b3d" -type f | sort > /tmp/all_with_arm_subjects.txt

TOTAL=$(wc -l < /tmp/all_with_arm_subjects.txt)
echo "Found $TOTAL subjects"
echo ""

# Split into 4 lists for 4 GPUs
split -n l/4 /tmp/all_with_arm_subjects.txt /tmp/gpu_
mv /tmp/gpu_aa /tmp/gpu0_subjects.txt
mv /tmp/gpu_ab /tmp/gpu1_subjects.txt
mv /tmp/gpu_ac /tmp/gpu2_subjects.txt
mv /tmp/gpu_ad /tmp/gpu3_subjects.txt

for i in 0 1 2 3; do
    count=$(wc -l < /tmp/gpu${i}_subjects.txt)
    echo "GPU $i: $count subjects"
done

echo ""
echo "Starting processing..."
echo ""

# Launch 4 parallel processes
PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_AllFrames"
SMPL_MODEL="models/smpl_model.pkl"

mkdir -p "$OUTPUT_DIR"

for GPU_ID in 0 1 2 3; do
    SUBJECTS_FILE="/tmp/gpu${GPU_ID}_subjects.txt"
    LOG_FILE="/tmp/gpu${GPU_ID}_with_arm_fullframes.log"

    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u batch_process_with_arm_gpu.py \
        --gpu_id=$GPU_ID \
        --subjects_file=$SUBJECTS_FILE \
        --output_dir=$OUTPUT_DIR \
        --log_file=$LOG_FILE \
        > $LOG_FILE 2>&1 &

    PID=$!
    echo "✓ GPU $GPU_ID launched (PID: $PID, Log: $LOG_FILE)"
done

echo ""
echo "================================================================================"
echo "ALL 4 GPUs LAUNCHED"
echo "================================================================================"
echo ""
echo "Monitor progress with:"
echo "  tail -f /tmp/gpu0_with_arm_fullframes.log"
echo "  tail -f /tmp/gpu1_with_arm_fullframes.log"
echo "  tail -f /tmp/gpu2_with_arm_fullframes.log"
echo "  tail -f /tmp/gpu3_with_arm_fullframes.log"
echo ""
echo "Or use this monitoring command:"
echo '  watch -n 10 '"'"'for i in 0 1 2 3; do echo "GPU $i:"; tail -5 /tmp/gpu${i}_with_arm_fullframes.log; echo ""; done'"'"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
echo "================================================================================"
