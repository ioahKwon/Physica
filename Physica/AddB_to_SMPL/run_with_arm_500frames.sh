#!/bin/bash
#
# Run With_Arm dataset processing with 500-frame adaptive sampling on 4 GPUs
# 435 subjects, distributed evenly across GPUs 0,1,2,3
# Long sequences (>500 frames) will be sampled uniformly (3-5× faster)
#

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

echo "================================================================================"
echo "WITH_ARM 500-FRAME SAMPLING PROCESSING - 4 GPUs (WITH HEAD/NECK)"
echo "================================================================================"
echo ""
echo "Dataset: 435 subjects from With_Arm"
echo "GPUs: 0, 1, 2, 3 (parallel processing)"
echo "Frames: Adaptive sampling (max 500 frames per subject)"
echo "Joints: ALL 24 SMPL joints (including head/neck)"
echo "Output: /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"
echo ""
echo "Speed improvement: 3-5× faster than full frames"
echo "Estimated time: 4-6 hours (vs 12-20 hours for full frames)"
echo ""
echo "================================================================================"
echo ""

# Get all .b3d files
find /egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm -name "*.b3d" -type f | sort > /tmp/all_with_arm_subjects_500f.txt

TOTAL=$(wc -l < /tmp/all_with_arm_subjects_500f.txt)
echo "Found $TOTAL subjects"
echo ""

# Split into 4 lists for 4 GPUs
split -n l/4 /tmp/all_with_arm_subjects_500f.txt /tmp/gpu_500f_
mv /tmp/gpu_500f_aa /tmp/gpu0_subjects_500f.txt
mv /tmp/gpu_500f_ab /tmp/gpu1_subjects_500f.txt
mv /tmp/gpu_500f_ac /tmp/gpu2_subjects_500f.txt
mv /tmp/gpu_500f_ad /tmp/gpu3_subjects_500f.txt

for i in 0 1 2 3; do
    count=$(wc -l < /tmp/gpu${i}_subjects_500f.txt)
    echo "GPU $i: $count subjects"
done

echo ""
echo "Starting processing..."
echo ""

# Launch 4 parallel processes
PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"
SMPL_MODEL="models/smpl_model.pkl"

mkdir -p "$OUTPUT_DIR"

for GPU_ID in 0 1 2 3; do
    SUBJECTS_FILE="/tmp/gpu${GPU_ID}_subjects_500f.txt"
    LOG_FILE="/tmp/gpu${GPU_ID}_with_arm_500frames.log"

    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u batch_process_with_arm_500frames.py \
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
echo "Monitor progress in real-time:"
echo "  ./monitor_with_arm_500frames.sh"
echo ""
echo "Or view individual GPU logs:"
echo "  tail -f /tmp/gpu0_with_arm_500frames.log"
echo "  tail -f /tmp/gpu1_with_arm_500frames.log"
echo "  tail -f /tmp/gpu2_with_arm_500frames.log"
echo "  tail -f /tmp/gpu3_with_arm_500frames.log"
echo ""
echo "Analyze MPJPE after completion:"
echo "  ./analyze_mpjpe.sh"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
echo "================================================================================"
