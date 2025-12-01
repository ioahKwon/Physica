#!/bin/bash
#
# Run With_Arm dataset processing with 500-frame adaptive sampling on 4 GPUs
# 435 subjects, distributed evenly across GPUs 0,1,2,3
# Long sequences (>500 frames) will be sampled uniformly (3-5× faster)
#
# NOHUP VERSION: 연결이 끊겨도 계속 실행됩니다!
#

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

echo "================================================================================"
echo "WITH_ARM 200-FRAME SAMPLING PROCESSING - 4 GPUs (NOHUP MODE + HEAD/NECK)"
echo "================================================================================"
echo ""
echo "Dataset: 435 subjects from With_Arm"
echo "GPUs: 0, 1, 2, 3 (parallel processing)"
echo "Frames: Adaptive sampling (max 200 frames per subject)"
echo "Joints: ALL 24 SMPL joints (including head/neck)"
echo "Output: /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"
echo ""
echo "⚠️  NOHUP MODE: 터미널/SSH 연결이 끊겨도 계속 실행됩니다!"
echo ""
echo "Speed improvement: Faster processing with 200-frame sampling"
echo "Estimated time: ~13 hours (vs 33 hours with 500 frames)"
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
echo "Starting processing with nohup..."
echo ""

# Launch 4 parallel processes WITH NOHUP
PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"
SMPL_MODEL="models/smpl_model.pkl"

mkdir -p "$OUTPUT_DIR"

for GPU_ID in 0 1 2 3; do
    SUBJECTS_FILE="/tmp/gpu${GPU_ID}_subjects_500f.txt"
    LOG_FILE="/tmp/gpu${GPU_ID}_with_arm_500frames.log"

    # Use nohup to keep process running after logout
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u batch_process_with_arm_500frames.py \
        --gpu_id=$GPU_ID \
        --subjects_file=$SUBJECTS_FILE \
        --output_dir=$OUTPUT_DIR \
        --log_file=$LOG_FILE \
        > $LOG_FILE 2>&1" &

    PID=$!
    echo "✓ GPU $GPU_ID launched with nohup (PID: $PID, Log: $LOG_FILE)"
done

echo ""
echo "================================================================================"
echo "ALL 4 GPUs LAUNCHED WITH NOHUP"
echo "================================================================================"
echo ""
echo "✅ 이제 터미널을 닫거나 SSH 연결이 끊겨도 계속 실행됩니다!"
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
echo "Check if still running:"
echo "  ps aux | grep batch_process_with_arm_500frames"
echo ""
echo "Stop all processes:"
echo "  pkill -f batch_process_with_arm_500frames"
echo ""
echo "Analyze MPJPE after completion:"
echo "  ./analyze_mpjpe.sh"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
echo "================================================================================"
