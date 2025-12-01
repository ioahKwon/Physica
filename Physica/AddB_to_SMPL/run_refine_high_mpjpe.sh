#!/bin/bash
#
# Refine 237 subjects with MPJPE > 50mm using warm-start optimization
# Distribution: 79, 79, 79 subjects across GPUs 0, 1, 2
# Strategy: 0.3× LR + 3× iterations from existing results
#

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

echo "================================================================================"
echo "HIGH-MPJPE REFINEMENT - Warm-start from existing results"
echo "================================================================================"
echo ""
echo "Subjects: 237 with MPJPE > 50mm"
echo "GPUs: 0, 1, 2, 3 (parallel processing)"
echo "Frames: 200 frames (fixed)"
echo "Joints: ALL 24 SMPL joints (including head/neck)"
echo "Output: /egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Refined"
echo ""
echo "Refinement strategy:"
echo "  - Warm-start: Load existing SMPL poses/betas from previous run"
echo "  - Learning rate: 0.7× (70% of original for faster convergence)"
echo "  - Iterations: 1.0× (same as original, for faster processing)"
echo ""
echo "Expected improvement: 3-7mm average MPJPE reduction"
echo "Estimated time: ~3 hours (237 subjects × ~45s/subject ÷ 4 GPUs)"
echo ""
echo "================================================================================"
echo ""

# Check if subject list exists
if [ ! -f /tmp/high_mpjpe_subjects.txt ]; then
    echo "❌ Error: /tmp/high_mpjpe_subjects.txt not found!"
    echo "Please run: python3 extract_high_mpjpe.py 50"
    exit 1
fi

TOTAL=$(wc -l < /tmp/high_mpjpe_subjects.txt)
echo "Found $TOTAL subjects to refine"
echo ""

# Split into 4 lists for 4 GPUs
split -n l/4 /tmp/high_mpjpe_subjects.txt /tmp/gpu_refine_
mv /tmp/gpu_refine_aa /tmp/gpu0_refine.txt
mv /tmp/gpu_refine_ab /tmp/gpu1_refine.txt
mv /tmp/gpu_refine_ac /tmp/gpu2_refine.txt
mv /tmp/gpu_refine_ad /tmp/gpu3_refine.txt

for i in 0 1 2 3; do
    count=$(wc -l < /tmp/gpu${i}_refine.txt)
    echo "GPU $i: $count subjects"
done

echo ""
echo "Starting refinement with nohup..."
echo ""

# Launch 4 parallel processes WITH NOHUP
PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Refined"
SMPL_MODEL="models/smpl_model.pkl"

mkdir -p "$OUTPUT_DIR"

for GPU_ID in 0 1 2 3; do
    SUBJECTS_FILE="/tmp/gpu${GPU_ID}_refine.txt"
    LOG_FILE="/tmp/gpu${GPU_ID}_refine.log"

    # Use nohup to keep process running after logout
    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u batch_refine_high_mpjpe.py \
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
echo "✅ 터미널을 닫거나 SSH 연결이 끊겨도 계속 실행됩니다!"
echo ""
echo "Monitor progress in real-time:"
echo "  ./monitor_refine.sh"
echo ""
echo "Or view individual GPU logs:"
echo "  tail -f /tmp/gpu0_refine.log"
echo "  tail -f /tmp/gpu1_refine.log"
echo "  tail -f /tmp/gpu2_refine.log"
echo ""
echo "Check if still running:"
echo "  ps aux | grep batch_refine_high_mpjpe"
echo ""
echo "Stop all processes:"
echo "  pkill -f batch_refine_high_mpjpe"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo ""
echo "================================================================================"
