#!/bin/bash
#
# Run ultra refinement on high-MPJPE subjects across 4 GPUs
#

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
INPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_Ultra_Refined"

echo "================================================================================

"
echo "🚀 ULTRA REFINEMENT - Full Arsenal Strategy"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Find subjects with 50mm <= MPJPE < 150mm from With_Arm_500Frames
# Exclude very bad cases (>= 150mm) as they make things worse
echo "Finding subjects with 50mm <= MPJPE < 150mm..."
SUBJECTS_FILE="/tmp/ultra_refine_subjects.txt"
> $SUBJECTS_FILE

python3 << EOF
import json
from pathlib import Path

input_dir = Path('$INPUT_DIR')
output_file = '$SUBJECTS_FILE'

subjects = []

for result_dir in input_dir.glob('*/'):
    meta_files = list(result_dir.glob('with_arm_*/meta.json'))
    if not meta_files:
        continue

    meta_file = meta_files[0]
    try:
        with open(meta_file) as f:
            meta = json.load(f)
        mpjpe = meta.get('metrics', {}).get('MPJPE', 0)

        # Only include subjects in 50-150mm range (exclude worst cases)
        if 50.0 <= mpjpe < 150.0:
            # Extract subject path
            dir_name = result_dir.name
            # Format: Study_Formatted_With_Arm_SubjectName
            parts = dir_name.split('_')

            # Find where study name ends (before "Formatted")
            formatted_idx = -1
            for i, part in enumerate(parts):
                if part == "Formatted":
                    formatted_idx = i
                    break

            if formatted_idx > 0:
                study = '_'.join(parts[:formatted_idx+3])  # Include "_With_Arm"
                subject = '_'.join(parts[formatted_idx+3:])

                subject_path = f"{study}/{subject}"
                subjects.append((subject_path, mpjpe))
    except:
        pass

# Sort by MPJPE (descending) - hardest first
subjects.sort(key=lambda x: x[1], reverse=True)

print(f"Found {len(subjects)} subjects with 50mm <= MPJPE < 150mm")

# Write to file
with open(output_file, 'w') as f:
    for subject_path, mpjpe in subjects:
        f.write(f"{subject_path}\n")

print(f"Subject list written to: {output_file}")
EOF

TOTAL=$(wc -l < $SUBJECTS_FILE)
echo "Total subjects to process: $TOTAL"
echo ""

if [ $TOTAL -eq 0 ]; then
    echo "No subjects found with 50mm <= MPJPE < 150mm. Exiting."
    exit 0
fi

# Split subjects across 4 GPUs
echo "Splitting subjects across 4 GPUs..."
split -n l/4 -d $SUBJECTS_FILE /tmp/ultra_gpu
mv /tmp/ultra_gpu00 /tmp/ultra_gpu0.txt
mv /tmp/ultra_gpu01 /tmp/ultra_gpu1.txt
mv /tmp/ultra_gpu02 /tmp/ultra_gpu2.txt
mv /tmp/ultra_gpu03 /tmp/ultra_gpu3.txt

for i in 0 1 2 3; do
    COUNT=$(wc -l < /tmp/ultra_gpu${i}.txt)
    echo "  GPU $i: $COUNT subjects"
done
echo ""

# Launch batch processing on 4 GPUs
echo "Launching ultra refinement on 4 GPUs..."
echo ""

for GPU_ID in 0 1 2 3; do
    SUBJECTS_FILE_GPU="/tmp/ultra_gpu${GPU_ID}.txt"
    LOG_FILE="/tmp/ultra_gpu${GPU_ID}.log"

    nohup bash -c "CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON -u batch_ultra_refine.py \
        --gpu_id=$GPU_ID \
        --subjects_file=$SUBJECTS_FILE_GPU \
        --input_dir=$INPUT_DIR \
        --output_dir=$OUTPUT_DIR \
        --log_file=$LOG_FILE \
        > $LOG_FILE 2>&1" &

    PID=$!
    COUNT=$(wc -l < $SUBJECTS_FILE_GPU)
    echo "✓ GPU $GPU_ID launched (PID: $PID, $COUNT subjects)"
done

echo ""
echo "================================================================================"
echo "🎯 All 4 GPUs launched!"
echo "================================================================================"
echo ""
echo "Monitor progress:"
echo "  ./monitor_ultra_refine.sh"
echo ""
echo "Or check individual logs:"
echo "  tail -f /tmp/ultra_gpu0.log"
echo "  tail -f /tmp/ultra_gpu1.log"
echo "  tail -f /tmp/ultra_gpu2.log"
echo "  tail -f /tmp/ultra_gpu3.log"
echo ""
echo "Results will be saved to: $OUTPUT_DIR"
echo "================================================================================"
