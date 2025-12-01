#!/bin/bash
# Extract physics data for 99 already-completed subjects

echo "================================================================================"
echo "EXTRACT PHYSICS DATA FOR 99 COMPLETED SUBJECTS"
echo "================================================================================"
echo ""
echo "This will add physics_data.npz to subjects that already have SMPL results"
echo "Expected time: ~12 minutes"
echo ""
echo "================================================================================"
echo ""

PYTHON="/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
OUTPUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_AllFrames"
DATASET_DIR="/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm"

cd /egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL

$PYTHON extract_physics_only.py \
    --output_dir=$OUTPUT_DIR \
    --dataset_dir=$DATASET_DIR

echo ""
echo "================================================================================"
echo "DONE!"
echo "================================================================================"
echo ""
echo "Check results in: $OUTPUT_DIR"
echo "Each subject directory should now have physics_data.npz"
echo ""
