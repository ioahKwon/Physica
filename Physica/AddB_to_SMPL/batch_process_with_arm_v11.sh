#!/bin/bash

# Batch processing script for all With_Arm subjects using FIX 11
SMPL_MODEL="/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
OUT_DIR="/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_v11_physics"
NUM_FRAMES=200

mkdir -p "$OUT_DIR"

# Get all .b3d files and split into 4 batches
find /egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm -name "*.b3d" -type f | sort > /tmp/with_arm_files.txt
TOTAL=$(wc -l < /tmp/with_arm_files.txt)
PER_GPU=$((TOTAL / 4))

echo "Total subjects: $TOTAL"
echo "Per GPU: ~$PER_GPU"

# Split files
head -n $PER_GPU /tmp/with_arm_files.txt > /tmp/gpu0_files.txt
tail -n +$(($PER_GPU+1)) /tmp/with_arm_files.txt | head -n $PER_GPU > /tmp/gpu1_files.txt
tail -n +$((2*$PER_GPU+1)) /tmp/with_arm_files.txt | head -n $PER_GPU > /tmp/gpu2_files.txt
tail -n +$((3*$PER_GPU+1)) /tmp/with_arm_files.txt > /tmp/gpu3_files.txt

# Process each GPU
for GPU_ID in 0 1 2 3; do
    BATCH_FILE="/tmp/gpu${GPU_ID}_files.txt"
    echo "GPU $GPU_ID: $(wc -l < $BATCH_FILE) subjects"
    
    (
        while IFS= read -r B3D_FILE; do
            SUBJECT_NAME=$(basename "$B3D_FILE" .b3d)
            DATASET_NAME=$(basename $(dirname "$B3D_FILE"))
            
            echo "[GPU $GPU_ID] Processing: $DATASET_NAME/$SUBJECT_NAME"
            
            CUDA_VISIBLE_DEVICES=$GPU_ID /egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python \
                addbiomechanics_to_smpl_v11_with_physics.py \
                --b3d "$B3D_FILE" \
                --smpl_model "$SMPL_MODEL" \
                --out_dir "$OUT_DIR" \
                --num_frames $NUM_FRAMES \
                --device cuda \
                >> "/tmp/gpu${GPU_ID}_v11.log" 2>&1
            
            echo "[GPU $GPU_ID] Completed: $DATASET_NAME/$SUBJECT_NAME" >> "/tmp/gpu${GPU_ID}_v11.log"
        done < "$BATCH_FILE"
        
        echo "[GPU $GPU_ID] All done!" >> "/tmp/gpu${GPU_ID}_v11.log"
    ) &
done

echo "All GPUs started!"
wait
echo "All processing complete!"
