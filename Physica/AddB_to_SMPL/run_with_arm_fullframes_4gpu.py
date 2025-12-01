#!/usr/bin/env python3
"""
Process all With_Arm subjects using all frames on 4 GPUs in parallel.
Goal: Complete all 435 subjects within 24 hours.

Distribution strategy:
- 435 subjects / 4 GPUs = ~109 subjects per GPU
- Use all frames (no subsampling except in Stage 1/3 as configured)
- Balanced workload across GPUs
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

# Configuration
PYTHON_BIN = "/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
SCRIPT = "addbiomechanics_to_smpl_v3_enhanced.py"
SMPL_MODEL = "models/smpl_model.pkl"
OUTPUT_DIR = "/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_AllFrames"
B3D_BASE = "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm"

# Get all .b3d files
b3d_files = sorted(list(Path(B3D_BASE).rglob("*.b3d")))
print(f"Found {len(b3d_files)} With_Arm subjects")

# Distribute subjects across 4 GPUs
subjects_per_gpu = [[] for _ in range(4)]
for i, b3d_file in enumerate(b3d_files):
    gpu_id = i % 4
    subjects_per_gpu[gpu_id].append(str(b3d_file))

# Print distribution
for gpu_id in range(4):
    print(f"GPU {gpu_id}: {len(subjects_per_gpu[gpu_id])} subjects")

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

# Launch one process per GPU
processes = []
log_files = []

for gpu_id in range(4):
    log_file = f"/tmp/gpu{gpu_id}_with_arm_fullframes.log"
    subjects_file = f"/tmp/gpu{gpu_id}_subjects.txt"

    # Write subjects list for this GPU
    with open(subjects_file, 'w') as f:
        for b3d_path in subjects_per_gpu[gpu_id]:
            f.write(f"{b3d_path}\n")

    # Launch batch processing script
    cmd = [
        PYTHON_BIN,
        "-u",  # Unbuffered output
        __file__.replace("run_with_arm_fullframes_4gpu.py", "batch_process_with_arm_gpu.py"),
        f"--gpu_id={gpu_id}",
        f"--subjects_file={subjects_file}",
        f"--output_dir={OUTPUT_DIR}",
        f"--log_file={log_file}"
    ]

    log_f = open(log_file, 'w')
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env={"CUDA_VISIBLE_DEVICES": str(gpu_id)}
    )

    processes.append(proc)
    log_files.append(log_file)
    print(f"✓ Launched GPU {gpu_id} processing (PID: {proc.pid}, Log: {log_file})")

print(f"\n{'='*80}")
print(f"ALL 4 GPUs LAUNCHED - Processing {len(b3d_files)} subjects")
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"{'='*80}\n")

# Monitor progress
print("Monitor logs with:")
for gpu_id, log_file in enumerate(log_files):
    print(f"  GPU {gpu_id}: tail -f {log_file}")

# Wait for all processes to complete
try:
    for gpu_id, proc in enumerate(processes):
        proc.wait()
        print(f"GPU {gpu_id} completed (exit code: {proc.returncode})")
except KeyboardInterrupt:
    print("\nInterrupted! Terminating processes...")
    for proc in processes:
        proc.terminate()
    print("All processes terminated.")

print(f"\n{'='*80}")
print(f"PROCESSING COMPLETE")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"{'='*80}")
