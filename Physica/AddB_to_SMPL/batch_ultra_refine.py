#!/usr/bin/env python3
"""
Batch Ultra Refinement - Process multiple subjects

Reads a list of subjects from a file and processes them sequentially
"""

import subprocess
import sys
from pathlib import Path
import json
import argparse
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--subjects_file', type=str, required=True)
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Directory containing initial results (With_Arm_500Frames)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for ultra-refined results')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--log_file', type=str, required=True)
    args = parser.parse_args()

    # Read subjects
    with open(args.subjects_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print(f"\n[GPU {args.gpu_id}] Processing {len(subjects)} subjects")
    print(f"[GPU {args.gpu_id}] Log: {args.log_file}")
    print(f"[GPU {args.gpu_id}] Input: {args.input_dir}")
    print(f"[GPU {args.gpu_id}] Output: {args.output_dir}\n")

    python_path = "/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
    ultra_refine_script = str(Path(__file__).parent / "ultra_refine.py")

    dataset_root = Path("/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm")
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    success_count = 0
    fail_count = 0

    for idx, subject_path in enumerate(subjects, 1):
        print(f"\n[GPU {args.gpu_id}] [{idx}/{len(subjects)}] Processing: {subject_path}")

        # Parse subject path
        parts = subject_path.split('/')
        if len(parts) < 2:
            print(f"[GPU {args.gpu_id}] ERROR: Invalid subject path format: {subject_path}")
            fail_count += 1
            continue

        study_name = parts[-2]  # e.g., "Han2023_Formatted_With_Arm"
        subject_name = parts[-1]  # e.g., "s004_split0"

        # Find B3D file
        b3d_path = dataset_root / study_name / subject_name / f"{subject_name}.b3d"

        if not b3d_path.exists():
            print(f"[GPU {args.gpu_id}] ERROR: B3D file not found: {b3d_path}")
            fail_count += 1
            continue

        # Find init result directory
        init_result = input_root / f"{study_name}_{subject_name}"

        if not init_result.exists():
            print(f"[GPU {args.gpu_id}] ERROR: Init result not found: {init_result}")
            fail_count += 1
            continue

        # Output directory
        out_dir = output_root / f"{study_name}_{subject_name}"

        # Call ultra_refine.py
        cmd = [
            python_path, ultra_refine_script,
            "--b3d", str(b3d_path),
            "--init_from", str(init_result),
            "--out_dir", str(out_dir),
            "--smpl_model", args.smpl_model,
            "--device", f"cuda"
        ]

        start_time = time.time()

        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            elapsed = time.time() - start_time

            # Parse output for MPJPE
            output_lines = result.stdout.split('\n')
            final_mpjpe = None
            prev_mpjpe = None
            improvement = None

            for line in output_lines:
                if "Previous MPJPE:" in line:
                    prev_mpjpe = float(line.split(':')[1].strip().replace('mm', ''))
                elif "Final MPJPE:" in line:
                    final_mpjpe = float(line.split(':')[1].strip().replace('mm', ''))
                elif "Improvement:" in line and '+' in line:
                    improvement = line.split(':')[1].strip()

            if final_mpjpe is not None:
                print(f"[GPU {args.gpu_id}] ✓ {subject_path}: {prev_mpjpe:.2f}mm → {final_mpjpe:.2f}mm ({improvement}) [{elapsed:.1f}s]")
                success_count += 1
            else:
                print(f"[GPU {args.gpu_id}] ✓ {subject_path} completed [{elapsed:.1f}s]")
                success_count += 1

        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start_time
            print(f"[GPU {args.gpu_id}] ✗ {subject_path} FAILED [{elapsed:.1f}s]")
            print(f"[GPU {args.gpu_id}]   Error: {e.stderr[:200]}")
            fail_count += 1

    print(f"\n{'='*80}")
    print(f"[GPU {args.gpu_id}] BATCH COMPLETE")
    print(f"{'='*80}")
    print(f"[GPU {args.gpu_id}] Success: {success_count}/{len(subjects)}")
    print(f"[GPU {args.gpu_id}] Failed:  {fail_count}/{len(subjects)}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
