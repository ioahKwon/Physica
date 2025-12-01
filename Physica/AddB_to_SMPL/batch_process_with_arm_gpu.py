#!/usr/bin/env python3
"""
Batch process subjects on a single GPU with ALL frames.
"""

import argparse
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=int, required=True)
    parser.add_argument('--subjects_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True)
    args = parser.parse_args()

    # Read subjects list
    with open(args.subjects_file, 'r') as f:
        subjects = [line.strip() for line in f if line.strip()]

    print(f"{'='*80}")
    print(f"GPU {args.gpu_id}: Processing {len(subjects)} subjects with ALL FRAMES")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    python_bin = "/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
    script = "addbiomechanics_to_smpl_v3_enhanced.py"
    smpl_model = "models/smpl_model.pkl"

    results = []
    start_time = time.time()

    for i, b3d_path in enumerate(subjects, 1):
        b3d_path_obj = Path(b3d_path)
        subject_name = b3d_path_obj.stem
        study_name = b3d_path_obj.parent.parent.name

        out_subj_dir = Path(args.output_dir) / f"{study_name}_{subject_name}"

        # Skip if already processed
        if (out_subj_dir / "meta.json").exists():
            print(f"[GPU {args.gpu_id}] [{i}/{len(subjects)}] SKIP {subject_name} (already done)")
            continue

        print(f"\n[GPU {args.gpu_id}] [{i}/{len(subjects)}] Processing {subject_name}...")
        print(f"  Study: {study_name}")
        print(f"  B3D: {b3d_path}")

        subject_start = time.time()

        try:
            # Run optimization with frame downsampling
            # Using --max_frames_optimize 200 for 10× speedup
            cmd = [
                python_bin,
                script,
                "--b3d", b3d_path,
                "--smpl_model", smpl_model,
                "--out_dir", str(out_subj_dir),
                "--num_frames", "-1",  # Load all frames first
                "--max_frames_optimize", "200",  # Then downsample to 200 for optimization
                "--device", "cuda"
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=None  # No timeout - let subjects complete naturally
            )

            subject_time = time.time() - subject_start

            # Parse result
            meta_file = out_subj_dir / "meta.json"
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                mpjpe = meta.get('MPJPE', None)
                num_frames = meta.get('num_frames', None)
                status = "success"
                print(f"  ✓ {subject_name}: MPJPE={mpjpe:.2f}mm, Time={subject_time:.1f}s, Frames={num_frames}")
            else:
                mpjpe = None
                num_frames = None
                status = "failed"
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                print(f"  ✗ {subject_name}: FAILED ({error_msg})")

            results.append({
                "subject": subject_name,
                "study": study_name,
                "mpjpe": mpjpe,
                "num_frames": num_frames,
                "time": subject_time,
                "status": status
            })

        except subprocess.TimeoutExpired:
            subject_time = time.time() - subject_start
            print(f"  ✗ {subject_name}: TIMEOUT (>{subject_time:.0f}s)")
            results.append({
                "subject": subject_name,
                "study": study_name,
                "mpjpe": None,
                "num_frames": None,
                "time": subject_time,
                "status": "timeout"
            })
        except Exception as e:
            subject_time = time.time() - subject_start
            print(f"  ✗ {subject_name}: ERROR ({str(e)})")
            results.append({
                "subject": subject_name,
                "study": study_name,
                "mpjpe": None,
                "num_frames": None,
                "time": subject_time,
                "status": "error",
                "error": str(e)
            })

    total_time = time.time() - start_time

    # Save summary
    summary_file = Path(args.output_dir) / f"gpu{args.gpu_id}_summary.json"
    summary = {
        "gpu_id": args.gpu_id,
        "num_subjects": len(subjects),
        "results": results,
        "total_time": total_time,
        "timestamp": datetime.now().isoformat()
    }

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*80}")
    print(f"GPU {args.gpu_id} SUMMARY")
    print(f"{'='*80}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Timeout: {sum(1 for r in results if r['status'] == 'timeout')}")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
