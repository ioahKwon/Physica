#!/usr/bin/env python3
"""
Batch refine subjects with high MPJPE (>50mm) using warm-start from existing results.
This script processes subjects on a single GPU with refined optimization settings.
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
    parser.add_argument('--subjects_file', type=str, required=True,
                       help='File with format: b3d_path|result_dir|mpjpe per line')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--log_file', type=str, required=True)
    args = parser.parse_args()

    # Read subjects list (format: b3d_path|result_dir|mpjpe)
    with open(args.subjects_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    subjects = []
    for line in lines:
        parts = line.split('|')
        if len(parts) >= 3:
            subjects.append({
                'b3d_path': parts[0],
                'init_from': parts[1],  # Previous result directory for warm-start
                'mpjpe_old': float(parts[2])
            })

    print(f"{'='*80}")
    print(f"GPU {args.gpu_id}: Refining {len(subjects)} subjects with MPJPE > 50mm")
    print(f"Strategy: Warm-start from existing results + 0.7× LR + 1.0× iterations")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    python_bin = "/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python"
    script = "addbiomechanics_to_smpl_v3_refined.py"
    smpl_model = "models/smpl_model.pkl"

    results = []
    start_time = time.time()

    for i, subj_info in enumerate(subjects, 1):
        b3d_path = subj_info['b3d_path']
        init_from = subj_info['init_from']
        mpjpe_old = subj_info['mpjpe_old']

        b3d_path_obj = Path(b3d_path)
        subject_name = b3d_path_obj.stem
        study_name = b3d_path_obj.parent.parent.name

        # Output to a new "refined" directory
        out_subj_dir = Path(args.output_dir) / f"{study_name}_{subject_name}_refined"

        # Skip if already refined
        if (out_subj_dir / "meta.json").exists():
            print(f"[GPU {args.gpu_id}] [{i}/{len(subjects)}] SKIP {subject_name} (already refined)")
            continue

        print(f"\n[GPU {args.gpu_id}] [{i}/{len(subjects)}] Refining {subject_name}...")
        print(f"  Study: {study_name}")
        print(f"  B3D: {b3d_path}")
        print(f"  Old MPJPE: {mpjpe_old:.2f}mm")
        print(f"  Warm-start from: {init_from}")

        subject_start = time.time()

        try:
            # Run refinement with warm-start
            cmd = [
                python_bin,
                script,
                "--b3d", b3d_path,
                "--smpl_model", smpl_model,
                "--out_dir", str(out_subj_dir),
                "--num_frames", "-1",  # Load all frames
                "--max_frames_optimize", "200",  # Sample to max 200 frames
                "--include_head_neck",  # Include head/neck joints (24 joints total)
                "--device", "cuda",
                # REFINEMENT ARGS:
                "--init_from", init_from,  # Warm-start from existing results
                "--refine_lr_scale", "0.7",  # 70% of original LR (faster convergence)
                "--refine_iters_scale", "1.0"  # 1.0× iterations (same as original)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            subject_time = time.time() - subject_start

            # Parse result
            meta_file = out_subj_dir / "with_arm_*/meta.json"
            meta_files = list(out_subj_dir.glob("with_arm_*/meta.json"))

            if meta_files:
                meta_file = meta_files[0]
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                mpjpe_new = meta.get('metrics', {}).get('MPJPE', None)
                improvement = mpjpe_old - mpjpe_new if mpjpe_new else 0
                status = "success"
                print(f"  ✓ {subject_name}: {mpjpe_old:.2f}mm → {mpjpe_new:.2f}mm (Δ {improvement:.2f}mm, {subject_time:.1f}s)")
            else:
                mpjpe_new = None
                improvement = 0
                status = "failed"
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                print(f"  ✗ {subject_name}: FAILED ({error_msg})")

            results.append({
                "subject": subject_name,
                "study": study_name,
                "mpjpe_old": mpjpe_old,
                "mpjpe_new": mpjpe_new,
                "improvement": improvement,
                "time": subject_time,
                "status": status
            })

        except subprocess.TimeoutExpired:
            subject_time = time.time() - subject_start
            print(f"  ✗ {subject_name}: TIMEOUT (>{subject_time:.0f}s)")
            results.append({
                "subject": subject_name,
                "study": study_name,
                "mpjpe_old": mpjpe_old,
                "mpjpe_new": None,
                "improvement": 0,
                "time": subject_time,
                "status": "timeout"
            })
        except Exception as e:
            subject_time = time.time() - subject_start
            print(f"  ✗ {subject_name}: ERROR ({str(e)})")
            results.append({
                "subject": subject_name,
                "study": study_name,
                "mpjpe_old": mpjpe_old,
                "mpjpe_new": None,
                "improvement": 0,
                "time": subject_time,
                "status": "error",
                "error": str(e)
            })

    total_time = time.time() - start_time

    # Save summary
    summary_file = Path(args.output_dir) / f"gpu{args.gpu_id}_refine_summary.json"
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
    successful = [r for r in results if r['status'] == 'success']
    avg_improvement = sum(r['improvement'] for r in successful) / len(successful) if successful else 0
    avg_mpjpe_old = sum(r['mpjpe_old'] for r in successful) / len(successful) if successful else 0
    avg_mpjpe_new = sum(r['mpjpe_new'] for r in successful if r['mpjpe_new']) / len(successful) if successful else 0

    print(f"\n{'='*80}")
    print(f"GPU {args.gpu_id} REFINEMENT SUMMARY")
    print(f"{'='*80}")
    print(f"Total subjects: {len(subjects)}")
    print(f"Successful: {sum(1 for r in results if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results if r['status'] == 'failed')}")
    print(f"Timeout: {sum(1 for r in results if r['status'] == 'timeout')}")
    print(f"Average MPJPE: {avg_mpjpe_old:.2f}mm → {avg_mpjpe_new:.2f}mm (Δ {avg_improvement:.2f}mm)")
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Summary saved to: {summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
