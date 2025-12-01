#!/usr/bin/env python3
"""
64-Frame Physics Pilot Test (Physics Loss ENABLED)

This script processes the same 6 With_Arm subjects with 64 frames each,
but with physics loss enabled to compare against the baseline.

Usage:
    CUDA_VISIBLE_DEVICES=0 python batch_process_physics_pilot.py --device cuda
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("64-FRAME PHYSICS PILOT TEST (Physics Enabled)")
print("=" * 80)
print()

# Load libraries
lib_load_start = time.time()
sys.path.insert(0, str(Path(__file__).parent))
from addbiomechanics_to_smpl_v3_enhanced import (
    process_single_b3d,
    SMPLModel,
    OptimisationConfig,
    torch,
)
lib_load_time = time.time() - lib_load_start
print(f"✓ Libraries loaded in {lib_load_time:.1f}s\n")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', help='Device (cpu or cuda)')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/physics_pilot_64frames',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--physics_foot_height_weight', type=float, default=0.1,
                        help='GRF loss weight (default: 0.1)')
    parser.add_argument('--physics_com_weight', type=float, default=0.05,
                        help='CoM loss weight (default: 0.05)')
    parser.add_argument('--physics_vel_smooth_weight', type=float, default=0.02,
                        help='Contact loss weight (default: 0.02)')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SMPL model
    print(f"⏳ Loading SMPL model on {args.device}...")
    smpl_model = SMPLModel(device=device, model_path=args.smpl_model)
    print(f"✓ SMPL model loaded\n")

    # Load pilot subjects (same as baseline)
    pilot_file = Path(__file__).parent / "pilot_baseline_64frames.json"
    with open(pilot_file, 'r') as f:
        pilot_data = json.load(f)

    subjects = pilot_data['subjects']
    print(f"✓ Loaded {len(subjects)} pilot subjects\n")

    # Configuration (same as baseline but with physics enabled)
    config = OptimisationConfig(
        shape_iters=80,
        pose_iters=50,
        pose_subsample_rate=48,  # Stage 3 speedup
        sequence_enhancement_iters=0,  # Skip Stage 4 for 64-frame pilot
        use_physics_loss=True,  # Physics ENABLED
        physics_foot_height_weight=args.physics_foot_height_weight,
        physics_com_weight=args.physics_com_weight,
        physics_vel_smooth_weight=args.physics_vel_smooth_weight
    )

    print("Configuration:")
    print(f"  Frames: 64")
    print(f"  lower_body_only: False (full body, 16 joints)")
    print(f"  Stage 1-2: shape_iters=80")
    print(f"  Stage 3: pose_iters=50, subsample=48")
    print(f"  Stage 4: SKIPPED (iters=0)")
    print(f"  Physics: ENABLED")
    print(f"    - GRF weight: {args.physics_foot_height_weight}")
    print(f"    - CoM weight: {args.physics_com_weight}")
    print(f"    - Contact weight: {args.physics_vel_smooth_weight}")
    print()

    # Process each subject
    results = []
    total_start = time.time()

    for idx, subj in enumerate(subjects, 1):
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(subjects)}] Processing {subj['name']}")
        print(f"{'='*80}")

        b3d_path = Path(subj['path'])
        subject_output_dir = output_dir / subj['name']

        # Check if already processed
        meta_file = subject_output_dir / "meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                if 'MPJPE' in meta:
                    print(f"✓ Already processed: MPJPE={meta['MPJPE']:.2f}mm")
                    results.append({
                        'subject': subj['name'],
                        'mpjpe': meta['MPJPE'],
                        'time': meta.get('optimization_time_seconds', 0),
                        'status': 'cached'
                    })
                    continue
            except:
                pass

        # Process
        subject_start = time.time()
        try:
            result = process_single_b3d(
                b3d_path=str(b3d_path),
                smpl_model=smpl_model,
                out_dir=str(subject_output_dir),
                num_frames=64,  # 64 frames only
                device=device,
                config=config,
                lower_body_only=False,  # Full body (16 joints)
                verbose=True
            )

            subject_time = time.time() - subject_start

            # Read MPJPE from saved meta.json since result structure differs
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            results.append({
                'subject': subj['name'],
                'mpjpe': meta['MPJPE'],
                'time': subject_time,
                'status': 'success'
            })

            print(f"\n✓ {subj['name']}: MPJPE={meta['MPJPE']:.2f}mm, Time={subject_time:.1f}s")

        except Exception as e:
            subject_time = time.time() - subject_start
            print(f"\n✗ {subj['name']}: FAILED - {str(e)}")
            results.append({
                'subject': subj['name'],
                'mpjpe': None,
                'time': subject_time,
                'status': 'failed',
                'error': str(e)
            })

    # Summary
    total_time = time.time() - total_start
    print(f"\n\n{'='*80}")
    print("PHYSICS PILOT TEST SUMMARY")
    print(f"{'='*80}")

    success_results = [r for r in results if r['status'] in ['success', 'cached'] and r['mpjpe'] is not None]
    if success_results:
        avg_mpjpe = sum(r['mpjpe'] for r in success_results) / len(success_results)
        avg_time = sum(r['time'] for r in success_results) / len(success_results)

        print(f"\nProcessed: {len(success_results)}/{len(subjects)}")
        print(f"Average MPJPE: {avg_mpjpe:.2f}mm")
        print(f"Average Time: {avg_time:.1f}s per subject")
        print(f"Total Time: {total_time/60:.1f} minutes")

        print(f"\nPer-subject results:")
        for r in results:
            if r['mpjpe'] is not None:
                print(f"  {r['subject']:30s} MPJPE={r['mpjpe']:6.2f}mm  Time={r['time']:6.1f}s  [{r['status']}]")
            else:
                print(f"  {r['subject']:30s} FAILED: {r.get('error', 'unknown')}")

        # Load baseline results for comparison
        baseline_summary = Path(__file__).parent.parent.parent / "Output/output_Physica/baseline_pilot_64frames/summary.json"
        if baseline_summary.exists():
            with open(baseline_summary, 'r') as f:
                baseline = json.load(f)

            baseline_results = {r['subject']: r['mpjpe'] for r in baseline['results'] if r['mpjpe'] is not None}

            print(f"\n{'='*80}")
            print("COMPARISON WITH BASELINE")
            print(f"{'='*80}")

            improvements = []
            for r in success_results:
                subj_name = r['subject']
                physics_mpjpe = r['mpjpe']
                baseline_mpjpe = baseline_results.get(subj_name)

                if baseline_mpjpe is not None:
                    improvement = baseline_mpjpe - physics_mpjpe
                    improvement_pct = (improvement / baseline_mpjpe) * 100
                    improvements.append(improvement_pct)

                    status = "↓ IMPROVED" if improvement > 0 else "↑ DEGRADED"
                    print(f"  {subj_name:30s} Baseline={baseline_mpjpe:6.2f}mm  Physics={physics_mpjpe:6.2f}mm  "
                          f"Δ={improvement:+6.2f}mm ({improvement_pct:+5.1f}%)  {status}")

            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                print(f"\nAverage improvement: {avg_improvement:+5.1f}%")
    else:
        avg_mpjpe = None
        avg_time = None

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'test_type': 'physics_64frames',
            'physics_enabled': True,
            'physics_weights': {
                'grf': args.physics_foot_height_weight,
                'com': args.physics_com_weight,
                'contact': args.physics_vel_smooth_weight
            },
            'num_subjects': len(subjects),
            'results': results,
            'average_mpjpe': avg_mpjpe if success_results else None,
            'average_time': avg_time if success_results else None,
            'total_time': total_time,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)

    print(f"\n✓ Summary saved to {summary_file}")
    print()

if __name__ == '__main__':
    main()
