#!/usr/bin/env python3
"""
64-Frame Baseline Pilot Test (Physics Loss DISABLED)

This script processes 6 With_Arm subjects with 64 frames each to establish
a baseline for comparison with physics-enabled version.

Usage:
    CUDA_VISIBLE_DEVICES=0 python batch_process_baseline_pilot.py --device cuda
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("64-FRAME BASELINE PILOT TEST (Physics Disabled)")
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
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/baseline_pilot_64frames',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load SMPL model
    print(f"⏳ Loading SMPL model on {args.device}...")
    smpl_model = SMPLModel(device=device, model_path=args.smpl_model)
    print(f"✓ SMPL model loaded\n")

    # Load pilot subjects
    pilot_file = Path(__file__).parent / "pilot_baseline_64frames.json"
    with open(pilot_file, 'r') as f:
        pilot_data = json.load(f)

    subjects = pilot_data['subjects']
    print(f"✓ Loaded {len(subjects)} pilot subjects\n")

    # Configuration (Stage 4 생략 for speed)
    config = OptimisationConfig(
        shape_iters=80,
        pose_iters=50,
        pose_subsample_rate=48,  # Stage 3 speedup
        sequence_enhancement_iters=0,  # Skip Stage 4 for 64-frame pilot
        use_physics_loss=False  # Physics disabled
    )

    print("Configuration:")
    print(f"  Frames: 64")
    print(f"  lower_body_only: False (full body, 16 joints)")
    print(f"  Stage 1-2: shape_iters=80")
    print(f"  Stage 3: pose_iters=50, subsample=48")
    print(f"  Stage 4: SKIPPED (iters=0)")
    print(f"  Physics: DISABLED")
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

            results.append({
                'subject': subj['name'],
                'mpjpe': result['MPJPE'],
                'time': subject_time,
                'status': 'success'
            })

            print(f"\n✓ {subj['name']}: MPJPE={result['MPJPE']:.2f}mm, Time={subject_time:.1f}s")

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
    print("BASELINE PILOT TEST SUMMARY")
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

    # Save summary
    summary_file = output_dir / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'test_type': 'baseline_64frames',
            'physics_enabled': False,
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
