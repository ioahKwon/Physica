#!/usr/bin/env python3
"""
Batch processing script for multiple subjects.

Usage:
    python run_batch.py --input_dir <path> --output_dir <path>
"""

import argparse
import os
import sys
import glob
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


def main():
    parser = argparse.ArgumentParser(
        description='Batch convert AddBiomechanics data to SKEL format'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing .b3d files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames per subject')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--subjects', type=str, nargs='+',
                        help='Specific subjects to process')
    args = parser.parse_args()

    from addb2skel.pipeline import convert_addb_to_skel, save_conversion_result
    from addb2skel.utils.io import load_b3d
    from addb2skel.config import OptimizationConfig

    # Find all .b3d files
    b3d_files = glob.glob(os.path.join(args.input_dir, '**/*.b3d'), recursive=True)
    print(f"Found {len(b3d_files)} .b3d files")

    if args.subjects:
        b3d_files = [f for f in b3d_files if any(s in f for s in args.subjects)]
        print(f"Filtered to {len(b3d_files)} files matching subjects")

    config = OptimizationConfig(device=args.device)

    results_summary = []

    for i, b3d_path in enumerate(b3d_files):
        subject_name = Path(b3d_path).stem
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(b3d_files)}] Processing: {subject_name}")
        print('='*60)

        try:
            # Load data
            addb_joints, joint_names, metadata = load_b3d(
                b3d_path, num_frames=args.num_frames
            )

            gender = 'male' if metadata['sex'] == 'male' else 'female'

            # Convert
            result = convert_addb_to_skel(
                addb_joints,
                gender=gender,
                config=config,
                height_m=metadata['height_m'],
                return_vertices=True,
                verbose=False,
            )

            # Save
            subject_output_dir = os.path.join(args.output_dir, subject_name)
            save_conversion_result(result, subject_output_dir, save_obj=True)

            results_summary.append({
                'subject': subject_name,
                'mpjpe_mm': result.mpjpe_mm,
                'status': 'success',
            })

            print(f"  MPJPE: {result.mpjpe_mm:.1f} mm")

        except Exception as e:
            print(f"  ERROR: {e}")
            results_summary.append({
                'subject': subject_name,
                'mpjpe_mm': None,
                'status': f'error: {e}',
            })

    # Print summary
    print("\n" + "=" * 60)
    print("BATCH SUMMARY")
    print("=" * 60)

    successful = [r for r in results_summary if r['status'] == 'success']
    if successful:
        mpjpes = [r['mpjpe_mm'] for r in successful]
        print(f"Successful: {len(successful)}/{len(results_summary)}")
        print(f"Mean MPJPE: {sum(mpjpes)/len(mpjpes):.1f} mm")
        print(f"Min MPJPE:  {min(mpjpes):.1f} mm")
        print(f"Max MPJPE:  {max(mpjpes):.1f} mm")

    failed = [r for r in results_summary if r['status'] != 'success']
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for r in failed:
            print(f"  {r['subject']}: {r['status']}")


if __name__ == '__main__':
    main()
