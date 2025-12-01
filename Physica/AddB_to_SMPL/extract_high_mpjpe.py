#!/usr/bin/env python3
"""
Extract subjects with MPJPE > 50mm for refinement
"""

import json
from pathlib import Path
import sys

def main():
    threshold = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    output_dir = Path("/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/With_Arm_500Frames")

    high_mpjpe_subjects = []

    # Find all meta.json files
    for meta_file in output_dir.glob("*/with_arm_*/meta.json"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)

            mpjpe = meta.get('metrics', {}).get('MPJPE', None)
            b3d_path = meta.get('b3d', None)  # Changed from 'b3d_file' to 'b3d'
            result_dir = meta_file.parent.parent  # Go up two levels

            if mpjpe is not None and mpjpe > threshold and b3d_path is not None:
                high_mpjpe_subjects.append({
                    'b3d_path': b3d_path,
                    'result_dir': str(result_dir),
                    'mpjpe': mpjpe
                })
        except Exception as e:
            print(f"Error reading {meta_file}: {e}", file=sys.stderr)
            continue

    # Sort by MPJPE (highest first)
    high_mpjpe_subjects.sort(key=lambda x: x['mpjpe'], reverse=True)

    # Save to 2 files:
    # 1. Full info (for debugging)
    output_file = Path("/tmp/high_mpjpe_subjects.txt")
    with open(output_file, 'w') as f:
        for subj in high_mpjpe_subjects:
            f.write(f"{subj['b3d_path']}|{subj['result_dir']}|{subj['mpjpe']:.2f}\n")

    # 2. B3D paths only (for batch processing)
    b3d_file = Path("/tmp/high_mpjpe_b3d_paths.txt")
    with open(b3d_file, 'w') as f:
        for subj in high_mpjpe_subjects:
            f.write(f"{subj['b3d_path']}\n")

    print(f"Found {len(high_mpjpe_subjects)} subjects with MPJPE > {threshold:.0f}mm")
    print(f"Saved to: {output_file}")

    # Print statistics
    if high_mpjpe_subjects:
        mpjpes = [s['mpjpe'] for s in high_mpjpe_subjects]
        print(f"\nStatistics:")
        print(f"  Highest MPJPE: {max(mpjpes):.2f}mm")
        print(f"  Lowest MPJPE: {min(mpjpes):.2f}mm")
        print(f"  Average MPJPE: {sum(mpjpes)/len(mpjpes):.2f}mm")

        # Show first 5
        print(f"\nTop 5 worst subjects:")
        for i, subj in enumerate(high_mpjpe_subjects[:5], 1):
            print(f"  {i}. {Path(subj['result_dir']).name}: {subj['mpjpe']:.2f}mm")

if __name__ == "__main__":
    main()
