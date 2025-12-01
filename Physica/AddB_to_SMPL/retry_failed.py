#!/usr/bin/env python3
"""
Retry failed subjects by processing specific b3d files
"""

import sys
import json
import time
from pathlib import Path
import argparse

# Import from v3_enhanced
sys.path.insert(0, str(Path(__file__).parent))
from addbiomechanics_to_smpl_v3_enhanced import (
    process_single_b3d,
    SMPLModel,
    OptimisationConfig,
    torch,
)

def main():
    parser = argparse.ArgumentParser(description='Retry failed subjects')
    parser.add_argument('--b3d_list', type=str, required=True,
                       help='File containing list of b3d files to process')
    parser.add_argument('--smpl_model', type=str, required=True,
                       help='Path to SMPL model')
    parser.add_argument('--out_dir', type=str, required=True,
                       help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--num_frames', type=int, default=64)

    args = parser.parse_args()

    # Load SMPL model
    print("Loading SMPL model...")
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    smpl_model = SMPLModel(args.smpl_model, device=device)
    print(f"✓ SMPL model loaded on {device}")

    # Read b3d list
    with open(args.b3d_list, 'r') as f:
        b3d_files = [line.strip() for line in f if line.strip()]

    print(f"\nFound {len(b3d_files)} files to process")

    # Create config
    config = OptimisationConfig()

    # Process each file
    success = 0
    failed = 0
    skipped = 0

    for idx, b3d_path in enumerate(b3d_files, 1):
        b3d_path = Path(b3d_path)

        # Generate output name
        subject_dir = b3d_path.parent.name
        subject_name = b3d_path.stem

        if '/test/' in str(b3d_path):
            output_name = f"test_{subject_dir}_{subject_name}"
        else:
            output_name = f"train_{subject_dir}_{subject_name}"

        subject_output_dir = Path(args.out_dir) / output_name

        # Check if already done
        if (subject_output_dir / 'meta.json').exists():
            print(f"[{idx}/{len(b3d_files)}] ⊙ {output_name} (already done)")
            skipped += 1
            continue

        # Process
        print(f"[{idx}/{len(b3d_files)}] Processing {output_name}...")
        start = time.time()

        try:
            result = process_single_b3d(
                b3d_path=str(b3d_path),
                smpl_model=smpl_model,
                out_dir=str(subject_output_dir),
                num_frames=args.num_frames,
                device=device,
                config=config,
                verbose=False
            )

            elapsed = time.time() - start
            mpjpe = result['mpjpe']
            print(f"[{idx}/{len(b3d_files)}] ✓ {output_name} | {mpjpe:.2f}mm | {elapsed:.1f}s")
            success += 1

        except Exception as e:
            elapsed = time.time() - start
            error_msg = str(e)[:50]
            print(f"[{idx}/{len(b3d_files)}] ✗ {output_name} | {error_msg} | {elapsed:.1f}s")
            failed += 1

    # Summary
    print("\n" + "="*80)
    print("RETRY COMPLETE")
    print("="*80)
    print(f"Success:  {success}")
    print(f"Failed:   {failed}")
    print(f"Skipped:  {skipped}")
    print(f"Total:    {len(b3d_files)}")
    print("="*80)

if __name__ == '__main__':
    main()
