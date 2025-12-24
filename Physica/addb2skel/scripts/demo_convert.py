#!/usr/bin/env python3
"""
Demo script for AddB → SKEL conversion.

Usage:
    python demo_convert.py --b3d <path> --output_dir <path>
    python demo_convert.py --npy <path> --output_dir <path>
"""

import argparse
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Convert AddBiomechanics data to SKEL format'
    )
    parser.add_argument('--b3d', type=str, help='Path to AddB .b3d file')
    parser.add_argument('--npy', type=str, help='Path to AddB joints .npy file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10,
                        help='Number of frames to process')
    parser.add_argument('--gender', type=str, default='male',
                        choices=['male', 'female'], help='Subject gender')
    parser.add_argument('--height', type=float, help='Subject height in meters')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'], help='Device')
    parser.add_argument('--pose_iters', type=int, default=300,
                        help='Number of pose optimization iterations')
    args = parser.parse_args()

    # Import after parsing to allow --help without loading torch
    from addb2skel.pipeline import convert_addb_to_skel, save_conversion_result
    from addb2skel.config import OptimizationConfig

    # Load data
    if args.b3d:
        from addb2skel.utils.io import load_b3d
        print(f"Loading {args.b3d}...")
        addb_joints, joint_names, metadata = load_b3d(
            args.b3d, num_frames=args.num_frames
        )
        gender = 'male' if metadata['sex'] == 'male' else 'female'
        height_m = metadata['height_m']
        print(f"  Loaded {len(addb_joints)} frames")
        print(f"  Subject: height={height_m:.2f}m, gender={gender}")
    elif args.npy:
        print(f"Loading {args.npy}...")
        addb_joints = np.load(args.npy)
        if args.num_frames:
            addb_joints = addb_joints[:args.num_frames]
        gender = args.gender
        height_m = args.height
        print(f"  Loaded {len(addb_joints)} frames")
    else:
        parser.error("Must specify either --b3d or --npy")

    # Create config
    config = OptimizationConfig(
        device=args.device,
        pose_iters=args.pose_iters,
    )

    # Run conversion
    result = convert_addb_to_skel(
        addb_joints,
        gender=gender,
        config=config,
        height_m=height_m,
        return_vertices=True,
        verbose=True,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    save_conversion_result(result, args.output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"MPJPE: {result.mpjpe_mm:.1f} mm")
    print(f"Output saved to: {args.output_dir}")

    if result.per_joint_error:
        print("\nPer-joint errors (mm):")
        for joint, err in sorted(result.per_joint_error.items(), key=lambda x: x[1], reverse=True):
            print(f"  {joint:20s}: {err:6.1f}")


if __name__ == '__main__':
    main()
