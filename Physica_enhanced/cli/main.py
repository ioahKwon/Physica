#!/usr/bin/env python3
"""
Command-line interface for Physica pipeline.

Usage:
    python -m physica_enhanced.cli.main --b3d <path> --smpl_model <path> --out_dir <path>
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from physica_enhanced.core import PipelineConfig
from physica_enhanced.physica_pipeline import PhysicaPipeline


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Physica Pipeline - AddBiomechanics → SMPL',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--b3d',
        required=True,
        help='Path to AddBiomechanics .b3d file'
    )
    parser.add_argument(
        '--smpl_model',
        required=True,
        help='Path to SMPL model .pkl file'
    )
    parser.add_argument(
        '--out_dir',
        required=True,
        help='Output directory for results'
    )

    # B3D loading options
    parser.add_argument('--trial', type=int, default=0, help='Trial index')
    parser.add_argument('--processing_pass', type=int, default=0, help='Processing pass index')
    parser.add_argument('--start_frame', type=int, default=0, help='Starting frame index')
    parser.add_argument('--num_frames', type=int, default=-1, help='Number of frames (-1 for all)')

    # Device options
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        help='Device for computation'
    )

    # Configuration presets
    parser.add_argument(
        '--preset',
        choices=['default', 'fast'],
        default='default',
        help='Configuration preset'
    )

    # Optimization hyperparameters
    parser.add_argument('--shape_iters', type=int, help='Shape optimization iterations')
    parser.add_argument('--pose_iters', type=int, help='Pose optimization iterations')
    parser.add_argument('--keyframe_ratio', type=float, help='Keyframe sampling ratio')

    # Other options
    parser.add_argument('--verbose', action='store_true', default=True, help='Verbose output')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create configuration
    if args.preset == 'fast':
        config = PipelineConfig.fast()
    else:
        config = PipelineConfig.default()

    # Override with command-line arguments
    if args.shape_iters is not None:
        config.shape_opt.max_iters = args.shape_iters
    if args.pose_iters is not None:
        config.pose_opt.max_iters = args.pose_iters
    if args.keyframe_ratio is not None:
        config.pose_opt.keyframe_ratio = args.keyframe_ratio

    config.verbose = args.verbose and not args.quiet
    config.device = args.device

    # Initialize pipeline
    pipeline = PhysicaPipeline(
        smpl_model_path=args.smpl_model,
        config=config,
        device=args.device
    )

    # Run pipeline
    try:
        result = pipeline.run(
            b3d_path=args.b3d,
            trial=args.trial,
            processing_pass=args.processing_pass,
            start_frame=args.start_frame,
            num_frames=args.num_frames
        )

        # Save results
        pipeline.save_result(result, args.out_dir)

        # Print summary
        if not args.quiet:
            print("\nSummary:")
            print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")
            print(f"  Output: {args.out_dir}")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if config.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
