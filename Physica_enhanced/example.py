#!/usr/bin/env python3
"""
Example usage of Physica Enhanced pipeline.

This script demonstrates how to use the pipeline both as a library and via CLI.
"""

import sys
from pathlib import Path

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physica_enhanced import PhysicaPipeline, PipelineConfig


def example_library_usage():
    """Example: Using pipeline as a Python library."""
    print("="*80)
    print("Example 1: Library Usage")
    print("="*80)

    # Configure pipeline
    config = PipelineConfig.fast()  # or PipelineConfig.default()

    # Initialize pipeline
    pipeline = PhysicaPipeline(
        smpl_model_path="/path/to/smpl_model.pkl",
        config=config,
        device="cuda"  # or "cpu"
    )

    # Run on a B3D file
    result = pipeline.run(
        b3d_path="/path/to/data.b3d",
        trial=0,
        num_frames=100  # Process first 100 frames
    )

    # Access results
    print(f"\nResults:")
    print(f"  Shape (β): {result.betas.shape}")
    print(f"  Poses (θ): {result.poses.shape}")
    print(f"  Translations: {result.trans.shape}")
    print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")

    # Save results
    pipeline.save_result(result, "/path/to/output")


def example_batch_processing():
    """Example: Batch processing multiple files."""
    print("="*80)
    print("Example 2: Batch Processing")
    print("="*80)

    # Initialize pipeline once
    pipeline = PhysicaPipeline(
        smpl_model_path="/path/to/smpl_model.pkl",
        config=PipelineConfig.fast(),
        device="cuda"
    )

    # Process multiple files
    b3d_files = [
        "/path/to/subject1.b3d",
        "/path/to/subject2.b3d",
        "/path/to/subject3.b3d",
    ]

    for i, b3d_path in enumerate(b3d_files):
        print(f"\nProcessing {i+1}/{len(b3d_files)}: {b3d_path}")

        result = pipeline.run(b3d_path)

        # Save with unique name
        output_dir = f"/path/to/output/subject_{i+1}"
        pipeline.save_result(result, output_dir)

        print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")


def example_custom_config():
    """Example: Custom configuration."""
    print("="*80)
    print("Example 3: Custom Configuration")
    print("="*80)

    # Start with default config
    config = PipelineConfig.default()

    # Customize shape optimization
    config.shape_opt.max_iters = 100
    config.shape_opt.sample_frames = 200
    config.shape_opt.batch_size = 64

    # Customize pose optimization
    config.pose_opt.max_iters = 15
    config.pose_opt.keyframe_ratio = 0.25
    config.pose_opt.interpolation_method = "slerp"

    # Enable optimizations
    config.use_torch_compile = True

    pipeline = PhysicaPipeline(
        smpl_model_path="/path/to/smpl_model.pkl",
        config=config
    )

    result = pipeline.run("/path/to/data.b3d")


def example_cli_usage():
    """Example: CLI usage (for reference)."""
    print("="*80)
    print("Example 4: CLI Usage")
    print("="*80)

    print("""
# Basic usage:
python -m physica_enhanced.cli.main \\
    --b3d /path/to/data.b3d \\
    --smpl_model /path/to/smpl_model.pkl \\
    --out_dir /path/to/output

# Fast preset:
python -m physica_enhanced.cli.main \\
    --b3d /path/to/data.b3d \\
    --smpl_model /path/to/smpl_model.pkl \\
    --out_dir /path/to/output \\
    --preset fast

# Custom parameters:
python -m physica_enhanced.cli.main \\
    --b3d /path/to/data.b3d \\
    --smpl_model /path/to/smpl_model.pkl \\
    --out_dir /path/to/output \\
    --shape_iters 100 \\
    --pose_iters 15 \\
    --keyframe_ratio 0.25

# Process subset of frames:
python -m physica_enhanced.cli.main \\
    --b3d /path/to/data.b3d \\
    --smpl_model /path/to/smpl_model.pkl \\
    --out_dir /path/to/output \\
    --start_frame 0 \\
    --num_frames 100
    """)


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Physica Enhanced - Example Usage")
    print("="*80 + "\n")

    # Uncomment to run examples:
    # example_library_usage()
    # example_batch_processing()
    # example_custom_config()
    example_cli_usage()
