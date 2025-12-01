#!/usr/bin/env python3
"""
Test script for Physica Enhanced pipeline.

Tests the pipeline on real AddBiomechanics data.
"""

import sys
from pathlib import Path
import torch

# Add to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from physica_enhanced import PhysicaPipeline, PipelineConfig


def test_basic_functionality():
    """Test basic pipeline functionality."""
    print("="*80)
    print("Test 1: Basic Functionality")
    print("="*80)

    # Use a small test file
    b3d_path = "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Camargo2021_Formatted_With_Arm/P002/P002_split0.b3d"
    smpl_model_path = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
    output_dir = "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/test_basic"

    # Fast configuration for quick test
    config = PipelineConfig.fast()
    config.verbose = True

    # Initialize pipeline
    pipeline = PhysicaPipeline(
        smpl_model_path=smpl_model_path,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Run on small subset
    result = pipeline.run(
        b3d_path=b3d_path,
        num_frames=50  # Only process first 50 frames for quick test
    )

    # Check results
    assert result.betas.shape == (10,), f"Unexpected betas shape: {result.betas.shape}"
    assert result.poses.shape[0] == 50, f"Unexpected poses frames: {result.poses.shape[0]}"
    assert result.poses.shape[1:] == (24, 3), f"Unexpected poses shape: {result.poses.shape}"
    assert result.trans.shape == (50, 3), f"Unexpected trans shape: {result.trans.shape}"

    print(f"\n✓ Results validated")
    print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")

    # Save results
    pipeline.save_result(result, output_dir)
    print(f"  Output: {output_dir}")

    return result


def test_full_sequence():
    """Test on full sequence (With Arm)."""
    print("\n" + "="*80)
    print("Test 2: Full Sequence (With Arm)")
    print("="*80)

    b3d_path = "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Camargo2021_Formatted_With_Arm/P002/P002_split0.b3d"
    smpl_model_path = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
    output_dir = "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/test_full_with_arm"

    config = PipelineConfig.fast()
    config.verbose = True

    pipeline = PhysicaPipeline(
        smpl_model_path=smpl_model_path,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    result = pipeline.run(b3d_path=b3d_path)

    print(f"\n✓ Full sequence processed")
    print(f"  Frames: {result.poses.shape[0]}")
    print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")

    pipeline.save_result(result, output_dir)

    return result


def test_no_arm_sequence():
    """Test on No Arm sequence."""
    print("\n" + "="*80)
    print("Test 3: No Arm Sequence")
    print("="*80)

    b3d_path = "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm/Tiziana2019_Formatted_No_Arm/Subject7/Subject7.b3d"
    smpl_model_path = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
    output_dir = "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/test_no_arm"

    config = PipelineConfig.fast()
    config.verbose = True

    pipeline = PhysicaPipeline(
        smpl_model_path=smpl_model_path,
        config=config,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    result = pipeline.run(
        b3d_path=b3d_path,
        num_frames=100  # Process subset for speed
    )

    print(f"\n✓ No Arm sequence processed")
    print(f"  Frames: {result.poses.shape[0]}")
    print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")
    print(f"  Synthesized joints: {len(result.retargeting_result.synthesized_indices)}")

    pipeline.save_result(result, output_dir)

    return result


def test_performance_comparison():
    """Compare performance with original implementation."""
    print("\n" + "="*80)
    print("Test 4: Performance Comparison")
    print("="*80)

    import time

    b3d_path = "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Camargo2021_Formatted_With_Arm/P002/P002_split0.b3d"
    smpl_model_path = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"

    # Test fast preset
    print("\nTesting FAST preset...")
    config_fast = PipelineConfig.fast()
    config_fast.verbose = False

    pipeline_fast = PhysicaPipeline(
        smpl_model_path=smpl_model_path,
        config=config_fast,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    result_fast = pipeline_fast.run(b3d_path=b3d_path, num_frames=100)
    time_fast = time.time() - start_time

    print(f"  Time: {time_fast:.2f}s")
    print(f"  MPJPE: {result_fast.metrics['mpjpe']:.2f} mm")

    # Test default preset
    print("\nTesting DEFAULT preset...")
    config_default = PipelineConfig.default()
    config_default.verbose = False

    pipeline_default = PhysicaPipeline(
        smpl_model_path=smpl_model_path,
        config=config_default,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    start_time = time.time()
    result_default = pipeline_default.run(b3d_path=b3d_path, num_frames=100)
    time_default = time.time() - start_time

    print(f"  Time: {time_default:.2f}s")
    print(f"  MPJPE: {result_default.metrics['mpjpe']:.2f} mm")

    print(f"\n✓ Performance comparison:")
    print(f"  Fast preset: {time_fast:.2f}s, {result_fast.metrics['mpjpe']:.2f}mm")
    print(f"  Default preset: {time_default:.2f}s, {result_default.metrics['mpjpe']:.2f}mm")
    print(f"  Speedup: {time_default/time_fast:.2f}x")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("Physica Enhanced - Test Suite")
    print("="*80 + "\n")

    try:
        # Run tests
        test_basic_functionality()
        test_full_sequence()
        test_no_arm_sequence()
        test_performance_comparison()

        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80 + "\n")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
