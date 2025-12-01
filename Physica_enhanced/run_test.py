#!/usr/bin/env python3
"""Simple test runner for Physica Enhanced."""

import sys
import os

# Set up path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Now import
from physica_pipeline import PhysicaPipeline
from core.config import PipelineConfig

print("✓ Imports successful!")

# Quick test
b3d_path = "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Camargo2021_Formatted_With_Arm/P002/P002_split0.b3d"
smpl_model_path = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl"
output_dir = "/egr/research-zijunlab/kwonjoon/Output/output_Physica_enhanced/quick_test"

print("\nInitializing pipeline...")
config = PipelineConfig.fast()
pipeline = PhysicaPipeline(
    smpl_model_path=smpl_model_path,
    config=config,
    device="cuda"
)

print("\nRunning pipeline on 50 frames...")
result = pipeline.run(b3d_path=b3d_path, num_frames=50)

print(f"\n✓ Pipeline completed!")
print(f"  Betas shape: {result.betas.shape}")
print(f"  Poses shape: {result.poses.shape}")
print(f"  Trans shape: {result.trans.shape}")
print(f"  MPJPE: {result.metrics['mpjpe']:.2f} mm")

pipeline.save_result(result, output_dir)
print(f"\n  Results saved to: {output_dir}")
