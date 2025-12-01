#!/usr/bin/env python3
"""
Test mmhuman3d visualization with baseline SMPL parameters
"""

import numpy as np
import torch
import argparse
from pathlib import Path

# Import mmhuman3d visualization
from mmhuman3d.core.visualization.visualize_smpl import visualize_smpl_pose


def test_mmhuman3d_visualization(smpl_params_path, output_video, smpl_model_path):
    """
    Test mmhuman3d visualization

    Args:
        smpl_params_path: Path to smpl_params.npz file
        output_video: Output MP4 file path
        smpl_model_path: Path to SMPL model directory
    """
    print(f"Loading SMPL parameters from {smpl_params_path}...")
    data = np.load(smpl_params_path)

    betas = data['betas']  # (10,) or (T, 10)
    poses = data['poses']  # (T, 24, 3) or (T, 72)
    trans = data['trans']  # (T, 3)

    T = poses.shape[0]
    print(f"  Loaded {T} frames")
    print(f"  Betas shape: {betas.shape}")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Prepare data for mmhuman3d
    # mmhuman3d expects poses in (T, 72) format
    if len(poses.shape) == 3:
        poses_flat = poses.reshape(T, -1)  # (T, 24, 3) -> (T, 72)
    else:
        poses_flat = poses  # Already (T, 72)

    # Expand betas if needed
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))  # (10,) -> (T, 10)
    else:
        betas_expanded = betas[:T]  # Already (T, 10)

    print(f"\nPrepared for mmhuman3d:")
    print(f"  Poses: {poses_flat.shape}")
    print(f"  Betas: {betas_expanded.shape}")
    print(f"  Trans: {trans.shape}")

    # Convert to torch tensors
    poses_tensor = torch.from_numpy(poses_flat).float()
    betas_tensor = torch.from_numpy(betas_expanded).float()
    trans_tensor = torch.from_numpy(trans).float()

    print(f"\nGenerating visualization with mmhuman3d...")
    print(f"  Output: {output_video}")
    print(f"  SMPL model: {smpl_model_path}")

    # Prepare body model config
    body_model_config = {
        'type': 'SMPL',
        'model_path': smpl_model_path,
    }

    # Use mmhuman3d visualization
    try:
        visualize_smpl_pose(
            poses=poses_tensor,
            betas=betas_tensor,
            transl=trans_tensor,
            body_model_config=body_model_config,
            output_path=output_video,
            render_choice='hq',  # High quality mesh rendering
            resolution=(1920, 1080),
            orbit_speed=0.0,  # Fixed camera
            palette='white'
        )
        print(f"\n✓ Successfully generated video: {output_video}")
    except Exception as e:
        print(f"\n✗ Error during visualization: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Test mmhuman3d visualization')
    parser.add_argument('--smpl_params', type=str, required=True,
                       help='Path to smpl_params.npz file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output MP4 video path')
    parser.add_argument('--smpl_model', type=str,
                       default='models/smpl_model.pkl',
                       help='Path to SMPL model file or directory')
    args = parser.parse_args()

    test_mmhuman3d_visualization(args.smpl_params, args.output, args.smpl_model)


if __name__ == '__main__':
    main()
