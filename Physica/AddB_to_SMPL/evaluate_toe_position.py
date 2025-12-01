#!/usr/bin/env python3
"""
Evaluate toe position accuracy between SMPL and AddBiomechanics.

This script provides an alternative evaluation metric to foot orientation,
focusing on the 3D position of toe joints rather than their rotational alignment.

Usage:
    python evaluate_toe_position.py \
        --b3d <path_to_b3d_file> \
        --smpl_result <path_to_smpl_params.npz> \
        --smpl_model <path_to_smpl_model.pkl> \
        --output <output_directory>
"""

import numpy as np
import torch
import argparse
import sys
import json
from pathlib import Path
import nimblephysics as nimble

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def load_addbiomechanics_data(b3d_path, num_frames=None):
    """Load AddBiomechanics data and extract joint positions."""
    print(f"Loading AddBiomechanics data from {b3d_path}...")

    subject = nimble.biomechanics.SubjectOnDisk(str(b3d_path))

    # Read frames
    frames = subject.readFrames(
        trial=0,
        startFrame=0,
        numFramesToRead=num_frames if num_frames else subject.getTrialLength(0),
        stride=1,
        contactThreshold=0.01
    )

    num_frames_loaded = len(frames)
    print(f"  Loaded {num_frames_loaded} frames from AddB")

    # Get skeleton to find joint indices
    skel = subject.readSkel(processingPass=0)

    # Find joint indices in AddB skeleton
    joint_map = {}
    for i in range(skel.getNumBodyNodes()):
        body = skel.getBodyNode(i)
        joint_map[body.getName()] = i

    print(f"  AddB joint mapping:")
    for name in ['talus_l', 'talus_r', 'calcn_l', 'calcn_r',
                 'mtp_l', 'mtp_r', 'subtalar_l', 'subtalar_r',
                 'toes_l', 'toes_r']:
        if name in joint_map:
            print(f"    {name} → index {joint_map[name]}")

    # Extract joint positions
    def frame_joint_centers(frame):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            return np.asarray(frame.processingPasses[0].jointCenters, dtype=np.float32)
        return np.asarray(frame.jointCenters, dtype=np.float32)

    first = frame_joint_centers(frames[0])
    num_joints_addb = first.size // 3

    joint_positions = np.zeros((num_frames_loaded, num_joints_addb, 3))
    for t, frame in enumerate(frames):
        joint_centers = frame_joint_centers(frame)
        joint_positions[t] = joint_centers.reshape(-1, 3)

    return {
        'joint_positions': joint_positions,
        'joint_map': joint_map,
        'num_frames': num_frames_loaded
    }


def convert_addb_to_smpl_coords(joints):
    """
    Convert AddBiomechanics coordinates (Z-up) to SMPL coordinates (Y-up).

    Args:
        joints: [..., 3] array in AddB coordinates

    Returns:
        [..., 3] array in SMPL coordinates
    """
    # AddB: Z-up, SMPL: Y-up
    # Conversion: [x, y, z]_addb -> [x, z, -y]_smpl
    return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)


def find_toe_joint_indices(joint_map):
    """
    Find toe joint indices in AddBiomechanics data.

    Priority:
    1. mtp_l/r (metatarsophalangeal joint - preferred)
    2. subtalar_l/r (fallback if mtp not available)
    3. calcn_l/r (calcaneus/heel - last resort)

    Returns:
        (left_toe_idx, right_toe_idx) or None if not found
    """
    # Try MTP first (most accurate for toe position)
    left_toe_idx = joint_map.get('mtp_l')
    right_toe_idx = joint_map.get('mtp_r')

    if left_toe_idx is not None and right_toe_idx is not None:
        print(f"  Using MTP joints: left={left_toe_idx}, right={right_toe_idx}")
        return left_toe_idx, right_toe_idx

    # Fallback to subtalar
    left_toe_idx = joint_map.get('subtalar_l')
    right_toe_idx = joint_map.get('subtalar_r')

    if left_toe_idx is not None and right_toe_idx is not None:
        print(f"  Using subtalar joints: left={left_toe_idx}, right={right_toe_idx}")
        return left_toe_idx, right_toe_idx

    # Last resort: calcn
    left_toe_idx = joint_map.get('calcn_l')
    right_toe_idx = joint_map.get('calcn_r')

    if left_toe_idx is not None and right_toe_idx is not None:
        print(f"  Using calcn joints: left={left_toe_idx}, right={right_toe_idx}")
        return left_toe_idx, right_toe_idx

    return None, None


def compute_toe_position_error(smpl_model, betas, poses, trans, addb_data, device):
    """
    Compute toe position error between SMPL and AddBiomechanics.

    Returns:
        dict with left_error, right_error, mean_error (all in mm)
    """
    T = poses.shape[0]

    # Get SMPL toe positions (joints 10 and 11)
    # Expand betas from (10,) to (T, 10)
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))  # (T, 10)
    else:
        betas_expanded = betas

    betas_t = torch.from_numpy(betas_expanded).float().to(device)

    # Reshape poses from (T, 24, 3) to (T, 72) if needed
    if len(poses.shape) == 3:
        poses_flat = poses.reshape(T, -1)  # (T, 72)
    else:
        poses_flat = poses
    poses_t = torch.from_numpy(poses_flat).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)

    with torch.no_grad():
        vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    joints_np = joints.cpu().numpy()  # (T, 24, 3)

    SMPL_LEFT_FOOT = 10
    SMPL_RIGHT_FOOT = 11

    smpl_left_toe = joints_np[:, SMPL_LEFT_FOOT, :]  # (T, 3)
    smpl_right_toe = joints_np[:, SMPL_RIGHT_FOOT, :]  # (T, 3)

    # Get AddB toe positions
    addb_positions = addb_data['joint_positions'][:T]  # (T, J, 3)
    joint_map = addb_data['joint_map']

    left_toe_idx, right_toe_idx = find_toe_joint_indices(joint_map)

    if left_toe_idx is None or right_toe_idx is None:
        print("ERROR: Could not find toe joints in AddBiomechanics data")
        return None

    addb_left_toe = addb_positions[:, left_toe_idx, :]  # (T, 3)
    addb_right_toe = addb_positions[:, right_toe_idx, :]  # (T, 3)

    # Convert AddB coordinates to SMPL coordinates
    addb_left_toe = convert_addb_to_smpl_coords(addb_left_toe)
    addb_right_toe = convert_addb_to_smpl_coords(addb_right_toe)

    # Compute 3D Euclidean distance
    left_diff = smpl_left_toe - addb_left_toe  # (T, 3)
    right_diff = smpl_right_toe - addb_right_toe  # (T, 3)

    left_dist = np.linalg.norm(left_diff, axis=1)  # (T,)
    right_dist = np.linalg.norm(right_diff, axis=1)  # (T,)

    # Convert to mm
    left_error_mm = left_dist.mean() * 1000
    right_error_mm = right_dist.mean() * 1000
    mean_error_mm = (left_error_mm + right_error_mm) / 2

    # Per-frame statistics
    left_min = left_dist.min() * 1000
    left_max = left_dist.max() * 1000
    left_std = left_dist.std() * 1000

    right_min = right_dist.min() * 1000
    right_max = right_dist.max() * 1000
    right_std = right_dist.std() * 1000

    return {
        'left_error_mm': float(left_error_mm),
        'left_min_mm': float(left_min),
        'left_max_mm': float(left_max),
        'left_std_mm': float(left_std),
        'right_error_mm': float(right_error_mm),
        'right_min_mm': float(right_min),
        'right_max_mm': float(right_max),
        'right_std_mm': float(right_std),
        'mean_error_mm': float(mean_error_mm),
        'num_frames': int(T),
        'per_frame_left': left_dist.tolist(),
        'per_frame_right': right_dist.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate toe position accuracy')
    parser.add_argument('--b3d', type=str, required=True, help='Path to .b3d file')
    parser.add_argument('--smpl_result', type=str, required=True,
                       help='Path to smpl_params.npz')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SMPL model
    print("\nLoading SMPL model...")
    smpl_model = SMPLModel(args.smpl_model, device=device)

    # Load SMPL results
    print(f"\nLoading SMPL results from {args.smpl_result}...")
    smpl_data = np.load(args.smpl_result)
    betas = smpl_data['betas']
    poses = smpl_data['poses']
    trans = smpl_data['trans']
    T = poses.shape[0]
    print(f"  Loaded {T} frames")

    # Load AddBiomechanics data
    addb_data = load_addbiomechanics_data(args.b3d, num_frames=T)

    # Compute toe position error
    print("\n" + "="*80)
    print("COMPUTING TOE POSITION ERROR")
    print("="*80)

    results = compute_toe_position_error(smpl_model, betas, poses, trans, addb_data, device)

    if results is None:
        print("ERROR: Failed to compute toe position error")
        sys.exit(1)

    # Print results
    print(f"\nToe Position Error (3D Euclidean Distance):")
    print(f"  Left Toe:")
    print(f"    Mean:  {results['left_error_mm']:7.2f} mm")
    print(f"    Std:   {results['left_std_mm']:7.2f} mm")
    print(f"    Min:   {results['left_min_mm']:7.2f} mm")
    print(f"    Max:   {results['left_max_mm']:7.2f} mm")

    print(f"\n  Right Toe:")
    print(f"    Mean:  {results['right_error_mm']:7.2f} mm")
    print(f"    Std:   {results['right_std_mm']:7.2f} mm")
    print(f"    Min:   {results['right_min_mm']:7.2f} mm")
    print(f"    Max:   {results['right_max_mm']:7.2f} mm")

    print(f"\n  Combined:")
    print(f"    Mean:  {results['mean_error_mm']:7.2f} mm")

    # Load comparison metrics if available
    smpl_result_path = Path(args.smpl_result)
    meta_file = smpl_result_path.parent / 'meta.json'

    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

        mpjpe = meta.get('metrics', {}).get('MPJPE', None)

        if mpjpe is not None:
            print(f"\n" + "="*80)
            print("COMPARISON WITH OTHER METRICS")
            print("="*80)
            print(f"  MPJPE (overall):         {mpjpe:7.2f} mm")
            print(f"  Toe Position Error:      {results['mean_error_mm']:7.2f} mm")
            print(f"  Ratio (Toe/MPJPE):       {results['mean_error_mm']/mpjpe:7.2%}")

    # Save results
    output_file = output_dir / 'toe_position_metrics.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved toe position metrics to {output_file}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nInterpretation:")
    print(f"  - Toe position error measures 3D distance between SMPL and AddB toe joints")
    print(f"  - This metric focuses on POSITIONAL accuracy rather than ORIENTATION")
    print(f"  - Lower values indicate better alignment of toe positions")
    print(f"  - This sidesteps the SMPL foot orientation limitation")


if __name__ == '__main__':
    main()
