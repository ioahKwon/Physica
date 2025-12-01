#!/usr/bin/env python3
"""
Compare foot orientation between AddBiomechanics and SMPL fitting results.

This script helps diagnose whether the "foot pointing backward" issue is:
1. A real problem with SMPL fitting
2. A visualization artifact

Usage:
    python compare_foot_orientation.py --b3d <path> --result_dir <path>
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import nimblephysics as nimble

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def load_addbiomechanics_data(b3d_path):
    """Load AddBiomechanics data from .b3d file."""
    print(f"Loading AddBiomechanics data from {b3d_path}...")
    subject = nimble.biomechanics.SubjectOnDisk(b3d_path)

    # Get first trial
    trial_idx = 0
    if subject.getNumTrials() == 0:
        raise ValueError("No trials found in B3D file")

    frames = subject.readFrames(
        trial=trial_idx,
        startFrame=0,
        numFramesToRead=subject.getTrialLength(trial_idx),
        stride=1,
        contactThreshold=0.01
    )

    num_frames = len(frames)
    print(f"  Loaded {num_frames} frames")

    # Extract joint positions from frames
    # Handle processing passes if they exist
    def frame_joint_centers(frame):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            return np.asarray(frame.processingPasses[0].jointCenters, dtype=np.float32)
        return np.asarray(frame.jointCenters, dtype=np.float32)

    first = frame_joint_centers(frames[0])
    num_dofs = first.size
    num_joints = num_dofs // 3

    joint_positions = np.zeros((num_frames, num_joints, 3))

    for t, frame in enumerate(frames):
        joint_centers = frame_joint_centers(frame)
        joint_positions[t] = joint_centers.reshape(-1, 3)

    # Get joint names from skeleton
    skel = subject.readSkel(processingPass=0)
    joint_names = []
    for i in range(skel.getNumBodyNodes()):
        body = skel.getBodyNode(i)
        joint_names.append(body.getName())

    return joint_positions, joint_names


def find_joint_idx(joint_names, target_names):
    """Find joint index by matching against multiple possible names."""
    for name in target_names:
        for i, jname in enumerate(joint_names):
            if name.lower() in jname.lower():
                return i
    return None


def compute_addb_foot_orientation(joint_positions, joint_names):
    """
    Compute foot orientation from AddBiomechanics data.

    Returns:
        left_foot_dirs: (T, 3) - normalized foot direction vectors
        right_foot_dirs: (T, 3) - normalized foot direction vectors
    """
    # Find ankle and foot joints
    # In AddB: talus = ankle, calcn = heel/foot
    left_ankle_idx = find_joint_idx(joint_names, ['talus_l', 'ankle_l', 'left_ankle', 'l_ankle'])
    right_ankle_idx = find_joint_idx(joint_names, ['talus_r', 'ankle_r', 'right_ankle', 'r_ankle'])
    left_foot_idx = find_joint_idx(joint_names, ['calcn_l', 'left_foot', 'l_foot', 'foot_l'])
    right_foot_idx = find_joint_idx(joint_names, ['calcn_r', 'right_foot', 'r_foot', 'foot_r'])

    if any(idx is None for idx in [left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx]):
        print(f"  Found indices: L_ankle={left_ankle_idx}, R_ankle={right_ankle_idx}, L_foot={left_foot_idx}, R_foot={right_foot_idx}")
        raise ValueError("Could not find all required joints in AddB data")

    print(f"  AddB joints: L_ankle={joint_names[left_ankle_idx]}, R_ankle={joint_names[right_ankle_idx]}")
    print(f"               L_foot={joint_names[left_foot_idx]}, R_foot={joint_names[right_foot_idx]}")

    # Compute foot direction vectors (ankle -> foot)
    left_ankle_pos = joint_positions[:, left_ankle_idx, :]
    right_ankle_pos = joint_positions[:, right_ankle_idx, :]
    left_foot_pos = joint_positions[:, left_foot_idx, :]
    right_foot_pos = joint_positions[:, right_foot_idx, :]

    left_foot_dirs = left_foot_pos - left_ankle_pos
    right_foot_dirs = right_foot_pos - right_ankle_pos

    # Normalize
    left_foot_dirs = left_foot_dirs / (np.linalg.norm(left_foot_dirs, axis=1, keepdims=True) + 1e-8)
    right_foot_dirs = right_foot_dirs / (np.linalg.norm(right_foot_dirs, axis=1, keepdims=True) + 1e-8)

    return left_foot_dirs, right_foot_dirs


def compute_smpl_foot_orientation(smpl_model, betas, poses, trans):
    """
    Compute foot orientation from SMPL fitting results.

    The foot's forward direction is the local X-axis after rotation.

    Returns:
        left_foot_dirs: (T, 3) - normalized foot forward direction vectors
        right_foot_dirs: (T, 3) - normalized foot forward direction vectors
    """
    T = poses.shape[0]
    device = betas.device

    LEFT_ANKLE_SMPL = 7
    RIGHT_ANKLE_SMPL = 8
    LEFT_FOOT_SMPL = 10
    RIGHT_FOOT_SMPL = 11

    left_foot_dirs = []
    right_foot_dirs = []

    for t in range(T):
        with torch.no_grad():
            # Get SMPL joints
            vertices, joints = smpl_model.forward(
                betas=betas,
                poses=poses[t],
                trans=trans[t]
            )

        joints_np = joints.cpu().numpy()  # (24, 3)
        poses_np = poses[t].cpu().numpy()  # (24, 3)

        # Get foot rotation matrices
        left_foot_rot = R.from_rotvec(poses_np[LEFT_FOOT_SMPL]).as_matrix()
        right_foot_rot = R.from_rotvec(poses_np[RIGHT_FOOT_SMPL]).as_matrix()

        # Foot forward direction = rotated X-axis (pointing forward in local frame)
        # In SMPL, the foot's default orientation has X-axis pointing downward from ankle
        # So we want the direction from ankle to foot position
        left_ankle_pos = joints_np[LEFT_ANKLE_SMPL]
        right_ankle_pos = joints_np[RIGHT_ANKLE_SMPL]
        left_foot_pos = joints_np[LEFT_FOOT_SMPL]
        right_foot_pos = joints_np[RIGHT_FOOT_SMPL]

        # Compute anatomical foot direction (ankle -> foot)
        left_foot_dir = left_foot_pos - left_ankle_pos
        right_foot_dir = right_foot_pos - right_ankle_pos

        # Normalize
        left_foot_dir = left_foot_dir / (np.linalg.norm(left_foot_dir) + 1e-8)
        right_foot_dir = right_foot_dir / (np.linalg.norm(right_foot_dir) + 1e-8)

        left_foot_dirs.append(left_foot_dir)
        right_foot_dirs.append(right_foot_dir)

    return np.array(left_foot_dirs), np.array(right_foot_dirs)


def compute_walking_direction(joint_positions):
    """
    Estimate walking direction from pelvis movement.

    Returns:
        walk_dir: (3,) - normalized walking direction in XY plane
    """
    # Use pelvis or first joint as reference
    pelvis_trajectory = joint_positions[:, 0, :]  # (T, 3)

    # Compute overall movement direction
    start_pos = pelvis_trajectory[0]
    end_pos = pelvis_trajectory[-1]
    walk_dir = end_pos - start_pos

    # Project to XY plane (assume Z is up)
    walk_dir[2] = 0
    walk_dir = walk_dir / (np.linalg.norm(walk_dir) + 1e-8)

    return walk_dir


def compute_angle_difference(vec1, vec2):
    """
    Compute angle difference between two vectors in degrees.

    Args:
        vec1: (T, 3) or (3,)
        vec2: (T, 3) or (3,)

    Returns:
        angles: (T,) or scalar - angle in degrees
    """
    # Ensure arrays are at least 2D
    if vec1.ndim == 1:
        vec1 = vec1[np.newaxis, :]
    if vec2.ndim == 1:
        vec2 = vec2[np.newaxis, :]

    # Normalize
    vec1_norm = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + 1e-8)
    vec2_norm = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + 1e-8)

    # Compute dot product
    dot_product = np.sum(vec1_norm * vec2_norm, axis=1)

    # Clip to valid range for arccos
    dot_product = np.clip(dot_product, -1.0, 1.0)

    # Compute angle
    angles = np.arccos(dot_product) * 180.0 / np.pi

    return angles if len(angles) > 1 else angles[0]


def plot_comparison(addb_left, addb_right, smpl_left, smpl_right, output_path):
    """
    Plot comparison of foot orientations over time.
    """
    T = len(addb_left)
    frames = np.arange(T)

    # Compute angles with respect to forward direction (Z-axis)
    forward = np.array([0, 1, 0])  # Y-axis is forward

    addb_left_angles = compute_angle_difference(addb_left, np.tile(forward, (T, 1)))
    addb_right_angles = compute_angle_difference(addb_right, np.tile(forward, (T, 1)))
    smpl_left_angles = compute_angle_difference(smpl_left, np.tile(forward, (T, 1)))
    smpl_right_angles = compute_angle_difference(smpl_right, np.tile(forward, (T, 1)))

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Left foot
    ax = axes[0]
    ax.plot(frames, addb_left_angles, 'b-', linewidth=2, label='AddBiomechanics', alpha=0.7)
    ax.plot(frames, smpl_left_angles, 'r--', linewidth=2, label='SMPL Fitting', alpha=0.7)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Angle from Forward (degrees)', fontsize=12)
    ax.set_title('Left Foot Orientation: AddB vs SMPL', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Right foot
    ax = axes[1]
    ax.plot(frames, addb_right_angles, 'b-', linewidth=2, label='AddBiomechanics', alpha=0.7)
    ax.plot(frames, smpl_right_angles, 'r--', linewidth=2, label='SMPL Fitting', alpha=0.7)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Angle from Forward (degrees)', fontsize=12)
    ax.set_title('Right Foot Orientation: AddB vs SMPL', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved comparison plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare AddB and SMPL foot orientations')
    parser.add_argument('--b3d', type=str, required=True,
                       help='Path to AddBiomechanics .b3d file')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Path to SMPL fitting result directory')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--frames', type=int, nargs='+', default=[0, 50, 100, 150],
                       help='Frames to analyze in detail')
    args = parser.parse_args()

    result_path = Path(args.result_dir)
    smpl_params_file = result_path / 'smpl_params.npz'

    if not smpl_params_file.exists():
        print(f"ERROR: {smpl_params_file} does not exist")
        sys.exit(1)

    device = torch.device('cpu')

    # Load AddBiomechanics data
    addb_joints, addb_joint_names = load_addbiomechanics_data(args.b3d)
    addb_left_foot, addb_right_foot = compute_addb_foot_orientation(addb_joints, addb_joint_names)

    # Load SMPL data
    print("\nLoading SMPL fitting results...")
    smpl = SMPLModel(args.smpl_model, device=device)
    data = np.load(smpl_params_file)
    betas = torch.from_numpy(data['betas']).float().to(device)
    poses = torch.from_numpy(data['poses']).float().to(device)
    trans = torch.from_numpy(data['trans']).float().to(device)

    T = min(poses.shape[0], len(addb_joints))
    print(f"  Loaded {T} frames")

    # Compute SMPL foot orientation
    print("\nComputing SMPL foot orientations...")
    smpl_left_foot, smpl_right_foot = compute_smpl_foot_orientation(
        smpl, betas, poses[:T], trans[:T]
    )

    # Compute differences
    print("\n" + "="*80)
    print("FOOT ORIENTATION COMPARISON: AddBiomechanics vs SMPL")
    print("="*80)

    left_diff_angles = compute_angle_difference(addb_left_foot[:T], smpl_left_foot)
    right_diff_angles = compute_angle_difference(addb_right_foot[:T], smpl_right_foot)

    print(f"\nOverall Statistics:")
    print(f"  Left Foot Angular Difference:")
    print(f"    Mean:   {np.mean(left_diff_angles):.2f}°")
    print(f"    Median: {np.median(left_diff_angles):.2f}°")
    print(f"    Std:    {np.std(left_diff_angles):.2f}°")
    print(f"    Max:    {np.max(left_diff_angles):.2f}°")

    print(f"\n  Right Foot Angular Difference:")
    print(f"    Mean:   {np.mean(right_diff_angles):.2f}°")
    print(f"    Median: {np.median(right_diff_angles):.2f}°")
    print(f"    Std:    {np.std(right_diff_angles):.2f}°")
    print(f"    Max:    {np.max(right_diff_angles):.2f}°")

    # Detailed frame analysis
    print(f"\nDetailed Frame Analysis:")
    for frame_idx in args.frames:
        if frame_idx >= T:
            continue

        print(f"\n--- Frame {frame_idx} ---")
        print(f"  Left Foot:")
        print(f"    AddB direction:  [{addb_left_foot[frame_idx, 0]:+.3f}, {addb_left_foot[frame_idx, 1]:+.3f}, {addb_left_foot[frame_idx, 2]:+.3f}]")
        print(f"    SMPL direction:  [{smpl_left_foot[frame_idx, 0]:+.3f}, {smpl_left_foot[frame_idx, 1]:+.3f}, {smpl_left_foot[frame_idx, 2]:+.3f}]")
        print(f"    Angular diff:    {left_diff_angles[frame_idx]:.2f}°")

        print(f"  Right Foot:")
        print(f"    AddB direction:  [{addb_right_foot[frame_idx, 0]:+.3f}, {addb_right_foot[frame_idx, 1]:+.3f}, {addb_right_foot[frame_idx, 2]:+.3f}]")
        print(f"    SMPL direction:  [{smpl_right_foot[frame_idx, 0]:+.3f}, {smpl_right_foot[frame_idx, 1]:+.3f}, {smpl_right_foot[frame_idx, 2]:+.3f}]")
        print(f"    Angular diff:    {right_diff_angles[frame_idx]:.2f}°")

    # Generate plot
    output_plot = result_path / 'foot_orientation_comparison.png'
    plot_comparison(
        addb_left_foot[:T], addb_right_foot[:T],
        smpl_left_foot, smpl_right_foot,
        output_plot
    )

    # Final verdict
    print("\n" + "="*80)
    print("VERDICT:")
    print("="*80)
    mean_diff = (np.mean(left_diff_angles) + np.mean(right_diff_angles)) / 2

    if mean_diff < 15:
        print(f"✓ Foot orientations are SIMILAR (mean diff: {mean_diff:.2f}°)")
        print("  → The issue is likely a VISUALIZATION problem")
        print("  → SMPL mesh orientation may need adjustment in rendering")
    elif mean_diff < 45:
        print(f"⚠ Foot orientations have MODERATE differences (mean diff: {mean_diff:.2f}°)")
        print("  → Some misalignment exists")
        print("  → May need better foot rotation optimization")
    else:
        print(f"✗ Foot orientations are VERY DIFFERENT (mean diff: {mean_diff:.2f}°)")
        print("  → This is a REAL PROBLEM with SMPL fitting")
        print("  → Need Option C: Direct foot orientation supervision")

    print("="*80)


if __name__ == '__main__':
    main()
