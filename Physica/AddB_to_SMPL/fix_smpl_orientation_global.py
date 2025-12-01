#!/usr/bin/env python3
"""
Fix SMPL orientation with global rotation

Simpler approach: Apply a global -90° rotation around X-axis to convert
from Z-up (standing) to Y-up (standing) coordinate system.

Usage:
    python fix_smpl_orientation_global.py \
        --input <path_to_smpl_params.npz> \
        --output <path_to_fixed_smpl_params.npz>
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def apply_global_rotation(poses, trans, angle_degrees=-90, axis='x'):
    """
    Apply a global rotation to align SMPL from Z-up to Y-up.

    Args:
        poses: (T, 24, 3) or (T, 72) SMPL poses
        trans: (T, 3) SMPL translations
        angle_degrees: Rotation angle in degrees (default: -90)
        axis: Rotation axis ('x', 'y', or 'z')

    Returns:
        poses_fixed, trans_fixed
    """
    T = poses.shape[0]

    # Create global rotation
    if axis == 'x':
        global_rot = R.from_euler('x', angle_degrees, degrees=True)
    elif axis == 'y':
        global_rot = R.from_euler('y', angle_degrees, degrees=True)
    else:
        global_rot = R.from_euler('z', angle_degrees, degrees=True)

    global_rot_vec = global_rot.as_rotvec()
    print(f"\nGlobal rotation: {angle_degrees}° around {axis.upper()}-axis")
    print(f"  Rotation vector: {global_rot_vec}")
    print(f"  Magnitude: {np.linalg.norm(global_rot_vec):.4f} rad ({np.degrees(np.linalg.norm(global_rot_vec)):.2f}°)")

    # Handle different pose shapes
    if len(poses.shape) == 3:
        # (T, 24, 3)
        poses_fixed = poses.copy()

        # Apply global rotation to ALL joint orientations
        for t in range(T):
            for j in range(24):
                joint_rot = R.from_rotvec(poses[t, j, :])
                # Compose rotations: new_joint = global_rot * old_joint
                new_joint_rot = global_rot * joint_rot
                poses_fixed[t, j, :] = new_joint_rot.as_rotvec()

    else:
        # (T, 72) - flattened
        poses_reshaped = poses.reshape(T, 24, 3)
        poses_fixed = poses_reshaped.copy()

        for t in range(T):
            for j in range(24):
                joint_rot = R.from_rotvec(poses_reshaped[t, j, :])
                new_joint_rot = global_rot * joint_rot
                poses_fixed[t, j, :] = new_joint_rot.as_rotvec()

        poses_fixed = poses_fixed.reshape(T, 72)

    # Apply global rotation to translations
    trans_fixed = trans.copy()
    for t in range(T):
        trans_fixed[t] = global_rot.apply(trans[t])

    return poses_fixed, trans_fixed


def main():
    parser = argparse.ArgumentParser(description='Fix SMPL orientation with global rotation')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input smpl_params.npz')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output fixed smpl_params.npz')
    parser.add_argument('--angle', type=float, default=-90,
                       help='Rotation angle in degrees (default: -90)')
    parser.add_argument('--axis', type=str, default='x', choices=['x', 'y', 'z'],
                       help='Rotation axis (default: x)')
    args = parser.parse_args()

    print(f"Loading SMPL parameters from {args.input}...")
    data = np.load(args.input)

    betas = data['betas']
    poses = data['poses']
    trans = data['trans']

    T = poses.shape[0]
    print(f"  Loaded {T} frames")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Check original values
    if len(poses.shape) == 3:
        root_rot_before = poses[0, 0, :]
    else:
        root_rot_before = poses[0, :3]

    trans_before = trans[0]

    print(f"\nOriginal values (frame 0):")
    print(f"  Root rotation: {root_rot_before}")
    print(f"  Root rotation (degrees): {np.degrees(root_rot_before)}")
    print(f"  Translation: {trans_before}")

    # Apply global rotation
    poses_fixed, trans_fixed = apply_global_rotation(poses, trans, args.angle, args.axis)

    # Check fixed values
    if len(poses_fixed.shape) == 3:
        root_rot_after = poses_fixed[0, 0, :]
    else:
        root_rot_after = poses_fixed[0, :3]

    trans_after = trans_fixed[0]

    print(f"\nFixed values (frame 0):")
    print(f"  Root rotation: {root_rot_after}")
    print(f"  Root rotation (degrees): {np.degrees(root_rot_after)}")
    print(f"  Translation: {trans_after}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        betas=betas,
        poses=poses_fixed,
        trans=trans_fixed
    )

    print(f"\n✓ Saved fixed SMPL parameters to {args.output}")
    print(f"\nChanges:")
    print(f"  Root rotation change: {np.linalg.norm(root_rot_after - root_rot_before):.4f} rad")
    print(f"  Translation change: {np.linalg.norm(trans_after - trans_before):.4f}")


if __name__ == '__main__':
    main()
