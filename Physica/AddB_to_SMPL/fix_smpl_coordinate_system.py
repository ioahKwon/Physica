#!/usr/bin/env python3
"""
Fix SMPL coordinate system mismatch

Problem: SMPL results are optimized with positions converted from AddBiomechanics (Z-up)
to SMPL (Y-up), but rotations are NOT converted. This causes meshes to appear in prone
position when visualized.

Solution: Apply proper coordinate frame transformation to all poses, especially root.

Coordinate Systems:
- AddBiomechanics: X-right, Y-forward, Z-up
- SMPL: X-right, Y-up, Z-forward

Transformation:
    [1,  0,  0]
    [0,  0,  1]   (Y_smpl = Z_addb)
    [0, -1,  0]   (Z_smpl = -Y_addb)

Usage:
    python fix_smpl_coordinate_system.py \
        --input <path_to_smpl_params.npz> \
        --output <path_to_fixed_smpl_params.npz>
"""

import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def convert_rotation_addb_to_smpl(rotation_vec_addb):
    """
    Convert rotation from AddBiomechanics (Z-up) to SMPL (Y-up) coordinate system.

    Args:
        rotation_vec_addb: (..., 3) axis-angle rotation in AddB coordinates

    Returns:
        (..., 3) axis-angle rotation in SMPL coordinates
    """
    original_shape = rotation_vec_addb.shape
    rotation_vec_addb = rotation_vec_addb.reshape(-1, 3)

    # Coordinate frame transformation matrix
    R_transform = np.array([
        [1,  0,  0],
        [0,  0,  1],   # Y_smpl = Z_addb
        [0, -1,  0]    # Z_smpl = -Y_addb
    ])

    converted_rots = []
    for rot_vec in rotation_vec_addb:
        # Convert axis-angle to rotation matrix
        r = R.from_rotvec(rot_vec)
        R_addb = r.as_matrix()

        # Apply transformation: R_smpl = R_transform @ R_addb @ R_transform.T
        R_smpl = R_transform @ R_addb @ R_transform.T

        # Convert back to axis-angle
        r_smpl = R.from_matrix(R_smpl)
        converted_rots.append(r_smpl.as_rotvec())

    result = np.array(converted_rots).reshape(original_shape)
    return result


def convert_translation_addb_to_smpl(trans_addb):
    """
    Convert translation from AddBiomechanics (Z-up) to SMPL (Y-up).

    This should already be done in the optimization, but included for completeness.

    Args:
        trans_addb: (..., 3) translation in AddB coordinates

    Returns:
        (..., 3) translation in SMPL coordinates
    """
    return np.stack([
        trans_addb[..., 0],   # X stays X
        trans_addb[..., 2],   # Y_smpl = Z_addb
        -trans_addb[..., 1]   # Z_smpl = -Y_addb
    ], axis=-1)


def fix_smpl_params(input_npz, output_npz, fix_all_poses=False, fix_translation=False):
    """
    Fix SMPL parameters by applying coordinate system transformation.

    Args:
        input_npz: Path to input smpl_params.npz
        output_npz: Path to output fixed smpl_params.npz
        fix_all_poses: If True, apply transformation to all joint poses.
                      If False, only fix root pose (index 0).
        fix_translation: If True, also re-apply translation conversion.
                        Usually not needed as it's already done.
    """
    print(f"Loading SMPL parameters from {input_npz}...")
    data = np.load(input_npz)

    betas = data['betas']
    poses = data['poses']
    trans = data['trans']

    T = poses.shape[0]
    print(f"  Loaded {T} frames")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Check original root orientation
    if len(poses.shape) == 3:
        root_rot_before = poses[0, 0, :]
    else:
        root_rot_before = poses[0, :3]

    print(f"\nOriginal root orientation (frame 0):")
    print(f"  Axis-angle: {root_rot_before}")
    print(f"  Degrees: {np.degrees(root_rot_before)}")
    print(f"  Magnitude: {np.linalg.norm(root_rot_before):.4f} rad ({np.degrees(np.linalg.norm(root_rot_before)):.2f}°)")

    # Fix poses
    poses_fixed = poses.copy()

    if len(poses.shape) == 3:
        # Shape: (T, 24, 3)
        if fix_all_poses:
            print(f"\nApplying coordinate transformation to ALL joint poses...")
            poses_fixed = convert_rotation_addb_to_smpl(poses_fixed)
        else:
            print(f"\nApplying coordinate transformation to ROOT pose only...")
            poses_fixed[:, 0, :] = convert_rotation_addb_to_smpl(poses[:, 0, :])
    else:
        # Shape: (T, 72) - flattened
        print(f"\nReshaping and applying coordinate transformation...")
        poses_reshaped = poses.reshape(T, 24, 3)

        if fix_all_poses:
            poses_reshaped = convert_rotation_addb_to_smpl(poses_reshaped)
        else:
            poses_reshaped[:, 0, :] = convert_rotation_addb_to_smpl(poses_reshaped[:, 0, :])

        poses_fixed = poses_reshaped.reshape(T, 72)

    # Check fixed root orientation
    if len(poses_fixed.shape) == 3:
        root_rot_after = poses_fixed[0, 0, :]
    else:
        root_rot_after = poses_fixed[0, :3]

    print(f"\nFixed root orientation (frame 0):")
    print(f"  Axis-angle: {root_rot_after}")
    print(f"  Degrees: {np.degrees(root_rot_after)}")
    print(f"  Magnitude: {np.linalg.norm(root_rot_after):.4f} rad ({np.degrees(np.linalg.norm(root_rot_after)):.2f}°)")

    # Optionally fix translation (usually not needed)
    trans_fixed = trans
    if fix_translation:
        print(f"\nApplying coordinate transformation to translations...")
        trans_fixed = convert_translation_addb_to_smpl(trans)
        print(f"  Original trans[0]: {trans[0]}")
        print(f"  Fixed trans[0]: {trans_fixed[0]}")

    # Save fixed parameters
    output_path = Path(output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        betas=betas,
        poses=poses_fixed,
        trans=trans_fixed
    )

    print(f"\n✓ Saved fixed SMPL parameters to {output_npz}")
    print(f"\nSummary:")
    print(f"  - Fixed {'all joint poses' if fix_all_poses else 'root pose only'}")
    print(f"  - {'Fixed' if fix_translation else 'Kept original'} translations")
    print(f"  - Root rotation change: {np.linalg.norm(root_rot_after - root_rot_before):.4f} rad")


def main():
    parser = argparse.ArgumentParser(description='Fix SMPL coordinate system')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input smpl_params.npz')
    parser.add_argument('--output', type=str, required=True,
                       help='Path to output fixed smpl_params.npz')
    parser.add_argument('--fix_all_poses', action='store_true',
                       help='Apply transformation to all joint poses (default: root only)')
    parser.add_argument('--fix_translation', action='store_true',
                       help='Re-apply translation conversion (usually not needed)')
    args = parser.parse_args()

    fix_smpl_params(args.input, args.output, args.fix_all_poses, args.fix_translation)


if __name__ == '__main__':
    main()
