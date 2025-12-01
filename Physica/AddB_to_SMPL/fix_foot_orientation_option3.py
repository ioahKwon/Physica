#!/usr/bin/env python3
"""
Option 3: Direct axis-angle replacement - Copy foot rotations from AddBiomechanics.

Key insight from DOF analysis:
- AddB subtalar joints are LOCKED (always 0), so we can't use them
- AddB ankle joints (DOF 10=ankle_r, DOF 17=ankle_l) have actual rotation values
- We'll extract ankle rotations from AddB and apply to SMPL foot joints

Strategy:
1. Load baseline SMPL results
2. For each frame, extract AddB ankle DOF values (single-axis rotation)
3. Convert from AddB coordinate system to SMPL coordinate system
4. Apply to SMPL foot joints (indices 10, 11)
5. Save corrected SMPL parameters

Usage:
    python fix_foot_orientation_option3.py \
        --b3d <path_to_b3d_file> \
        --smpl_result <path_to_baseline_smpl_params.npz> \
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
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def find_dof_indices(skel):
    """
    Find DOF indices for ankle joints in AddBiomechanics skeleton.

    Returns:
        (ankle_r_idx, ankle_l_idx) or (None, None) if not found
    """
    print("Identifying DOF indices...")

    dof_idx = 0
    ankle_r_idx = None
    ankle_l_idx = None

    for i in range(skel.getNumBodyNodes()):
        body = skel.getBodyNode(i)
        joint = body.getParentJoint()
        num_dofs = joint.getNumDofs()

        joint_name = joint.getName()

        # ankle_r joint has 1 DOF (RevoluteJoint)
        if joint_name == 'ankle_r':
            ankle_r_idx = dof_idx
            print(f"  Found ankle_r: DOF index {ankle_r_idx}")
        elif joint_name == 'ankle_l':
            ankle_l_idx = dof_idx
            print(f"  Found ankle_l: DOF index {ankle_l_idx}")

        dof_idx += num_dofs

    return ankle_r_idx, ankle_l_idx


def extract_ankle_rotations(b3d_path, num_frames):
    """
    Extract ankle rotation angles from AddBiomechanics data.

    Returns:
        dict with:
            - ankle_r_angles: (T,) array of right ankle rotations in radians
            - ankle_l_angles: (T,) array of left ankle rotations in radians
    """
    print(f"\nLoading AddBiomechanics data from {b3d_path}...")
    subject = nimble.biomechanics.SubjectOnDisk(str(b3d_path))
    skel = subject.readSkel(processingPass=0)

    # Find DOF indices
    ankle_r_idx, ankle_l_idx = find_dof_indices(skel)

    if ankle_r_idx is None or ankle_l_idx is None:
        print("ERROR: Could not find ankle DOF indices")
        return None

    # Read frames
    frames = subject.readFrames(
        trial=0,
        startFrame=0,
        numFramesToRead=num_frames,
        stride=1,
        contactThreshold=0.01
    )

    T = len(frames)
    print(f"  Loaded {T} frames")

    # Extract ankle angles
    ankle_r_angles = np.zeros(T)
    ankle_l_angles = np.zeros(T)

    for t, frame in enumerate(frames):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            pos = frame.processingPasses[0].pos
        else:
            pos = frame.pos

        ankle_r_angles[t] = pos[ankle_r_idx]
        ankle_l_angles[t] = pos[ankle_l_idx]

    print(f"\n  Ankle rotation statistics:")
    print(f"    Right ankle: min={np.degrees(ankle_r_angles.min()):.2f}°, "
          f"max={np.degrees(ankle_r_angles.max()):.2f}°, "
          f"mean={np.degrees(ankle_r_angles.mean()):.2f}°")
    print(f"    Left ankle:  min={np.degrees(ankle_l_angles.min()):.2f}°, "
          f"max={np.degrees(ankle_l_angles.max()):.2f}°, "
          f"mean={np.degrees(ankle_l_angles.mean()):.2f}°")

    return {
        'ankle_r_angles': ankle_r_angles,
        'ankle_l_angles': ankle_l_angles
    }


def ankle_angle_to_smpl_foot_rotation(ankle_angle, side='right'):
    """
    Convert AddB ankle rotation (1D revolute joint) to SMPL foot rotation (3D axis-angle).

    AddB ankle is a RevoluteJoint rotating around a specific axis.
    SMPL foot joint expects 3D axis-angle rotation.

    Coordinate system conversion:
    - AddB: Z-up, ankle rotates around a medio-lateral axis
    - SMPL: Y-up

    Args:
        ankle_angle: scalar rotation in radians (AddB ankle DOF value)
        side: 'right' or 'left'

    Returns:
        (3,) axis-angle rotation for SMPL foot joint
    """
    # AddB ankle joint rotates around the medio-lateral axis (approximately X-axis in AddB)
    # This corresponds to plantar/dorsiflexion

    # In AddB coordinates (Z-up):
    # - Positive rotation = dorsiflexion (toes up)
    # - Negative rotation = plantarflexion (toes down)
    # - Rotation axis is approximately [1, 0, 0] (medio-lateral)

    # Create rotation in AddB coordinates
    # For a RevoluteJoint, the rotation is around the X-axis
    axis_addb = np.array([1.0, 0.0, 0.0])
    rot_addb = axis_addb * ankle_angle

    # Convert to rotation matrix in AddB coordinates
    R_addb = R.from_rotvec(rot_addb)

    # Convert coordinate system from AddB (Z-up) to SMPL (Y-up)
    # AddB: [X, Y, Z] where Z is up
    # SMPL: [X, Y, Z] where Y is up
    # Conversion: [x, y, z]_addb -> [x, z, -y]_smpl

    # This is equivalent to rotating -90° around X-axis
    # R_convert = Rx(-90°)
    R_convert = R.from_euler('x', -90, degrees=True)

    # Apply conversion: R_smpl = R_convert * R_addb * R_convert^T
    R_smpl = R_convert * R_addb * R_convert.inv()

    # Convert to axis-angle
    rot_smpl = R_smpl.as_rotvec()

    return rot_smpl


def apply_foot_rotations(poses, ankle_data):
    """
    Apply AddB ankle rotations to SMPL foot joints.

    Args:
        poses: (T, 24, 3) SMPL pose parameters
        ankle_data: dict from extract_ankle_rotations()

    Returns:
        (T, 24, 3) modified SMPL poses with corrected foot rotations
    """
    poses_fixed = poses.copy()
    T = poses.shape[0]

    ankle_r_angles = ankle_data['ankle_r_angles']
    ankle_l_angles = ankle_data['ankle_l_angles']

    print(f"\nApplying ankle rotations to SMPL foot joints...")

    for t in range(T):
        # Right foot (SMPL joint 11)
        rot_r = ankle_angle_to_smpl_foot_rotation(ankle_r_angles[t], side='right')
        poses_fixed[t, 11] = rot_r

        # Left foot (SMPL joint 10)
        rot_l = ankle_angle_to_smpl_foot_rotation(ankle_l_angles[t], side='left')
        poses_fixed[t, 10] = rot_l

    print(f"  Applied rotations to {T} frames")
    print(f"  Modified SMPL joints: 10 (left_foot), 11 (right_foot)")

    return poses_fixed


def compute_mpjpe(smpl_model, betas, poses, trans, target_joints, device):
    """Compute MPJPE between SMPL and target joints."""
    T = poses.shape[0]

    # Expand betas if needed
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))
    else:
        betas_expanded = betas

    # Reshape poses if needed
    if len(poses.shape) == 3:
        poses_flat = poses.reshape(T, -1)
    else:
        poses_flat = poses

    betas_t = torch.from_numpy(betas_expanded).float().to(device)
    poses_t = torch.from_numpy(poses_flat).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)

    with torch.no_grad():
        vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    joints_np = joints.cpu().numpy()

    # For MPJPE, we need to map SMPL joints to AddB joints
    # This is complex because they have different joint structures
    # For now, just compute average distance across first N joints
    n_joints = min(16, joints_np.shape[1], target_joints.shape[1])

    errors = []
    for t in range(T):
        error = np.linalg.norm(joints_np[t, :n_joints] - target_joints[t, :n_joints], axis=1)
        errors.append(error.mean())

    mpjpe = np.mean(errors) * 1000  # Convert to mm
    return mpjpe


def main():
    parser = argparse.ArgumentParser(description='Option 3: Direct axis-angle replacement')
    parser.add_argument('--b3d', type=str, required=True, help='Path to .b3d file')
    parser.add_argument('--smpl_result', type=str, required=True,
                       help='Path to baseline smpl_params.npz')
    parser.add_argument('--smpl_model', type=str,
                       default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load SMPL model
    print("Loading SMPL model...")
    smpl_model = SMPLModel(args.smpl_model, device=device)

    # Load baseline SMPL results
    print(f"\nLoading baseline SMPL results from {args.smpl_result}...")
    smpl_data = np.load(args.smpl_result)
    betas = smpl_data['betas']
    poses_baseline = smpl_data['poses']
    trans = smpl_data['trans']
    T = poses_baseline.shape[0]
    print(f"  Loaded {T} frames")
    print(f"  Betas shape: {betas.shape}")
    print(f"  Poses shape: {poses_baseline.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Extract ankle rotations from AddB
    ankle_data = extract_ankle_rotations(args.b3d, num_frames=T)

    if ankle_data is None:
        print("\nERROR: Failed to extract ankle rotations from AddBiomechanics")
        sys.exit(1)

    # Apply foot rotations
    poses_fixed = apply_foot_rotations(poses_baseline, ankle_data)

    # Compare foot rotation magnitudes
    print(f"\n" + "="*80)
    print("FOOT ROTATION COMPARISON")
    print("="*80)

    baseline_l_norm = np.linalg.norm(poses_baseline[:, 10, :], axis=1).mean()
    baseline_r_norm = np.linalg.norm(poses_baseline[:, 11, :], axis=1).mean()
    fixed_l_norm = np.linalg.norm(poses_fixed[:, 10, :], axis=1).mean()
    fixed_r_norm = np.linalg.norm(poses_fixed[:, 11, :], axis=1).mean()

    print(f"Left foot rotation magnitude (axis-angle norm):")
    print(f"  Baseline: {np.degrees(baseline_l_norm):.2f}°")
    print(f"  Fixed:    {np.degrees(fixed_l_norm):.2f}°")
    print(f"\nRight foot rotation magnitude (axis-angle norm):")
    print(f"  Baseline: {np.degrees(baseline_r_norm):.2f}°")
    print(f"  Fixed:    {np.degrees(fixed_r_norm):.2f}°")

    # Save fixed results
    output_file = output_dir / 'smpl_params.npz'
    np.savez(
        output_file,
        betas=betas,
        poses=poses_fixed,
        trans=trans
    )
    print(f"\n✓ Saved corrected SMPL parameters to {output_file}")

    # Save metadata
    meta_file = output_dir / 'meta.json'
    meta = {
        'method': 'Option 3: Direct axis-angle replacement from AddB ankle',
        'num_frames': int(T),
        'input_b3d': str(args.b3d),
        'input_smpl_result': str(args.smpl_result),
        'baseline_foot_rot_magnitude': {
            'left': float(np.degrees(baseline_l_norm)),
            'right': float(np.degrees(baseline_r_norm))
        },
        'fixed_foot_rot_magnitude': {
            'left': float(np.degrees(fixed_l_norm)),
            'right': float(np.degrees(fixed_r_norm))
        },
        'ankle_rotation_stats': {
            'right': {
                'min_deg': float(np.degrees(ankle_data['ankle_r_angles'].min())),
                'max_deg': float(np.degrees(ankle_data['ankle_r_angles'].max())),
                'mean_deg': float(np.degrees(ankle_data['ankle_r_angles'].mean()))
            },
            'left': {
                'min_deg': float(np.degrees(ankle_data['ankle_l_angles'].min())),
                'max_deg': float(np.degrees(ankle_data['ankle_l_angles'].max())),
                'mean_deg': float(np.degrees(ankle_data['ankle_l_angles'].mean()))
            }
        }
    }

    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved metadata to {meta_file}")

    print("\n" + "="*80)
    print("OPTION 3 COMPLETE")
    print("="*80)
    print(f"\nNext steps:")
    print(f"1. Run foot orientation evaluation:")
    print(f"   python compare_foot_orientation.py --b3d {args.b3d} \\")
    print(f"       --smpl_result {output_file} --smpl_model {args.smpl_model}")
    print(f"\n2. Create visualization:")
    print(f"   python create_foot_comparison_video.py \\")
    print(f"       --baseline {args.smpl_result} --fixed {output_file} ...")


if __name__ == '__main__':
    main()
