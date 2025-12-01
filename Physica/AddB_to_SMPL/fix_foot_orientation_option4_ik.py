#!/usr/bin/env python3
"""
Option 4: Position-based Inverse Kinematics for foot orientation

Key insight:
- AddB의 mtp (toe) 회전은 잠겨있음 (DOF locked)
- 하지만 mtp의 **위치**는 사용 가능
- talus (ankle) → mtp (toe) 방향 벡터를 계산
- SMPL의 ankle → foot 방향이 이와 일치하도록 foot rotation 계산

Usage:
    python fix_foot_orientation_option4_ik.py \
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


def extract_foot_positions(b3d_path, num_frames):
    """
    Extract ankle and toe positions from AddBiomechanics.

    Returns:
        dict with:
            - talus_l_pos: (T, 3) left ankle positions
            - talus_r_pos: (T, 3) right ankle positions
            - mtp_l_pos: (T, 3) left toe positions
            - mtp_r_pos: (T, 3) right toe positions
    """
    print(f"\nLoading AddBiomechanics data from {b3d_path}...")
    subject = nimble.biomechanics.SubjectOnDisk(str(b3d_path))

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

    # Get skeleton to find joint indices
    skel = subject.readSkel(processingPass=0)

    # Find joint indices
    joint_map = {}
    for i in range(skel.getNumBodyNodes()):
        body = skel.getBodyNode(i)
        joint_map[body.getName()] = i

    # Find the joints we need
    talus_l_idx = joint_map.get('talus_l')
    talus_r_idx = joint_map.get('talus_r')
    mtp_l_idx = joint_map.get('toes_l')  # toes_l is the mtp body
    mtp_r_idx = joint_map.get('toes_r')

    if any(idx is None for idx in [talus_l_idx, talus_r_idx, mtp_l_idx, mtp_r_idx]):
        print(f"ERROR: Could not find all required joints")
        print(f"  talus_l: {talus_l_idx}, talus_r: {talus_r_idx}")
        print(f"  toes_l: {mtp_l_idx}, toes_r: {mtp_r_idx}")
        return None

    print(f"  Found joints:")
    print(f"    Left ankle (talus_l): index {talus_l_idx}")
    print(f"    Right ankle (talus_r): index {talus_r_idx}")
    print(f"    Left toe (toes_l): index {mtp_l_idx}")
    print(f"    Right toe (toes_r): index {mtp_r_idx}")

    # Extract positions
    talus_l_pos = np.zeros((T, 3))
    talus_r_pos = np.zeros((T, 3))
    mtp_l_pos = np.zeros((T, 3))
    mtp_r_pos = np.zeros((T, 3))

    for t, frame in enumerate(frames):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            joint_centers = np.asarray(frame.processingPasses[0].jointCenters, dtype=np.float32)
        else:
            joint_centers = np.asarray(frame.jointCenters, dtype=np.float32)

        joint_positions = joint_centers.reshape(-1, 3)

        talus_l_pos[t] = joint_positions[talus_l_idx]
        talus_r_pos[t] = joint_positions[talus_r_idx]
        mtp_l_pos[t] = joint_positions[mtp_l_idx]
        mtp_r_pos[t] = joint_positions[mtp_r_idx]

    return {
        'talus_l_pos': talus_l_pos,
        'talus_r_pos': talus_r_pos,
        'mtp_l_pos': mtp_l_pos,
        'mtp_r_pos': mtp_r_pos
    }


def convert_addb_to_smpl_coords(pos):
    """
    Convert AddBiomechanics coordinates (Z-up) to SMPL coordinates (Y-up).

    Args:
        pos: (..., 3) array in AddB coordinates

    Returns:
        (..., 3) array in SMPL coordinates
    """
    # AddB: Z-up, SMPL: Y-up
    # Conversion: [x, y, z]_addb -> [x, z, -y]_smpl
    return np.stack([pos[..., 0], pos[..., 2], -pos[..., 1]], axis=-1)


def compute_rotation_from_vectors(v_from, v_to):
    """
    Compute rotation (axis-angle) that aligns v_from to v_to.

    Args:
        v_from: (3,) source vector
        v_to: (3,) target vector

    Returns:
        (3,) axis-angle rotation
    """
    # Normalize vectors
    v_from = v_from / (np.linalg.norm(v_from) + 1e-8)
    v_to = v_to / (np.linalg.norm(v_to) + 1e-8)

    # Compute rotation axis and angle
    rot_axis = np.cross(v_from, v_to)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-6:
        # Parallel or anti-parallel
        if np.dot(v_from, v_to) > 0:
            # Already aligned
            return np.zeros(3)
        else:
            # Anti-parallel, rotate 180 degrees around any perpendicular axis
            # Find a perpendicular vector
            if abs(v_from[0]) < 0.9:
                perp = np.cross(v_from, np.array([1, 0, 0]))
            else:
                perp = np.cross(v_from, np.array([0, 1, 0]))
            perp = perp / np.linalg.norm(perp)
            return perp * np.pi

    rot_axis = rot_axis / rot_axis_norm
    angle = np.arccos(np.clip(np.dot(v_from, v_to), -1, 1))

    return rot_axis * angle


def apply_foot_rotations_ik(poses, foot_data, smpl_model, betas, trans, device):
    """
    Apply foot rotations using IK based on mtp positions.

    Strategy:
    1. Get current SMPL ankle positions (from baseline)
    2. Compute target foot direction: ankle → mtp (from AddB)
    3. Compute SMPL foot rotation to align with target direction

    Args:
        poses: (T, 24, 3) baseline SMPL poses
        foot_data: dict from extract_foot_positions()
        smpl_model: SMPL model
        betas, trans: SMPL shape and translation

    Returns:
        (T, 24, 3) modified poses with corrected foot rotations
    """
    poses_fixed = poses.copy()
    T = poses.shape[0]

    # Convert AddB positions to SMPL coordinate system
    talus_l_smpl = convert_addb_to_smpl_coords(foot_data['talus_l_pos'][:T])
    talus_r_smpl = convert_addb_to_smpl_coords(foot_data['talus_r_pos'][:T])
    mtp_l_smpl = convert_addb_to_smpl_coords(foot_data['mtp_l_pos'][:T])
    mtp_r_smpl = convert_addb_to_smpl_coords(foot_data['mtp_r_pos'][:T])

    print(f"\nComputing foot rotations using position-based IK...")

    # SMPL joint indices
    SMPL_LEFT_ANKLE = 7
    SMPL_RIGHT_ANKLE = 8
    SMPL_LEFT_FOOT = 10
    SMPL_RIGHT_FOOT = 11

    # Generate SMPL joints for baseline to get ankle positions
    if len(betas.shape) == 1:
        betas_expanded = np.tile(betas, (T, 1))
    else:
        betas_expanded = betas[:T]

    poses_flat = poses[:T].reshape(T, -1)
    betas_t = torch.from_numpy(betas_expanded).float().to(device)
    poses_t = torch.from_numpy(poses_flat).float().to(device)
    trans_t = torch.from_numpy(trans[:T]).float().to(device)

    with torch.no_grad():
        _, joints_baseline = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    joints_baseline_np = joints_baseline.cpu().numpy()  # (T, 24, 3)

    # For each frame, compute foot rotation
    for t in range(T):
        # Get SMPL ankle positions (from current baseline)
        smpl_left_ankle = joints_baseline_np[t, SMPL_LEFT_ANKLE]
        smpl_right_ankle = joints_baseline_np[t, SMPL_RIGHT_ANKLE]
        smpl_left_foot = joints_baseline_np[t, SMPL_LEFT_FOOT]
        smpl_right_foot = joints_baseline_np[t, SMPL_RIGHT_FOOT]

        # Compute target foot directions (from AddB)
        # Note: AddB uses global positions, but we need to align directions
        target_left_dir = mtp_l_smpl[t] - talus_l_smpl[t]
        target_right_dir = mtp_r_smpl[t] - talus_r_smpl[t]

        # Compute current SMPL foot directions
        current_left_dir = smpl_left_foot - smpl_left_ankle
        current_right_dir = smpl_right_foot - smpl_right_ankle

        # Compute rotations to align current → target
        left_foot_rot = compute_rotation_from_vectors(current_left_dir, target_left_dir)
        right_foot_rot = compute_rotation_from_vectors(current_right_dir, target_right_dir)

        # Apply to SMPL foot joints
        poses_fixed[t, SMPL_LEFT_FOOT] = left_foot_rot
        poses_fixed[t, SMPL_RIGHT_FOOT] = right_foot_rot

    print(f"  Computed foot rotations for {T} frames")

    # Compute statistics
    left_rot_norms = np.linalg.norm(poses_fixed[:T, SMPL_LEFT_FOOT], axis=1)
    right_rot_norms = np.linalg.norm(poses_fixed[:T, SMPL_RIGHT_FOOT], axis=1)

    print(f"\n  Foot rotation magnitudes:")
    print(f"    Left:  mean={np.degrees(left_rot_norms.mean()):.2f}°, "
          f"max={np.degrees(left_rot_norms.max()):.2f}°")
    print(f"    Right: mean={np.degrees(right_rot_norms.mean()):.2f}°, "
          f"max={np.degrees(right_rot_norms.max()):.2f}°")

    return poses_fixed


def main():
    parser = argparse.ArgumentParser(description='Option 4: Position-based IK for foot orientation')
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

    # Extract foot positions from AddB
    foot_data = extract_foot_positions(args.b3d, num_frames=T)

    if foot_data is None:
        print("\nERROR: Failed to extract foot positions from AddBiomechanics")
        sys.exit(1)

    # Apply foot rotations using IK
    poses_fixed = apply_foot_rotations_ik(
        poses_baseline, foot_data, smpl_model, betas, trans, device
    )

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
    baseline_l_norm = np.linalg.norm(poses_baseline[:, 10, :], axis=1).mean()
    baseline_r_norm = np.linalg.norm(poses_baseline[:, 11, :], axis=1).mean()
    fixed_l_norm = np.linalg.norm(poses_fixed[:, 10, :], axis=1).mean()
    fixed_r_norm = np.linalg.norm(poses_fixed[:, 11, :], axis=1).mean()

    meta = {
        'method': 'Option 4: Position-based IK using AddB mtp (toe) positions',
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
        }
    }

    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved metadata to {meta_file}")

    print("\n" + "="*80)
    print("OPTION 4 COMPLETE")
    print("="*80)
    print(f"\nApproach:")
    print(f"  - Extracted ankle (talus) and toe (mtp) positions from AddB")
    print(f"  - Computed target foot direction: ankle → toe")
    print(f"  - Calculated SMPL foot rotation to match this direction")
    print(f"\nNext steps:")
    print(f"1. Run foot orientation evaluation:")
    print(f"   python compare_foot_orientation.py --b3d {args.b3d} \\")
    print(f"       --result_dir {output_dir} --smpl_model {args.smpl_model}")


if __name__ == '__main__':
    main()
