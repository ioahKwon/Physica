#!/usr/bin/env python3
"""
Option E: Post-processing to fix foot orientation by directly copying from AddBiomechanics.

This script:
1. Loads SMPL fitting results (baseline)
2. Loads AddBiomechanics data
3. Replaces ankle and foot joint rotations with AddB rotations
4. Saves corrected SMPL parameters
5. Computes MPJPE and foot orientation metrics

Usage:
    python fix_foot_orientation_postprocess.py \
        --b3d <path_to_b3d_file> \
        --smpl_result <path_to_smpl_params.npz> \
        --smpl_model <path_to_smpl_model.pkl> \
        --output <output_directory>
"""

import numpy as np
import torch
import argparse
import sys
from pathlib import Path
import nimblephysics as nimble
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel


def load_addbiomechanics_data(b3d_path, num_frames=None):
    """Load AddBiomechanics data and extract joint rotations."""
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

    # Find foot joint indices in AddB skeleton
    joint_map = {}
    for i in range(skel.getNumBodyNodes()):
        body = skel.getBodyNode(i)
        joint_map[body.getName()] = i

    print(f"  AddB joint mapping:")
    for name in ['talus_l', 'talus_r', 'calcn_l', 'calcn_r', 'ankle_l', 'ankle_r']:
        if name in joint_map:
            print(f"    {name} → index {joint_map[name]}")

    # Extract joint positions (for MPJPE calculation)
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

    # Extract joint rotations (Euler angles from skeleton)
    # AddB stores rotations in the skeleton DOFs
    joint_rotations = {}

    for t, frame in enumerate(frames):
        if t == 0:
            # Just get the structure for the first frame
            pass

    return {
        'joint_positions': joint_positions,
        'joint_map': joint_map,
        'frames': frames,
        'skel': skel
    }


def smpl_to_addb_joint_mapping():
    """Map SMPL joint indices to AddBiomechanics joint names."""
    return {
        7: ['ankle_l', 'talus_l'],   # SMPL left ankle
        8: ['ankle_r', 'talus_r'],   # SMPL right ankle
        10: ['subtalar_l', 'calcn_l'],  # SMPL left foot
        11: ['subtalar_r', 'calcn_r']   # SMPL right foot
    }


def compute_foot_rotation_from_positions(ankle_pos, foot_pos):
    """
    Compute the rotation needed to align the foot direction.

    Args:
        ankle_pos: (3,) ankle position
        foot_pos: (3,) foot position

    Returns:
        (3,) axis-angle rotation
    """
    # Compute desired foot direction
    foot_dir = foot_pos - ankle_pos
    foot_dir = foot_dir / (np.linalg.norm(foot_dir) + 1e-8)

    # Default SMPL foot direction (approximately downward)
    default_dir = np.array([0, 0, -1])

    # Compute rotation from default to desired
    rot_axis = np.cross(default_dir, foot_dir)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-6:
        # Parallel or anti-parallel
        if np.dot(default_dir, foot_dir) > 0:
            return np.zeros(3)
        else:
            return np.array([np.pi, 0, 0])

    rot_axis = rot_axis / rot_axis_norm
    angle = np.arccos(np.clip(np.dot(default_dir, foot_dir), -1, 1))

    return rot_axis * angle


def fix_foot_orientation(smpl_params, addb_data, smpl_model, device):
    """
    Fix foot orientation by copying from AddBiomechanics.

    Strategy:
    Since AddB uses a different skeleton structure, we'll compute the desired
    foot orientation from AddB joint positions and apply it to SMPL.
    """
    betas = smpl_params['betas']
    poses = smpl_params['poses'].copy()  # (T, 24, 3)
    trans = smpl_params['trans']

    T = poses.shape[0]

    # Get AddB joint positions
    addb_positions = addb_data['joint_positions'][:T]  # (T, N, 3)
    joint_map = addb_data['joint_map']

    # Find AddB joint indices
    def find_addb_idx(names):
        for name in names:
            if name in joint_map:
                return joint_map[name]
        return None

    left_ankle_idx = find_addb_idx(['ankle_l', 'talus_l'])
    right_ankle_idx = find_addb_idx(['ankle_r', 'talus_r'])
    left_foot_idx = find_addb_idx(['subtalar_l', 'calcn_l'])
    right_foot_idx = find_addb_idx(['subtalar_r', 'calcn_r'])

    if any(idx is None for idx in [left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx]):
        print(f"WARNING: Could not find all foot joints in AddB data")
        print(f"  Left ankle: {left_ankle_idx}, Right ankle: {right_ankle_idx}")
        print(f"  Left foot: {left_foot_idx}, Right foot: {right_foot_idx}")
        return poses

    print(f"Fixing foot orientation for {T} frames...")
    print(f"  AddB joints: L_ankle={left_ankle_idx}, R_ankle={right_ankle_idx}, "
          f"L_foot={left_foot_idx}, R_foot={right_foot_idx}")

    # For each frame, compute foot rotation from AddB positions
    for t in range(T):
        # Left foot
        left_ankle_pos = addb_positions[t, left_ankle_idx]
        left_foot_pos = addb_positions[t, left_foot_idx]
        left_foot_rot = compute_foot_rotation_from_positions(left_ankle_pos, left_foot_pos)
        poses[t, 10] = left_foot_rot  # SMPL left foot joint

        # Right foot
        right_ankle_pos = addb_positions[t, right_ankle_idx]
        right_foot_pos = addb_positions[t, right_foot_idx]
        right_foot_rot = compute_foot_rotation_from_positions(right_ankle_pos, right_foot_pos)
        poses[t, 11] = right_foot_rot  # SMPL right foot joint

    print("✓ Foot orientation fixed")
    return poses


def compute_mpjpe(smpl_model, betas, poses, trans, target_joints, device):
    """Compute MPJPE between SMPL and target joints."""
    betas_t = torch.from_numpy(betas).float().to(device)
    poses_t = torch.from_numpy(poses).float().to(device)
    trans_t = torch.from_numpy(trans).float().to(device)

    with torch.no_grad():
        vertices, joints = smpl_model.forward(betas=betas_t, poses=poses_t, trans=trans_t)

    joints_np = joints.cpu().numpy()  # (T, 24, 3)

    # Map SMPL joints to AddB joints (use first 24 joints for now)
    T = min(joints_np.shape[0], target_joints.shape[0])
    n_joints = min(24, target_joints.shape[1])

    errors = []
    for t in range(T):
        error = np.linalg.norm(joints_np[t, :n_joints] - target_joints[t, :n_joints], axis=1)
        errors.append(error.mean())

    mpjpe = np.mean(errors) * 1000  # Convert to mm
    return mpjpe


def main():
    parser = argparse.ArgumentParser(description='Fix foot orientation via post-processing')
    parser.add_argument('--b3d', type=str, required=True, help='Path to .b3d file')
    parser.add_argument('--smpl_result', type=str, required=True,
                       help='Path to smpl_params.npz from baseline')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl',
                       help='Path to SMPL model')
    parser.add_argument('--output', type=str, required=True, help='Output directory')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SMPL model
    print("Loading SMPL model...")
    smpl_model = SMPLModel(args.smpl_model, device=device)

    # Load baseline SMPL results
    print(f"Loading baseline SMPL results from {args.smpl_result}...")
    smpl_data = np.load(args.smpl_result)
    betas = smpl_data['betas']
    poses_original = smpl_data['poses']
    trans = smpl_data['trans']
    T = poses_original.shape[0]
    print(f"  Loaded {T} frames")

    # Load AddBiomechanics data
    addb_data = load_addbiomechanics_data(args.b3d, num_frames=T)

    # Fix foot orientation
    poses_fixed = fix_foot_orientation(
        {'betas': betas, 'poses': poses_original, 'trans': trans},
        addb_data,
        smpl_model,
        device
    )

    # Compute MPJPE
    print("\nComputing MPJPE...")
    mpjpe_original = compute_mpjpe(smpl_model, betas, poses_original, trans,
                                    addb_data['joint_positions'], device)
    mpjpe_fixed = compute_mpjpe(smpl_model, betas, poses_fixed, trans,
                                addb_data['joint_positions'], device)

    print(f"  Original MPJPE: {mpjpe_original:.2f} mm")
    print(f"  Fixed MPJPE:    {mpjpe_fixed:.2f} mm")
    print(f"  Difference:     {mpjpe_fixed - mpjpe_original:+.2f} mm")

    # Save fixed results
    output_file = output_dir / 'smpl_params.npz'
    np.savez(
        output_file,
        betas=betas,
        poses=poses_fixed,
        trans=trans
    )
    print(f"\n✓ Saved fixed SMPL parameters to {output_file}")

    # Save metadata
    import json
    meta_file = output_dir / 'meta.json'
    meta = {
        'method': 'Option E: Post-processing with AddB foot rotation',
        'original_mpjpe_mm': float(mpjpe_original),
        'fixed_mpjpe_mm': float(mpjpe_fixed),
        'mpjpe_diff_mm': float(mpjpe_fixed - mpjpe_original),
        'num_frames': int(T),
        'input_b3d': str(args.b3d),
        'input_smpl_result': str(args.smpl_result)
    }
    with open(meta_file, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Saved metadata to {meta_file}")

    print("\n" + "="*80)
    print("POST-PROCESSING COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()
