#!/usr/bin/env python3
"""
Re-fit SMPL from saved target_joints.npy
This script performs SMPL fitting without needing nimblephysics,
using pre-extracted joint positions.
"""

import torch
import numpy as np
from pathlib import Path
import argparse
import sys
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

# SMPL joint names
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# Joint mapping: AddB index -> SMPL index
# FIX: Remove back->spine2 mapping to prevent forward lean
JOINT_MAPPING = {
    0: 0,    # ground_pelvis -> pelvis
    1: 2,    # hip_r -> right_hip
    2: 5,    # walker_knee_r -> right_knee
    3: 8,    # ankle_r -> right_ankle
    5: 11,   # mtp_r -> right_foot
    6: 1,    # hip_l -> left_hip
    7: 4,    # walker_knee_l -> left_knee
    8: 7,    # ankle_l -> left_ankle
    10: 10,  # mtp_l -> left_foot
    # 11: 6, # back -> spine2  # DISABLED - causes forward lean!
    12: 17,  # acromial_r -> right_shoulder
    13: 19,  # elbow_r -> right_elbow
    15: 21,  # radius_hand_r -> right_wrist
    16: 16,  # acromial_l -> left_shoulder
    17: 18,  # elbow_l -> left_elbow
    19: 20,  # radius_hand_l -> left_wrist
}

# Joint weights: AddB index -> weight (default 1.0)
# Higher weight = more importance in fitting
JOINT_WEIGHTS = {
    0: 1.0,   # ground_pelvis
    1: 1.0,   # hip_r
    2: 1.0,   # walker_knee_r
    3: 1.0,   # ankle_r
    5: 1.0,   # mtp_r
    6: 1.0,   # hip_l
    7: 1.0,   # walker_knee_l
    8: 1.0,   # ankle_l
    10: 1.0,  # mtp_l
    12: 1.0,  # acromial_r
    13: 1.0,  # elbow_r
    15: 1.0,  # radius_hand_r
    16: 1.0,  # acromial_l
    17: 1.0,  # elbow_l
    19: 1.0,  # radius_hand_l
}

# Bone pairs for bone length loss: (AddB parent, AddB child, SMPL parent, SMPL child)
BONE_PAIRS = [
    (0, 1, 0, 2),    # pelvis -> hip_r : pelvis -> right_hip (NEW)
    (0, 6, 0, 1),    # pelvis -> hip_l : pelvis -> left_hip (NEW)
    (1, 2, 2, 5),    # hip_r -> knee_r : right_hip -> right_knee
    (2, 3, 5, 8),    # knee_r -> ankle_r : right_knee -> right_ankle
    (6, 7, 1, 4),    # hip_l -> knee_l : left_hip -> left_knee
    (7, 8, 4, 7),    # knee_l -> ankle_l : left_knee -> left_ankle
    (12, 13, 17, 19),  # shoulder_r -> elbow_r
    (13, 15, 19, 21),  # elbow_r -> wrist_r
    (16, 17, 16, 18),  # shoulder_l -> elbow_l
    (17, 19, 18, 20),  # elbow_l -> wrist_l
]

# Anatomical constraints for joints
# SMPL joint indices: 1=left_hip, 2=right_hip, 4=left_knee, 5=right_knee
# 18=left_elbow, 19=right_elbow
# Knees and elbows should primarily rotate around X-axis (flexion/extension)
# Hips have more freedom but internal rotation (Y-axis) should be limited
KNEE_JOINTS = [4, 5]  # left_knee, right_knee
HIP_JOINTS = [1, 2]   # left_hip, right_hip
ELBOW_JOINTS = [18, 19]  # left_elbow, right_elbow


def fit_smpl(smpl: SMPLModel,
             target_joints: np.ndarray,
             joint_mapping: dict,
             device: torch.device,
             shape_iters: int = 300,
             pose_iters: int = 300,
             joint_weights: dict = None) -> tuple:
    """
    Fit SMPL to target joint positions.

    Args:
        smpl: SMPL model
        target_joints: Target joint positions [T, N, 3]
        joint_mapping: Dict mapping AddB indices to SMPL indices
        device: torch device
        shape_iters: Number of shape optimization iterations
        pose_iters: Number of pose optimization iterations per frame
        joint_weights: Dict mapping AddB indices to weights (default 1.0)

    Returns:
        betas, poses, trans
    """
    T = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    addb_indices = torch.tensor(list(joint_mapping.keys()), dtype=torch.long, device=device)
    smpl_indices = torch.tensor(list(joint_mapping.values()), dtype=torch.long, device=device)

    # Build weight tensor
    if joint_weights is None:
        joint_weights = {k: 1.0 for k in joint_mapping.keys()}
    weights = torch.tensor([joint_weights.get(k, 1.0) for k in joint_mapping.keys()],
                           dtype=torch.float32, device=device)

    # Find root index
    root_addb_idx = None
    for addb_idx, smpl_idx in joint_mapping.items():
        if smpl_idx == 0:
            root_addb_idx = addb_idx
            break

    # Stage 1: Optimize shape using sampled frames
    print("Stage 1: Shape optimization...")
    betas = torch.zeros(10, device=device, requires_grad=True)

    # Sample frames for shape estimation
    sample_indices = np.linspace(0, T-1, min(50, T), dtype=int)

    optimizer = torch.optim.Adam([betas], lr=0.1)
    for it in tqdm(range(shape_iters), desc="Shape"):
        optimizer.zero_grad()

        total_loss = 0.0
        for frame_idx in sample_indices:
            target_frame = target[frame_idx]
            zero_pose = torch.zeros(24, 3, device=device)

            if root_addb_idx is not None:
                trans = target_frame[root_addb_idx].clone()
            else:
                trans = torch.zeros(3, device=device)

            _, pred_joints = smpl.forward(betas, zero_pose, trans)

            target_subset = target_frame[addb_indices]
            pred_subset = pred_joints[smpl_indices]
            mask = ~torch.isnan(target_subset).any(dim=1)

            if mask.sum() > 0:
                diff = pred_subset[mask] - target_subset[mask]
                w = weights[mask]
                # Weighted MSE
                total_loss += ((diff ** 2).sum(dim=1) * w).sum() / w.sum()

            # Bone length loss - critical for correct body size
            bone_loss = 0.0
            for addb_p, addb_c, smpl_p, smpl_c in BONE_PAIRS:
                addb_bone = torch.norm(target_frame[addb_c] - target_frame[addb_p])
                smpl_bone = torch.norm(pred_joints[smpl_c] - pred_joints[smpl_p])
                if not torch.isnan(addb_bone):
                    bone_loss += (smpl_bone - addb_bone) ** 2
            total_loss += 10.0 * bone_loss  # Strong weight on bone length

        total_loss = total_loss / len(sample_indices)
        total_loss += 0.001 * (betas ** 2).sum()  # Regularization

        total_loss.backward()
        optimizer.step()

    betas = betas.detach()
    print(f"  Final betas: {betas.cpu().numpy()}")

    # Stage 2: Optimize pose per frame
    print("Stage 2: Pose optimization...")
    poses = torch.zeros(T, 24, 3, device=device)
    trans = torch.zeros(T, 3, device=device)

    for t in tqdm(range(T), desc="Pose"):
        target_frame = target[t]

        # Initialize translation from pelvis
        if root_addb_idx is not None and not torch.isnan(target_frame[root_addb_idx]).any():
            trans[t] = target_frame[root_addb_idx].clone()

        pose_param = torch.nn.Parameter(poses[t].clone())
        trans_param = torch.nn.Parameter(trans[t].clone())

        optimizer = torch.optim.Adam([pose_param, trans_param], lr=0.02)

        for it in range(pose_iters):
            optimizer.zero_grad()

            _, pred_joints = smpl.forward(betas.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0))
            pred_joints = pred_joints.squeeze(0)

            target_subset = target_frame[addb_indices]
            pred_subset = pred_joints[smpl_indices]
            mask = ~torch.isnan(target_subset).any(dim=1)

            if mask.sum() == 0:
                break

            diff = pred_subset[mask] - target_subset[mask]
            w = weights[mask]
            # Weighted MSE
            loss = ((diff ** 2).sum(dim=1) * w).sum() / w.sum()

            # Pose regularization
            loss += 0.01 * (pose_param ** 2).mean()

            # Anatomical constraints for legs
            # Knees should rotate primarily around X-axis (flexion)
            # Penalize Y and Z axis rotations heavily
            for knee_idx in KNEE_JOINTS:
                knee_rot = pose_param[knee_idx]
                # Penalize non-X rotations (Y and Z components)
                loss += 0.5 * (knee_rot[1] ** 2 + knee_rot[2] ** 2)

            # Hips: penalize excessive internal rotation (Y-axis)
            for hip_idx in HIP_JOINTS:
                hip_rot = pose_param[hip_idx]
                # Penalize Y-axis rotation (internal/external rotation)
                loss += 0.3 * (hip_rot[1] ** 2)
                # Also penalize Z-axis (abduction/adduction) if excessive
                loss += 0.2 * (hip_rot[2] ** 2)

            # Temporal smoothness
            if t > 0:
                pose_diff = pose_param - poses[t-1].detach()
                trans_diff = trans_param - trans[t-1].detach()
                loss += 0.3 * ((pose_diff ** 2).mean() + (trans_diff ** 2).mean())

            loss.backward()
            optimizer.step()

        poses[t] = pose_param.detach()
        trans[t] = trans_param.detach()

    return betas.cpu().numpy(), poses.cpu().numpy(), trans.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Re-fit SMPL from target joints")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing target_joints.npy')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/01_Code/Physica/models/SMPL_NEUTRAL.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--start_frame', type=int, default=0)
    parser.add_argument('--end_frame', type=int, default=None)

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load target joints
    target_path = input_dir / 'target_joints.npy'
    if not target_path.exists():
        print(f"Error: {target_path} not found")
        sys.exit(1)

    target_joints = np.load(target_path)
    print(f"Loaded target joints: {target_joints.shape}")

    # Apply frame range
    if args.end_frame is not None:
        target_joints = target_joints[args.start_frame:args.end_frame]
    elif args.start_frame > 0:
        target_joints = target_joints[args.start_frame:]

    print(f"Using frames: {target_joints.shape[0]}")

    # Load SMPL
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    smpl = SMPLModel(args.smpl_model, device=device)

    # Fit
    print("\n" + "="*60)
    print("SMPL Fitting (without back->spine2 mapping)")
    print("="*60)

    betas, poses, trans = fit_smpl(smpl, target_joints, JOINT_MAPPING, device,
                                    joint_weights=JOINT_WEIGHTS)

    # Save results
    output_path = output_dir / 'smpl_params.npz'
    np.savez(output_path, betas=betas, poses=poses, trans=trans)
    print(f"\nSaved to: {output_path}")

    # Copy target joints for reference
    np.save(output_dir / 'target_joints.npy', target_joints)

    print("\nDone!")


if __name__ == '__main__':
    main()
