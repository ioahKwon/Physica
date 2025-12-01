#!/usr/bin/env python3
"""
Shoulder Correction Experiment Script

This script runs a SKEL optimization with the new shoulder correction losses
and compares against the baseline (without correction).

Usage:
    python run_experiment.py --b3d <path> --out_dir <path>
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# Add parent directory to path for imports
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

# Import from existing codebase
from models.smpl_model import SMPLModel, SMPL_NUM_BETAS, SMPL_NUM_JOINTS
from models.skel_model import (
    SKELModelWrapper,
    SKEL_NUM_BETAS,
    SKEL_NUM_JOINTS,
    SKEL_NUM_POSE_DOF,
    AUTO_JOINT_NAME_MAP_SKEL,
    SKEL_JOINT_NAMES
)

# Import shoulder correction
from shoulder_correction import (
    ACROMIAL_VERTEX_IDX,
    ADDB_JOINT_IDX,
    compute_virtual_acromial,
    loss_acromial,
    loss_shoulder_width,
    loss_shoulder_direction,
    compute_shoulder_losses,
)

# Import utilities from compare_smpl_skel
from compare_smpl_skel import (
    load_b3d_data,
    build_joint_mapping,
    build_bone_pair_mapping,
    build_bone_length_pairs,
    compute_bone_direction_loss,
    compute_bone_length_loss,
    compute_mpjpe,
    create_joint_spheres,
    create_skeleton_bones,
    save_obj,
    SKEL_BONE_PAIRS,
    SKEL_ADDB_BONE_LENGTH_PAIRS,
)

try:
    import nimblephysics as nimble
except ImportError as exc:
    raise RuntimeError("nimblephysics required. Install: pip install nimblephysics") from exc


# Constants
SMPL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/models/SMPL_NEUTRAL.pkl'
SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'


def optimize_skel_with_shoulder_correction(
    target_joints: np.ndarray,
    addb_joint_names: List[str],
    device: torch.device,
    num_iters: int = 200,
    lambda_acromial: float = 1.0,
    lambda_width: float = 1.0,
    lambda_direction: float = 0.5,
) -> Dict:
    """
    Optimize SKEL with shoulder correction losses.
    """
    print("\n=== SKEL Optimization with Shoulder Correction ===")
    print(f"  lambda_acromial={lambda_acromial}, lambda_width={lambda_width}, lambda_direction={lambda_direction}")

    # Load model
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    # Build joint mapping
    addb_indices, skel_indices = build_joint_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL
    )
    print(f"  Mapped {len(addb_indices)} joints")

    # Build bone pair mapping
    bone_pairs = build_bone_pair_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL, SKEL_BONE_PAIRS
    )
    bone_length_pairs = build_bone_length_pairs(
        addb_joint_names, SKEL_JOINT_NAMES, SKEL_ADDB_BONE_LENGTH_PAIRS
    )

    T = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    # Get AddB acromial indices
    addb_acromial_r_idx = addb_joint_names.index('acromial_r') if 'acromial_r' in addb_joint_names else None
    addb_acromial_l_idx = addb_joint_names.index('acromial_l') if 'acromial_l' in addb_joint_names else None
    addb_back_idx = addb_joint_names.index('back') if 'back' in addb_joint_names else None

    if addb_acromial_r_idx is None or addb_acromial_l_idx is None:
        print("  WARNING: acromial joints not found in AddB data!")
        lambda_acromial = 0.0
        lambda_width = 0.0
        lambda_direction = 0.0

    print(f"  AddB acromial indices: R={addb_acromial_r_idx}, L={addb_acromial_l_idx}, back={addb_back_idx}")

    # Initialize parameters
    betas = torch.zeros(SKEL_NUM_BETAS, device=device, requires_grad=True)
    poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=device, requires_grad=True)
    trans = torch.zeros(T, 3, device=device, requires_grad=True)

    # Initialize translation from pelvis
    pelvis_idx = addb_joint_names.index('ground_pelvis') if 'ground_pelvis' in addb_joint_names else 0
    with torch.no_grad():
        trans[:] = target[:, pelvis_idx]
    trans.requires_grad = True

    # Optimizer
    optimizer = torch.optim.Adam([
        {'params': [betas], 'lr': 0.05},
        {'params': [poses], 'lr': 0.02},
        {'params': [trans], 'lr': 0.01}
    ])

    # Loss weights
    bone_dir_weight = 0.3
    bone_length_weight = 1.0

    # Joint importance weights
    joint_weights = torch.ones(len(skel_indices), device=device)
    important_skel_joints = ['pelvis', 'femur_r', 'femur_l', 'humerus_r', 'humerus_l', 'thorax']
    for i, skel_idx in enumerate(skel_indices):
        joint_name = SKEL_JOINT_NAMES[skel_idx].lower()
        if joint_name in important_skel_joints:
            joint_weights[i] = 2.0

    # Tracking for logging
    losses_history = {
        'total': [], 'joint': [], 'acromial': [], 'width': [], 'direction': [],
        'bone_dir': [], 'bone_len': [], 'mpjpe': []
    }

    # Optimization loop
    for it in range(num_iters):
        optimizer.zero_grad()

        # Forward pass - get vertices too for shoulder correction
        verts, joints = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans
        )

        pred_joints = joints  # [T, 24, 3]

        # 1. Joint position loss
        pred_subset = pred_joints[:, skel_indices]
        target_subset = target[:, addb_indices]

        valid_mask = ~torch.isnan(target_subset).any(dim=-1)
        if valid_mask.sum() > 0:
            diff = pred_subset - target_subset
            sq_diff = (diff ** 2).sum(dim=-1)
            weighted_sq_diff = sq_diff * joint_weights.unsqueeze(0)
            loss_joint = weighted_sq_diff[valid_mask].mean()
        else:
            loss_joint = torch.tensor(0.0, device=device)

        # 2. Bone direction loss
        bone_dir_loss = compute_bone_direction_loss(pred_joints, target, bone_pairs)

        # 3. Bone length loss
        bone_len_loss = compute_bone_length_loss(pred_joints, target, bone_length_pairs)

        # 4. Shoulder correction losses (NEW!)
        if addb_acromial_r_idx is not None:
            addb_acromial_r = target[:, addb_acromial_r_idx, :]
            addb_acromial_l = target[:, addb_acromial_l_idx, :]
            addb_back = target[:, addb_back_idx, :] if addb_back_idx is not None else pred_joints[:, 12, :]

            # Compute virtual acromial from vertices
            virtual_acr_r, virtual_acr_l = compute_virtual_acromial(verts)

            # Acromial position loss
            loss_acr = loss_acromial(verts, addb_acromial_r, addb_acromial_l)

            # Shoulder width loss
            loss_w = loss_shoulder_width(virtual_acr_r, virtual_acr_l, addb_acromial_r, addb_acromial_l)

            # Shoulder direction loss
            skel_thorax = pred_joints[:, 12, :]  # thorax index
            loss_dir = loss_shoulder_direction(
                skel_thorax, virtual_acr_r, virtual_acr_l,
                addb_back, addb_acromial_r, addb_acromial_l
            )
        else:
            loss_acr = torch.tensor(0.0, device=device)
            loss_w = torch.tensor(0.0, device=device)
            loss_dir = torch.tensor(0.0, device=device)

        # Total loss
        loss = (
            loss_joint
            + bone_dir_weight * bone_dir_loss
            + bone_length_weight * bone_len_loss
            + lambda_acromial * loss_acr
            + lambda_width * loss_w
            + lambda_direction * loss_dir
        )

        # Regularization
        loss = loss + 0.01 * (poses ** 2).mean()
        loss = loss + 0.005 * (betas ** 2).mean()

        # Spine regularization
        spine_dof_indices = list(range(17, 26))
        spine_dofs = poses[:, spine_dof_indices]
        loss = loss + 0.05 * (spine_dofs ** 2).mean()

        # Scapula elevation regularization
        scapula_elev_indices = [27, 37]
        scapula_elev = poses[:, scapula_elev_indices]
        loss = loss + 0.1 * (scapula_elev ** 2).mean()

        loss.backward()
        optimizer.step()

        # Logging
        if (it + 1) % 20 == 0:
            mpjpe = compute_mpjpe(pred_joints.detach().cpu().numpy(),
                                  target_joints, skel_indices, addb_indices)

            # Also compute acromial-specific error
            if addb_acromial_r_idx is not None:
                with torch.no_grad():
                    vr, vl = compute_virtual_acromial(verts)
                    acr_err_r = torch.norm(vr - addb_acromial_r, dim=-1).mean().item() * 1000
                    acr_err_l = torch.norm(vl - addb_acromial_l, dim=-1).mean().item() * 1000
                    shoulder_width_skel = torch.norm(vr - vl, dim=-1).mean().item() * 1000
                    shoulder_width_addb = torch.norm(addb_acromial_r - addb_acromial_l, dim=-1).mean().item() * 1000
            else:
                acr_err_r = acr_err_l = 0.0
                shoulder_width_skel = shoulder_width_addb = 0.0

            print(f"  Iter {it+1}/{num_iters}: MPJPE={mpjpe:.1f}mm | "
                  f"Acr_R={acr_err_r:.1f}mm, Acr_L={acr_err_l:.1f}mm | "
                  f"Width: SKEL={shoulder_width_skel:.1f}mm, AddB={shoulder_width_addb:.1f}mm")

            losses_history['mpjpe'].append(mpjpe)

    # Final forward pass
    with torch.no_grad():
        verts, joints, skel_verts = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans,
            return_skeleton=True
        )

        # Final metrics
        mpjpe_final = compute_mpjpe(joints.cpu().numpy(), target_joints, skel_indices, addb_indices)

        if addb_acromial_r_idx is not None:
            vr, vl = compute_virtual_acromial(verts)
            acr_err_r = torch.norm(vr - target[:, addb_acromial_r_idx], dim=-1).mean().item() * 1000
            acr_err_l = torch.norm(vl - target[:, addb_acromial_l_idx], dim=-1).mean().item() * 1000
            shoulder_width_skel = torch.norm(vr - vl, dim=-1).mean().item() * 1000
            shoulder_width_addb = torch.norm(target[:, addb_acromial_r_idx] - target[:, addb_acromial_l_idx], dim=-1).mean().item() * 1000
        else:
            acr_err_r = acr_err_l = shoulder_width_skel = shoulder_width_addb = 0.0

    print(f"\n  Final Results:")
    print(f"    MPJPE: {mpjpe_final:.2f}mm")
    print(f"    Acromial Error: R={acr_err_r:.2f}mm, L={acr_err_l:.2f}mm")
    print(f"    Shoulder Width: SKEL={shoulder_width_skel:.1f}mm, AddB={shoulder_width_addb:.1f}mm, diff={abs(shoulder_width_skel-shoulder_width_addb):.1f}mm")

    skel_parents = skel.parents.tolist()

    return {
        'betas': betas.detach().cpu().numpy(),
        'poses': poses.detach().cpu().numpy(),
        'trans': trans.detach().cpu().numpy(),
        'vertices': verts.cpu().numpy(),
        'joints': joints.cpu().numpy(),
        'skel_vertices': skel_verts.cpu().numpy(),
        'faces': skel.faces,
        'skel_faces': skel.skel_faces,
        'parents': skel_parents,
        'addb_indices': addb_indices,
        'model_indices': skel_indices,
        'metrics': {
            'mpjpe_mm': mpjpe_final,
            'acromial_r_error_mm': acr_err_r,
            'acromial_l_error_mm': acr_err_l,
            'shoulder_width_skel_mm': shoulder_width_skel,
            'shoulder_width_addb_mm': shoulder_width_addb,
            'shoulder_width_diff_mm': abs(shoulder_width_skel - shoulder_width_addb),
        }
    }


def main():
    parser = argparse.ArgumentParser(description='SKEL optimization with shoulder correction')
    parser.add_argument('--b3d', type=str,
                        default='/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d',
                        help='Path to .b3d file')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_shoulder_correction',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=10, help='Number of frames')
    parser.add_argument('--num_iters', type=int, default=200, help='Optimization iterations')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--lambda_acromial', type=float, default=1.0, help='Acromial loss weight')
    parser.add_argument('--lambda_width', type=float, default=1.0, help='Width loss weight')
    parser.add_argument('--lambda_direction', type=float, default=0.5, help='Direction loss weight')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = args.out_dir + f'_{timestamp}'
    skel_dir = os.path.join(out_dir, 'skel')
    addb_dir = os.path.join(out_dir, 'addb')
    os.makedirs(skel_dir, exist_ok=True)
    os.makedirs(addb_dir, exist_ok=True)

    print(f"Output directory: {out_dir}")

    # Load data
    target_joints, joint_names, addb_parents = load_b3d_data(args.b3d, args.num_frames)
    print(f"Loaded {len(target_joints)} frames with {len(joint_names)} joints")
    print(f"AddB joint names: {joint_names}")

    # Run optimization with shoulder correction
    result = optimize_skel_with_shoulder_correction(
        target_joints, joint_names, device, args.num_iters,
        lambda_acromial=args.lambda_acromial,
        lambda_width=args.lambda_width,
        lambda_direction=args.lambda_direction,
    )

    # Save results
    print("\n=== Saving Results ===")

    # Save AddB joints
    np.save(os.path.join(addb_dir, 'joints.npy'), target_joints)
    for t in range(len(target_joints)):
        verts, faces = create_joint_spheres(target_joints[t], radius=0.02)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(addb_dir, f'joints_frame_{t:04d}.obj'))
        verts, faces = create_skeleton_bones(target_joints[t], addb_parents, radius=0.01)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(addb_dir, f'skeleton_frame_{t:04d}.obj'))

    # Save SKEL
    np.savez(os.path.join(skel_dir, 'skel_params.npz'),
             betas=result['betas'],
             poses=result['poses'],
             trans=result['trans'])
    np.save(os.path.join(skel_dir, 'joints.npy'), result['joints'])

    for t in range(len(result['vertices'])):
        if result['faces'] is not None:
            save_obj(result['vertices'][t], result['faces'],
                     os.path.join(skel_dir, f'mesh_frame_{t:04d}.obj'))
        verts, faces = create_joint_spheres(result['joints'][t], radius=0.02)
        if len(verts) > 0:
            save_obj(verts, faces, os.path.join(skel_dir, f'joints_frame_{t:04d}.obj'))

    # Save metrics
    metrics = {
        'input_file': args.b3d,
        'num_frames': args.num_frames,
        'num_iters': args.num_iters,
        'lambda_acromial': args.lambda_acromial,
        'lambda_width': args.lambda_width,
        'lambda_direction': args.lambda_direction,
        'addb_joint_names': joint_names,
        'skel_metrics': result['metrics'],
        'vertex_indices': {
            'right': ACROMIAL_VERTEX_IDX['right'],
            'left': ACROMIAL_VERTEX_IDX['left'],
        }
    }

    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {out_dir}")
    print(f"\nFinal Metrics:")
    print(f"  MPJPE: {result['metrics']['mpjpe_mm']:.2f}mm")
    print(f"  Acromial Error: R={result['metrics']['acromial_r_error_mm']:.2f}mm, L={result['metrics']['acromial_l_error_mm']:.2f}mm")
    print(f"  Shoulder Width Diff: {result['metrics']['shoulder_width_diff_mm']:.2f}mm")


if __name__ == '__main__':
    main()
