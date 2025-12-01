#!/usr/bin/env python3
"""
SKEL Optimization with Shoulder Correction

This module provides an enhanced SKEL optimization that includes
shoulder correction loss to better align virtual acromial with AddB acromial.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from datetime import datetime

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

from models.skel_model import (
    SKELModelWrapper,
    SKEL_NUM_BETAS,
    SKEL_NUM_JOINTS,
    SKEL_NUM_POSE_DOF,
    AUTO_JOINT_NAME_MAP_SKEL,
    SKEL_JOINT_NAMES,
)

# Import original functions from compare_smpl_skel
from compare_smpl_skel import (
    build_joint_mapping,
    build_bone_pair_mapping,
    build_bone_length_pairs,
    compute_bone_direction_loss,
    compute_bone_length_loss,
    compute_mpjpe,
    SKEL_BONE_PAIRS,
    SKEL_ADDB_BONE_LENGTH_PAIRS,
    SKEL_MODEL_PATH,
)

# Import shoulder correction
from shoulder_correction import (
    ACROMIAL_VERTEX_IDX,
    compute_virtual_acromial,
    loss_acromial,
    loss_shoulder_width_from_vertices,
)


def optimize_skel_with_shoulder(
    target_joints: np.ndarray,
    addb_joint_names: List[str],
    device: torch.device,
    num_iters: int = 200,
    shoulder_weight: float = 1.0,
    width_weight: float = 0.5,
    exclude_acromial_from_joint_loss: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    Optimize SKEL parameters with shoulder correction loss.

    This version adds virtual acromial matching loss to improve
    shoulder alignment between SKEL and AddB.

    Args:
        target_joints: [T, N, 3] AddB joint positions
        addb_joint_names: List of AddB joint names
        device: torch device
        num_iters: Number of optimization iterations
        shoulder_weight: Weight for acromial position loss
        width_weight: Weight for shoulder width loss
        exclude_acromial_from_joint_loss: If True, remove acromial->humerus from joint loss
            to avoid conflict with shoulder correction loss
        verbose: Print progress

    Returns:
        Dict with optimized parameters, vertices, joints, and metrics
    """
    if verbose:
        print("\n=== SKEL Optimization with Shoulder Correction ===")

    # Load model
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    # Build joint mapping
    addb_indices_full, skel_indices_full = build_joint_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL
    )

    # Get acromial indices from AddB (do this early to filter joint mapping)
    acr_r_idx = addb_joint_names.index('acromial_r') if 'acromial_r' in addb_joint_names else None
    acr_l_idx = addb_joint_names.index('acromial_l') if 'acromial_l' in addb_joint_names else None
    has_acromial = acr_r_idx is not None and acr_l_idx is not None

    # Keep full mapping for MPJPE calculation
    addb_indices = addb_indices_full
    skel_indices = skel_indices_full

    # Optionally exclude acromial from joint LOSS (but keep for MPJPE calculation)
    if exclude_acromial_from_joint_loss and has_acromial and shoulder_weight > 0:
        # Remove acromial->humerus mapping from joint loss
        addb_indices_loss = []
        skel_indices_loss = []
        for ai, si in zip(addb_indices_full, skel_indices_full):
            # Skip acromial joints (they're mapped to humerus)
            if ai == acr_r_idx or ai == acr_l_idx:
                continue
            addb_indices_loss.append(ai)
            skel_indices_loss.append(si)
        if verbose:
            print(f"  Original mapping: {len(addb_indices_full)} joints")
            print(f"  Joint loss uses: {len(addb_indices_loss)} joints (excluding acromial)")
    else:
        addb_indices_loss = addb_indices_full
        skel_indices_loss = skel_indices_full
        if verbose:
            print(f"  Mapped {len(addb_indices_full)} joints")

    # Build bone pair mapping for bone direction loss
    bone_pairs = build_bone_pair_mapping(
        addb_joint_names, SKEL_JOINT_NAMES, AUTO_JOINT_NAME_MAP_SKEL, SKEL_BONE_PAIRS
    )
    if verbose:
        print(f"  Mapped {len(bone_pairs)} bone pairs for direction loss")

    # Build bone length pairs for body shape matching
    bone_length_pairs = build_bone_length_pairs(
        addb_joint_names, SKEL_JOINT_NAMES, SKEL_ADDB_BONE_LENGTH_PAIRS
    )
    if verbose:
        print(f"  Mapped {len(bone_length_pairs)} bone pairs for length loss")

    # Log acromial info
    if has_acromial:
        if verbose:
            print(f"  Acromial indices found: R={acr_r_idx}, L={acr_l_idx}")
            print(f"  Shoulder correction enabled: weight={shoulder_weight}, width_weight={width_weight}")
    else:
        if verbose:
            print("  WARNING: Acromial joints not found, shoulder correction disabled")
        shoulder_weight = 0.0
        width_weight = 0.0

    T = target_joints.shape[0]
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

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

    # Per-joint weights for important joints (use loss indices)
    joint_weights = torch.ones(len(skel_indices_loss), device=device)
    important_skel_joints = ['pelvis', 'femur_r', 'femur_l', 'thorax']

    for i, skel_idx in enumerate(skel_indices_loss):
        joint_name = SKEL_JOINT_NAMES[skel_idx].lower()
        if joint_name in important_skel_joints:
            joint_weights[i] = 2.0

    # Get AddB acromial positions
    if has_acromial:
        addb_acr_r = target[:, acr_r_idx, :]
        addb_acr_l = target[:, acr_l_idx, :]

    # Track losses
    loss_history = {
        'total': [],
        'joint': [],
        'bone_dir': [],
        'bone_len': [],
        'shoulder': [],
        'width': [],
        'mpjpe': [],
    }

    # Optimization loop
    for it in range(num_iters):
        optimizer.zero_grad()

        # Forward - get vertices AND joints
        verts, joints = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans
        )

        pred_joints = joints  # [T, 24, 3]

        # Joint position loss with per-joint weights (using loss-specific indices)
        pred_subset = pred_joints[:, skel_indices_loss]
        target_subset = target[:, addb_indices_loss]

        valid_mask = ~torch.isnan(target_subset).any(dim=-1)
        if valid_mask.sum() > 0:
            diff = pred_subset - target_subset
            sq_diff = (diff ** 2).sum(dim=-1)
            weighted_sq_diff = sq_diff * joint_weights.unsqueeze(0)
            joint_loss = weighted_sq_diff[valid_mask].mean()
        else:
            joint_loss = torch.tensor(0.0, device=device)

        loss = joint_loss

        # Bone direction loss
        bone_dir_loss = compute_bone_direction_loss(pred_joints, target, bone_pairs)
        loss = loss + bone_dir_weight * bone_dir_loss

        # Bone length loss
        bone_len_loss = compute_bone_length_loss(pred_joints, target, bone_length_pairs)
        loss = loss + bone_length_weight * bone_len_loss

        # ===== SHOULDER CORRECTION LOSSES =====
        shoulder_loss = torch.tensor(0.0, device=device)
        width_loss = torch.tensor(0.0, device=device)

        if has_acromial and shoulder_weight > 0:
            # Acromial position loss (virtual acromial → AddB acromial)
            shoulder_loss = loss_acromial(
                verts, addb_acr_r, addb_acr_l,
                v_idx_r=ACROMIAL_VERTEX_IDX['right'],
                v_idx_l=ACROMIAL_VERTEX_IDX['left'],
            )
            loss = loss + shoulder_weight * shoulder_loss

        if has_acromial and width_weight > 0:
            # Shoulder width loss
            width_loss = loss_shoulder_width_from_vertices(
                verts, addb_acr_r, addb_acr_l,
                v_idx_r=ACROMIAL_VERTEX_IDX['right'],
                v_idx_l=ACROMIAL_VERTEX_IDX['left'],
            )
            loss = loss + width_weight * width_loss

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
        if (it + 1) % 20 == 0 or it == 0:
            with torch.no_grad():
                mpjpe = compute_mpjpe(
                    pred_joints.detach().cpu().numpy(),
                    target_joints, skel_indices, addb_indices
                )

            loss_history['total'].append(loss.item())
            loss_history['joint'].append(joint_loss.item())
            loss_history['bone_dir'].append(bone_dir_loss.item())
            loss_history['bone_len'].append(bone_len_loss.item())
            loss_history['shoulder'].append(shoulder_loss.item())
            loss_history['width'].append(width_loss.item())
            loss_history['mpjpe'].append(mpjpe)

            if verbose:
                print(f"  Iter {it+1}/{num_iters}: Loss={loss.item():.4f}, "
                      f"MPJPE={mpjpe:.1f}mm, Shoulder={shoulder_loss.item():.4f}, "
                      f"Width={width_loss.item():.6f}")

    # Final forward pass
    with torch.no_grad():
        verts, joints, skel_verts = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans,
            return_skeleton=True
        )

        # Final MPJPE (full mapping including acromial->humerus for comparison)
        final_mpjpe = compute_mpjpe(
            joints.cpu().numpy(),
            target_joints, skel_indices_full, addb_indices_full
        )

        # MPJPE excluding acromial (for fair comparison when using shoulder correction)
        final_mpjpe_no_acr = compute_mpjpe(
            joints.cpu().numpy(),
            target_joints, skel_indices_loss, addb_indices_loss
        )

        # Compute final shoulder metrics
        if has_acromial:
            virtual_r, virtual_l = compute_virtual_acromial(verts)

            humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
            humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
            humerus_r = joints[:, humerus_r_idx, :]
            humerus_l = joints[:, humerus_l_idx, :]

            # Widths
            addb_width = torch.norm(addb_acr_r - addb_acr_l, dim=-1).mean().item() * 1000
            humerus_width = torch.norm(humerus_r - humerus_l, dim=-1).mean().item() * 1000
            virtual_width = torch.norm(virtual_r - virtual_l, dim=-1).mean().item() * 1000

            # Errors
            humerus_err_r = torch.norm(humerus_r - addb_acr_r, dim=-1).mean().item() * 1000
            humerus_err_l = torch.norm(humerus_l - addb_acr_l, dim=-1).mean().item() * 1000
            virtual_err_r = torch.norm(virtual_r - addb_acr_r, dim=-1).mean().item() * 1000
            virtual_err_l = torch.norm(virtual_l - addb_acr_l, dim=-1).mean().item() * 1000

            shoulder_metrics = {
                'addb_width_mm': addb_width,
                'humerus_width_mm': humerus_width,
                'virtual_width_mm': virtual_width,
                'humerus_error_r_mm': humerus_err_r,
                'humerus_error_l_mm': humerus_err_l,
                'virtual_error_r_mm': virtual_err_r,
                'virtual_error_l_mm': virtual_err_l,
                'humerus_avg_error_mm': (humerus_err_r + humerus_err_l) / 2,
                'virtual_avg_error_mm': (virtual_err_r + virtual_err_l) / 2,
            }
        else:
            shoulder_metrics = {}

    skel_parents = skel.parents.tolist()

    result = {
        'betas': betas.detach().cpu().numpy(),
        'poses': poses.detach().cpu().numpy(),
        'trans': trans.detach().cpu().numpy(),
        'vertices': verts.cpu().numpy(),
        'joints': joints.cpu().numpy(),
        'skel_vertices': skel_verts.cpu().numpy(),
        'faces': skel.faces,
        'skel_faces': skel.skel_faces,
        'parents': skel_parents,
        'addb_indices': addb_indices_full,
        'model_indices': skel_indices_full,
        'metrics': {
            'mpjpe_mm': final_mpjpe,
            'mpjpe_no_acromial_mm': final_mpjpe_no_acr,
            **shoulder_metrics,
        },
        'loss_history': loss_history,
    }

    if verbose:
        print(f"\n  Final MPJPE (with acromial->humerus): {final_mpjpe:.1f}mm")
        print(f"  Final MPJPE (without acromial): {final_mpjpe_no_acr:.1f}mm")
        if has_acromial:
            print(f"  Shoulder Metrics:")
            print(f"    AddB width:     {addb_width:.1f}mm")
            print(f"    Humerus width:  {humerus_width:.1f}mm (error: {(humerus_err_r + humerus_err_l)/2:.1f}mm)")
            print(f"    Virtual width:  {virtual_width:.1f}mm (error: {(virtual_err_r + virtual_err_l)/2:.1f}mm)")

    return result


def test_shoulder_optimization():
    """
    Test the shoulder correction optimization using saved data.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved AddB joints
    output_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_baseline_check'
    addb_joints = np.load(os.path.join(output_dir, 'addb', 'joints.npy'))

    with open(os.path.join(output_dir, 'comparison_metrics.json'), 'r') as f:
        metrics = json.load(f)
    joint_names = metrics['addb_joint_names']

    print(f"Loaded AddB joints: {addb_joints.shape}")
    print(f"Joint names: {joint_names}")

    # Test different weight configurations
    configs = [
        {'name': 'Baseline (no shoulder)', 'shoulder': 0.0, 'width': 0.0, 'iters': 300},
        {'name': 'Shoulder w=0.2', 'shoulder': 0.2, 'width': 0.0, 'iters': 300},
        {'name': 'Shoulder w=0.3', 'shoulder': 0.3, 'width': 0.0, 'iters': 300},
        {'name': 'Shoulder w=0.4', 'shoulder': 0.4, 'width': 0.0, 'iters': 300},
        {'name': 'Shoulder w=0.5', 'shoulder': 0.5, 'width': 0.0, 'iters': 300},
    ]

    results = {}
    for cfg in configs:
        print("\n" + "=" * 60)
        print(f"Testing: {cfg['name']}")
        print("=" * 60)
        result = optimize_skel_with_shoulder(
            addb_joints, joint_names, device,
            num_iters=cfg['iters'],
            shoulder_weight=cfg['shoulder'],
            width_weight=cfg['width'],
        )
        results[cfg['name']] = result

    # Compare results
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"\n{'Config':<35} {'MPJPE':<12} {'MPJPE(no acr)':<15} {'Virtual Err':<12} {'Humerus Err':<12}")
    print("-" * 86)
    for name, result in results.items():
        m = result['metrics']
        print(f"{name:<35} {m['mpjpe_mm']:<12.1f} {m['mpjpe_no_acromial_mm']:<15.1f} "
              f"{m.get('virtual_avg_error_mm', 0):<12.1f} {m.get('humerus_avg_error_mm', 0):<12.1f}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/shoulder_test_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)

    comparison = {name: result['metrics'] for name, result in results.items()}

    with open(os.path.join(save_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nResults saved to: {save_dir}")

    return results


if __name__ == '__main__':
    test_shoulder_optimization()
