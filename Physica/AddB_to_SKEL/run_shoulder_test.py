#!/usr/bin/env python3
"""
Shoulder Correction Test - Using existing compare_smpl_skel.py as base

This script runs the existing SKEL optimization and adds shoulder correction losses.
"""

import os
import sys
import json
from datetime import datetime

import numpy as np
import torch

# Add paths
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

# Import from existing code
from compare_smpl_skel import (
    load_b3d_data,
    optimize_skel,
    create_joint_spheres,
    create_skeleton_bones,
    save_obj,
    compute_mpjpe,
    SKEL_MODEL_PATH,
)
from models.skel_model import (
    SKELModelWrapper,
    SKEL_NUM_BETAS,
    SKEL_NUM_POSE_DOF,
    AUTO_JOINT_NAME_MAP_SKEL,
    SKEL_JOINT_NAMES,
)

# Import shoulder correction
from shoulder_correction import (
    ACROMIAL_VERTEX_IDX,
    compute_virtual_acromial,
)


def test_shoulder_correction():
    """
    Test shoulder correction by comparing baseline SKEL with AddB acromial positions.
    """
    # Settings
    b3d_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject1/Subject1.b3d'
    num_frames = 10
    num_iters = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Using device: {device}")
    print(f"Loading data from: {b3d_path}")

    # Load data
    target_joints, joint_names, addb_parents = load_b3d_data(b3d_path, num_frames)
    print(f"Loaded {len(target_joints)} frames with {len(joint_names)} joints")
    print(f"Joint names: {joint_names}")

    # Get acromial indices
    acromial_r_idx = joint_names.index('acromial_r') if 'acromial_r' in joint_names else None
    acromial_l_idx = joint_names.index('acromial_l') if 'acromial_l' in joint_names else None
    print(f"Acromial indices: R={acromial_r_idx}, L={acromial_l_idx}")

    # Run baseline SKEL optimization (existing code)
    print("\n" + "="*60)
    print("Running baseline SKEL optimization...")
    print("="*60)
    result = optimize_skel(target_joints, joint_names, device, num_iters)

    # Compute shoulder metrics
    print("\n" + "="*60)
    print("Analyzing shoulder alignment...")
    print("="*60)

    # Load SKEL model for vertex access
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    T = len(target_joints)
    target = torch.tensor(target_joints, dtype=torch.float32, device=device)

    # Get optimized parameters
    betas = torch.tensor(result['betas'], device=device)
    poses = torch.tensor(result['poses'], device=device)
    trans = torch.tensor(result['trans'], device=device)

    # Forward pass to get vertices
    with torch.no_grad():
        verts, joints = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans
        )

        # Compute virtual acromial from vertices
        virtual_acr_r, virtual_acr_l = compute_virtual_acromial(verts)

        # Get AddB acromial positions
        addb_acr_r = target[:, acromial_r_idx, :]
        addb_acr_l = target[:, acromial_l_idx, :]

        # Compute errors
        acr_err_r = torch.norm(virtual_acr_r - addb_acr_r, dim=-1).mean().item() * 1000
        acr_err_l = torch.norm(virtual_acr_l - addb_acr_l, dim=-1).mean().item() * 1000

        # Shoulder widths
        skel_width = torch.norm(virtual_acr_r - virtual_acr_l, dim=-1).mean().item() * 1000
        addb_width = torch.norm(addb_acr_r - addb_acr_l, dim=-1).mean().item() * 1000

        # Get humerus positions (current mapping target)
        humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
        humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
        skel_humerus_r = joints[:, humerus_r_idx, :]
        skel_humerus_l = joints[:, humerus_l_idx, :]

        humerus_err_r = torch.norm(skel_humerus_r - addb_acr_r, dim=-1).mean().item() * 1000
        humerus_err_l = torch.norm(skel_humerus_l - addb_acr_l, dim=-1).mean().item() * 1000
        humerus_width = torch.norm(skel_humerus_r - skel_humerus_l, dim=-1).mean().item() * 1000

    print("\n=== Shoulder Alignment Results ===")
    print(f"AddB shoulder width:        {addb_width:.1f} mm")
    print()
    print("Current (humerus → acromial mapping):")
    print(f"  SKEL humerus width:       {humerus_width:.1f} mm (diff: {abs(humerus_width - addb_width):.1f} mm)")
    print(f"  Humerus→Acromial error:   R={humerus_err_r:.1f} mm, L={humerus_err_l:.1f} mm")
    print()
    print("Virtual acromial (vertex-based):")
    print(f"  SKEL virtual width:       {skel_width:.1f} mm (diff: {abs(skel_width - addb_width):.1f} mm)")
    print(f"  Virtual→Acromial error:   R={acr_err_r:.1f} mm, L={acr_err_l:.1f} mm")
    print()
    print(f"Vertex indices used:")
    print(f"  Right: {ACROMIAL_VERTEX_IDX['right']}")
    print(f"  Left:  {ACROMIAL_VERTEX_IDX['left']}")

    # Check positions frame by frame
    print("\n=== Per-frame Analysis (first 3 frames) ===")
    for t in range(min(3, T)):
        print(f"\nFrame {t}:")
        print(f"  AddB acromial R: {addb_acr_r[t].cpu().numpy()}")
        print(f"  SKEL humerus R:  {skel_humerus_r[t].cpu().numpy()}")
        print(f"  SKEL virtual R:  {virtual_acr_r[t].cpu().numpy()}")
        print(f"  Error humerus:   {torch.norm(skel_humerus_r[t] - addb_acr_r[t]).item()*1000:.1f} mm")
        print(f"  Error virtual:   {torch.norm(virtual_acr_r[t] - addb_acr_r[t]).item()*1000:.1f} mm")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = f'/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/shoulder_analysis_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    metrics = {
        'baseline_mpjpe_mm': result['metrics']['mpjpe_mm'] if 'metrics' in result else None,
        'addb_shoulder_width_mm': addb_width,
        'skel_humerus_width_mm': humerus_width,
        'skel_virtual_width_mm': skel_width,
        'humerus_error_r_mm': humerus_err_r,
        'humerus_error_l_mm': humerus_err_l,
        'virtual_error_r_mm': acr_err_r,
        'virtual_error_l_mm': acr_err_l,
        'vertex_indices_r': ACROMIAL_VERTEX_IDX['right'],
        'vertex_indices_l': ACROMIAL_VERTEX_IDX['left'],
    }

    with open(os.path.join(out_dir, 'shoulder_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nResults saved to: {out_dir}")

    return metrics


if __name__ == '__main__':
    test_shoulder_correction()
