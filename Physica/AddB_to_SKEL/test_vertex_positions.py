#!/usr/bin/env python3
"""
Quick test: Verify the corrected SKEL vertex indices for acromion.
This script avoids loading AddB data (which causes segfault) and only tests SKEL.
"""

import sys
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

import numpy as np
import torch

from models.skel_model import SKELModelWrapper, SKEL_JOINT_NAMES

SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

# Updated SKEL-specific vertex indices (found via find_shoulder_vertices.py)
SKEL_ACROMIAL_VERTEX_IDX = {
    'left': [635, 636, 1830, 1829],
    'right': [4125, 4124, 5293, 5290],
}

# Wrong SMPLify-X indices (for comparison)
SMPLIFY_ACROMIAL_VERTEX_IDX = {
    'left': [3321, 3325, 3290, 3340],
    'right': [5624, 5630, 5690, 5700],
}


def test_vertex_positions():
    """Test that corrected vertex indices are at shoulder height."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load SKEL model
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    # Get T-pose mesh
    betas = torch.zeros(1, 10, device=device)
    poses = torch.zeros(1, 46, device=device)
    trans = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        verts, joints = skel.forward(betas, poses, trans)

    verts = verts[0].cpu().numpy()  # [6890, 3]
    joints = joints[0].cpu().numpy()  # [24, 3]

    # Get humerus (shoulder joint) positions
    humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
    humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
    humerus_r = joints[humerus_r_idx]
    humerus_l = joints[humerus_l_idx]

    print("\n" + "=" * 60)
    print("SKEL HUMERUS (shoulder joint) POSITIONS")
    print("=" * 60)
    print(f"  humerus_r: {humerus_r} (y={humerus_r[1]*1000:.1f}mm)")
    print(f"  humerus_l: {humerus_l} (y={humerus_l[1]*1000:.1f}mm)")
    print(f"  Width: {np.linalg.norm(humerus_r - humerus_l)*1000:.1f}mm")

    print("\n" + "=" * 60)
    print("CORRECTED SKEL VERTEX INDICES (should be near humerus)")
    print("=" * 60)

    # Test corrected SKEL indices
    for side in ['right', 'left']:
        idx = SKEL_ACROMIAL_VERTEX_IDX[side]
        positions = verts[idx]
        avg = positions.mean(axis=0)
        print(f"\n{side.upper()} acromion vertices: {idx}")
        for i, v_idx in enumerate(idx):
            v = verts[v_idx]
            print(f"  Vertex {v_idx}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
        print(f"  Average: [{avg[0]:.4f}, {avg[1]:.4f}, {avg[2]:.4f}]")
        print(f"  Height (y): {avg[1]*1000:.1f}mm")

    # Virtual acromial positions
    virt_r = verts[SKEL_ACROMIAL_VERTEX_IDX['right']].mean(axis=0)
    virt_l = verts[SKEL_ACROMIAL_VERTEX_IDX['left']].mean(axis=0)
    virt_width = np.linalg.norm(virt_r - virt_l)

    print("\n" + "=" * 60)
    print("WRONG SMPLify-X VERTEX INDICES (for comparison)")
    print("=" * 60)

    # Test wrong SMPLify-X indices
    for side in ['right', 'left']:
        idx = SMPLIFY_ACROMIAL_VERTEX_IDX[side]
        positions = verts[idx]
        avg = positions.mean(axis=0)
        print(f"\n{side.upper()} SMPLify-X vertices: {idx}")
        for i, v_idx in enumerate(idx):
            v = verts[v_idx]
            print(f"  Vertex {v_idx}: [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]")
        print(f"  Average: [{avg[0]:.4f}, {avg[1]:.4f}, {avg[2]:.4f}]")
        print(f"  Height (y): {avg[1]*1000:.1f}mm")

        # Check if position is reasonable (should be ~0.28m height for shoulders)
        if abs(avg[1] - 0.28) > 0.5:
            print(f"  ⚠️  WARNING: y={avg[1]*1000:.1f}mm is FAR from shoulder height (~280mm)")
        else:
            print(f"  ✓ Position is reasonable for shoulder")

    # SMPLify-X virtual acromial (wrong)
    smplify_virt_r = verts[SMPLIFY_ACROMIAL_VERTEX_IDX['right']].mean(axis=0)
    smplify_virt_l = verts[SMPLIFY_ACROMIAL_VERTEX_IDX['left']].mean(axis=0)
    smplify_virt_width = np.linalg.norm(smplify_virt_r - smplify_virt_l)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"\nHumerus width (joints):    {np.linalg.norm(humerus_r - humerus_l)*1000:.1f}mm")
    print(f"Virtual acromial width (CORRECTED):  {virt_width*1000:.1f}mm")
    print(f"Virtual acromial width (SMPLify-X):  {smplify_virt_width*1000:.1f}mm  ⚠️ WRONG")

    print(f"\nCorrected SKEL vertices are at height: R={virt_r[1]*1000:.1f}mm, L={virt_l[1]*1000:.1f}mm")
    print(f"SMPLify-X vertices are at height: R={smplify_virt_r[1]*1000:.1f}mm, L={smplify_virt_l[1]*1000:.1f}mm  ⚠️ WRONG")

    # Validate
    print("\n" + "=" * 60)
    print("VALIDATION")
    print("=" * 60)

    if abs(virt_r[1] - 0.28) < 0.1 and abs(virt_l[1] - 0.28) < 0.1:
        print("✓ CORRECTED indices are at shoulder height (~280mm)")
    else:
        print("⚠️ CORRECTED indices may not be at shoulder height")

    if abs(smplify_virt_r[1] - 0.28) > 0.5 or abs(smplify_virt_l[1] - 0.28) > 0.5:
        print("✓ CONFIRMED: SMPLify-X indices are WRONG for SKEL mesh")

    return {
        'humerus_width_mm': np.linalg.norm(humerus_r - humerus_l)*1000,
        'skel_virtual_width_mm': virt_width*1000,
        'smplify_virtual_width_mm': smplify_virt_width*1000,
        'skel_virt_r': virt_r,
        'skel_virt_l': virt_l,
        'smplify_virt_r': smplify_virt_r,
        'smplify_virt_l': smplify_virt_l,
    }


if __name__ == '__main__':
    test_vertex_positions()
