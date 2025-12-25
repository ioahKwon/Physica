#!/usr/bin/env python3
"""
Test Collar Joint Widening - The CORRECT Shoulder Widening

Purpose: Widen shoulders by moving collar joints (actual shoulder tops)
NOT shoulder joints (which are actually arm/humerus joints)

Joint naming clarification:
- left_collar (idx 13): 1505 vertices - actual shoulder top
- right_collar (idx 14): 605 vertices - actual shoulder top
- left_shoulder (idx 16): 445 vertices - ARM/humerus (NOT shoulder top!)
- right_shoulder (idx 17): 445 vertices - ARM/humerus (NOT shoulder top!)
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from models.skel_model import SKELModelWrapper
from models.smpl_model import SMPLModel

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_collar_widen'
    os.makedirs(out_dir, exist_ok=True)

    # Load pkl
    skel_pkl = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
    with open(skel_pkl, 'rb') as f:
        skel_data = pickle.load(f, encoding='latin1')

    weights = skel_data['skin_weights'].toarray()

    # Load params
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    skel_beta = torch.from_numpy(skel_params['betas']).float().to(device)
    skel_poses = torch.from_numpy(skel_params['poses']).float().to(device)
    skel_trans = torch.from_numpy(skel_params['trans']).float().to(device)

    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    frame = 0

    print("=" * 70)
    print("Collar Joint Widening - CORRECT Shoulder Widening")
    print("=" * 70)

    # Initialize SKEL
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Get original mesh
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy().copy()
    joints_np = skel_joints_orig[0].detach().cpu().numpy().copy()

    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_original.obj'))

    print(f"  Mesh shape: {verts_np.shape}")

    # Get COLLAR weights from pkl (indices 13=left_collar, 14=right_collar)
    weights_left_collar = weights[:, 13]
    weights_right_collar = weights[:, 14]

    print(f"\nCollar weights info:")
    print(f"  Left collar (idx 13) - vertices with weight > 0.01: {np.sum(weights_left_collar > 0.01)}")
    print(f"  Right collar (idx 14) - vertices with weight > 0.01: {np.sum(weights_right_collar > 0.01)}")

    # Check if collar weights overlap
    overlap = np.sum((weights_left_collar > 0.01) & (weights_right_collar > 0.01))
    print(f"  Overlapping vertices: {overlap}")

    # Check if they're identical
    if overlap > 0:
        overlap_mask = (weights_left_collar > 0.01) & (weights_right_collar > 0.01)
        left_vals = weights_left_collar[overlap_mask]
        right_vals = weights_right_collar[overlap_mask]
        max_diff = np.abs(left_vals - right_vals).max()
        print(f"  Max weight difference in overlap: {max_diff:.6f}")
        if max_diff < 1e-6:
            print(f"  WARNING: Collar weights are IDENTICAL (like shoulder weights were)")

    # Test different offsets - larger values to clearly see effect
    offsets = [0.05, 0.10, 0.15, 0.20]

    for offset in offsets:
        print(f"\n[Offset {offset:.2f}m = {int(offset*100)}cm]")

        # Force deep copy
        verts_modified = np.array(verts_np, copy=True)

        # Count how many vertices we modify
        count_left = 0
        count_right = 0

        # Check if we need X-coordinate splitting (if weights overlap)
        if overlap > 0 and max_diff < 1e-6:
            print(f"  Using X-coordinate splitting (collar weights overlap)")

            # Move vertices based on collar weights AND position
            for v_idx in range(len(verts_modified)):
                w_collar = max(weights_left_collar[v_idx], weights_right_collar[v_idx])

                if w_collar > 0.001:
                    x_pos = verts_modified[v_idx, 0]

                    # If on right side (negative X), move more negative (outward right)
                    if x_pos < 0:
                        verts_modified[v_idx, 0] -= offset * w_collar
                        count_right += 1
                    # If on left side (positive X), move more positive (outward left)
                    else:
                        verts_modified[v_idx, 0] += offset * w_collar
                        count_left += 1
        else:
            print(f"  Using separate weights (collar weights don't overlap)")

            # Move vertices based on separate collar weights
            for v_idx in range(len(verts_modified)):
                w_left = weights_left_collar[v_idx]
                w_right = weights_right_collar[v_idx]

                # Right collar: move outward (negative X)
                if w_right > 0.001:
                    verts_modified[v_idx, 0] -= offset * w_right
                    count_right += 1

                # Left collar: move outward (positive X)
                if w_left > 0.001:
                    verts_modified[v_idx, 0] += offset * w_left
                    count_left += 1

        print(f"  Modified {count_left} left collar vertices")
        print(f"  Modified {count_right} right collar vertices")

        save_obj(verts_modified, skel.faces,
                 os.path.join(out_dir, f'2_collar_offset_{int(offset*100)}cm.obj'))

        # Check if actually different
        diff = np.abs(verts_modified - verts_np).max()
        print(f"  Max vertex displacement: {diff:.6f}m")

        # Check specific collar vertices
        left_verts_idx = np.where(weights_left_collar > 0.01)[0]
        if len(left_verts_idx) > 0:
            v_idx = left_verts_idx[0]
            print(f"  Sample left collar vertex {v_idx}:")
            print(f"    Original: {verts_np[v_idx]}")
            print(f"    Modified: {verts_modified[v_idx]}")
            print(f"    Weight: {weights_left_collar[v_idx]:.4f}")
            print(f"    Displacement: {verts_modified[v_idx] - verts_np[v_idx]}")

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_verts, _ = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl.obj'))

    print("\n" + "=" * 70)
    print("Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles:")
    print("  1_original.obj")
    print("  2_collar_offset_5cm.obj   ← 5cm collar offset")
    print("  2_collar_offset_10cm.obj  ← 10cm collar offset")
    print("  2_collar_offset_15cm.obj  ← 15cm collar offset")
    print("  2_collar_offset_20cm.obj  ← 20cm collar offset")
    print("  3_smpl.obj")
    print("\nNow widening COLLAR joints (actual shoulder tops)")
    print("NOT shoulder joints (which are actually arms)!")

if __name__ == '__main__':
    main()
