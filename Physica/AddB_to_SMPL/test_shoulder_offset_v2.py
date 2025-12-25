#!/usr/bin/env python3
"""
Test Direct Shoulder Vertex Offset (v2 - Simplified)

Purpose: Directly modify shoulder vertices to widen shoulders
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
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_shoulder_offset_v2'
    os.makedirs(out_dir, exist_ok=True)

    # Load SKEL pkl
    skel_pkl_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
    with open(skel_pkl_path, 'rb') as f:
        skel_data = pickle.load(f, encoding='latin1')

    weights = skel_data['skin_weights'].toarray()  # [6890, 24]
    joint_names = skel_data['joints_name']

    print("=" * 70)
    print("Direct Shoulder Vertex Offset (v2)")
    print("=" * 70)
    print(f"\nSkinning weights shape: {weights.shape}")
    print(f"Joint names: {joint_names}")

    # Find shoulder joint indices (in pkl file they're called left/right_shoulder)
    shoulder_r_idx = joint_names.index('right_shoulder')
    shoulder_l_idx = joint_names.index('left_shoulder')
    print(f"\nright_shoulder index: {shoulder_r_idx}")
    print(f"left_shoulder index: {shoulder_l_idx}")

    # Get shoulder weights (1D arrays)
    weights_r = weights[:, shoulder_r_idx]  # [6890]
    weights_l = weights[:, shoulder_l_idx]  # [6890]

    # Find affected vertices
    threshold = 0.05
    verts_r = np.where(weights_r > threshold)[0]
    verts_l = np.where(weights_l > threshold)[0]

    print(f"\nRight shoulder vertices (weight > {threshold}): {len(verts_r)}")
    print(f"Left shoulder vertices (weight > {threshold}): {len(verts_l)}")

    # Load parameters
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    skel_beta = torch.from_numpy(skel_params['betas']).float().to(device)
    skel_poses = torch.from_numpy(skel_params['poses']).float().to(device)
    skel_trans = torch.from_numpy(skel_params['trans']).float().to(device)

    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    frame = 0

    # Initialize models
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    smpl = SMPLModel(gender='male', device=device)

    # Generate original SKEL
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy()
    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_skel_original.obj'))

    right_shoulder = skel_joints_orig[0, 15].cpu().numpy()
    left_shoulder = skel_joints_orig[0, 20].cpu().numpy()
    orig_width = np.linalg.norm(right_shoulder - left_shoulder)
    print(f"  Shoulder width: {orig_width:.4f}m")

    # Test different offsets
    offset_scales = [0.01, 0.02, 0.03, 0.05]

    for offset_m in offset_scales:
        print(f"\n[Offset ±{offset_m:.2f}m]")

        verts_modified = verts_np.copy()

        # Right shoulder: move outward (negative X direction)
        for v_idx in verts_r:
            w = weights_r[v_idx]
            verts_modified[v_idx, 0] -= offset_m * w

        # Left shoulder: move outward (positive X direction)
        for v_idx in verts_l:
            w = weights_l[v_idx]
            verts_modified[v_idx, 0] += offset_m * w

        save_obj(verts_modified, skel.faces,
                 os.path.join(out_dir, f'2_skel_offset_{offset_m:.2f}m.obj'))
        print(f"  Saved: 2_skel_offset_{offset_m:.2f}m.obj")

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl_verts, smpl_joints = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl_reference.obj'))

    smpl_right = smpl_joints[0, 17].cpu().numpy()
    smpl_left = smpl_joints[0, 16].cpu().numpy()
    smpl_width = np.linalg.norm(smpl_right - smpl_left)
    print(f"  SMPL shoulder width: {smpl_width:.4f}m")
    print(f"  Target offset needed: ~{(smpl_width - orig_width)/2:.3f}m per side")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")

if __name__ == '__main__':
    main()
