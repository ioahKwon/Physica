#!/usr/bin/env python3
"""
Test Large Shoulder Joint Offset

Purpose: Apply much larger offsets to clearly see the effect
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
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_large_offset'
    os.makedirs(out_dir, exist_ok=True)

    # Load SKEL pkl
    skel_pkl = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
    with open(skel_pkl, 'rb') as f:
        skel_data = pickle.load(f, encoding='latin1')

    weights = skel_data['skin_weights'].toarray()

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

    print("=" * 70)
    print("Large Shoulder Offset Test")
    print("=" * 70)

    # Initialize SKEL
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    # Original
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy()
    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_skel_original.obj'))

    joints_np = skel_joints_orig[0].detach().cpu().numpy()
    orig_width = np.linalg.norm(joints_np[15] - joints_np[20])
    print(f"  Shoulder width: {orig_width:.4f}m")

    # Large offsets: 5cm, 10cm, 15cm, 20cm
    offsets = [0.05, 0.10, 0.15, 0.20]

    weights_r = weights[:, 17]  # right_shoulder
    weights_l = weights[:, 16]  # left_shoulder

    for offset in offsets:
        print(f"\n[Offset ±{offset:.2f}m = ±{offset*100:.0f}cm]")

        verts_modified = verts_np.copy()

        # Move ALL vertices with shoulder weights
        for v_idx in range(len(verts_modified)):
            w_r = weights_r[v_idx]
            w_l = weights_l[v_idx]

            # Right shoulder: move outward (negative X)
            if w_r > 0.001:
                verts_modified[v_idx, 0] -= offset * w_r

            # Left shoulder: move outward (positive X)
            if w_l > 0.001:
                verts_modified[v_idx, 0] += offset * w_l

        save_obj(verts_modified, skel.faces,
                 os.path.join(out_dir, f'2_skel_offset_{int(offset*100)}cm.obj'))

        # Calculate approximate new width
        # Find shoulder vertices
        r_verts = verts_modified[weights_r > 0.1]
        l_verts = verts_modified[weights_l > 0.1]

        if len(r_verts) > 0 and len(l_verts) > 0:
            r_center = r_verts.mean(axis=0)
            l_center = l_verts.mean(axis=0)
            new_width = np.linalg.norm(r_center - l_center)
            width_change = (new_width - orig_width) / orig_width * 100
            print(f"  Approx new width: {new_width:.4f}m ({width_change:+.1f}%)")

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl = SMPLModel(gender='male', device=device)
    smpl_verts, smpl_joints = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl_reference.obj'))

    smpl_joints_np = smpl_joints[0].detach().cpu().numpy()
    smpl_width = np.linalg.norm(smpl_joints_np[17] - smpl_joints_np[16])
    print(f"  SMPL shoulder width: {smpl_width:.4f}m")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles:")
    print("  1_skel_original.obj")
    print("  2_skel_offset_5cm.obj   ← 5cm offset")
    print("  2_skel_offset_10cm.obj  ← 10cm offset")
    print("  2_skel_offset_15cm.obj  ← 15cm offset")
    print("  2_skel_offset_20cm.obj  ← 20cm offset")
    print("  3_smpl_reference.obj")

if __name__ == '__main__':
    main()
