#!/usr/bin/env python3
"""
Skeleton-based Shoulder Widening

Instead of directly moving vertices, we modify the skeleton joint positions
(scapula/humerus) and let LBS naturally deform the mesh.

SKEL wrapper joint indices:
  14: scapula_r (right shoulder blade)
  15: humerus_r (right upper arm)
  19: scapula_l (left shoulder blade)
  20: humerus_l (left upper arm)
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/SKEL')
sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')

from skel.skel_model import SKEL

def save_obj(vertices: np.ndarray, faces: np.ndarray, filepath: str):
    """Save mesh as OBJ file"""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_skeleton_offset'
    os.makedirs(out_dir, exist_ok=True)

    # Load params
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    betas = torch.from_numpy(skel_params['betas']).float().to(device).unsqueeze(0)
    poses = torch.from_numpy(skel_params['poses'][0:1]).float().to(device)
    trans = torch.from_numpy(skel_params['trans'][0:1]).float().to(device)

    print("=" * 70)
    print("Skeleton-based Shoulder Widening")
    print("=" * 70)

    # Load SKEL model directly (not wrapper) to access internal functions
    skel = SKEL(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male'
    ).to(device)

    print(f"\nSKEL model loaded")
    print(f"  Joints: {skel.num_joints}")

    # Get original output
    output = skel(
        betas=betas,
        poses=poses,
        trans=trans
    )

    verts_orig = output.skin_verts[0].detach().cpu().numpy()
    joints_orig = output.joints[0].detach().cpu().numpy()
    faces = skel.skin_f.cpu().numpy()

    save_obj(verts_orig, faces, os.path.join(out_dir, '1_original.obj'))

    print(f"\nOriginal joint positions (relevant joints):")
    # SKEL uses different joint naming - let's check
    joint_names = [
        'pelvis', 'femur_r', 'tibia_r', 'talus_r', 'calcn_r', 'toes_r',
        'femur_l', 'tibia_l', 'talus_l', 'calcn_l', 'toes_l',
        'lumbar_body', 'thorax', 'head',
        'scapula_r', 'humerus_r', 'ulna_r', 'radius_r', 'hand_r',
        'scapula_l', 'humerus_l', 'ulna_l', 'radius_l', 'hand_l'
    ]

    for idx in [14, 15, 19, 20]:
        print(f"  {idx:2d} ({joint_names[idx]:12s}): {joints_orig[idx]}")

    scapula_r = joints_orig[14]
    scapula_l = joints_orig[19]
    orig_width = np.abs(scapula_l[0] - scapula_r[0])
    print(f"\nOriginal shoulder width (X distance): {orig_width:.4f}m")

    # Check if we can modify J_regressor or template joints
    print("\n" + "=" * 70)
    print("Attempting to modify skeleton...")

    # SKEL uses skin_template_v instead of v_template
    print(f"\nSKEL model attributes:")
    print(f"  skin_template_v shape: {skel.skin_template_v.shape}")
    print(f"  J_regressor shape: {skel.J_regressor.shape}")

    # Get J_regressor for scapula joints (sparse tensor)
    J_reg = skel.J_regressor.to_dense().cpu().numpy()

    # Find which vertices contribute to scapula_r (joint 14) and scapula_l (joint 19)
    scapula_r_weights = J_reg[14]
    scapula_l_weights = J_reg[19]

    scapula_r_verts = np.where(scapula_r_weights > 0.001)[0]
    scapula_l_verts = np.where(scapula_l_weights > 0.001)[0]

    print(f"\nVertices contributing to scapula joints:")
    print(f"  scapula_r (14): {len(scapula_r_verts)} vertices")
    print(f"  scapula_l (19): {len(scapula_l_verts)} vertices")

    if len(scapula_r_verts) > 0:
        print(f"  scapula_r vertex indices: {scapula_r_verts[:5]}...")
    if len(scapula_l_verts) > 0:
        print(f"  scapula_l vertex indices: {scapula_l_verts[:5]}...")

    # Modify skin_template_v to shift shoulder vertices outward
    offsets = [0.02, 0.05, 0.08]

    for offset in offsets:
        print(f"\n[Offset {offset:.2f}m = {int(offset*100)}cm]")

        # Create modified template
        skin_template_modified = skel.skin_template_v.clone()

        # Move scapula_l vertices outward (+X direction for left side)
        for v_idx in scapula_l_verts:
            w = scapula_l_weights[v_idx]
            skin_template_modified[v_idx, 0] += offset * w

        # Move scapula_r vertices outward (-X direction for right side)
        for v_idx in scapula_r_verts:
            w = scapula_r_weights[v_idx]
            skin_template_modified[v_idx, 0] -= offset * w

        # Temporarily replace skin_template_v
        orig_template = skel.skin_template_v.data.clone()
        skel.skin_template_v.data = skin_template_modified

        # Forward pass with modified template
        output_mod = skel(
            betas=betas,
            poses=poses,
            trans=trans
        )

        verts_mod = output_mod.skin_verts[0].detach().cpu().numpy()
        joints_mod = output_mod.joints[0].detach().cpu().numpy()

        # Restore original template
        skel.skin_template_v.data = orig_template

        save_obj(verts_mod, faces, os.path.join(out_dir, f'2_skeleton_offset_{int(offset*100)}cm.obj'))

        # Check new shoulder width
        new_scapula_r = joints_mod[14]
        new_scapula_l = joints_mod[19]
        new_width = np.abs(new_scapula_l[0] - new_scapula_r[0])

        print(f"  New shoulder width: {new_width:.4f}m (was {orig_width:.4f}m)")
        print(f"  Width increase: {(new_width - orig_width)*100:.2f}cm")

    # Save SMPL for reference
    print("\n[Reference] SMPL...")
    sys.path.append('/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
    from models.smpl_model import SMPLModel

    smpl = SMPLModel(gender='male', device=device)
    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)
    smpl_poses = torch.from_numpy(smpl_params['poses']).float().to(device)
    smpl_trans = torch.from_numpy(smpl_params['trans']).float().to(device)

    smpl_verts, _ = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[0:1],
        trans=smpl_trans[0:1]
    )
    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl.obj'))

    print("\n" + "=" * 70)
    print("Files saved to:")
    print(f"  {out_dir}")
    print("\nThis modifies the skeleton (v_template → joint positions)")
    print("and lets LBS naturally deform the mesh.")

if __name__ == '__main__':
    main()
