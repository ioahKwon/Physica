#!/usr/bin/env python3
"""
Test Moving Shoulder Joints Outward

Purpose: Modify SKEL model to have wider shoulder joints
- Load SKEL model
- Modify J_regressor to move shoulder joints outward
- Generate mesh with wider shoulders

Usage:
    python test_move_shoulder_joints.py
"""

import os
import sys
import pickle
import numpy as np
import torch
import copy

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

class ModifiedSKEL(SKELModelWrapper):
    """SKEL with modified shoulder joint positions"""

    def __init__(self, *args, shoulder_offset=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.shoulder_offset = shoulder_offset

        # Modify J_regressor to move shoulder joints outward
        if shoulder_offset != 0.0:
            self._modify_joint_regressor()

    def _modify_joint_regressor(self):
        """Modify J_regressor to move shoulder joints laterally"""
        # J_regressor shape: [24, 6890]
        # We want to shift the weighted average for shoulder joints

        # Get joint names and find shoulder indices
        joint_names = ['pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee',
                       'right_knee', 'spine2', 'left_ankle', 'right_ankle', 'spine3',
                       'left_foot', 'right_foot', 'neck', 'left_collar', 'right_collar',
                       'head', 'left_shoulder', 'right_shoulder', 'left_elbow',
                       'right_elbow', 'left_wrist', 'right_wrist']

        left_shoulder_idx = 16
        right_shoulder_idx = 17

        print(f"\nModifying J_regressor:")
        print(f"  Shoulder offset: {self.shoulder_offset}m")
        print(f"  Left shoulder joint index: {left_shoulder_idx}")
        print(f"  Right shoulder joint index: {right_shoulder_idx}")

        # For SKEL, we need to modify the base template vertices
        # that are used to compute joint positions
        # Instead, we'll directly add offset to computed joints in forward pass

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_move_joints'
    os.makedirs(out_dir, exist_ok=True)

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
    print("Moving Shoulder Joints Outward Test")
    print("=" * 70)

    # Test 1: Original SKEL
    print("\n[1] Original SKEL...")
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    verts_np = skel_verts_orig[0].detach().cpu().numpy()
    joints_np = skel_joints_orig[0].detach().cpu().numpy()

    save_obj(verts_np, skel.faces, os.path.join(out_dir, '1_skel_original.obj'))

    right_shoulder = joints_np[15]  # humerus_r in SKEL wrapper
    left_shoulder = joints_np[20]   # humerus_l in SKEL wrapper
    orig_width = np.linalg.norm(right_shoulder - left_shoulder)

    print(f"  Right shoulder joint [15]: {right_shoulder}")
    print(f"  Left shoulder joint [20]: {left_shoulder}")
    print(f"  Shoulder width: {orig_width:.4f}m")

    # Test 2-5: Move shoulder joints with different offsets
    # We'll manually add offset to joint positions after forward pass
    offsets = [0.01, 0.02, 0.03, 0.05]

    for offset in offsets:
        print(f"\n[Offset {offset:.2f}m]")

        # Forward pass
        verts, joints = skel.forward(
            betas=skel_beta.unsqueeze(0),
            poses=skel_poses[frame:frame+1],
            trans=skel_trans[frame:frame+1]
        )

        verts_np = verts[0].detach().cpu().numpy()
        joints_np = joints[0].detach().cpu().numpy()

        # Manually shift shoulder joints
        # Right shoulder (index 15): move in negative X (left/outward)
        joints_np[15, 0] -= offset

        # Left shoulder (index 20): move in positive X (right/outward)
        joints_np[20, 0] += offset

        # Also need to shift related vertices
        # Use skinning weights to determine which vertices to move
        skel_pkl = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1/skel_male.pkl'
        with open(skel_pkl, 'rb') as f:
            skel_data = pickle.load(f, encoding='latin1')

        weights = skel_data['skin_weights'].toarray()
        weights_r = weights[:, 17]  # right_shoulder in pkl
        weights_l = weights[:, 16]  # left_shoulder in pkl

        # Move vertices proportional to their weights
        for v_idx in range(len(verts_np)):
            w_r = weights_r[v_idx]
            w_l = weights_l[v_idx]

            if w_r > 0.01:
                verts_np[v_idx, 0] -= offset * w_r

            if w_l > 0.01:
                verts_np[v_idx, 0] += offset * w_l

        save_obj(verts_np, skel.faces,
                 os.path.join(out_dir, f'2_skel_offset_{offset:.2f}m.obj'))

        new_width = np.linalg.norm(joints_np[15] - joints_np[20])
        width_increase = (new_width - orig_width) / orig_width * 100

        print(f"  New shoulder width: {new_width:.4f}m ({width_increase:+.1f}%)")

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
    print(f"  Target offset needed: ~{(smpl_width - orig_width)/2:.3f}m per side")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")

if __name__ == '__main__':
    main()
