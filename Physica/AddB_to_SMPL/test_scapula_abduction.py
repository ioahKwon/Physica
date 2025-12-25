#!/usr/bin/env python3
"""
Test Scapula Abduction to Widen Shoulders

Purpose: Modify SKEL scapula abduction parameters to widen shoulders
- SKEL pose parameter indices:
  - 26: scapula_abduction_r (right scapula outward movement)
  - 36: scapula_abduction_l (left scapula outward movement)
- Increase abduction = wider shoulders
- Test different abduction angles

Usage:
    python test_scapula_abduction.py
"""

import os
import sys
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
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_scapula_abduction'
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
    print("Scapula Abduction Test (Widen Shoulders via Pose)")
    print("=" * 70)

    # Initialize models
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    smpl = SMPLModel(gender='male', device=device)

    print(f"\nOriginal SKEL scapula parameters (frame {frame}):")
    print(f"  Right scapula [26-28]: {skel_poses[frame, 26:29].numpy()}")
    print(f"    - scapula_abduction_r [26]: {skel_poses[frame, 26].item():.4f}")
    print(f"  Left scapula [36-38]: {skel_poses[frame, 36:39].numpy()}")
    print(f"    - scapula_abduction_l [36]: {skel_poses[frame, 36].item():.4f}")

    # Test 1: Original SKEL
    print("\n[1] Original SKEL...")
    skel_verts_orig, skel_joints_orig = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )
    save_obj(skel_verts_orig[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, '1_skel_original.obj'))

    # Calculate original shoulder width
    right_shoulder = skel_joints_orig[0, 15].cpu().numpy()  # humerus_r
    left_shoulder = skel_joints_orig[0, 20].cpu().numpy()   # humerus_l
    orig_width = np.linalg.norm(right_shoulder - left_shoulder)
    print(f"  Original shoulder width: {orig_width:.4f}m")

    # Test different abduction angles
    # Positive abduction = move scapula away from spine = wider shoulders
    abduction_angles = [0.0, 0.1, 0.2, 0.3, 0.5]  # radians

    for angle in abduction_angles:
        print(f"\n[Abduction {angle:.1f} rad = {np.degrees(angle):.1f}°]")

        skel_pose_modified = skel_poses[frame].clone()

        # Increase scapula abduction bilaterally
        skel_pose_modified[26] = angle  # Right scapula abduction
        skel_pose_modified[36] = angle  # Left scapula abduction

        print(f"  Modified scapula abduction:")
        print(f"    Right [26]: {skel_pose_modified[26].item():.4f}")
        print(f"    Left [36]:  {skel_pose_modified[36].item():.4f}")

        skel_verts_mod, skel_joints_mod = skel.forward(
            betas=skel_beta.unsqueeze(0),
            poses=skel_pose_modified.unsqueeze(0),
            trans=skel_trans[frame:frame+1]
        )

        save_obj(skel_verts_mod[0].detach().cpu().numpy(), skel.faces,
                 os.path.join(out_dir, f'2_skel_abduction_{angle:.1f}rad.obj'))

        # Calculate new shoulder width
        right_shoulder = skel_joints_mod[0, 15].cpu().numpy()
        left_shoulder = skel_joints_mod[0, 20].cpu().numpy()
        new_width = np.linalg.norm(right_shoulder - left_shoulder)
        width_increase = (new_width - orig_width) / orig_width * 100

        print(f"  New shoulder width: {new_width:.4f}m ({width_increase:+.1f}%)")

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

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles generated:")
    print("  1_skel_original.obj              ← Original SKEL")
    print("  2_skel_abduction_0.0rad.obj      ← Zero abduction")
    print("  2_skel_abduction_0.1rad.obj      ← 5.7° abduction")
    print("  2_skel_abduction_0.2rad.obj      ← 11.5° abduction")
    print("  2_skel_abduction_0.3rad.obj      ← 17.2° abduction")
    print("  2_skel_abduction_0.5rad.obj      ← 28.6° abduction")
    print("  3_smpl_reference.obj             ← SMPL reference")
    print("\nFind which abduction angle matches SMPL shoulder width!")

if __name__ == '__main__':
    main()
