#!/usr/bin/env python3
"""
Test Artificial Shoulder Widening for SKEL

Purpose: Artificially widen SKEL shoulders by manipulating vertices
- Move shoulder vertices outward (X direction)
- Test different widening scales
- Compare with SMPL

Usage:
    python test_widen_shoulders.py
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

def widen_shoulders(vertices: np.ndarray, shoulder_joint_positions: np.ndarray,
                    scale_factor: float = 1.2, y_threshold: float = 0.2) -> np.ndarray:
    """
    Widen shoulder area by pushing vertices outward in X direction

    Args:
        vertices: [N, 3] vertex positions
        shoulder_joint_positions: [2, 3] right and left shoulder positions
        scale_factor: How much to widen (1.0 = no change, 1.2 = 20% wider)
        y_threshold: Vertical threshold from shoulder height to affect

    Returns:
        Modified vertices
    """
    verts = vertices.copy()

    # Average shoulder height
    shoulder_y = (shoulder_joint_positions[0, 1] + shoulder_joint_positions[1, 1]) / 2

    # Find vertices in shoulder region (by Y coordinate)
    # Typically shoulders are upper body, so we threshold by Y
    shoulder_mask = np.abs(verts[:, 1] - shoulder_y) < y_threshold

    # Also check if vertex is in upper body (Y > some threshold)
    upper_body_mask = verts[:, 1] > (shoulder_y - y_threshold)

    # Combine masks
    affected_mask = shoulder_mask & upper_body_mask

    # For affected vertices, scale X coordinate from center
    center_x = 0.0  # Assume body is centered at X=0

    for i in np.where(affected_mask)[0]:
        x_offset = verts[i, 0] - center_x
        # Scale outward
        verts[i, 0] = center_x + x_offset * scale_factor

    return verts

def main():
    device = torch.device('cpu')
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_widen_shoulders'
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
    print("Artificial Shoulder Widening Test")
    print("=" * 70)

    # Initialize models
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    smpl = SMPLModel(gender='male', device=device)

    # Generate SKEL mesh
    print("\n[1] Original SKEL...")
    skel_verts, skel_joints = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_poses[frame:frame+1],
        trans=skel_trans[frame:frame+1]
    )

    skel_verts_np = skel_verts[0].detach().cpu().numpy()
    skel_joints_np = skel_joints[0].detach().cpu().numpy()

    save_obj(skel_verts_np, skel.faces,
             os.path.join(out_dir, '1_skel_original.obj'))

    # Joint indices for SKEL (from SKELModelWrapper)
    # 14: scapula_r, 15: humerus_r (right shoulder)
    # 19: scapula_l, 20: humerus_l (left shoulder)
    right_shoulder_joint = skel_joints_np[15]  # humerus_r
    left_shoulder_joint = skel_joints_np[20]   # humerus_l

    shoulder_joints = np.array([right_shoulder_joint, left_shoulder_joint])

    print(f"  Right shoulder joint position: {right_shoulder_joint}")
    print(f"  Left shoulder joint position:  {left_shoulder_joint}")
    print(f"  Shoulder width: {np.linalg.norm(right_shoulder_joint - left_shoulder_joint):.3f}m")

    # Test different widening scales
    scales = [1.1, 1.2, 1.3, 1.5]
    y_threshold = 0.3  # 30cm vertical threshold from shoulder height

    for scale in scales:
        print(f"\n[Scale {scale:.1f}] Widening SKEL shoulders by {(scale-1)*100:.0f}%...")

        widened_verts = widen_shoulders(
            skel_verts_np,
            shoulder_joints,
            scale_factor=scale,
            y_threshold=y_threshold
        )

        save_obj(widened_verts, skel.faces,
                 os.path.join(out_dir, f'2_skel_widened_{scale:.1f}x.obj'))

    # SMPL reference
    print("\n[Reference] SMPL...")
    smpl_verts, smpl_joints = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_poses[frame:frame+1],
        trans=smpl_trans[frame:frame+1]
    )

    smpl_verts_np = smpl_verts[0].detach().cpu().numpy()
    smpl_joints_np = smpl_joints[0].detach().cpu().numpy()

    save_obj(smpl_verts_np, smpl.faces.cpu().numpy(),
             os.path.join(out_dir, '3_smpl_reference.obj'))

    # SMPL shoulder width for comparison
    smpl_right_shoulder = smpl_joints_np[17]  # right shoulder
    smpl_left_shoulder = smpl_joints_np[16]   # left shoulder
    print(f"  SMPL Right shoulder joint: {smpl_right_shoulder}")
    print(f"  SMPL Left shoulder joint:  {smpl_left_shoulder}")
    print(f"  SMPL Shoulder width: {np.linalg.norm(smpl_right_shoulder - smpl_left_shoulder):.3f}m")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nFiles generated:")
    print("  1_skel_original.obj        ← Original SKEL (narrow shoulders)")
    print("  2_skel_widened_1.1x.obj    ← SKEL widened by 10%")
    print("  2_skel_widened_1.2x.obj    ← SKEL widened by 20%")
    print("  2_skel_widened_1.3x.obj    ← SKEL widened by 30%")
    print("  2_skel_widened_1.5x.obj    ← SKEL widened by 50%")
    print("  3_smpl_reference.obj       ← SMPL reference")
    print("\nCompare to find which scale matches SMPL shoulder width best!")

if __name__ == '__main__':
    main()
