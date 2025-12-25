#!/usr/bin/env python3
"""
Test T-pose Visualization

Purpose: Test if pose (not beta) is causing shoulder issues
- Use T-pose (all pose parameters = 0)
- Use optimized beta
- Compare with SMPL T-pose

Usage:
    python test_tpose_visualization.py
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
    out_dir = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_tpose'
    os.makedirs(out_dir, exist_ok=True)

    # Load optimized betas
    skel_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/skel/skel_params.npz')
    smpl_params = np.load('/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/final_Subject1/smpl/smpl_params.npz')

    skel_beta = torch.from_numpy(skel_params['betas']).float().to(device)
    smpl_beta = torch.from_numpy(smpl_params['betas']).float().to(device)

    print("=" * 70)
    print("T-pose Visualization Test")
    print("=" * 70)
    print(f"\nUsing optimized betas, but T-pose (pose = zeros)")
    print(f"SKEL beta: {skel_beta.numpy()}")
    print(f"SMPL beta: {smpl_beta.numpy()}")

    # SKEL T-pose
    print("\n[1] SKEL with T-pose...")
    skel = SKELModelWrapper(
        model_path='/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1',
        gender='male',
        device=device,
        add_acromial_joints=False
    )

    skel_tpose = torch.zeros(46, device=device)  # T-pose
    skel_trans = torch.zeros(3, device=device)

    skel_verts, skel_joints = skel.forward(
        betas=skel_beta.unsqueeze(0),
        poses=skel_tpose.unsqueeze(0),
        trans=skel_trans.unsqueeze(0)
    )

    save_obj(skel_verts[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_tpose_optimized_beta.obj'))

    print(f"  Saved: skel_tpose_optimized_beta.obj")

    # SMPL T-pose
    print("\n[2] SMPL with T-pose...")
    smpl = SMPLModel(gender='male', device=device)

    smpl_tpose = torch.zeros(72, device=device)  # T-pose (24*3)
    smpl_trans = torch.zeros(3, device=device)

    smpl_verts, smpl_joints = smpl.forward(
        betas=smpl_beta.unsqueeze(0),
        poses=smpl_tpose.unsqueeze(0),
        trans=smpl_trans.unsqueeze(0)
    )

    save_obj(smpl_verts[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, 'smpl_tpose_optimized_beta.obj'))

    print(f"  Saved: smpl_tpose_optimized_beta.obj")

    # Also save with default beta for comparison
    print("\n[3] SKEL T-pose with default beta...")
    skel_verts_default, _ = skel.forward(
        betas=torch.zeros(10, device=device).unsqueeze(0),
        poses=skel_tpose.unsqueeze(0),
        trans=skel_trans.unsqueeze(0)
    )

    save_obj(skel_verts_default[0].detach().cpu().numpy(), skel.faces,
             os.path.join(out_dir, 'skel_tpose_default_beta.obj'))

    print(f"  Saved: skel_tpose_default_beta.obj")

    print("\n[4] SMPL T-pose with default beta...")
    smpl_verts_default, _ = smpl.forward(
        betas=torch.zeros(10, device=device).unsqueeze(0),
        poses=smpl_tpose.unsqueeze(0),
        trans=smpl_trans.unsqueeze(0)
    )

    save_obj(smpl_verts_default[0].detach().cpu().numpy(), smpl.faces.cpu().numpy(),
             os.path.join(out_dir, 'smpl_tpose_default_beta.obj'))

    print(f"  Saved: smpl_tpose_default_beta.obj")

    print("\n" + "=" * 70)
    print("Complete! Files saved to:")
    print(f"  {out_dir}")
    print("\nCompare:")
    print("  1. skel_tpose_optimized_beta.obj  ← SKEL T-pose with optimized beta")
    print("  2. smpl_tpose_optimized_beta.obj  ← SMPL T-pose with optimized beta")
    print("  3. skel_tpose_default_beta.obj    ← SKEL T-pose with default beta")
    print("  4. smpl_tpose_default_beta.obj    ← SMPL T-pose with default beta")
    print("\nIf SKEL shoulders look good in T-pose → POSE was the problem!")
    print("If SKEL shoulders still wrong in T-pose → BETA or MODEL problem")

if __name__ == '__main__':
    main()
