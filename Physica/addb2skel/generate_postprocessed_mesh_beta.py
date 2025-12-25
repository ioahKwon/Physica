"""
Generate mesh with post-processed humerus position by optimizing betas.

This script:
1. Loads optimized SKEL parameters
2. Optimizes betas to move humerus onto AddB arm line
3. Forward pass with modified betas to generate correct mesh
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add paths
SKEL_REPO_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/SKEL'
if SKEL_REPO_PATH not in sys.path:
    sys.path.insert(0, SKEL_REPO_PATH)

ADDB2SKEL_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/Physica/addb2skel'
if ADDB2SKEL_PATH not in sys.path:
    sys.path.insert(0, ADDB2SKEL_PATH)

from skel.skel_model import SKEL
from utils.geometry import project_point_onto_line, convert_addb_to_skel_coords
from utils.io import load_b3d
from config import SKEL_MODEL_PATH


def save_mesh_obj(filepath, vertices, faces):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def optimize_betas_for_humerus_target(
    skel_model,
    poses: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    target_humerus_r: torch.Tensor,
    target_humerus_l: torch.Tensor,
    num_iters: int = 200,
    lr: float = 0.01,
):
    """
    Optimize betas to move humerus joint to target position.

    Args:
        skel_model: SKEL model
        poses: Pose parameters [46]
        betas: Original shape parameters [10]
        trans: Translation [3]
        target_humerus_r: Target right humerus position [3]
        target_humerus_l: Target left humerus position [3]
        num_iters: Number of optimization iterations
        lr: Learning rate

    Returns:
        Optimized betas [10]
    """
    device = betas.device

    # Make betas optimizable
    opt_betas = betas.clone().detach().requires_grad_(True)
    original_betas = betas.clone().detach()

    optimizer = torch.optim.Adam([opt_betas], lr=lr)

    # SKEL joint indices
    SKEL_HUMERUS_R = 15
    SKEL_HUMERUS_L = 20

    for i in range(num_iters):
        optimizer.zero_grad()

        # Forward pass
        output = skel_model(
            poses=poses.unsqueeze(0),
            betas=opt_betas.unsqueeze(0),
            trans=trans.unsqueeze(0),
            poses_type='skel',
            skelmesh=False,
        )

        joints = output.joints[0]  # [24, 3]

        # Loss: humerus should be at target position
        loss_r = torch.sum((joints[SKEL_HUMERUS_R] - target_humerus_r) ** 2)
        loss_l = torch.sum((joints[SKEL_HUMERUS_L] - target_humerus_l) ** 2)

        # Regularization: prefer small changes from original betas (very weak)
        reg = 0.0001 * torch.sum((opt_betas - original_betas) ** 2)

        loss = loss_r + loss_l + reg

        loss.backward()
        optimizer.step()

        if i % 50 == 0:
            print(f"      Iter {i}: loss_r={loss_r.item()*1000:.2f}mm², loss_l={loss_l.item()*1000:.2f}mm²")

    return opt_betas.detach()


def main():
    parser = argparse.ArgumentParser(description='Generate mesh with beta-optimized humerus')
    parser.add_argument('--params_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/Subject11_v2/skel_params.npz',
                        help='Path to optimized SKEL parameters')
    parser.add_argument('--b3d_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d',
                        help='Path to AddB b3d file')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/postprocessed_mesh_beta',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to process')
    parser.add_argument('--opt_iters', type=int, default=300, help='Optimization iterations')
    parser.add_argument('--opt_lr', type=float, default=0.02, help='Optimization learning rate')
    parser.add_argument('--gender', type=str, default='male', help='Gender for SKEL model')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading SKEL model...")
    skel_model = SKEL(
        gender=args.gender,
        model_path=SKEL_MODEL_PATH,
    ).to(device)
    skel_model.eval()

    # Get faces
    skin_faces = skel_model.skin_f.cpu().numpy().astype(np.int32)
    skel_faces = skel_model.skel_f.cpu().numpy().astype(np.int32)

    # Load optimized parameters
    print(f"Loading optimized parameters from {args.params_path}...")
    params = np.load(args.params_path)
    poses = torch.from_numpy(params['poses']).float().to(device)
    betas = torch.from_numpy(params['betas']).float().to(device)
    trans = torch.from_numpy(params['trans']).float().to(device)

    print(f"  Original betas: {betas.cpu().numpy()}")

    num_frames = min(args.num_frames, poses.shape[0])
    print(f"  Processing {num_frames} frames")

    # Load AddB joints
    print(f"Loading AddB joints from {args.b3d_path}...")
    addb_joints, joint_names, metadata = load_b3d(args.b3d_path)
    addb_joints = addb_joints[:num_frames]
    # Convert to SKEL coordinate system (flip Z)
    addb_joints = convert_addb_to_skel_coords(addb_joints)
    addb_joints = torch.from_numpy(addb_joints).float().to(device)
    print(f"  AddB joints shape: {addb_joints.shape}")

    # AddB joint indices
    ADDB_ACROMIAL_R = 12
    ADDB_ACROMIAL_L = 16
    ADDB_ELBOW_R = 13
    ADDB_ELBOW_L = 17

    # SKEL joint indices
    SKEL_HUMERUS_R = 15
    SKEL_HUMERUS_L = 20

    # Optimize betas using first frame (betas are shared across all frames)
    print("\nOptimizing betas using first frame...")
    frame_idx = 0

    # Get original SKEL joints
    with torch.no_grad():
        output_orig = skel_model(
            poses=poses[frame_idx:frame_idx+1],
            betas=betas.unsqueeze(0),
            trans=trans[frame_idx:frame_idx+1],
            poses_type='skel',
            skelmesh=False,
        )

    orig_joints = output_orig.joints[0]
    orig_hum_r = orig_joints[SKEL_HUMERUS_R]
    orig_hum_l = orig_joints[SKEL_HUMERUS_L]

    # Compute target humerus position (on arm line)
    addb_frame = addb_joints[frame_idx]

    # Right arm
    addb_acr_r = addb_frame[ADDB_ACROMIAL_R]
    addb_elbow_r = addb_frame[ADDB_ELBOW_R]
    target_hum_r = project_point_onto_line(orig_hum_r, addb_acr_r, addb_elbow_r)

    # Left arm
    addb_acr_l = addb_frame[ADDB_ACROMIAL_L]
    addb_elbow_l = addb_frame[ADDB_ELBOW_L]
    target_hum_l = project_point_onto_line(orig_hum_l, addb_acr_l, addb_elbow_l)

    print(f"  Original humerus R: {orig_hum_r.cpu().numpy()}")
    print(f"  Target humerus R:   {target_hum_r.cpu().numpy()}")
    print(f"  Delta R: {torch.norm(target_hum_r - orig_hum_r).item()*1000:.2f}mm")

    # Optimize betas
    optimized_betas = optimize_betas_for_humerus_target(
        skel_model,
        poses[frame_idx],
        betas,
        trans[frame_idx],
        target_hum_r,
        target_hum_l,
        num_iters=args.opt_iters,
        lr=args.opt_lr,
    )

    print(f"\n  Optimized betas: {optimized_betas.cpu().numpy()}")
    print(f"  Beta change: {(optimized_betas - betas).cpu().numpy()}")

    # Check result
    with torch.no_grad():
        output_opt = skel_model(
            poses=poses[frame_idx:frame_idx+1],
            betas=optimized_betas.unsqueeze(0),
            trans=trans[frame_idx:frame_idx+1],
            poses_type='skel',
            skelmesh=False,
        )
    opt_joints = output_opt.joints[0]
    opt_hum_r = opt_joints[SKEL_HUMERUS_R]
    print(f"  Optimized humerus R: {opt_hum_r.cpu().numpy()}")
    print(f"  Residual error R: {torch.norm(target_hum_r - opt_hum_r).item()*1000:.2f}mm")

    # Generate meshes for all frames with optimized betas
    print("\nGenerating meshes with optimized betas...")
    for frame_idx in range(num_frames):
        print(f"\n  Frame {frame_idx}:")

        # Original mesh
        with torch.no_grad():
            output_orig = skel_model(
                poses=poses[frame_idx:frame_idx+1],
                betas=betas.unsqueeze(0),
                trans=trans[frame_idx:frame_idx+1],
                poses_type='skel',
                skelmesh=True,
            )

        # Optimized mesh
        with torch.no_grad():
            output_opt = skel_model(
                poses=poses[frame_idx:frame_idx+1],
                betas=optimized_betas.unsqueeze(0),
                trans=trans[frame_idx:frame_idx+1],
                poses_type='skel',
                skelmesh=True,
            )

        # Check humerus position
        orig_hum = output_orig.joints[0, SKEL_HUMERUS_R]
        opt_hum = output_opt.joints[0, SKEL_HUMERUS_R]

        addb_frame = addb_joints[frame_idx]
        target_hum = project_point_onto_line(
            orig_hum, addb_frame[ADDB_ACROMIAL_R], addb_frame[ADDB_ELBOW_R]
        )

        print(f"    Original delta: {torch.norm(target_hum - orig_hum).item()*1000:.2f}mm")
        print(f"    Optimized delta: {torch.norm(target_hum - opt_hum).item()*1000:.2f}mm")

        # Save meshes
        skin_verts_orig = output_orig.skin_verts[0].cpu().numpy()
        skel_verts_orig = output_orig.skel_verts[0].cpu().numpy()
        skin_verts_opt = output_opt.skin_verts[0].cpu().numpy()
        skel_verts_opt = output_opt.skel_verts[0].cpu().numpy()

        # Original meshes
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_skin_original_frame{frame_idx:03d}.obj'),
            skin_verts_orig, skin_faces
        )
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_bones_original_frame{frame_idx:03d}.obj'),
            skel_verts_orig, skel_faces
        )

        # Beta-optimized meshes
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_skin_beta_frame{frame_idx:03d}.obj'),
            skin_verts_opt, skin_faces
        )
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_bones_beta_frame{frame_idx:03d}.obj'),
            skel_verts_opt, skel_faces
        )

        print(f"    Saved meshes for frame {frame_idx}")

    # Save optimized betas
    np.savez(
        os.path.join(args.out_dir, 'optimized_params.npz'),
        poses=poses.cpu().numpy(),
        betas=optimized_betas.cpu().numpy(),
        trans=trans.cpu().numpy(),
        original_betas=betas.cpu().numpy(),
    )

    print(f"\nDone! Output saved to {args.out_dir}")


if __name__ == '__main__':
    main()
