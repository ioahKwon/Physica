"""
Generate mesh with post-processed humerus position using IK.

This script:
1. Loads optimized SKEL parameters
2. Uses IK to adjust scapula pose so humerus lies on AddB arm line
3. Forward pass with modified pose to generate correct mesh
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
from config import SKEL_MODEL_PATH, SCAPULA_DOF_INDICES


def save_mesh_obj(filepath, vertices, faces):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def optimize_shoulder_for_humerus_target(
    skel_model,
    poses: torch.Tensor,
    betas: torch.Tensor,
    trans: torch.Tensor,
    target_humerus_r: torch.Tensor,
    target_humerus_l: torch.Tensor,
    num_iters: int = 100,
    lr: float = 0.01,
):
    """
    Optimize scapula AND humerus DOFs to move humerus joint to target position.

    Args:
        skel_model: SKEL model
        poses: Original pose [46]
        betas: Shape parameters [10]
        trans: Translation [3]
        target_humerus_r: Target right humerus position [3]
        target_humerus_l: Target left humerus position [3]
        num_iters: Number of optimization iterations
        lr: Learning rate

    Returns:
        Optimized poses [46]
    """
    device = poses.device

    # Clone poses
    optimized_poses = poses.clone().detach()

    # DOF indices:
    # Right scapula: 26, 27, 28
    # Right humerus (shoulder): 29, 30, 31
    # Left scapula: 36, 37, 38
    # Left humerus (shoulder): 39, 40, 41

    # Extract shoulder DOFs (scapula + humerus) as optimizable parameters
    # Total: 12 DOFs (6 per side)
    shoulder_dofs = torch.zeros(12, device=device, requires_grad=True)

    # Right side
    shoulder_dofs.data[0] = poses[26].clone()  # scapula abduction
    shoulder_dofs.data[1] = poses[27].clone()  # scapula elevation
    shoulder_dofs.data[2] = poses[28].clone()  # scapula upward_rot
    shoulder_dofs.data[3] = poses[29].clone()  # humerus x
    shoulder_dofs.data[4] = poses[30].clone()  # humerus y
    shoulder_dofs.data[5] = poses[31].clone()  # humerus z

    # Left side
    shoulder_dofs.data[6] = poses[36].clone()   # scapula abduction
    shoulder_dofs.data[7] = poses[37].clone()   # scapula elevation
    shoulder_dofs.data[8] = poses[38].clone()   # scapula upward_rot
    shoulder_dofs.data[9] = poses[39].clone()   # humerus x
    shoulder_dofs.data[10] = poses[40].clone()  # humerus y
    shoulder_dofs.data[11] = poses[41].clone()  # humerus z

    # Save original values for regularization
    original_dofs = shoulder_dofs.data.clone()

    optimizer = torch.optim.Adam([shoulder_dofs], lr=lr)

    # SKEL joint indices
    SKEL_HUMERUS_R = 15
    SKEL_HUMERUS_L = 20
    SKEL_ULNA_R = 16  # elbow
    SKEL_ULNA_L = 21

    for i in range(num_iters):
        optimizer.zero_grad()

        # Update poses with current shoulder DOFs
        current_poses = optimized_poses.clone()

        # Right side
        current_poses[26] = shoulder_dofs[0]
        current_poses[27] = shoulder_dofs[1]
        current_poses[28] = shoulder_dofs[2]
        current_poses[29] = shoulder_dofs[3]
        current_poses[30] = shoulder_dofs[4]
        current_poses[31] = shoulder_dofs[5]

        # Left side
        current_poses[36] = shoulder_dofs[6]
        current_poses[37] = shoulder_dofs[7]
        current_poses[38] = shoulder_dofs[8]
        current_poses[39] = shoulder_dofs[9]
        current_poses[40] = shoulder_dofs[10]
        current_poses[41] = shoulder_dofs[11]

        # Forward pass
        output = skel_model(
            poses=current_poses.unsqueeze(0),
            betas=betas.unsqueeze(0),
            trans=trans.unsqueeze(0),
            poses_type='skel',
            skelmesh=False,
        )

        joints = output.joints[0]  # [24, 3]

        # Loss: humerus should be at target position
        loss_r = torch.sum((joints[SKEL_HUMERUS_R] - target_humerus_r) ** 2)
        loss_l = torch.sum((joints[SKEL_HUMERUS_L] - target_humerus_l) ** 2)

        # Regularization: prefer small changes from original (very weak)
        reg = 0.0001 * torch.sum((shoulder_dofs - original_dofs) ** 2)

        loss = loss_r + loss_l + reg

        loss.backward()
        optimizer.step()

        # Clamp DOFs to reasonable range
        with torch.no_grad():
            # Scapula DOFs: smaller range
            shoulder_dofs.data[0:3].clamp_(-0.8, 0.8)
            shoulder_dofs.data[6:9].clamp_(-0.8, 0.8)
            # Humerus DOFs: larger range
            shoulder_dofs.data[3:6].clamp_(-2.0, 2.0)
            shoulder_dofs.data[9:12].clamp_(-2.0, 2.0)

    # Apply final DOFs to poses
    final_poses = optimized_poses.clone()

    # Right side
    final_poses[26] = shoulder_dofs[0].detach()
    final_poses[27] = shoulder_dofs[1].detach()
    final_poses[28] = shoulder_dofs[2].detach()
    final_poses[29] = shoulder_dofs[3].detach()
    final_poses[30] = shoulder_dofs[4].detach()
    final_poses[31] = shoulder_dofs[5].detach()

    # Left side
    final_poses[36] = shoulder_dofs[6].detach()
    final_poses[37] = shoulder_dofs[7].detach()
    final_poses[38] = shoulder_dofs[8].detach()
    final_poses[39] = shoulder_dofs[9].detach()
    final_poses[40] = shoulder_dofs[10].detach()
    final_poses[41] = shoulder_dofs[11].detach()

    return final_poses


def main():
    parser = argparse.ArgumentParser(description='Generate mesh with IK-corrected humerus')
    parser.add_argument('--params_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/Subject11_v2/skel_params.npz',
                        help='Path to optimized SKEL parameters')
    parser.add_argument('--b3d_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d',
                        help='Path to AddB b3d file')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/postprocessed_mesh_ik',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to process')
    parser.add_argument('--ik_iters', type=int, default=200, help='IK iterations per frame')
    parser.add_argument('--ik_lr', type=float, default=0.02, help='IK learning rate')
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

    print("\nProcessing frames with IK...")
    for frame_idx in range(num_frames):
        print(f"\n  Frame {frame_idx}:")

        # Get original SKEL joints
        with torch.no_grad():
            output_orig = skel_model(
                poses=poses[frame_idx:frame_idx+1],
                betas=betas.unsqueeze(0),
                trans=trans[frame_idx:frame_idx+1],
                poses_type='skel',
                skelmesh=True,
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

        print(f"    Original humerus R: {orig_hum_r.cpu().numpy()}")
        print(f"    Target humerus R:   {target_hum_r.cpu().numpy()}")
        print(f"    Delta R: {torch.norm(target_hum_r - orig_hum_r).item()*1000:.2f}mm")

        # Run IK to optimize shoulder pose (scapula + humerus)
        optimized_poses = optimize_shoulder_for_humerus_target(
            skel_model,
            poses[frame_idx],
            betas,
            trans[frame_idx],
            target_hum_r,
            target_hum_l,
            num_iters=args.ik_iters,
            lr=args.ik_lr,
        )

        # Forward pass with optimized pose
        with torch.no_grad():
            output_opt = skel_model(
                poses=optimized_poses.unsqueeze(0),
                betas=betas.unsqueeze(0),
                trans=trans[frame_idx:frame_idx+1],
                poses_type='skel',
                skelmesh=True,
            )

        opt_joints = output_opt.joints[0]
        opt_hum_r = opt_joints[SKEL_HUMERUS_R]

        print(f"    Optimized humerus R: {opt_hum_r.cpu().numpy()}")
        print(f"    Residual error R: {torch.norm(target_hum_r - opt_hum_r).item()*1000:.2f}mm")

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

        # IK-optimized meshes
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_skin_ik_frame{frame_idx:03d}.obj'),
            skin_verts_opt, skin_faces
        )
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_bones_ik_frame{frame_idx:03d}.obj'),
            skel_verts_opt, skel_faces
        )

        print(f"    Saved meshes for frame {frame_idx}")

    print(f"\nDone! Output saved to {args.out_dir}")
    print("\nGenerated files:")
    print("  - skel_skin_original_frame*.obj: Original skin mesh")
    print("  - skel_skin_ik_frame*.obj: Skin mesh with IK-corrected humerus")
    print("  - skel_bones_original_frame*.obj: Original skeleton mesh")
    print("  - skel_bones_ik_frame*.obj: Skeleton mesh with IK-corrected humerus")


if __name__ == '__main__':
    main()
