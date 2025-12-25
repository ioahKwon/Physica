"""
Generate mesh with post-processed humerus position.

This script:
1. Loads optimized SKEL parameters
2. Post-processes humerus to lie on AddB arm line
3. Uses skinning weights to move affected vertices
4. Saves both skin and skeleton meshes
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
from utils.geometry import postprocess_humerus_to_arm_line, project_point_onto_line, convert_addb_to_skel_coords
from utils.io import load_b3d
from config import SKEL_MODEL_PATH


def save_mesh_obj(filepath, vertices, faces):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def apply_humerus_correction_to_mesh(
    vertices: np.ndarray,
    skin_weights: np.ndarray,
    humerus_delta: np.ndarray,
    humerus_joint_idx: int,
    child_joint_indices: list = None,
) -> np.ndarray:
    """
    Apply humerus position correction to mesh vertices using skinning weights.

    Args:
        vertices: Mesh vertices [V, 3]
        skin_weights: Skinning weights [V, J]
        humerus_delta: Delta to apply to humerus [3]
        humerus_joint_idx: Index of humerus joint
        child_joint_indices: Indices of child joints to also apply correction

    Returns:
        Corrected vertices [V, 3]
    """
    result = vertices.copy()

    # Get weight for humerus joint
    humerus_weights = skin_weights[:, humerus_joint_idx]

    # Add child weights if specified
    if child_joint_indices:
        for child_idx in child_joint_indices:
            humerus_weights = np.maximum(humerus_weights, skin_weights[:, child_idx])

    # Apply delta weighted by skinning weight
    for i in range(3):
        result[:, i] += humerus_weights * humerus_delta[i]

    return result


def main():
    parser = argparse.ArgumentParser(description='Generate mesh with post-processed humerus')
    parser.add_argument('--params_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/Subject11_v2/skel_params.npz',
                        help='Path to optimized SKEL parameters')
    parser.add_argument('--b3d_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d',
                        help='Path to AddB b3d file')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/postprocessed_mesh',
                        help='Output directory')
    parser.add_argument('--num_frames', type=int, default=5, help='Number of frames to process')
    parser.add_argument('--blend_factor', type=float, default=1.0, help='Blend factor for humerus correction')
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

    # Get skinning weights and faces
    skin_weights = skel_model.skin_weights.to_dense().cpu().numpy()  # [V, J]
    skin_faces = skel_model.skin_f.cpu().numpy().astype(np.int32)
    skel_faces = skel_model.skel_f.cpu().numpy().astype(np.int32)

    print(f"  Skin weights shape: {skin_weights.shape}")
    print(f"  Skin faces shape: {skin_faces.shape}")
    print(f"  Skel faces shape: {skel_faces.shape}")

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
    print(f"  AddB joints shape: {addb_joints.shape}")

    # SKEL joint indices
    SKEL_HUMERUS_R = 15
    SKEL_HUMERUS_L = 20
    # Child joints of humerus (ulna, radius, hand)
    SKEL_CHILDREN_R = [16, 17, 18]  # ulna_r, radius_r, hand_r
    SKEL_CHILDREN_L = [21, 22, 23]  # ulna_l, radius_l, hand_l

    # AddB joint indices
    ADDB_ACROMIAL_R = 12
    ADDB_ACROMIAL_L = 16
    ADDB_ELBOW_R = 13
    ADDB_ELBOW_L = 17

    print("\nProcessing frames...")
    for frame_idx in range(num_frames):
        print(f"\n  Frame {frame_idx}:")

        # Get SKEL output
        with torch.no_grad():
            output = skel_model(
                poses=poses[frame_idx:frame_idx+1],
                betas=betas.unsqueeze(0),
                trans=trans[frame_idx:frame_idx+1],
                poses_type='skel',
                skelmesh=True,
            )

        skin_verts = output.skin_verts[0].cpu().numpy()
        skel_verts = output.skel_verts[0].cpu().numpy()
        joints = output.joints[0].cpu().numpy()

        print(f"    Original skin verts: {skin_verts.shape}")
        print(f"    Original skel verts: {skel_verts.shape}")
        print(f"    Joints: {joints.shape}")

        # Get current AddB joints for this frame
        addb_frame = addb_joints[frame_idx]

        # Compute post-processed humerus position
        # Right arm
        addb_acr_r = addb_frame[ADDB_ACROMIAL_R]
        addb_elbow_r = addb_frame[ADDB_ELBOW_R]
        skel_hum_r = joints[SKEL_HUMERUS_R]

        projected_r = project_point_onto_line(skel_hum_r, addb_acr_r, addb_elbow_r)
        target_hum_r = (1 - args.blend_factor) * skel_hum_r + args.blend_factor * projected_r
        delta_r = target_hum_r - skel_hum_r

        # Left arm
        addb_acr_l = addb_frame[ADDB_ACROMIAL_L]
        addb_elbow_l = addb_frame[ADDB_ELBOW_L]
        skel_hum_l = joints[SKEL_HUMERUS_L]

        projected_l = project_point_onto_line(skel_hum_l, addb_acr_l, addb_elbow_l)
        target_hum_l = (1 - args.blend_factor) * skel_hum_l + args.blend_factor * projected_l
        delta_l = target_hum_l - skel_hum_l

        print(f"    Humerus R delta: {np.linalg.norm(delta_r)*1000:.2f}mm")
        print(f"    Humerus L delta: {np.linalg.norm(delta_l)*1000:.2f}mm")

        # Apply correction to skin mesh
        corrected_skin = skin_verts.copy()

        # Right arm correction
        corrected_skin = apply_humerus_correction_to_mesh(
            corrected_skin, skin_weights, delta_r,
            SKEL_HUMERUS_R, SKEL_CHILDREN_R
        )

        # Left arm correction
        corrected_skin = apply_humerus_correction_to_mesh(
            corrected_skin, skin_weights, delta_l,
            SKEL_HUMERUS_L, SKEL_CHILDREN_L
        )

        # Apply correction to skeleton mesh using skel_weights
        skel_weights = skel_model.skel_weights.to_dense().cpu().numpy()
        corrected_skel = skel_verts.copy()

        # Right arm correction
        corrected_skel = apply_humerus_correction_to_mesh(
            corrected_skel, skel_weights, delta_r,
            SKEL_HUMERUS_R, SKEL_CHILDREN_R
        )

        # Left arm correction
        corrected_skel = apply_humerus_correction_to_mesh(
            corrected_skel, skel_weights, delta_l,
            SKEL_HUMERUS_L, SKEL_CHILDREN_L
        )

        # Save meshes
        # Original meshes
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_skin_original_frame{frame_idx:03d}.obj'),
            skin_verts, skin_faces
        )
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_bones_original_frame{frame_idx:03d}.obj'),
            skel_verts, skel_faces
        )

        # Post-processed meshes
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_skin_postprocessed_frame{frame_idx:03d}.obj'),
            corrected_skin, skin_faces
        )
        save_mesh_obj(
            os.path.join(args.out_dir, f'skel_bones_postprocessed_frame{frame_idx:03d}.obj'),
            corrected_skel, skel_faces
        )

        print(f"    Saved meshes for frame {frame_idx}")

    print(f"\nDone! Output saved to {args.out_dir}")
    print("\nGenerated files:")
    print("  - skel_skin_original_frame*.obj: Original skin mesh")
    print("  - skel_skin_postprocessed_frame*.obj: Skin mesh with humerus on arm line")
    print("  - skel_bones_original_frame*.obj: Original skeleton mesh")
    print("  - skel_bones_postprocessed_frame*.obj: Skeleton mesh with humerus on arm line")


if __name__ == '__main__':
    main()
