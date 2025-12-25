"""
Test beta parameter variations on the best result.

Vary each beta component and see how it affects MPJPE and mesh.
"""

import os
import sys
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
from utils.geometry import convert_addb_to_skel_coords
from utils.io import load_b3d
from config import SKEL_MODEL_PATH

# Paths
PARAMS_PATH = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/best_scapula_mapping_21mm/skel_params.npz'
B3D_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d'
OUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/beta_variations'


def save_mesh_obj(filepath, vertices, faces):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def compute_mpjpe(pred_joints, target_joints, joint_mapping):
    """Compute MPJPE for mapped joints."""
    errors = []
    for addb_idx, skel_idx in joint_mapping:
        error = np.linalg.norm(pred_joints[skel_idx] - target_joints[addb_idx])
        errors.append(error)
    return np.mean(errors) * 1000  # mm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading SKEL model...")
    skel_model = SKEL(
        gender='male',
        model_path=SKEL_MODEL_PATH,
    ).to(device)
    skel_model.eval()

    skin_faces = skel_model.skin_f.cpu().numpy().astype(np.int32)

    # Load optimized parameters
    print(f"Loading params from {PARAMS_PATH}...")
    params = np.load(PARAMS_PATH)
    poses = torch.from_numpy(params['poses']).float().to(device)
    betas_orig = torch.from_numpy(params['betas']).float().to(device)
    trans = torch.from_numpy(params['trans']).float().to(device)

    print(f"  Original betas: {betas_orig.cpu().numpy()}")

    # Load AddB joints
    print(f"Loading AddB joints from {B3D_PATH}...")
    addb_joints, joint_names, metadata = load_b3d(B3D_PATH)
    addb_joints = convert_addb_to_skel_coords(addb_joints)
    addb_joints = addb_joints[0]  # First frame
    print(f"  AddB joints shape: {addb_joints.shape}")

    # Joint mapping (AddB idx, SKEL idx) - using scapula mapping
    # Based on AUTO_JOINT_NAME_MAP_SKEL with acromial→scapula
    joint_mapping = [
        (0, 0),   # pelvis
        (1, 1),   # hip_r → femur_r
        (2, 2),   # walker_knee_r → tibia_r
        (3, 3),   # ankle_r → talus_r
        (6, 6),   # hip_l → femur_l
        (7, 7),   # walker_knee_l → tibia_l
        (8, 8),   # ankle_l → talus_l
        (11, 11), # back → lumbar
        (12, 14), # acromial_r → scapula_r
        (13, 16), # elbow_r → ulna_r
        (15, 18), # radius_hand_r → hand_r
        (16, 19), # acromial_l → scapula_l
        (17, 21), # elbow_l → ulna_l
        (19, 23), # radius_hand_l → hand_l
    ]

    # Get original MPJPE
    with torch.no_grad():
        output_orig = skel_model(
            poses=poses[0:1],
            betas=betas_orig.unsqueeze(0),
            trans=trans[0:1],
            poses_type='skel',
            skelmesh=False,
        )
    orig_joints = output_orig.joints[0].cpu().numpy()
    orig_mpjpe = compute_mpjpe(orig_joints, addb_joints, joint_mapping)
    print(f"\nOriginal MPJPE: {orig_mpjpe:.2f}mm")

    # Save original mesh
    with torch.no_grad():
        output_mesh = skel_model(
            poses=poses[0:1],
            betas=betas_orig.unsqueeze(0),
            trans=trans[0:1],
            poses_type='skel',
            skelmesh=False,
        )
    save_mesh_obj(
        os.path.join(OUT_DIR, 'original.obj'),
        output_mesh.skin_verts[0].cpu().numpy(),
        skin_faces
    )

    # Test variations
    print("\n" + "="*70)
    print("BETA VARIATIONS TEST")
    print("="*70)

    # Test delta values
    deltas = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]

    results = []

    for beta_idx in range(10):
        print(f"\n--- Beta[{beta_idx}] (original: {betas_orig[beta_idx].item():.3f}) ---")

        for delta in deltas:
            # Create modified betas
            betas_mod = betas_orig.clone()
            betas_mod[beta_idx] += delta

            # Forward pass
            with torch.no_grad():
                output = skel_model(
                    poses=poses[0:1],
                    betas=betas_mod.unsqueeze(0),
                    trans=trans[0:1],
                    poses_type='skel',
                    skelmesh=False,
                )
            mod_joints = output.joints[0].cpu().numpy()
            mod_mpjpe = compute_mpjpe(mod_joints, addb_joints, joint_mapping)

            mpjpe_change = mod_mpjpe - orig_mpjpe
            arrow = "↑" if mpjpe_change > 0 else "↓" if mpjpe_change < 0 else "="

            print(f"  delta={delta:+.1f}: MPJPE={mod_mpjpe:.2f}mm ({arrow}{abs(mpjpe_change):.2f}mm)")

            results.append({
                'beta_idx': beta_idx,
                'delta': delta,
                'mpjpe': mod_mpjpe,
                'mpjpe_change': mpjpe_change,
            })

            # Save mesh for significant changes
            if abs(delta) == 2.0:
                with torch.no_grad():
                    output_mesh = skel_model(
                        poses=poses[0:1],
                        betas=betas_mod.unsqueeze(0),
                        trans=trans[0:1],
                        poses_type='skel',
                        skelmesh=False,
                    )
                save_mesh_obj(
                    os.path.join(OUT_DIR, f'beta{beta_idx}_delta{delta:+.1f}.obj'),
                    output_mesh.skin_verts[0].cpu().numpy(),
                    skin_faces
                )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Best improvements by beta")
    print("="*70)

    for beta_idx in range(10):
        beta_results = [r for r in results if r['beta_idx'] == beta_idx]
        best = min(beta_results, key=lambda x: x['mpjpe'])
        worst = max(beta_results, key=lambda x: x['mpjpe'])

        print(f"Beta[{beta_idx}]: best delta={best['delta']:+.1f} ({best['mpjpe']:.2f}mm), "
              f"worst delta={worst['delta']:+.1f} ({worst['mpjpe']:.2f}mm)")

    # Find best overall
    best_overall = min(results, key=lambda x: x['mpjpe'])
    print(f"\nBest overall: Beta[{best_overall['beta_idx']}] delta={best_overall['delta']:+.1f} "
          f"→ MPJPE={best_overall['mpjpe']:.2f}mm (improvement: {-best_overall['mpjpe_change']:.2f}mm)")

    print(f"\nMeshes saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
