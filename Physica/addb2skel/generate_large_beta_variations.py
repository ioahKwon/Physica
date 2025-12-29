"""
Generate meshes with LARGE beta variations for Subject11.
Beta[0] and Beta[1] with ±1.0, ±2.0 changes.
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
from config import SKEL_MODEL_PATH


def save_mesh_obj(filepath, vertices, faces):
    """Save mesh as OBJ file."""
    with open(filepath, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def compute_shoulder_width(joints):
    """Compute shoulder width (scapula_r to scapula_l distance)."""
    SKEL_SCAPULA_R = 14
    SKEL_SCAPULA_L = 19
    return np.linalg.norm(joints[SKEL_SCAPULA_R] - joints[SKEL_SCAPULA_L]) * 1000  # mm


def main():
    # Output directory
    OUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/beta_large_variations'
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("Loading SKEL model...")
    skel_model = SKEL(
        gender='male',
        model_path=SKEL_MODEL_PATH,
    ).to(device)
    skel_model.eval()

    skin_faces = skel_model.skin_f.cpu().numpy().astype(np.int32)

    # Load Subject11 current parameters
    PARAMS_PATH = '/egr/research-zijunlab/kwonjoon/03_Output/addb2skel/multi_subject_test/Subject11/skel/skel_params.npz'
    print(f"Loading params from {PARAMS_PATH}...")
    params = np.load(PARAMS_PATH)
    poses = torch.from_numpy(params['poses']).float().to(device)
    betas_orig = torch.from_numpy(params['betas']).float().to(device)
    trans = torch.from_numpy(params['trans']).float().to(device)

    print(f"Original betas: {betas_orig.cpu().numpy()}")

    # Generate original mesh
    print("\nGenerating original mesh...")
    with torch.no_grad():
        output = skel_model(
            poses=poses[0:1],
            betas=betas_orig.unsqueeze(0),
            trans=trans[0:1],
            poses_type='skel',
            skelmesh=False,
        )
    orig_joints = output.joints[0].cpu().numpy()
    orig_sw = compute_shoulder_width(orig_joints)
    print(f"Original shoulder width: {orig_sw:.1f}mm")

    save_mesh_obj(
        os.path.join(OUT_DIR, 'original.obj'),
        output.skin_verts[0].cpu().numpy(),
        skin_faces
    )

    # Large variations for Beta[0] and Beta[1]
    variations = {
        'beta0': [
            ('minus2.0', -2.0),
            ('minus1.0', -1.0),
            ('plus1.0', +1.0),
            ('plus2.0', +2.0),
        ],
        'beta1': [
            ('minus2.0', -2.0),
            ('minus1.0', -1.0),
            ('plus1.0', +1.0),
            ('plus2.0', +2.0),
        ],
    }

    results = []

    for beta_name, deltas in variations.items():
        beta_idx = int(beta_name.replace('beta', ''))
        print(f"\n=== {beta_name.upper()} (original: {betas_orig[beta_idx].item():.3f}) ===")

        for delta_name, delta in deltas:
            # Modify beta
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
            mod_sw = compute_shoulder_width(mod_joints)
            sw_change = mod_sw - orig_sw

            print(f"  {delta_name}: SW={mod_sw:.1f}mm (Δ{sw_change:+.1f}mm), beta[{beta_idx}]={betas_mod[beta_idx].item():.3f}")

            # Save mesh
            mesh_path = os.path.join(OUT_DIR, f'{beta_name}_{delta_name}.obj')
            save_mesh_obj(
                mesh_path,
                output.skin_verts[0].cpu().numpy(),
                skin_faces
            )

            results.append({
                'beta': beta_name,
                'delta': delta_name,
                'delta_val': delta,
                'sw': mod_sw,
                'sw_change': sw_change,
                'mesh_path': mesh_path,
            })

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Original: SW={orig_sw:.1f}mm")
    print()
    for r in results:
        print(f"{r['beta']} {r['delta']}: SW={r['sw']:.1f}mm (Δ{r['sw_change']:+.1f}mm)")

    print(f"\nMeshes saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
