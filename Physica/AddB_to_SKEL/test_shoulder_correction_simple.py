#!/usr/bin/env python3
"""
Simple Shoulder Correction Test

This script loads pre-optimized SKEL results from compare_smpl_skel.py output
and evaluates the virtual acromial vs AddB acromial alignment.

This avoids running the full optimization which causes segfault.
"""

import os
import sys
import json
import numpy as np
import torch

sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SMPL')
sys.path.insert(0, '/egr/research-zijunlab/kwonjoon/01_Code/Physica/AddB_to_SKEL')

from models.skel_model import SKELModelWrapper, SKEL_JOINT_NAMES

# Import shoulder correction
from shoulder_correction import (
    ACROMIAL_VERTEX_IDX,
    compute_virtual_acromial,
)

SKEL_MODEL_PATH = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'
OUTPUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare'


def find_latest_output():
    """Find the most recent output directory."""
    dirs = [d for d in os.listdir(OUTPUT_DIR) if d.startswith('compare_') or d.startswith('test_')]
    if not dirs:
        return None
    dirs.sort(reverse=True)

    for d in dirs:
        skel_json = os.path.join(OUTPUT_DIR, d, 'skel_params.json')
        if os.path.exists(skel_json):
            return os.path.join(OUTPUT_DIR, d)
    return None


def load_addb_from_npz(output_dir):
    """Load AddB joint data from npz file if exists."""
    npz_path = os.path.join(output_dir, 'target_joints.npz')
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data['joints'], list(data['joint_names'])
    return None, None


def test_shoulder_with_precomputed():
    """Test shoulder correction using pre-computed SKEL optimization results."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Find latest output
    output_dir = find_latest_output()
    if output_dir is None:
        print("No pre-computed output found. Looking for npz files...")
        # List available directories
        dirs = sorted(os.listdir(OUTPUT_DIR))
        print(f"Available directories: {dirs}")
        return

    print(f"Loading from: {output_dir}")

    # Load SKEL params
    skel_json = os.path.join(output_dir, 'skel_params.json')
    with open(skel_json, 'r') as f:
        skel_data = json.load(f)

    betas = np.array(skel_data['betas'])
    poses = np.array(skel_data['poses'])
    trans = np.array(skel_data['trans'])

    print(f"Loaded SKEL params: betas={betas.shape}, poses={poses.shape}, trans={trans.shape}")

    # Load AddB target joints if available
    addb_joints, joint_names = load_addb_from_npz(output_dir)

    if addb_joints is None:
        print("No target_joints.npz found. Creating one from sample...")
        # We need to load AddB data - but this might crash
        # Let's use known values from previous runs instead

        # From previous run_shoulder_test.py output:
        # AddB shoulder width: 370.5mm
        # This is the typical value for Subject1
        print("\nUsing typical AddB values from previous runs:")
        print(f"  Typical AddB shoulder width: ~370-380mm")
        print(f"  Typical SKEL humerus width:  ~351mm")
        addb_joints = None
    else:
        print(f"Loaded AddB joints: {addb_joints.shape} with names: {joint_names}")

    # Load SKEL model
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    T = len(poses)
    betas_t = torch.tensor(betas, device=device, dtype=torch.float32)
    poses_t = torch.tensor(poses, device=device, dtype=torch.float32)
    trans_t = torch.tensor(trans, device=device, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        verts, joints = skel.forward(
            betas_t.unsqueeze(0).expand(T, -1),
            poses_t,
            trans_t
        )

        # Compute virtual acromial
        virtual_r, virtual_l = compute_virtual_acromial(verts)

        # Get humerus positions
        humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
        humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
        humerus_r = joints[:, humerus_r_idx, :]
        humerus_l = joints[:, humerus_l_idx, :]

        # Widths
        humerus_width = torch.norm(humerus_r - humerus_l, dim=-1).mean().item() * 1000
        virtual_width = torch.norm(virtual_r - virtual_l, dim=-1).mean().item() * 1000

        print("\n" + "=" * 60)
        print("SHOULDER WIDTH COMPARISON")
        print("=" * 60)
        print(f"SKEL humerus width:  {humerus_width:.1f}mm")
        print(f"SKEL virtual width:  {virtual_width:.1f}mm")

        if addb_joints is not None and 'acromial_r' in joint_names and 'acromial_l' in joint_names:
            addb_joints_t = torch.tensor(addb_joints, device=device, dtype=torch.float32)
            acr_r_idx = joint_names.index('acromial_r')
            acr_l_idx = joint_names.index('acromial_l')

            addb_r = addb_joints_t[:, acr_r_idx, :]
            addb_l = addb_joints_t[:, acr_l_idx, :]

            addb_width = torch.norm(addb_r - addb_l, dim=-1).mean().item() * 1000

            # Errors
            humerus_err_r = torch.norm(humerus_r - addb_r, dim=-1).mean().item() * 1000
            humerus_err_l = torch.norm(humerus_l - addb_l, dim=-1).mean().item() * 1000
            virtual_err_r = torch.norm(virtual_r - addb_r, dim=-1).mean().item() * 1000
            virtual_err_l = torch.norm(virtual_l - addb_l, dim=-1).mean().item() * 1000

            print(f"AddB shoulder width: {addb_width:.1f}mm")
            print()
            print("Mapping: humerus → acromial:")
            print(f"  Width diff: {abs(humerus_width - addb_width):.1f}mm")
            print(f"  Position error: R={humerus_err_r:.1f}mm, L={humerus_err_l:.1f}mm")
            print()
            print("Mapping: virtual acromial → acromial:")
            print(f"  Width diff: {abs(virtual_width - addb_width):.1f}mm")
            print(f"  Position error: R={virtual_err_r:.1f}mm, L={virtual_err_l:.1f}mm")
        else:
            print("AddB acromial data not available for comparison")
            print(f"Expected AddB width: ~370-380mm (typical)")
            print(f"  Humerus is ~{370 - humerus_width:.0f}mm narrower than expected")
            print(f"  Virtual is ~{370 - virtual_width:.0f}mm narrower than expected")

        print("\n" + "=" * 60)
        print("VIRTUAL ACROMIAL POSITIONS (first frame)")
        print("=" * 60)
        print(f"Virtual R: {virtual_r[0].cpu().numpy()}")
        print(f"Virtual L: {virtual_l[0].cpu().numpy()}")
        print(f"Humerus R: {humerus_r[0].cpu().numpy()}")
        print(f"Humerus L: {humerus_l[0].cpu().numpy()}")

        print(f"\nVertex indices used:")
        print(f"  Right: {ACROMIAL_VERTEX_IDX['right']}")
        print(f"  Left:  {ACROMIAL_VERTEX_IDX['left']}")


if __name__ == '__main__':
    test_shoulder_with_precomputed()
