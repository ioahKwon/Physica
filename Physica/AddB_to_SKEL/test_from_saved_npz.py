#!/usr/bin/env python3
"""
Test Shoulder Correction from Saved NPZ Files

Load pre-optimized SKEL results and AddB target joints, then evaluate
the virtual acromial vs AddB acromial alignment.
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
OUTPUT_DIR = '/egr/research-zijunlab/kwonjoon/03_Output/smpl_skel_compare/test_baseline_check'


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load saved data
    skel_params = np.load(os.path.join(OUTPUT_DIR, 'skel', 'skel_params.npz'))
    skel_joints = np.load(os.path.join(OUTPUT_DIR, 'skel', 'joints.npy'))
    addb_joints = np.load(os.path.join(OUTPUT_DIR, 'addb', 'joints.npy'))

    print(f"SKEL params: betas={skel_params['betas'].shape}, poses={skel_params['poses'].shape}")
    print(f"SKEL joints: {skel_joints.shape}")
    print(f"AddB joints: {addb_joints.shape}")

    # Load joint names from metrics
    with open(os.path.join(OUTPUT_DIR, 'comparison_metrics.json'), 'r') as f:
        metrics = json.load(f)
    joint_names = metrics['addb_joint_names']
    print(f"Joint names: {joint_names}")

    # Get acromial indices
    acr_r_idx = joint_names.index('acromial_r')
    acr_l_idx = joint_names.index('acromial_l')
    print(f"Acromial indices: R={acr_r_idx}, L={acr_l_idx}")

    # Load SKEL model to get vertices
    skel = SKELModelWrapper(model_path=SKEL_MODEL_PATH, gender='male', device=device)

    T = len(skel_params['poses'])
    betas = torch.tensor(skel_params['betas'], device=device, dtype=torch.float32)
    poses = torch.tensor(skel_params['poses'], device=device, dtype=torch.float32)
    trans = torch.tensor(skel_params['trans'], device=device, dtype=torch.float32)

    # Forward pass to get vertices
    with torch.no_grad():
        verts, joints = skel.forward(
            betas.unsqueeze(0).expand(T, -1),
            poses,
            trans
        )

        # Compute virtual acromial
        virtual_r, virtual_l = compute_virtual_acromial(verts)

        # Get humerus positions
        humerus_r_idx = SKEL_JOINT_NAMES.index('humerus_r')
        humerus_l_idx = SKEL_JOINT_NAMES.index('humerus_l')
        humerus_r = joints[:, humerus_r_idx, :]
        humerus_l = joints[:, humerus_l_idx, :]

        # AddB acromial positions
        addb_t = torch.tensor(addb_joints, device=device, dtype=torch.float32)
        addb_acr_r = addb_t[:, acr_r_idx, :]
        addb_acr_l = addb_t[:, acr_l_idx, :]

        # Compute metrics
        humerus_width = torch.norm(humerus_r - humerus_l, dim=-1).mean().item() * 1000
        virtual_width = torch.norm(virtual_r - virtual_l, dim=-1).mean().item() * 1000
        addb_width = torch.norm(addb_acr_r - addb_acr_l, dim=-1).mean().item() * 1000

        # Position errors
        humerus_err_r = torch.norm(humerus_r - addb_acr_r, dim=-1).mean().item() * 1000
        humerus_err_l = torch.norm(humerus_l - addb_acr_l, dim=-1).mean().item() * 1000
        virtual_err_r = torch.norm(virtual_r - addb_acr_r, dim=-1).mean().item() * 1000
        virtual_err_l = torch.norm(virtual_l - addb_acr_l, dim=-1).mean().item() * 1000

    print("\n" + "=" * 60)
    print("SHOULDER ALIGNMENT RESULTS (with corrected SKEL vertex indices)")
    print("=" * 60)

    print(f"\nAddB shoulder width:        {addb_width:.1f} mm")
    print()
    print("Current mapping (humerus → acromial):")
    print(f"  SKEL humerus width:       {humerus_width:.1f} mm (diff: {abs(humerus_width - addb_width):.1f} mm)")
    print(f"  Humerus→Acromial error:   R={humerus_err_r:.1f} mm, L={humerus_err_l:.1f} mm")
    print(f"  Avg humerus error:        {(humerus_err_r + humerus_err_l) / 2:.1f} mm")
    print()
    print("Virtual acromial (vertex-based, CORRECTED indices):")
    print(f"  SKEL virtual width:       {virtual_width:.1f} mm (diff: {abs(virtual_width - addb_width):.1f} mm)")
    print(f"  Virtual→Acromial error:   R={virtual_err_r:.1f} mm, L={virtual_err_l:.1f} mm")
    print(f"  Avg virtual error:        {(virtual_err_r + virtual_err_l) / 2:.1f} mm")
    print()
    print(f"Vertex indices used:")
    print(f"  Right: {ACROMIAL_VERTEX_IDX['right']}")
    print(f"  Left:  {ACROMIAL_VERTEX_IDX['left']}")

    print("\n" + "=" * 60)
    print("POSITION COMPARISON (first frame)")
    print("=" * 60)
    print(f"\nRight shoulder:")
    print(f"  AddB acromial:   {addb_acr_r[0].cpu().numpy()}")
    print(f"  SKEL humerus:    {humerus_r[0].cpu().numpy()}")
    print(f"  SKEL virtual:    {virtual_r[0].cpu().numpy()}")
    print(f"\nLeft shoulder:")
    print(f"  AddB acromial:   {addb_acr_l[0].cpu().numpy()}")
    print(f"  SKEL humerus:    {humerus_l[0].cpu().numpy()}")
    print(f"  SKEL virtual:    {virtual_l[0].cpu().numpy()}")

    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    improvement_r = humerus_err_r - virtual_err_r
    improvement_l = humerus_err_l - virtual_err_l
    width_improvement = abs(humerus_width - addb_width) - abs(virtual_width - addb_width)

    if improvement_r > 0:
        print(f"✓ Virtual acromial is {improvement_r:.1f}mm BETTER than humerus for RIGHT")
    else:
        print(f"✗ Virtual acromial is {-improvement_r:.1f}mm WORSE than humerus for RIGHT")

    if improvement_l > 0:
        print(f"✓ Virtual acromial is {improvement_l:.1f}mm BETTER than humerus for LEFT")
    else:
        print(f"✗ Virtual acromial is {-improvement_l:.1f}mm WORSE than humerus for LEFT")

    if width_improvement > 0:
        print(f"✓ Virtual width is {width_improvement:.1f}mm CLOSER to AddB width")
    else:
        print(f"✗ Virtual width is {-width_improvement:.1f}mm FARTHER from AddB width")


if __name__ == '__main__':
    main()
