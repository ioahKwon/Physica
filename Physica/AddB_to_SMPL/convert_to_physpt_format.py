#!/usr/bin/env python3
"""
최적화된 SMPL 데이터 + AddB physics 데이터를
PhysPT 학습 format으로 변환

Converts:
- Axis-angle (24, 3) → 6D rotation (144,)
- AddB body-level forces → SMPL vertex-level forces (29, 3)
- AddB DOF torques → SMPL 69 DOFs

Usage:
    python convert_to_physpt_format.py \
        --smpl_params /path/to/smpl_params.npz \
        --physics_data /path/to/physics_data.npz \
        --physics_metadata /path/to/physics_metadata.json \
        --contact_mapping /path/to/contact_mapping.json \
        --dof_mapping /path/to/dof_mapping.json \
        --output /path/to/output_physpt.npz
"""

import numpy as np
import torch
import argparse
import json
from pathlib import Path
import sys


def axis_angle_to_rotation_6d(poses: np.ndarray) -> np.ndarray:
    """
    Axis-angle (24, 3) → 6D rotation (144,)

    6D rotation: rotation matrix의 첫 2개 column vectors
    [R[:, 0], R[:, 1]] → (3, 2) → flatten → (6,)
    24 joints → 24 * 6 = 144

    Args:
        poses: (T, 24, 3) axis-angle

    Returns:
        poses_6d: (T, 144) 6D rotation
    """
    try:
        import pytorch3d.transforms as transforms
    except ImportError:
        raise ImportError("pytorch3d is required for rotation conversion. "
                         "Install with: pip install pytorch3d")

    T, n_joints, _ = poses.shape  # (T, 24, 3)
    poses_flat = poses.reshape(T * n_joints, 3)

    # Axis-angle → Rotation matrix
    poses_torch = torch.from_numpy(poses_flat).float()
    rot_mats = transforms.axis_angle_to_matrix(poses_torch)  # (T*24, 3, 3)

    # Rotation matrix → 6D (첫 2 columns)
    rot_6d = rot_mats[:, :, :2].reshape(T * n_joints, 6)  # (T*24, 6)
    rot_6d = rot_6d.reshape(T, n_joints * 6)  # (T, 144)

    return rot_6d.numpy()


def convert_to_physpt_format(
    smpl_params_path: str,
    physics_data_path: str,
    physics_metadata_path: str,
    contact_mapping_path: str,
    dof_mapping_path: str,
    output_path: str
):
    """
    SMPL params + physics data → PhysPT format

    PhysPT 입력 format:
    - poses_6d: (T, 144) - 6D rotation
    - betas: (10,) - shape parameters
    - trans: (T, 3) - root translation

    PhysPT GT format:
    - gt_grf: (T, 29, 3) - vertex-level forces
    - gt_torques: (T, 69) - joint torques
    - contacts: (T, 29) - contact states
    """

    print("=" * 80)
    print("Converting to PhysPT Format")
    print("=" * 80)
    print()

    # Load SMPL params
    print("[1/6] Loading SMPL parameters...")
    smpl_data = np.load(smpl_params_path)
    poses = smpl_data['poses']  # (T, 24, 3) axis-angle
    betas = smpl_data['betas']  # (10,)
    trans = smpl_data['trans']  # (T, 3)

    T = poses.shape[0]
    print(f"  Frames: {T}")
    print(f"  Poses shape: {poses.shape}")
    print(f"  Betas shape: {betas.shape}")
    print(f"  Trans shape: {trans.shape}")

    # Convert axis-angle → 6D rotation
    print("\n[2/6] Converting axis-angle to 6D rotation...")
    poses_6d = axis_angle_to_rotation_6d(poses)  # (T, 144)
    print(f"  Poses (6D) shape: {poses_6d.shape}")

    # Load physics data
    print("\n[3/6] Loading physics data...")
    physics_data = np.load(physics_data_path)
    addb_grf = physics_data['grf']  # (T, N_addb_bodies, 3)
    addb_cop = physics_data['cop']  # (T, N_addb_bodies, 3)
    addb_torques = physics_data['torques']  # (T, N_addb_dof)
    addb_contacts = physics_data['contacts']  # (T, N_addb_bodies)

    print(f"  AddB GRF shape: {addb_grf.shape}")
    print(f"  AddB Torques shape: {addb_torques.shape}")
    print(f"  AddB Contacts shape: {addb_contacts.shape}")

    # Load metadata
    print("\n[4/6] Loading metadata...")
    with open(physics_metadata_path) as f:
        physics_meta = json.load(f)
    addb_body_names = physics_meta['contact_body_names']
    addb_dof_names = physics_meta['dof_names']
    print(f"  Contact bodies: {len(addb_body_names)}")
    print(f"  DOFs: {len(addb_dof_names)}")

    # Load mappings
    print("\n[5/6] Loading mappings...")
    with open(contact_mapping_path) as f:
        contact_mapping = json.load(f)
    with open(dof_mapping_path) as f:
        dof_mapping = json.load(f)
    print(f"  Contact mapping: {len(contact_mapping)} bodies")

    # Map physics to SMPL format
    print("\n[6/6] Mapping physics to SMPL format...")

    # Map torques to SMPL DOFs
    print("  - Mapping torques to SMPL 69 DOFs...")
    sys.path.insert(0, str(Path(__file__).parent))
    from create_dof_mapping import map_torques_to_smpl
    gt_torques = map_torques_to_smpl(addb_torques, addb_dof_names, dof_mapping)
    print(f"    GT torques shape: {gt_torques.shape}")

    # Map GRF to SMPL vertices
    # NOTE: This requires SMPL vertex positions from forward pass
    # For now, create placeholder (needs SMPL model forward pass)
    print("  - Creating placeholder for GRF mapping (requires SMPL vertices)...")
    print("    WARNING: GRF mapping requires SMPL model forward pass to get vertex positions")
    print("    Creating zero placeholder. Implement SMPL forward pass for actual mapping.")
    gt_grf = np.zeros((T, 29, 3), dtype=np.float32)

    # Map contact states
    print("  - Creating placeholder for contact states...")
    gt_contacts = np.zeros((T, 29), dtype=np.uint8)

    # TODO: Implement actual GRF and contact mapping
    # This requires:
    # 1. Load SMPL model
    # 2. Forward pass with poses, betas, trans to get vertex positions
    # 3. Use create_contact_mapping.map_grf_to_smpl_vertices()

    # Save PhysPT format
    print(f"\n[Saving] Writing to {output_path}...")
    np.savez_compressed(
        output_path,
        poses_6d=poses_6d,      # (T, 144)
        betas=betas,            # (10,)
        trans=trans,            # (T, 3)
        gt_grf=gt_grf,         # (T, 29, 3) - PLACEHOLDER
        gt_torques=gt_torques, # (T, 69)
        contacts=gt_contacts,   # (T, 29) - PLACEHOLDER
    )

    print("\n" + "=" * 80)
    print("Conversion complete!")
    print("=" * 80)
    print(f"Output saved to: {output_path}")
    print()
    print("PhysPT format contents:")
    print(f"  - poses_6d: {poses_6d.shape}")
    print(f"  - betas: {betas.shape}")
    print(f"  - trans: {trans.shape}")
    print(f"  - gt_grf: {gt_grf.shape} (PLACEHOLDER - needs SMPL vertices)")
    print(f"  - gt_torques: {gt_torques.shape}")
    print(f"  - contacts: {gt_contacts.shape} (PLACEHOLDER)")
    print()
    print("=" * 80)
    print("TODO: Implement GRF and contact mapping")
    print("=" * 80)
    print("""
To complete GRF mapping, you need to:

1. Load SMPL model:
   from models.smpl_model import SMPLModel
   smpl = SMPLModel('path/to/smpl_model.pkl', device='cpu')

2. Forward pass to get vertices:
   for t in range(T):
       vertices = smpl.forward(betas, poses[t], trans[t])
       # vertices shape: (6890, 3)

3. Map GRF using contact mapping:
   from create_contact_mapping import map_grf_to_smpl_vertices
   gt_grf = map_grf_to_smpl_vertices(
       addb_grf, addb_cop, addb_body_names,
       vertices, contact_mapping
   )

4. Map contacts similarly
""")


def main():
    parser = argparse.ArgumentParser(
        description='Convert SMPL + Physics to PhysPT format'
    )
    parser.add_argument('--smpl_params', required=True,
                       help='Path to smpl_params.npz')
    parser.add_argument('--physics_data', required=True,
                       help='Path to physics_data.npz')
    parser.add_argument('--physics_metadata', required=True,
                       help='Path to physics_metadata.json')
    parser.add_argument('--contact_mapping', required=True,
                       help='Path to contact_mapping.json')
    parser.add_argument('--dof_mapping', required=True,
                       help='Path to dof_mapping.json')
    parser.add_argument('--output', required=True,
                       help='Output path for PhysPT format .npz')

    args = parser.parse_args()

    convert_to_physpt_format(
        args.smpl_params,
        args.physics_data,
        args.physics_metadata,
        args.contact_mapping,
        args.dof_mapping,
        args.output
    )


if __name__ == '__main__':
    main()
