#!/usr/bin/env python3
"""
AddBiomechanics DOFs → SMPL 69 DOFs 매핑 생성

SMPL 69 DOFs 구조:
- Pelvis: 6 DOF (translation 3 + rotation 3)
- Body joints (23): 3 DOF each = 69 total
- 손가락, 발가락 포함

Usage:
    python create_dof_mapping.py

Output:
    dof_mapping.json - AddB DOF → SMPL DOF mapping
"""

import json
import numpy as np
from pathlib import Path


# SMPL joint names (24 joints, pelvis부터 시작)
SMPL_JOINT_NAMES = [
    'pelvis',           # 0: 6 DOF (tx, ty, tz, rx, ry, rz)
    'left_hip',         # 1: 3 DOF
    'right_hip',        # 2: 3 DOF
    'spine1',           # 3: 3 DOF
    'left_knee',        # 4: 3 DOF
    'right_knee',       # 5: 3 DOF
    'spine2',           # 6: 3 DOF
    'left_ankle',       # 7: 3 DOF
    'right_ankle',      # 8: 3 DOF
    'spine3',           # 9: 3 DOF
    'left_foot',        # 10: 3 DOF
    'right_foot',       # 11: 3 DOF
    'neck',             # 12: 3 DOF
    'left_collar',      # 13: 3 DOF
    'right_collar',     # 14: 3 DOF
    'head',             # 15: 3 DOF
    'left_shoulder',    # 16: 3 DOF
    'right_shoulder',   # 17: 3 DOF
    'left_elbow',       # 18: 3 DOF
    'right_elbow',      # 19: 3 DOF
    'left_wrist',       # 20: 3 DOF
    'right_wrist',      # 21: 3 DOF
    'left_hand',        # 22: 3 DOF
    'right_hand',       # 23: 3 DOF
]

# AddBiomechanics DOF names (OpenSim 기준, 각 subject마다 다를 수 있음)
# 일반적인 패턴만 제공 (template)
ADDB_TO_SMPL_DOF_MAPPING_TEMPLATE = {
    # Pelvis (root) - 6 DOF
    'pelvis_tx': (0, 'pelvis_tx'),
    'pelvis_ty': (1, 'pelvis_ty'),
    'pelvis_tz': (2, 'pelvis_tz'),
    'pelvis_tilt': (3, 'pelvis_rx'),      # tilt = rotation around x
    'pelvis_list': (4, 'pelvis_ry'),      # list = rotation around y
    'pelvis_rotation': (5, 'pelvis_rz'),  # rotation = rotation around z

    # Hip - 각 3 DOF
    'hip_flexion_r': (6, 'right_hip_rx'),
    'hip_adduction_r': (7, 'right_hip_ry'),
    'hip_rotation_r': (8, 'right_hip_rz'),
    'hip_flexion_l': (9, 'left_hip_rx'),
    'hip_adduction_l': (10, 'left_hip_ry'),
    'hip_rotation_l': (11, 'left_hip_rz'),

    # Lumbar (spine1) - 3 DOF
    'lumbar_extension': (12, 'spine1_rx'),
    'lumbar_bending': (13, 'spine1_ry'),
    'lumbar_rotation': (14, 'spine1_rz'),

    # Knee - 주로 1 DOF만 사용 (flexion)
    'knee_angle_r': (15, 'right_knee_rx'),
    'knee_angle_l': (18, 'left_knee_rx'),

    # Ankle - 2-3 DOF
    'ankle_angle_r': (21, 'right_ankle_rx'),
    'ankle_angle_l': (24, 'left_ankle_rx'),

    # Subtalar (foot) - 발목 회전
    'subtalar_angle_r': (33, 'right_foot_rx'),
    'subtalar_angle_l': (30, 'left_foot_rx'),

    # 추가 매핑 (데이터에 따라)
    # Shoulder, elbow, wrist 등
}


def create_dof_mapping() -> dict:
    """
    AddB DOFs → SMPL 69 DOFs 매핑 생성

    Returns:
        {
            "template_mappings": {addb_dof_name: (smpl_dof_idx, smpl_dof_name)},
            "smpl_dof_names": [ordered list of SMPL DOF names],
            "approximations": {
                smpl_dof_range: {
                    "source": addb_dof_name,
                    "scale": float,
                    "description": str
                }
            },
            "zeros": [smpl_dof_indices to set to zero],
            "notes": str
        }
    """

    # SMPL 69 DOF names (expanded)
    smpl_dof_names = []

    # Pelvis: 6 DOF
    smpl_dof_names += ['pelvis_tx', 'pelvis_ty', 'pelvis_tz',
                       'pelvis_rx', 'pelvis_ry', 'pelvis_rz']  # 0-5

    # Other joints: 3 DOF each (rx, ry, rz)
    for joint_name in SMPL_JOINT_NAMES[1:]:  # Skip pelvis
        smpl_dof_names += [f'{joint_name}_rx',
                          f'{joint_name}_ry',
                          f'{joint_name}_rz']

    # SMPL actually has 72 DOFs: pelvis 6 + 23 joints * 3 = 75 - 3 (pelvis already counted) = 72
    # But PhysPT uses 69 DOFs (likely excluding some end effectors)
    # For now, we'll use 72 and trim/pad as needed
    assert len(smpl_dof_names) == 6 + 23 * 3, f"Expected {6 + 23 * 3} DOFs, got {len(smpl_dof_names)}"

    # Template direct mappings (확실한 대응)
    template_mappings = ADDB_TO_SMPL_DOF_MAPPING_TEMPLATE.copy()

    # Approximations (근사 매핑)
    # 예: 발가락 DOF는 ankle torque의 10%로 근사
    approximations = {
        '30-32': {  # left_foot DOFs
            'source': 'ankle_angle_l',
            'scale': 0.1,
            'description': 'Approximate toe DOFs from ankle'
        },
        '33-35': {  # right_foot DOFs
            'source': 'ankle_angle_r',
            'scale': 0.1,
            'description': 'Approximate toe DOFs from ankle'
        },
        '66-68': {  # right_hand DOFs
            'source': 'wrist_angle_r',
            'scale': 0.05,
            'description': 'Approximate hand DOFs from wrist (if available)'
        },
        '63-65': {  # left_hand DOFs
            'source': 'wrist_angle_l',
            'scale': 0.05,
            'description': 'Approximate hand DOFs from wrist (if available)'
        }
    }

    # Zeros (손가락 등, AddB에 없는 DOF) - 대부분 approximation으로 대체 가능
    zeros = []

    return {
        'template_mappings': {k: {'smpl_dof_idx': v[0], 'smpl_dof_name': v[1]}
                             for k, v in template_mappings.items()},
        'smpl_dof_names': smpl_dof_names,
        'approximations': approximations,
        'zeros': zeros,
        'notes': (
            'This is a TEMPLATE mapping. Actual AddB DOF names vary by subject/dataset. '
            'Use this as a reference and adjust mapping based on actual DOF names from physics_metadata.json. '
            'Common variations: hip_flexion vs hip_flexion_r, knee_angle vs knee_flexion, etc.'
        )
    }


def map_torques_to_smpl(
    addb_torques: np.ndarray,  # (T, N_addb_dof)
    addb_dof_names: list,      # (N_addb_dof,)
    dof_mapping: dict
) -> np.ndarray:
    """
    AddB joint torques → SMPL 69 DOFs 변환

    Args:
        addb_torques: AddB joint torques (T, N_addb_dof)
        addb_dof_names: AddB DOF names (N_addb_dof,)
        dof_mapping: Mapping dict from create_dof_mapping()

    Returns:
        smpl_torques: (T, 69)
    """
    T = addb_torques.shape[0]
    smpl_torques = np.zeros((T, 69), dtype=np.float32)

    # Create name to index mapping for AddB
    addb_name_to_idx = {name.lower(): idx for idx, name in enumerate(addb_dof_names)}

    # Direct mappings
    template_mappings = dof_mapping['template_mappings']
    for addb_name, mapping_info in template_mappings.items():
        addb_name_lower = addb_name.lower()
        if addb_name_lower in addb_name_to_idx:
            addb_idx = addb_name_to_idx[addb_name_lower]
            smpl_idx = mapping_info['smpl_dof_idx']
            if smpl_idx < 69:
                smpl_torques[:, smpl_idx] = addb_torques[:, addb_idx]

    # Approximations
    approximations = dof_mapping['approximations']
    for smpl_range_str, approx_info in approximations.items():
        start, end = map(int, smpl_range_str.split('-'))
        source_name = approx_info['source'].lower()
        scale = approx_info['scale']

        if source_name in addb_name_to_idx:
            addb_idx = addb_name_to_idx[source_name]
            for smpl_idx in range(start, end + 1):
                if smpl_idx < 69:
                    smpl_torques[:, smpl_idx] = addb_torques[:, addb_idx] * scale

    # Zeros already initialized to 0

    return smpl_torques


def main():
    print("=" * 80)
    print("AddBiomechanics → SMPL DOF Mapping Generator")
    print("=" * 80)
    print()

    # Create mapping
    dof_mapping = create_dof_mapping()

    # Save to JSON
    output_path = Path(__file__).parent / 'dof_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(dof_mapping, f, indent=2)

    print(f"✓ DOF mapping saved to: {output_path}")
    print()
    print(f"Template direct mappings: {len(dof_mapping['template_mappings'])}")
    print(f"Approximations: {len(dof_mapping['approximations'])}")
    print(f"Zeros: {len(dof_mapping['zeros'])}")
    print()
    print("SMPL DOF structure (69 DOFs total):")
    print("  - Pelvis: 6 DOF (translation 3 + rotation 3)")
    print("  - Other joints (23): 3 DOF each (rx, ry, rz)")
    print()
    print("=" * 80)
    print("IMPORTANT NOTE:")
    print("=" * 80)
    print(dof_mapping['notes'])
    print()
    print("=" * 80)
    print("Usage Example:")
    print("=" * 80)
    print("""
# Load mapping
with open('dof_mapping.json') as f:
    mapping = json.load(f)

# Load physics metadata to get actual DOF names
with open('physics_metadata.json') as f:
    meta = json.load(f)
    addb_dof_names = meta['dof_names']

# Map torques to SMPL
from create_dof_mapping import map_torques_to_smpl
smpl_torques = map_torques_to_smpl(
    addb_torques=torques,      # (T, N_addb_dof)
    addb_dof_names=addb_dof_names,  # list of DOF names
    dof_mapping=mapping
)
# smpl_torques shape: (T, 69)
""")


if __name__ == '__main__':
    main()
