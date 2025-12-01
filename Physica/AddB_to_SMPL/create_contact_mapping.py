#!/usr/bin/env python3
"""
AddBiomechanics contact bodies → SMPL 29 contact vertices 매핑 생성

CoP(Center of Pressure) 기반 Gaussian distribution으로
body-level forces를 vertex-level forces로 분배

Usage:
    python create_contact_mapping.py

Output:
    contact_mapping.json - AddB body → SMPL vertices mapping
"""

import json
import numpy as np
from pathlib import Path


# SMPL 29 contact vertices (PhysPT에서 사용)
# 실제 vertex indices는 SMPL mesh topology에 따라 조정 필요
SMPL_CONTACT_VERTICES = {
    'left_heel': [3387, 3388, 3389, 3390, 3391],
    'left_toes': [3460, 3461, 3462, 3463, 3464, 3465],
    'right_heel': [6728, 6729, 6730, 6731, 6732],
    'right_toes': [6801, 6802, 6803, 6804, 6805, 6806],
    # 손 (With_Arm 데이터용)
    'left_hand': [2445, 2446, 2447, 2448, 2449],
    'right_hand': [5782, 5783, 5784, 5785, 5786],
}

# AddBiomechanics contact body names → SMPL body parts
# OpenSim 표준 body names 기준
ADDB_TO_SMPL_CONTACT_MAPPING = {
    # 발
    'calcn_r': 'right_heel',
    'calcn_l': 'left_heel',
    'toes_r': 'right_toes',
    'toes_l': 'left_toes',

    # 손 (With_Arm 데이터용)
    'hand_r': 'right_hand',
    'hand_l': 'left_hand',

    # 추가 가능한 contact bodies (필요시)
    'talus_r': 'right_heel',  # 발목 → heel로 근사
    'talus_l': 'left_heel',

    # 다른 가능한 naming conventions
    'calcn_right': 'right_heel',
    'calcn_left': 'left_heel',
    'toes_right': 'right_toes',
    'toes_left': 'left_toes',
}


def create_contact_mapping(sigma: float = 0.05) -> dict:
    """
    AddB contact bodies → SMPL vertices 매핑 테이블 생성

    Args:
        sigma: Gaussian weighting의 표준편차 (미터 단위)
               작을수록 CoP 근처에 force 집중
               기본값 0.05m = 5cm

    Returns:
        매핑 딕셔너리:
        {
            "calcn_r": {
                "smpl_part": "right_heel",
                "smpl_vertices": [6728, 6729, ...],
                "sigma": 0.05,
                "distribution_method": "gaussian_cop"
            },
            ...
        }
    """

    mapping = {}

    for addb_body, smpl_part in ADDB_TO_SMPL_CONTACT_MAPPING.items():
        if smpl_part in SMPL_CONTACT_VERTICES:
            mapping[addb_body] = {
                'smpl_part': smpl_part,
                'smpl_vertices': SMPL_CONTACT_VERTICES[smpl_part],
                'sigma': sigma,
                'distribution_method': 'gaussian_cop',
                'description': f'Map {addb_body} GRF to SMPL {smpl_part} vertices using CoP-based Gaussian distribution'
            }

    return mapping


def map_grf_to_smpl_vertices(
    addb_grf: np.ndarray,  # (T, N_addb_bodies, 3)
    addb_cop: np.ndarray,  # (T, N_addb_bodies, 3)
    addb_body_names: list,  # (N_addb_bodies,)
    smpl_vertices: np.ndarray,  # (6890, 3) SMPL vertex positions
    contact_mapping: dict
) -> np.ndarray:
    """
    AddB body-level GRF → SMPL vertex-level GRF 변환

    CoP 기반 Gaussian distribution 사용:
    - CoP로부터 가까운 vertices에 더 큰 weight
    - weight_i = exp(-dist_i² / (2*σ²))
    - Normalize하여 총 force 보존

    Args:
        addb_grf: AddB ground reaction forces (T, N_addb_bodies, 3)
        addb_cop: AddB center of pressure (T, N_addb_bodies, 3)
        addb_body_names: AddB contact body names (N_addb_bodies,)
        smpl_vertices: SMPL vertex positions (6890, 3) - from SMPL forward pass
        contact_mapping: Mapping dict from create_contact_mapping()

    Returns:
        smpl_grf: (T, 29, 3) SMPL contact vertices의 force
    """
    T = addb_grf.shape[0]
    smpl_grf = np.zeros((T, 29, 3), dtype=np.float32)

    # 각 AddB contact body에 대해
    for body_idx, body_name in enumerate(addb_body_names):
        if body_name not in contact_mapping:
            continue

        mapping_info = contact_mapping[body_name]
        smpl_vertex_ids = mapping_info['smpl_vertices']
        sigma = mapping_info['sigma']

        # 각 프레임에 대해
        for t in range(T):
            force = addb_grf[t, body_idx]  # (3,)
            cop_pos = addb_cop[t, body_idx]  # (3,)

            # Skip if no force
            if np.linalg.norm(force) < 1e-6:
                continue

            # SMPL vertices의 위치
            vertex_positions = smpl_vertices[smpl_vertex_ids]  # (N_verts, 3)

            # CoP로부터 거리 계산
            distances = np.linalg.norm(vertex_positions - cop_pos, axis=1)  # (N_verts,)

            # Gaussian weighting
            weights = np.exp(- distances**2 / (2 * sigma**2))
            weights /= (weights.sum() + 1e-10)  # Normalize

            # Force 분배
            for i, vid in enumerate(smpl_vertex_ids):
                smpl_grf[t, vid] += force * weights[i]

    return smpl_grf


def main():
    print("=" * 80)
    print("AddBiomechanics → SMPL Contact Mapping Generator")
    print("=" * 80)
    print()

    # Create mapping with default sigma
    sigma = 0.05  # 5cm
    contact_mapping = create_contact_mapping(sigma=sigma)

    # Save to JSON
    output_path = Path(__file__).parent / 'contact_mapping.json'
    with open(output_path, 'w') as f:
        json.dump(contact_mapping, f, indent=2)

    print(f"✓ Contact mapping saved to: {output_path}")
    print()
    print(f"Mapped {len(contact_mapping)} AddB contact bodies:")
    print()

    # Group by part
    parts = {}
    for addb_body, info in contact_mapping.items():
        part = info['smpl_part']
        if part not in parts:
            parts[part] = []
        parts[part].append((addb_body, len(info['smpl_vertices'])))

    for part, bodies in sorted(parts.items()):
        print(f"  {part}:")
        for addb_body, num_verts in bodies:
            print(f"    - {addb_body:15s} → {num_verts} vertices")
        print()

    print(f"Gaussian sigma: {sigma}m ({sigma*1000:.0f}mm)")
    print()
    print("=" * 80)
    print("Usage Example:")
    print("=" * 80)
    print("""
# Load mapping
with open('contact_mapping.json') as f:
    mapping = json.load(f)

# Map GRF to SMPL vertices
from create_contact_mapping import map_grf_to_smpl_vertices
smpl_grf = map_grf_to_smpl_vertices(
    addb_grf=grf,              # (T, N_bodies, 3)
    addb_cop=cop,              # (T, N_bodies, 3)
    addb_body_names=body_names,  # list of body names
    smpl_vertices=vertices,    # (6890, 3) from SMPL forward pass
    contact_mapping=mapping
)
# smpl_grf shape: (T, 29, 3)
""")


if __name__ == '__main__':
    main()
