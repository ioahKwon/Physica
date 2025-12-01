#!/usr/bin/env python3
"""
Joint mapping between AddBiomechanics/OpenSim and SMPL.

Provides automatic mapping based on joint names and manual override support.
"""

from typing import Dict, List, Set, Optional, Tuple
import json


# SMPL joint names (24 joints)
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

# Automatic mapping from common OpenSim/AddBiomechanics joint names to SMPL
AUTO_JOINT_MAPPING: Dict[str, str] = {
    # Pelvis/Root
    'ground_pelvis': 'pelvis',
    'pelvis': 'pelvis',
    'root': 'pelvis',
    'hip': 'pelvis',

    # Hips
    'hip_r': 'right_hip',
    'hip_right': 'right_hip',
    'r_hip': 'right_hip',
    'hip_l': 'left_hip',
    'hip_left': 'left_hip',
    'l_hip': 'left_hip',

    # Knees
    'walker_knee_r': 'right_knee',
    'knee_r': 'right_knee',
    'knee_right': 'right_knee',
    'r_knee': 'right_knee',
    'walker_knee_l': 'left_knee',
    'knee_l': 'left_knee',
    'knee_left': 'left_knee',
    'l_knee': 'left_knee',

    # Ankles
    'ankle_r': 'right_ankle',
    'ankle_right': 'right_ankle',
    'r_ankle': 'right_ankle',
    'ankle_l': 'left_ankle',
    'ankle_left': 'left_ankle',
    'l_ankle': 'left_ankle',
    'talus_r': 'right_ankle',
    'talus_l': 'left_ankle',

    # Feet
    'subtalar_r': 'right_foot',
    'mtp_r': 'right_foot',
    'calcn_r': 'right_foot',
    'foot_r': 'right_foot',
    'subtalar_l': 'left_foot',
    'mtp_l': 'left_foot',
    'calcn_l': 'left_foot',
    'foot_l': 'left_foot',

    # Spine
    'back': 'spine1',
    'torso': 'spine1',
    'spine': 'spine1',
    'lumbar': 'spine1',

    # Shoulders/Clavicles
    'acromial_r': 'right_shoulder',
    'shoulder_r': 'right_shoulder',
    'r_shoulder': 'right_shoulder',
    'acromial_l': 'left_shoulder',
    'shoulder_l': 'left_shoulder',
    'l_shoulder': 'left_shoulder',
    'clav_r': 'right_collar',
    'clav_l': 'left_collar',

    # Elbows
    'elbow_r': 'right_elbow',
    'r_elbow': 'right_elbow',
    'elbow_l': 'left_elbow',
    'l_elbow': 'left_elbow',

    # Wrists (skip intermediate joints like radioulnar)
    'wrist_r': 'right_wrist',
    'r_wrist': 'right_wrist',
    'radius_hand_r': 'right_wrist',
    'wrist_l': 'left_wrist',
    'l_wrist': 'left_wrist',
    'radius_hand_l': 'left_wrist',

    # Hands
    'hand_r': 'right_hand',
    'r_hand': 'right_hand',
    'hand_l': 'left_hand',
    'l_hand': 'left_hand',

    # Head/Neck
    'head': 'head',
    'skull': 'head',
    'neck': 'neck',
}


class JointMapper:
    """
    Maps joints from AddBiomechanics/OpenSim to SMPL.

    Supports automatic mapping based on joint names and manual overrides.
    """

    def __init__(
        self,
        source_joint_names: List[str],
        mapping_overrides: Optional[Dict[int, int]] = None
    ):
        """
        Initialize joint mapper.

        Args:
            source_joint_names: List of source joint names from AddBiomechanics
            mapping_overrides: Optional manual mapping overrides {source_idx: smpl_idx}
        """
        self.source_joint_names = source_joint_names
        self.num_source_joints = len(source_joint_names)

        # Create automatic mapping
        self.auto_mapping = self._create_auto_mapping()

        # Apply overrides if provided
        if mapping_overrides is not None:
            self.mapping = self.auto_mapping.copy()
            self.mapping.update(mapping_overrides)
        else:
            self.mapping = self.auto_mapping

        # Create reverse mapping (SMPL → source)
        self.reverse_mapping = {v: k for k, v in self.mapping.items()}

    def _create_auto_mapping(self) -> Dict[int, int]:
        """
        Create automatic joint mapping based on joint names.

        Returns:
            Dictionary mapping source joint indices to SMPL joint indices
        """
        mapping: Dict[int, int] = {}
        used_smpl: Set[int] = set()

        for source_idx, name in enumerate(self.source_joint_names):
            # Normalize name (lowercase, remove spaces/underscores)
            key = name.lower().replace(' ', '').replace('_', '')

            # Also try with underscores kept
            key_with_underscore = name.lower().replace(' ', '')

            # Try to find match in AUTO_JOINT_MAPPING
            smpl_name = None
            if key in {k.replace('_', ''): k for k in AUTO_JOINT_MAPPING}.values():
                # Find original key
                for orig_key, orig_val in AUTO_JOINT_MAPPING.items():
                    if orig_key.replace('_', '') == key:
                        smpl_name = orig_val
                        break

            if smpl_name is None and key_with_underscore in AUTO_JOINT_MAPPING:
                smpl_name = AUTO_JOINT_MAPPING[key_with_underscore]

            # If found, map to SMPL index
            if smpl_name is not None and smpl_name in SMPL_JOINT_NAMES:
                smpl_idx = SMPL_JOINT_NAMES.index(smpl_name)

                # Skip if already mapped (avoid duplicates)
                if smpl_idx in used_smpl:
                    continue

                mapping[source_idx] = smpl_idx
                used_smpl.add(smpl_idx)

        return mapping

    def get_mapped_indices(self) -> Tuple[List[int], List[int]]:
        """
        Get lists of mapped source and SMPL indices.

        Returns:
            (source_indices, smpl_indices) - paired lists of mapped joints
        """
        source_indices = sorted(self.mapping.keys())
        smpl_indices = [self.mapping[i] for i in source_indices]
        return source_indices, smpl_indices

    def get_unmapped_source_joints(self) -> List[str]:
        """Get list of unmapped source joint names."""
        mapped_indices = set(self.mapping.keys())
        unmapped = [
            self.source_joint_names[i]
            for i in range(self.num_source_joints)
            if i not in mapped_indices
        ]
        return unmapped

    def get_unmapped_smpl_joints(self) -> List[str]:
        """Get list of unmapped SMPL joint names."""
        mapped_smpl = set(self.mapping.values())
        unmapped = [
            SMPL_JOINT_NAMES[i]
            for i in range(len(SMPL_JOINT_NAMES))
            if i not in mapped_smpl
        ]
        return unmapped

    def is_joint_mapped(self, source_idx: int) -> bool:
        """Check if source joint is mapped to SMPL."""
        return source_idx in self.mapping

    def get_smpl_idx(self, source_idx: int) -> Optional[int]:
        """Get SMPL index for source joint (None if unmapped)."""
        return self.mapping.get(source_idx)

    def get_source_idx(self, smpl_idx: int) -> Optional[int]:
        """Get source index for SMPL joint (None if unmapped)."""
        return self.reverse_mapping.get(smpl_idx)

    def save_mapping(self, filepath: str) -> None:
        """Save mapping to JSON file."""
        data = {
            'source_joint_names': self.source_joint_names,
            'mapping': {
                str(k): int(v) for k, v in self.mapping.items()
            },
            'mapping_named': {
                self.source_joint_names[k]: SMPL_JOINT_NAMES[v]
                for k, v in self.mapping.items()
            }
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_mapping(cls, filepath: str, source_joint_names: List[str]) -> "JointMapper":
        """Load mapping from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Convert string keys back to int
        mapping_overrides = {
            int(k): int(v) for k, v in data['mapping'].items()
        }

        return cls(source_joint_names, mapping_overrides)

    def __repr__(self) -> str:
        num_mapped = len(self.mapping)
        num_unmapped_source = len(self.get_unmapped_source_joints())
        num_unmapped_smpl = len(self.get_unmapped_smpl_joints())

        return (
            f"JointMapper(\n"
            f"  source_joints={self.num_source_joints},\n"
            f"  mapped={num_mapped},\n"
            f"  unmapped_source={num_unmapped_source},\n"
            f"  unmapped_smpl={num_unmapped_smpl}\n"
            f")"
        )
