"""
Joint definitions and mapping between AddBiomechanics and SKEL.

Based on the specification:
- AddB has 20 joints (internal anatomical joint centers from IK, not surface markers)
- SKEL has 24 joints

Key facts:
1. AddB joints are NOT surface markers - they are anatomical joint centers.
2. acromial_* in AddB are surface-aligned scapular landmarks, NOT glenohumeral centers.
3. SKEL has explicit scapula_* and humerus_* (glenohumeral) joints.
4. elbow_* (AddB) corresponds to ulna_* (SKEL elbow)
5. radius_* (AddB) corresponds to hand_* (SKEL wrist)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional


# =============================================================================
# AddBiomechanics Joint Definitions (20 joints)
# =============================================================================

ADDB_JOINTS = [
    'pelvis',        # 0
    'femur_r',       # 1
    'tibia_r',       # 2
    'talus_r',       # 3
    'calcn_r',       # 4  (rigid body frame / hindfoot segment ref)
    'toes_r',        # 5
    'femur_l',       # 6
    'tibia_l',       # 7
    'talus_l',       # 8
    'calcn_l',       # 9  (rigid body frame)
    'toes_l',        # 10
    'torso',         # 11 (simplified trunk)
    'acromial_r',    # 12 (surface landmark, NOT glenohumeral)
    'elbow_r',       # 13
    'radius_r',      # 14
    'hand_r',        # 15 (rigid body frame)
    'acromial_l',    # 16 (surface landmark, NOT glenohumeral)
    'elbow_l',       # 17
    'radius_l',      # 18
    'hand_l',        # 19 (rigid body frame)
]

ADDB_JOINT_TO_IDX = {name: idx for idx, name in enumerate(ADDB_JOINTS)}


# =============================================================================
# SKEL Joint Definitions (24 joints)
# =============================================================================

SKEL_JOINTS = [
    'pelvis',        # 0
    'femur_r',       # 1
    'tibia_r',       # 2
    'talus_r',       # 3
    'calcn_r',       # 4
    'toes_r',        # 5
    'femur_l',       # 6
    'tibia_l',       # 7
    'talus_l',       # 8
    'calcn_l',       # 9
    'toes_l',        # 10
    'lumbar',        # 11
    'thorax',        # 12
    'head',          # 13
    'scapula_r',     # 14
    'humerus_r',     # 15 (glenohumeral = true shoulder center)
    'ulna_r',        # 16 (elbow)
    'radius_r',      # 17
    'hand_r',        # 18 (wrist)
    'scapula_l',     # 19
    'humerus_l',     # 20 (glenohumeral)
    'ulna_l',        # 21 (elbow)
    'radius_l',      # 22
    'hand_l',        # 23 (wrist)
]

SKEL_JOINT_TO_IDX = {name: idx for idx, name in enumerate(SKEL_JOINTS)}


# =============================================================================
# Mapping Types
# =============================================================================

class MappingType(Enum):
    """Type of mapping between AddB and SKEL joints."""
    DIRECT = "direct"           # 1:1 correspondence
    PROXY = "proxy"             # Surface landmark to virtual point
    APPROXIMATE = "approximate" # Multiple joints involved
    SEGMENT = "segment"         # Segment frame reference
    NONE = "none"               # No direct mapping


@dataclass
class JointMapping:
    """Mapping from AddB joint to SKEL joint(s)."""
    addb_joint: str
    skel_joints: List[str]  # Can be multiple for APPROXIMATE
    mapping_type: MappingType
    description: str = ""
    use_in_loss: bool = True  # Whether to use in optimization loss


# =============================================================================
# AddB → SKEL Joint Correspondence
# =============================================================================

ADDB_TO_SKEL_MAPPING: Dict[str, JointMapping] = {
    # Lower body - Direct mappings
    'pelvis': JointMapping(
        'pelvis', ['pelvis'], MappingType.DIRECT,
        "Root joint - direct 1:1", use_in_loss=True
    ),
    'femur_r': JointMapping(
        'femur_r', ['femur_r'], MappingType.DIRECT,
        "Right hip - direct 1:1", use_in_loss=True
    ),
    'femur_l': JointMapping(
        'femur_l', ['femur_l'], MappingType.DIRECT,
        "Left hip - direct 1:1", use_in_loss=True
    ),
    'tibia_r': JointMapping(
        'tibia_r', ['tibia_r'], MappingType.DIRECT,
        "Right knee - direct 1:1", use_in_loss=True
    ),
    'tibia_l': JointMapping(
        'tibia_l', ['tibia_l'], MappingType.DIRECT,
        "Left knee - direct 1:1", use_in_loss=True
    ),
    'talus_r': JointMapping(
        'talus_r', ['talus_r'], MappingType.DIRECT,
        "Right ankle - direct 1:1", use_in_loss=True
    ),
    'talus_l': JointMapping(
        'talus_l', ['talus_l'], MappingType.DIRECT,
        "Left ankle - direct 1:1", use_in_loss=True
    ),
    'toes_r': JointMapping(
        'toes_r', ['toes_r'], MappingType.DIRECT,
        "Right toes - direct 1:1", use_in_loss=True
    ),
    'toes_l': JointMapping(
        'toes_l', ['toes_l'], MappingType.DIRECT,
        "Left toes - direct 1:1", use_in_loss=True
    ),

    # Segment frames - less reliable for loss
    'calcn_r': JointMapping(
        'calcn_r', ['calcn_r'], MappingType.SEGMENT,
        "Right hindfoot segment frame", use_in_loss=False
    ),
    'calcn_l': JointMapping(
        'calcn_l', ['calcn_l'], MappingType.SEGMENT,
        "Left hindfoot segment frame", use_in_loss=False
    ),

    # Spine - Approximate mapping
    'torso': JointMapping(
        'torso', ['lumbar', 'thorax'], MappingType.APPROXIMATE,
        "Torso → lumbar+thorax average", use_in_loss=True
    ),

    # Shoulder/Scapula - PROXY mappings (most complex)
    'acromial_r': JointMapping(
        'acromial_r', ['scapula_r', 'humerus_r'], MappingType.PROXY,
        "Right acromial landmark → virtual acromial on scapula/humerus",
        use_in_loss=True
    ),
    'acromial_l': JointMapping(
        'acromial_l', ['scapula_l', 'humerus_l'], MappingType.PROXY,
        "Left acromial landmark → virtual acromial on scapula/humerus",
        use_in_loss=True
    ),

    # Upper limb - Direct mappings with naming difference
    'elbow_r': JointMapping(
        'elbow_r', ['ulna_r'], MappingType.DIRECT,
        "Right elbow (AddB) → ulna_r (SKEL elbow)", use_in_loss=True
    ),
    'elbow_l': JointMapping(
        'elbow_l', ['ulna_l'], MappingType.DIRECT,
        "Left elbow (AddB) → ulna_l (SKEL elbow)", use_in_loss=True
    ),
    'radius_r': JointMapping(
        'radius_r', ['hand_r'], MappingType.DIRECT,
        "Right radius (AddB wrist-ish) → hand_r (SKEL wrist)", use_in_loss=True
    ),
    'radius_l': JointMapping(
        'radius_l', ['hand_l'], MappingType.DIRECT,
        "Left radius (AddB wrist-ish) → hand_l (SKEL wrist)", use_in_loss=True
    ),

    # Hand segment frames
    'hand_r': JointMapping(
        'hand_r', ['hand_r'], MappingType.SEGMENT,
        "Right hand segment frame", use_in_loss=False
    ),
    'hand_l': JointMapping(
        'hand_l', ['hand_l'], MappingType.SEGMENT,
        "Left hand segment frame", use_in_loss=False
    ),
}


# =============================================================================
# Build Index Mappings
# =============================================================================

def build_direct_joint_mapping() -> Tuple[List[int], List[int]]:
    """
    Build lists of (AddB indices, SKEL indices) for DIRECT mappings only.

    Returns:
        Tuple of (addb_indices, skel_indices) for joints with direct 1:1 mapping.
    """
    addb_indices = []
    skel_indices = []

    for addb_name, mapping in ADDB_TO_SKEL_MAPPING.items():
        if mapping.mapping_type == MappingType.DIRECT and mapping.use_in_loss:
            addb_idx = ADDB_JOINT_TO_IDX[addb_name]
            skel_idx = SKEL_JOINT_TO_IDX[mapping.skel_joints[0]]
            addb_indices.append(addb_idx)
            skel_indices.append(skel_idx)

    return addb_indices, skel_indices


def build_all_joint_mapping(
    include_types: Optional[List[MappingType]] = None
) -> Tuple[List[int], List[int]]:
    """
    Build joint mapping for specified mapping types.

    Args:
        include_types: List of MappingType to include. Default: DIRECT only.

    Returns:
        Tuple of (addb_indices, skel_indices).
    """
    if include_types is None:
        include_types = [MappingType.DIRECT]

    addb_indices = []
    skel_indices = []

    for addb_name, mapping in ADDB_TO_SKEL_MAPPING.items():
        if mapping.mapping_type in include_types and mapping.use_in_loss:
            addb_idx = ADDB_JOINT_TO_IDX[addb_name]
            # Use first SKEL joint for multi-joint mappings
            skel_idx = SKEL_JOINT_TO_IDX[mapping.skel_joints[0]]
            addb_indices.append(addb_idx)
            skel_indices.append(skel_idx)

    return addb_indices, skel_indices


# =============================================================================
# Bone Pair Definitions
# =============================================================================

# AddB bone pairs for direction/length loss
ADDB_BONE_PAIRS = [
    ('pelvis', 'femur_r'),
    ('pelvis', 'femur_l'),
    ('femur_r', 'tibia_r'),
    ('femur_l', 'tibia_l'),
    ('tibia_r', 'talus_r'),
    ('tibia_l', 'talus_l'),
    ('talus_r', 'toes_r'),
    ('talus_l', 'toes_l'),
    ('acromial_r', 'elbow_r'),   # Upper arm
    ('acromial_l', 'elbow_l'),
    ('elbow_r', 'radius_r'),     # Forearm
    ('elbow_l', 'radius_l'),
]

# Corresponding SKEL bone pairs
SKEL_BONE_PAIRS = [
    ('pelvis', 'femur_r'),
    ('pelvis', 'femur_l'),
    ('femur_r', 'tibia_r'),
    ('femur_l', 'tibia_l'),
    ('tibia_r', 'talus_r'),
    ('tibia_l', 'talus_l'),
    ('talus_r', 'toes_r'),
    ('talus_l', 'toes_l'),
    ('humerus_r', 'ulna_r'),     # Upper arm (from glenohumeral)
    ('humerus_l', 'ulna_l'),
    ('ulna_r', 'hand_r'),        # Forearm
    ('ulna_l', 'hand_l'),
]

# Reliable bones for scale estimation (not affected by scapula uncertainty)
RELIABLE_BONE_PAIRS_ADDB = [
    ('pelvis', 'femur_r'),
    ('pelvis', 'femur_l'),
    ('femur_r', 'tibia_r'),
    ('femur_l', 'tibia_l'),
    ('tibia_r', 'talus_r'),
    ('tibia_l', 'talus_l'),
    ('elbow_r', 'radius_r'),
    ('elbow_l', 'radius_l'),
]

RELIABLE_BONE_PAIRS_SKEL = [
    ('pelvis', 'femur_r'),
    ('pelvis', 'femur_l'),
    ('femur_r', 'tibia_r'),
    ('femur_l', 'tibia_l'),
    ('tibia_r', 'talus_r'),
    ('tibia_l', 'talus_l'),
    ('ulna_r', 'hand_r'),
    ('ulna_l', 'hand_l'),
]


def get_bone_indices(
    bone_pairs: List[Tuple[str, str]],
    joint_to_idx: Dict[str, int]
) -> List[Tuple[int, int]]:
    """Convert bone pairs from names to indices."""
    return [(joint_to_idx[a], joint_to_idx[b]) for a, b in bone_pairs]


# =============================================================================
# Shoulder-specific indices
# =============================================================================

# AddB acromial indices
ADDB_ACROMIAL_R_IDX = ADDB_JOINT_TO_IDX['acromial_r']  # 12
ADDB_ACROMIAL_L_IDX = ADDB_JOINT_TO_IDX['acromial_l']  # 16

# SKEL shoulder-related indices
SKEL_SCAPULA_R_IDX = SKEL_JOINT_TO_IDX['scapula_r']    # 14
SKEL_SCAPULA_L_IDX = SKEL_JOINT_TO_IDX['scapula_l']    # 19
SKEL_HUMERUS_R_IDX = SKEL_JOINT_TO_IDX['humerus_r']    # 15
SKEL_HUMERUS_L_IDX = SKEL_JOINT_TO_IDX['humerus_l']    # 20

# SKEL spine indices
SKEL_LUMBAR_IDX = SKEL_JOINT_TO_IDX['lumbar']          # 11
SKEL_THORAX_IDX = SKEL_JOINT_TO_IDX['thorax']          # 12
