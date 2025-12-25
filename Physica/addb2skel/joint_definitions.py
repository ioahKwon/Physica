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
# NOTE: These are the ACTUAL joint names from b3d files (e.g., Subject11.b3d)
# The names are nimblephysics-style, not OpenSim-style

ADDB_JOINTS = [
    'ground_pelvis', # 0  (pelvis root)
    'hip_r',         # 1  (right hip / femur_r)
    'walker_knee_r', # 2  (right knee / tibia_r)
    'ankle_r',       # 3  (right ankle / talus_r)
    'subtalar_r',    # 4  (right hindfoot / calcn_r)
    'mtp_r',         # 5  (right toe / toes_r)
    'hip_l',         # 6  (left hip / femur_l)
    'walker_knee_l', # 7  (left knee / tibia_l)
    'ankle_l',       # 8  (left ankle / talus_l)
    'subtalar_l',    # 9  (left hindfoot / calcn_l)
    'mtp_l',         # 10 (left toe / toes_l)
    'back',          # 11 (spine / lumbar)
    'acromial_r',    # 12 (right shoulder surface landmark)
    'elbow_r',       # 13 (right elbow)
    'radioulnar_r',  # 14 (right wrist / radius_r)
    'radius_hand_r', # 15 (right hand frame)
    'acromial_l',    # 16 (left shoulder surface landmark)
    'elbow_l',       # 17 (left elbow)
    'radioulnar_l',  # 18 (left wrist / radius_l)
    'radius_hand_l', # 19 (left hand frame)
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
# Uses actual nimblephysics joint names from b3d files

ADDB_TO_SKEL_MAPPING: Dict[str, JointMapping] = {
    # Pelvis (root)
    'ground_pelvis': JointMapping(
        'ground_pelvis', ['pelvis'], MappingType.DIRECT,
        "Root joint - direct 1:1", use_in_loss=True
    ),

    # Right leg
    'hip_r': JointMapping(
        'hip_r', ['femur_r'], MappingType.DIRECT,
        "Right hip → femur_r", use_in_loss=True
    ),
    'walker_knee_r': JointMapping(
        'walker_knee_r', ['tibia_r'], MappingType.DIRECT,
        "Right knee → tibia_r", use_in_loss=True
    ),
    'ankle_r': JointMapping(
        'ankle_r', ['talus_r'], MappingType.DIRECT,
        "Right ankle → talus_r", use_in_loss=True
    ),
    'subtalar_r': JointMapping(
        'subtalar_r', ['calcn_r'], MappingType.SEGMENT,
        "Right hindfoot segment frame", use_in_loss=False
    ),
    'mtp_r': JointMapping(
        'mtp_r', ['toes_r'], MappingType.DIRECT,
        "Right toe → toes_r", use_in_loss=True
    ),

    # Left leg
    'hip_l': JointMapping(
        'hip_l', ['femur_l'], MappingType.DIRECT,
        "Left hip → femur_l", use_in_loss=True
    ),
    'walker_knee_l': JointMapping(
        'walker_knee_l', ['tibia_l'], MappingType.DIRECT,
        "Left knee → tibia_l", use_in_loss=True
    ),
    'ankle_l': JointMapping(
        'ankle_l', ['talus_l'], MappingType.DIRECT,
        "Left ankle → talus_l", use_in_loss=True
    ),
    'subtalar_l': JointMapping(
        'subtalar_l', ['calcn_l'], MappingType.SEGMENT,
        "Left hindfoot segment frame", use_in_loss=False
    ),
    'mtp_l': JointMapping(
        'mtp_l', ['toes_l'], MappingType.DIRECT,
        "Left toe → toes_l", use_in_loss=True
    ),

    # Spine - back → lumbar (per working compare_smpl_skel.py: back → lumbar_body)
    'back': JointMapping(
        'back', ['lumbar'], MappingType.DIRECT,
        "Back → lumbar for optimization", use_in_loss=True
    ),

    # Right arm
    # Acromial → scapula (not humerus) - scapula is more lateral, closer to surface
    # Note: humerus (glenohumeral) is ~30-40mm medial to acromial, causing narrow shoulders
    'acromial_r': JointMapping(
        'acromial_r', ['scapula_r'], MappingType.DIRECT,
        "Right acromial → scapula (more lateral than humerus)",
        use_in_loss=True
    ),
    'elbow_r': JointMapping(
        'elbow_r', ['ulna_r'], MappingType.DIRECT,
        "Right elbow → ulna_r", use_in_loss=True
    ),
    'radioulnar_r': JointMapping(
        'radioulnar_r', ['radius_r'], MappingType.DIRECT,
        "Right wrist → radius_r", use_in_loss=True
    ),
    'radius_hand_r': JointMapping(
        'radius_hand_r', ['hand_r'], MappingType.DIRECT,
        "Right hand → hand_r", use_in_loss=True
    ),

    # Left arm
    'acromial_l': JointMapping(
        'acromial_l', ['scapula_l'], MappingType.DIRECT,
        "Left acromial → scapula (more lateral than humerus)",
        use_in_loss=True
    ),
    'elbow_l': JointMapping(
        'elbow_l', ['ulna_l'], MappingType.DIRECT,
        "Left elbow → ulna_l", use_in_loss=True
    ),
    'radioulnar_l': JointMapping(
        'radioulnar_l', ['radius_l'], MappingType.DIRECT,
        "Left wrist → radius_l", use_in_loss=True
    ),
    'radius_hand_l': JointMapping(
        'radius_hand_l', ['hand_l'], MappingType.DIRECT,
        "Left hand → hand_l", use_in_loss=True
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
# Uses actual nimblephysics joint names from b3d files

# AddB bone pairs for direction/length loss
ADDB_BONE_PAIRS = [
    ('ground_pelvis', 'hip_r'),        # Pelvis to right hip
    ('ground_pelvis', 'hip_l'),        # Pelvis to left hip
    ('hip_r', 'walker_knee_r'),        # Right thigh
    ('hip_l', 'walker_knee_l'),        # Left thigh
    ('walker_knee_r', 'ankle_r'),      # Right shin
    ('walker_knee_l', 'ankle_l'),      # Left shin
    ('ankle_r', 'mtp_r'),              # Right foot
    ('ankle_l', 'mtp_l'),              # Left foot
    ('acromial_r', 'elbow_r'),         # Right upper arm
    ('acromial_l', 'elbow_l'),         # Left upper arm
    ('elbow_r', 'radioulnar_r'),       # Right forearm
    ('elbow_l', 'radioulnar_l'),       # Left forearm
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
    ('humerus_r', 'ulna_r'),           # Upper arm (from glenohumeral)
    ('humerus_l', 'ulna_l'),
    ('ulna_r', 'radius_r'),             # Forearm
    ('ulna_l', 'radius_l'),
]

# Reliable bones for scale estimation (not affected by scapula uncertainty)
RELIABLE_BONE_PAIRS_ADDB = [
    ('ground_pelvis', 'hip_r'),
    ('ground_pelvis', 'hip_l'),
    ('hip_r', 'walker_knee_r'),
    ('hip_l', 'walker_knee_l'),
    ('walker_knee_r', 'ankle_r'),
    ('walker_knee_l', 'ankle_l'),
    ('elbow_r', 'radioulnar_r'),
    ('elbow_l', 'radioulnar_l'),
]

RELIABLE_BONE_PAIRS_SKEL = [
    ('pelvis', 'femur_r'),
    ('pelvis', 'femur_l'),
    ('femur_r', 'tibia_r'),
    ('femur_l', 'tibia_l'),
    ('tibia_r', 'talus_r'),
    ('tibia_l', 'talus_l'),
    ('ulna_r', 'radius_r'),
    ('ulna_l', 'radius_l'),
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
