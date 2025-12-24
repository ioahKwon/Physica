"""
Test joint mapping definitions.
"""

import pytest
from addb2skel.joint_definitions import (
    ADDB_JOINTS,
    SKEL_JOINTS,
    ADDB_TO_SKEL_MAPPING,
    ADDB_JOINT_TO_IDX,
    SKEL_JOINT_TO_IDX,
    build_direct_joint_mapping,
    MappingType,
)


def test_addb_joint_count():
    """AddB should have exactly 20 joints."""
    assert len(ADDB_JOINTS) == 20


def test_skel_joint_count():
    """SKEL should have exactly 24 joints."""
    assert len(SKEL_JOINTS) == 24


def test_joint_index_mapping():
    """Joint name to index mapping should be correct."""
    for i, name in enumerate(ADDB_JOINTS):
        assert ADDB_JOINT_TO_IDX[name] == i

    for i, name in enumerate(SKEL_JOINTS):
        assert SKEL_JOINT_TO_IDX[name] == i


def test_all_addb_joints_have_mapping():
    """Every AddB joint should have a mapping defined."""
    for joint in ADDB_JOINTS:
        assert joint in ADDB_TO_SKEL_MAPPING, f"Missing mapping for {joint}"


def test_direct_mappings():
    """Direct mappings should be 1:1."""
    for addb_name, mapping in ADDB_TO_SKEL_MAPPING.items():
        if mapping.mapping_type == MappingType.DIRECT:
            assert len(mapping.skel_joints) == 1, \
                f"Direct mapping {addb_name} should have exactly 1 SKEL joint"


def test_build_direct_joint_mapping():
    """build_direct_joint_mapping should return valid indices."""
    addb_idx, skel_idx = build_direct_joint_mapping()

    # Should have same length
    assert len(addb_idx) == len(skel_idx)

    # All indices should be valid
    for i in addb_idx:
        assert 0 <= i < len(ADDB_JOINTS)
    for i in skel_idx:
        assert 0 <= i < len(SKEL_JOINTS)


def test_acromial_mapping():
    """Acromial joints should have PROXY mapping type."""
    for side in ['acromial_r', 'acromial_l']:
        mapping = ADDB_TO_SKEL_MAPPING[side]
        assert mapping.mapping_type == MappingType.PROXY
        assert 'scapula' in mapping.skel_joints[0] or 'humerus' in mapping.skel_joints[0]


def test_elbow_to_ulna_mapping():
    """Elbow (AddB) should map to ulna (SKEL)."""
    mapping_r = ADDB_TO_SKEL_MAPPING['elbow_r']
    mapping_l = ADDB_TO_SKEL_MAPPING['elbow_l']

    assert 'ulna_r' in mapping_r.skel_joints
    assert 'ulna_l' in mapping_l.skel_joints


def test_radius_to_hand_mapping():
    """Radius (AddB) should map to hand (SKEL wrist)."""
    mapping_r = ADDB_TO_SKEL_MAPPING['radius_r']
    mapping_l = ADDB_TO_SKEL_MAPPING['radius_l']

    assert 'hand_r' in mapping_r.skel_joints
    assert 'hand_l' in mapping_l.skel_joints


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
