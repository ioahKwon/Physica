"""
Roundtrip test: SKEL → AddB (simulated) → SKEL reconstruction.

This tests the full pipeline by:
1. Generating SKEL poses and computing joints
2. Mapping SKEL joints to AddB-like format (simulated AddB data)
3. Running the conversion pipeline
4. Verifying the reconstruction matches the original
"""

import pytest
import numpy as np
import torch

# Skip if SKEL model not available
try:
    from addb2skel.skel_interface import create_skel_interface
    SKEL_AVAILABLE = True
except Exception:
    SKEL_AVAILABLE = False

from addb2skel.joint_definitions import (
    SKEL_JOINT_TO_IDX,
    ADDB_JOINT_TO_IDX,
    ADDB_TO_SKEL_MAPPING,
    MappingType,
)
from addb2skel.config import SKEL_NUM_POSE_DOF, SKEL_NUM_BETAS


def skel_to_addb_joints(skel_joints: np.ndarray) -> np.ndarray:
    """
    Map SKEL joints to AddB format (simulated).

    This is the inverse of what the pipeline does, used for testing.

    Args:
        skel_joints: SKEL joint positions [T, 24, 3] or [24, 3].

    Returns:
        addb_joints: Simulated AddB joints [T, 20, 3] or [20, 3].
    """
    if skel_joints.ndim == 2:
        skel_joints = skel_joints[np.newaxis]

    T = skel_joints.shape[0]
    addb_joints = np.zeros((T, 20, 3))

    # Direct mappings (reverse lookup)
    skel_to_addb_direct = {
        'pelvis': 'pelvis',
        'femur_r': 'femur_r',
        'femur_l': 'femur_l',
        'tibia_r': 'tibia_r',
        'tibia_l': 'tibia_l',
        'talus_r': 'talus_r',
        'talus_l': 'talus_l',
        'calcn_r': 'calcn_r',
        'calcn_l': 'calcn_l',
        'toes_r': 'toes_r',
        'toes_l': 'toes_l',
        'ulna_r': 'elbow_r',      # SKEL ulna = AddB elbow
        'ulna_l': 'elbow_l',
        'hand_r': 'radius_r',     # SKEL hand = AddB radius (wrist)
        'hand_l': 'radius_l',
    }

    for skel_name, addb_name in skel_to_addb_direct.items():
        skel_idx = SKEL_JOINT_TO_IDX[skel_name]
        addb_idx = ADDB_JOINT_TO_IDX[addb_name]
        addb_joints[:, addb_idx, :] = skel_joints[:, skel_idx, :]

    # Torso = average of lumbar and thorax
    addb_joints[:, ADDB_JOINT_TO_IDX['torso'], :] = (
        skel_joints[:, SKEL_JOINT_TO_IDX['lumbar'], :] +
        skel_joints[:, SKEL_JOINT_TO_IDX['thorax'], :]
    ) / 2

    # Acromial = approximate from humerus (slightly lateral/superior)
    # In reality AddB acromial is a surface landmark, but for testing
    # we approximate it from glenohumeral center
    offset_r = np.array([0.03, 0.02, 0])
    offset_l = np.array([-0.03, 0.02, 0])
    addb_joints[:, ADDB_JOINT_TO_IDX['acromial_r'], :] = (
        skel_joints[:, SKEL_JOINT_TO_IDX['humerus_r'], :] + offset_r
    )
    addb_joints[:, ADDB_JOINT_TO_IDX['acromial_l'], :] = (
        skel_joints[:, SKEL_JOINT_TO_IDX['humerus_l'], :] + offset_l
    )

    # Hand segment frames (same as wrist for testing)
    addb_joints[:, ADDB_JOINT_TO_IDX['hand_r'], :] = skel_joints[:, SKEL_JOINT_TO_IDX['hand_r'], :]
    addb_joints[:, ADDB_JOINT_TO_IDX['hand_l'], :] = skel_joints[:, SKEL_JOINT_TO_IDX['hand_l'], :]

    if T == 1:
        return addb_joints[0]
    return addb_joints


@pytest.mark.skipif(not SKEL_AVAILABLE, reason="SKEL model not available")
def test_roundtrip_tpose():
    """Test roundtrip with T-pose."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create SKEL interface
    skel = create_skel_interface(gender='male', device=str(device))

    # Generate T-pose
    betas = torch.zeros(SKEL_NUM_BETAS, device=device)
    poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=device)
    trans = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        _, skel_joints, _ = skel.forward(betas.unsqueeze(0), poses, trans)
        skel_joints_np = skel_joints[0].cpu().numpy()

    # Map to AddB format
    addb_joints = skel_to_addb_joints(skel_joints_np)

    # Verify shape
    assert addb_joints.shape == (20, 3)

    # Verify pelvis matches
    np.testing.assert_allclose(
        addb_joints[0],  # pelvis
        skel_joints_np[0],  # pelvis
        atol=1e-6
    )

    # Verify leg joints match
    for joint in ['femur_r', 'tibia_r', 'talus_r', 'toes_r']:
        addb_idx = ADDB_JOINT_TO_IDX[joint]
        skel_idx = SKEL_JOINT_TO_IDX[joint]
        np.testing.assert_allclose(
            addb_joints[addb_idx],
            skel_joints_np[skel_idx],
            atol=1e-6,
            err_msg=f"Mismatch at {joint}"
        )


@pytest.mark.skipif(not SKEL_AVAILABLE, reason="SKEL model not available")
def test_roundtrip_random_pose():
    """Test roundtrip with random pose."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    skel = create_skel_interface(gender='male', device=str(device))

    # Generate random pose (small values to avoid extreme poses)
    np.random.seed(42)
    betas = torch.zeros(SKEL_NUM_BETAS, device=device)
    poses = torch.from_numpy(np.random.randn(1, SKEL_NUM_POSE_DOF) * 0.2).float().to(device)
    trans = torch.zeros(1, 3, device=device)

    with torch.no_grad():
        _, skel_joints, _ = skel.forward(betas.unsqueeze(0), poses, trans)
        skel_joints_np = skel_joints[0].cpu().numpy()

    # Map to AddB format
    addb_joints = skel_to_addb_joints(skel_joints_np)

    # Verify shape
    assert addb_joints.shape == (20, 3)

    # Verify elbow matches ulna
    np.testing.assert_allclose(
        addb_joints[ADDB_JOINT_TO_IDX['elbow_r']],
        skel_joints_np[SKEL_JOINT_TO_IDX['ulna_r']],
        atol=1e-6
    )


@pytest.mark.skipif(not SKEL_AVAILABLE, reason="SKEL model not available")
def test_full_roundtrip():
    """Full roundtrip: SKEL → AddB → SKEL reconstruction."""
    from addb2skel.pipeline import convert_addb_to_skel
    from addb2skel.config import OptimizationConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate SKEL data
    skel = create_skel_interface(gender='male', device=str(device))

    betas_orig = torch.zeros(SKEL_NUM_BETAS, device=device)
    betas_orig[0] = -2.0  # Some shape variation

    np.random.seed(123)
    poses_orig = torch.from_numpy(np.random.randn(5, SKEL_NUM_POSE_DOF) * 0.1).float().to(device)
    trans_orig = torch.zeros(5, 3, device=device)

    with torch.no_grad():
        _, skel_joints_orig, _ = skel.forward(
            betas_orig.unsqueeze(0).expand(5, -1),
            poses_orig,
            trans_orig
        )
        skel_joints_np = skel_joints_orig.cpu().numpy()

    # Map to AddB format
    addb_joints = skel_to_addb_joints(skel_joints_np)

    # Quick config for faster test
    config = OptimizationConfig(
        scale_iters=50,
        pose_iters=100,
    )

    # Run conversion
    result = convert_addb_to_skel(
        addb_joints,
        gender='male',
        config=config,
        verbose=False,
    )

    # Check MPJPE is reasonable (should be < 50mm for this synthetic test)
    assert result.mpjpe_mm < 50, f"MPJPE too high: {result.mpjpe_mm:.1f}mm"

    print(f"Roundtrip MPJPE: {result.mpjpe_mm:.1f}mm")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
