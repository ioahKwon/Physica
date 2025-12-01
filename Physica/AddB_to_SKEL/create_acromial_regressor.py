"""
Create Extended SKEL Joint Regressor with Acromial Joints

This script expands the SKEL J_regressor from 24 joints to 26 joints
by adding two new "acromial" measurement joints.

New joints:
- Index 24: acromial_r (right acromial / shoulder tip)
- Index 25: acromial_l (left acromial / shoulder tip)

These are "measurement-only" joints - they are NOT part of the kinematic chain
(LBS or FK), only used for loss computation.

Usage:
    python create_acromial_regressor.py --skel_path /path/to/skel_models --output /path/to/output.pkl

Acromion Vertex Indices (SMPL/SKEL standard):
- Left: [3321, 3325, 3290, 3340]
- Right: [5624, 5630, 5690, 5700]
"""

import os
import argparse
import pickle as pkl
import numpy as np
import scipy.sparse as sp
from typing import List, Optional, Dict, Tuple


# =============================================================================
# Default Acromion Vertex Indices
# =============================================================================

ACROMIAL_VERTEX_IDX = {
    # Left acromion cluster (SMPLify-X, ExPose standard)
    'left': [3321, 3325, 3290, 3340],
    'left_main': 3321,

    # Right acromion cluster
    'right': [5624, 5630, 5690, 5700],
    'right_main': 5624,
}


# =============================================================================
# Core Functions
# =============================================================================

def load_skel_regressor(skel_model_path: str, gender: str = 'male') -> np.ndarray:
    """
    Load the original SKEL J_regressor_osim.

    Args:
        skel_model_path: Path to SKEL model directory
        gender: 'male' or 'female'

    Returns:
        J_regressor: numpy array [24, V] where V is number of vertices
    """
    # Try different possible file locations
    possible_paths = [
        os.path.join(skel_model_path, f'skel_{gender}.pkl'),
        os.path.join(skel_model_path, f'SKEL_{gender}.pkl'),
        os.path.join(skel_model_path, gender, 'model.pkl'),
    ]

    model_data = None
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading SKEL model from: {path}")
            with open(path, 'rb') as f:
                model_data = pkl.load(f, encoding='latin1')
            break

    if model_data is None:
        raise FileNotFoundError(
            f"Could not find SKEL model. Tried paths:\n" +
            "\n".join(possible_paths)
        )

    # Extract J_regressor_osim
    if 'J_regressor_osim' in model_data:
        regressor = model_data['J_regressor_osim']
    elif 'J_regressor' in model_data:
        print("Warning: Using J_regressor instead of J_regressor_osim")
        regressor = model_data['J_regressor']
    else:
        raise KeyError("Could not find J_regressor or J_regressor_osim in model")

    # Convert to dense if sparse
    if sp.issparse(regressor):
        regressor = regressor.toarray()

    print(f"Original regressor shape: {regressor.shape}")
    return regressor


def create_acromial_regressor_rows(
    num_vertices: int,
    v_idx_r: List[int],
    v_idx_l: List[int],
    weights_r: Optional[np.ndarray] = None,
    weights_l: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create regressor rows for acromial joints.

    Each row is a sparse vector where vertices in the cluster have non-zero weights.

    Args:
        num_vertices: Number of vertices in the mesh (6890 for SMPL/SKEL)
        v_idx_r: Right acromial vertex indices
        v_idx_l: Left acromial vertex indices
        weights_r: Weights for right vertices (default: uniform)
        weights_l: Weights for left vertices (default: uniform)

    Returns:
        row_r: Regressor row for right acromial [V]
        row_l: Regressor row for left acromial [V]
    """
    # Default: uniform weights
    if weights_r is None:
        weights_r = np.ones(len(v_idx_r)) / len(v_idx_r)
    else:
        weights_r = weights_r / weights_r.sum()

    if weights_l is None:
        weights_l = np.ones(len(v_idx_l)) / len(v_idx_l)
    else:
        weights_l = weights_l / weights_l.sum()

    # Create rows
    row_r = np.zeros(num_vertices, dtype=np.float32)
    row_l = np.zeros(num_vertices, dtype=np.float32)

    for idx, w in zip(v_idx_r, weights_r):
        row_r[idx] = w

    for idx, w in zip(v_idx_l, weights_l):
        row_l[idx] = w

    return row_r, row_l


def extend_regressor(
    original_regressor: np.ndarray,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[np.ndarray] = None,
    weights_l: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Extend the J_regressor from [24, V] to [26, V].

    Args:
        original_regressor: Original [24, V] regressor
        v_idx_r: Right acromial vertex indices
        v_idx_l: Left acromial vertex indices
        weights_r: Optional custom weights for right
        weights_l: Optional custom weights for left

    Returns:
        extended_regressor: New [26, V] regressor
    """
    if v_idx_r is None:
        v_idx_r = ACROMIAL_VERTEX_IDX['right']
    if v_idx_l is None:
        v_idx_l = ACROMIAL_VERTEX_IDX['left']

    num_joints, num_vertices = original_regressor.shape
    print(f"Original shape: [{num_joints}, {num_vertices}]")

    # Create new rows
    row_r, row_l = create_acromial_regressor_rows(
        num_vertices, v_idx_r, v_idx_l, weights_r, weights_l
    )

    # Stack: [24, V] + [2, V] = [26, V]
    extended = np.vstack([
        original_regressor,
        row_r[np.newaxis, :],  # index 24: acromial_r
        row_l[np.newaxis, :],  # index 25: acromial_l
    ])

    print(f"Extended shape: {extended.shape}")
    return extended.astype(np.float32)


def save_extended_regressor(
    extended_regressor: np.ndarray,
    output_path: str,
    save_as_sparse: bool = False,
) -> None:
    """
    Save the extended regressor to a pickle file.

    Args:
        extended_regressor: The [26, V] regressor
        output_path: Path to save the pickle file
        save_as_sparse: If True, save as scipy sparse matrix
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    if save_as_sparse:
        regressor_to_save = sp.csr_matrix(extended_regressor)
    else:
        regressor_to_save = extended_regressor

    with open(output_path, 'wb') as f:
        pkl.dump(regressor_to_save, f)

    print(f"Saved extended regressor to: {output_path}")
    print(f"Shape: {extended_regressor.shape}")
    print(f"Format: {'sparse' if save_as_sparse else 'dense'}")


def verify_regressor(regressor_path: str) -> None:
    """
    Load and verify the saved regressor.
    """
    print("\n" + "=" * 50)
    print("Verifying saved regressor...")

    with open(regressor_path, 'rb') as f:
        loaded = pkl.load(f)

    if sp.issparse(loaded):
        loaded = loaded.toarray()

    print(f"Shape: {loaded.shape}")
    print(f"Dtype: {loaded.dtype}")

    # Check acromial rows
    row_r = loaded[24]
    row_l = loaded[25]

    print(f"\nRight acromial (idx 24):")
    print(f"  Non-zero entries: {np.count_nonzero(row_r)}")
    print(f"  Non-zero indices: {np.where(row_r > 0)[0].tolist()}")
    print(f"  Weights sum: {row_r.sum():.4f}")

    print(f"\nLeft acromial (idx 25):")
    print(f"  Non-zero entries: {np.count_nonzero(row_l)}")
    print(f"  Non-zero indices: {np.where(row_l > 0)[0].tolist()}")
    print(f"  Weights sum: {row_l.sum():.4f}")


# =============================================================================
# Extended Joint Names (for reference)
# =============================================================================

SKEL_JOINT_NAMES_EXTENDED = [
    # Original 24 joints (0-23)
    'pelvis',       # 0
    'femur_r',      # 1
    'tibia_r',      # 2
    'talus_r',      # 3
    'calcn_r',      # 4
    'toes_r',       # 5
    'femur_l',      # 6
    'tibia_l',      # 7
    'talus_l',      # 8
    'calcn_l',      # 9
    'toes_l',       # 10
    'lumbar',       # 11
    'thorax',       # 12
    'head',         # 13
    'scapula_r',    # 14
    'humerus_r',    # 15
    'ulna_r',       # 16
    'radius_r',     # 17
    'hand_r',       # 18
    'scapula_l',    # 19
    'humerus_l',    # 20
    'ulna_l',       # 21
    'radius_l',     # 22
    'hand_l',       # 23
    # New measurement joints (24-25)
    'acromial_r',   # 24 (NEW)
    'acromial_l',   # 25 (NEW)
]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create extended SKEL joint regressor with acromial joints'
    )
    parser.add_argument(
        '--skel_path',
        type=str,
        default='/egr/research-zijunlab/kwonjoon/01_Code/SKEL/skel/skel_models_v1.1',
        help='Path to SKEL model directory'
    )
    parser.add_argument(
        '--gender',
        type=str,
        default='male',
        choices=['male', 'female'],
        help='Gender of the model'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='./J_regressor_osim_acromial.pkl',
        help='Output path for extended regressor'
    )
    parser.add_argument(
        '--sparse',
        action='store_true',
        help='Save as sparse matrix'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify the saved regressor after creation'
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Creating Extended SKEL Joint Regressor")
    print("=" * 60)
    print()

    print("Acromial Vertex Indices:")
    print(f"  Right: {ACROMIAL_VERTEX_IDX['right']}")
    print(f"  Left:  {ACROMIAL_VERTEX_IDX['left']}")
    print()

    # Load original regressor
    original = load_skel_regressor(args.skel_path, args.gender)

    # Extend regressor
    extended = extend_regressor(original)

    # Save
    save_extended_regressor(extended, args.output, save_as_sparse=args.sparse)

    # Verify
    if args.verify:
        verify_regressor(args.output)

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)
    print()
    print("To use this regressor in SKEL:")
    print()
    print("  from skel import SKEL")
    print()
    print("  skel = SKEL(")
    print(f"      model_path='{args.skel_path}',")
    print(f"      custom_joint_reg_path='{args.output}'")
    print("  )")
    print()
    print("New joint indices:")
    print("  24: acromial_r")
    print("  25: acromial_l")


if __name__ == '__main__':
    main()
