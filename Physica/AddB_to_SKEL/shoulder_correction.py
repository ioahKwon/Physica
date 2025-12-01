"""
Shoulder Correction Module for AddB → SKEL Optimization

This module provides functions to correct the shoulder joint definition mismatch
between AddBiomechanics and SKEL models.

Key Components:
1. Virtual Acromial Joint computation (vertex-based)
2. Acromial matching loss
3. Shoulder width constraint loss
4. Shoulder direction constraint loss

Acromion Vertex Indices (SMPL/SKEL standard landmarks):
- Left acromion cluster: [3321, 3325, 3290, 3340] (main: 3321)
- Right acromion cluster: [5624, 5630, 5690, 5700] (main: 5624)
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict


# =============================================================================
# Constants: Acromion Vertex Indices
# =============================================================================

ACROMIAL_VERTEX_IDX = {
    # SKEL-specific acromion vertices (found via find_shoulder_vertices.py)
    # These are vertices near the humerus joint, slightly lateral/superior
    # NOTE: SMPLify-X/ExPose indices do NOT work for SKEL (different topology)

    # Left acromion cluster (SKEL)
    'left': [635, 636, 1830, 1829],
    'left_main': 635,

    # Right acromion cluster (SKEL)
    'right': [4125, 4124, 5293, 5290],
    'right_main': 4125,

    # Legacy SMPLify-X indices (DO NOT USE FOR SKEL)
    # 'smplify_left': [3321, 3325, 3290, 3340],
    # 'smplify_right': [5624, 5630, 5690, 5700],
}

# AddBiomechanics joint indices
ADDB_JOINT_IDX = {
    'acromial_r': 12,
    'acromial_l': 16,
    'back': 11,  # torso
}

# SKEL joint indices (original 24 joints)
SKEL_JOINT_IDX = {
    'thorax': 12,
    'scapula_r': 14,
    'humerus_r': 15,
    'scapula_l': 19,
    'humerus_l': 20,
}

# Extended SKEL joint indices (after adding acromial joints)
SKEL_EXTENDED_JOINT_IDX = {
    'acromial_r': 24,  # new measurement joint
    'acromial_l': 25,  # new measurement joint
}


# =============================================================================
# (A) Virtual Acromial Joint Functions
# =============================================================================

def compute_virtual_acromial(
    vertices: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[torch.Tensor] = None,
    weights_l: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute virtual acromial joint positions from mesh vertices.

    The virtual acromial is computed as a weighted average of vertices
    near the acromion (shoulder tip) region.

    Args:
        vertices: SKEL mesh vertices [B, V, 3] where V=6890
        v_idx_r: Right acromion vertex indices. Default: ACROMIAL_VERTEX_IDX['right']
        v_idx_l: Left acromion vertex indices. Default: ACROMIAL_VERTEX_IDX['left']
        weights_r: Weights for right vertices [len(v_idx_r)]. Default: uniform
        weights_l: Weights for left vertices [len(v_idx_l)]. Default: uniform

    Returns:
        acromial_r: Right acromial position [B, 3]
        acromial_l: Left acromial position [B, 3]
    """
    # Use default vertex indices if not provided
    if v_idx_r is None:
        v_idx_r = ACROMIAL_VERTEX_IDX['right']
    if v_idx_l is None:
        v_idx_l = ACROMIAL_VERTEX_IDX['left']

    device = vertices.device
    dtype = vertices.dtype

    # Get vertices for each side
    verts_r = vertices[:, v_idx_r, :]  # [B, N_r, 3]
    verts_l = vertices[:, v_idx_l, :]  # [B, N_l, 3]

    # Set up weights (default: uniform)
    if weights_r is None:
        weights_r = torch.ones(len(v_idx_r), device=device, dtype=dtype) / len(v_idx_r)
    else:
        weights_r = weights_r.to(device=device, dtype=dtype)
        weights_r = weights_r / weights_r.sum()  # normalize

    if weights_l is None:
        weights_l = torch.ones(len(v_idx_l), device=device, dtype=dtype) / len(v_idx_l)
    else:
        weights_l = weights_l.to(device=device, dtype=dtype)
        weights_l = weights_l / weights_l.sum()  # normalize

    # Compute weighted average
    # verts: [B, N, 3], weights: [N] -> result: [B, 3]
    acromial_r = torch.einsum('bni,n->bi', verts_r, weights_r)
    acromial_l = torch.einsum('bni,n->bi', verts_l, weights_l)

    return acromial_r, acromial_l


def loss_acromial(
    skel_vertices: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[torch.Tensor] = None,
    weights_l: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute acromial matching loss between SKEL virtual acromial and AddB acromial.

    L_acromial = MSE(virtual_acromial_skel, acromial_addb)

    Args:
        skel_vertices: SKEL mesh vertices [B, 6890, 3]
        addb_acromial_r: AddB right acromial position [B, 3]
        addb_acromial_l: AddB left acromial position [B, 3]
        v_idx_r: Right acromion vertex indices
        v_idx_l: Left acromion vertex indices
        weights_r: Weights for right vertices
        weights_l: Weights for left vertices
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar loss value (or [B] if reduction='none')
    """
    # Compute virtual acromial positions
    virtual_r, virtual_l = compute_virtual_acromial(
        skel_vertices, v_idx_r, v_idx_l, weights_r, weights_l
    )

    # Compute MSE loss for each side
    loss_r = F.mse_loss(virtual_r, addb_acromial_r, reduction='none')  # [B, 3]
    loss_l = F.mse_loss(virtual_l, addb_acromial_l, reduction='none')  # [B, 3]

    # Sum over coordinates
    loss_r = loss_r.sum(dim=-1)  # [B]
    loss_l = loss_l.sum(dim=-1)  # [B]

    # Combine losses
    loss = loss_r + loss_l  # [B]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:  # 'none'
        return loss


def loss_acromial_l1(
    skel_vertices: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[torch.Tensor] = None,
    weights_l: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute acromial matching loss using L1 distance (more robust to outliers).

    L_acromial = L1(virtual_acromial_skel, acromial_addb)
    """
    virtual_r, virtual_l = compute_virtual_acromial(
        skel_vertices, v_idx_r, v_idx_l, weights_r, weights_l
    )

    loss_r = F.l1_loss(virtual_r, addb_acromial_r, reduction='none').sum(dim=-1)
    loss_l = F.l1_loss(virtual_l, addb_acromial_l, reduction='none').sum(dim=-1)

    loss = loss_r + loss_l

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


# =============================================================================
# (C) Shoulder Width and Direction Losses
# =============================================================================

def loss_shoulder_width(
    skel_acromial_r: torch.Tensor,
    skel_acromial_l: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute shoulder width matching loss.

    L_width = (distance(skel_acr_R, skel_acr_L) - distance(addb_acr_R, addb_acr_L))^2

    This ensures the predicted SKEL shoulder width matches AddB shoulder width.

    Args:
        skel_acromial_r: SKEL right acromial position [B, 3]
        skel_acromial_l: SKEL left acromial position [B, 3]
        addb_acromial_r: AddB right acromial position [B, 3]
        addb_acromial_l: AddB left acromial position [B, 3]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar or [B] depending on reduction
    """
    # Compute shoulder widths
    skel_width = torch.norm(skel_acromial_r - skel_acromial_l, dim=-1)  # [B]
    addb_width = torch.norm(addb_acromial_r - addb_acromial_l, dim=-1)  # [B]

    # Squared difference
    loss = (skel_width - addb_width) ** 2  # [B]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def loss_shoulder_width_from_vertices(
    skel_vertices: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[torch.Tensor] = None,
    weights_l: Optional[torch.Tensor] = None,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute shoulder width loss directly from vertices (for Step 1 virtual joint approach).
    """
    skel_acromial_r, skel_acromial_l = compute_virtual_acromial(
        skel_vertices, v_idx_r, v_idx_l, weights_r, weights_l
    )

    return loss_shoulder_width(
        skel_acromial_r, skel_acromial_l,
        addb_acromial_r, addb_acromial_l,
        reduction=reduction
    )


def loss_shoulder_direction(
    skel_thorax: torch.Tensor,
    skel_acromial_r: torch.Tensor,
    skel_acromial_l: torch.Tensor,
    addb_back: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute shoulder direction matching loss.

    L_direction = (1 - cos_sim(skel_dir, addb_dir)) for both left and right

    This ensures the direction from thorax/back to acromial matches between models.

    Args:
        skel_thorax: SKEL thorax position [B, 3]
        skel_acromial_r: SKEL right acromial position [B, 3]
        skel_acromial_l: SKEL left acromial position [B, 3]
        addb_back: AddB back/torso position [B, 3]
        addb_acromial_r: AddB right acromial position [B, 3]
        addb_acromial_l: AddB left acromial position [B, 3]
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Scalar or [B] depending on reduction
    """
    # Compute direction vectors (thorax/back → acromial)
    skel_dir_r = F.normalize(skel_acromial_r - skel_thorax, dim=-1)  # [B, 3]
    skel_dir_l = F.normalize(skel_acromial_l - skel_thorax, dim=-1)  # [B, 3]
    addb_dir_r = F.normalize(addb_acromial_r - addb_back, dim=-1)    # [B, 3]
    addb_dir_l = F.normalize(addb_acromial_l - addb_back, dim=-1)    # [B, 3]

    # Compute cosine similarity
    cos_sim_r = (skel_dir_r * addb_dir_r).sum(dim=-1)  # [B]
    cos_sim_l = (skel_dir_l * addb_dir_l).sum(dim=-1)  # [B]

    # Loss: 1 - cos_sim (0 when directions match, 2 when opposite)
    loss_r = 1 - cos_sim_r
    loss_l = 1 - cos_sim_l

    loss = loss_r + loss_l  # [B]

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def loss_shoulder_direction_from_vertices(
    skel_vertices: torch.Tensor,
    skel_joints: torch.Tensor,
    addb_joints: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[torch.Tensor] = None,
    weights_l: Optional[torch.Tensor] = None,
    thorax_idx: int = 12,
    addb_back_idx: int = 11,
    addb_acromial_r_idx: int = 12,
    addb_acromial_l_idx: int = 16,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute shoulder direction loss directly from vertices (for Step 1).
    """
    skel_acromial_r, skel_acromial_l = compute_virtual_acromial(
        skel_vertices, v_idx_r, v_idx_l, weights_r, weights_l
    )

    skel_thorax = skel_joints[:, thorax_idx, :]
    addb_back = addb_joints[:, addb_back_idx, :]
    addb_acromial_r = addb_joints[:, addb_acromial_r_idx, :]
    addb_acromial_l = addb_joints[:, addb_acromial_l_idx, :]

    return loss_shoulder_direction(
        skel_thorax, skel_acromial_r, skel_acromial_l,
        addb_back, addb_acromial_r, addb_acromial_l,
        reduction=reduction
    )


# =============================================================================
# (D) Combined Shoulder Loss (Convenience Function)
# =============================================================================

def compute_shoulder_losses(
    skel_vertices: torch.Tensor,
    skel_joints: torch.Tensor,
    addb_joints: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    weights_r: Optional[torch.Tensor] = None,
    weights_l: Optional[torch.Tensor] = None,
    lambda_acromial: float = 1.0,
    lambda_width: float = 1.0,
    lambda_direction: float = 0.5,
    thorax_idx: int = 12,
    addb_back_idx: int = 11,
    addb_acromial_r_idx: int = 12,
    addb_acromial_l_idx: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Compute all shoulder correction losses at once.

    This is a convenience function for Step 1 (virtual joint approach).

    Args:
        skel_vertices: SKEL mesh vertices [B, 6890, 3]
        skel_joints: SKEL joints [B, 24, 3]
        addb_joints: AddB joints [B, 20, 3]
        v_idx_r, v_idx_l: Acromion vertex indices
        weights_r, weights_l: Vertex weights
        lambda_*: Loss weights
        *_idx: Joint indices for each model

    Returns:
        Dictionary with individual losses and combined loss:
        {
            'loss_acromial': ...,
            'loss_width': ...,
            'loss_direction': ...,
            'loss_total': ...,
            'virtual_acromial_r': ...,
            'virtual_acromial_l': ...,
        }
    """
    # Extract AddB acromial positions
    addb_acromial_r = addb_joints[:, addb_acromial_r_idx, :]
    addb_acromial_l = addb_joints[:, addb_acromial_l_idx, :]
    addb_back = addb_joints[:, addb_back_idx, :]

    # Compute virtual acromial positions
    virtual_r, virtual_l = compute_virtual_acromial(
        skel_vertices, v_idx_r, v_idx_l, weights_r, weights_l
    )

    # Get SKEL thorax
    skel_thorax = skel_joints[:, thorax_idx, :]

    # Compute individual losses
    l_acromial = loss_acromial(
        skel_vertices, addb_acromial_r, addb_acromial_l,
        v_idx_r, v_idx_l, weights_r, weights_l
    )

    l_width = loss_shoulder_width(
        virtual_r, virtual_l,
        addb_acromial_r, addb_acromial_l
    )

    l_direction = loss_shoulder_direction(
        skel_thorax, virtual_r, virtual_l,
        addb_back, addb_acromial_r, addb_acromial_l
    )

    # Combined loss
    l_total = (
        lambda_acromial * l_acromial
        + lambda_width * l_width
        + lambda_direction * l_direction
    )

    return {
        'loss_acromial': l_acromial,
        'loss_width': l_width,
        'loss_direction': l_direction,
        'loss_total': l_total,
        'virtual_acromial_r': virtual_r,
        'virtual_acromial_l': virtual_l,
    }


# =============================================================================
# Step 2: Functions for Extended Joint Regressor (26 joints)
# =============================================================================

def loss_acromial_from_joints(
    skel_joints: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    acromial_r_idx: int = 24,
    acromial_l_idx: int = 25,
    reduction: str = 'mean',
) -> torch.Tensor:
    """
    Compute acromial loss when using extended SKEL joints (26 joints).

    This is for Step 2 when the regressor has been expanded.

    Args:
        skel_joints: Extended SKEL joints [B, 26, 3]
        addb_acromial_r: AddB right acromial [B, 3]
        addb_acromial_l: AddB left acromial [B, 3]
        acromial_r_idx: Index of right acromial in extended joints (default: 24)
        acromial_l_idx: Index of left acromial in extended joints (default: 25)
        reduction: 'mean', 'sum', or 'none'

    Returns:
        loss: Acromial matching loss
    """
    skel_acromial_r = skel_joints[:, acromial_r_idx, :]
    skel_acromial_l = skel_joints[:, acromial_l_idx, :]

    loss_r = F.mse_loss(skel_acromial_r, addb_acromial_r, reduction='none').sum(dim=-1)
    loss_l = F.mse_loss(skel_acromial_l, addb_acromial_l, reduction='none').sum(dim=-1)

    loss = loss_r + loss_l

    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss


def compute_shoulder_losses_extended(
    skel_joints: torch.Tensor,
    addb_joints: torch.Tensor,
    lambda_acromial: float = 1.0,
    lambda_width: float = 1.0,
    lambda_direction: float = 0.5,
    thorax_idx: int = 12,
    acromial_r_idx: int = 24,
    acromial_l_idx: int = 25,
    addb_back_idx: int = 11,
    addb_acromial_r_idx: int = 12,
    addb_acromial_l_idx: int = 16,
) -> Dict[str, torch.Tensor]:
    """
    Compute all shoulder losses for Step 2 (extended regressor with 26 joints).

    Args:
        skel_joints: Extended SKEL joints [B, 26, 3]
        addb_joints: AddB joints [B, 20, 3]
        lambda_*: Loss weights
        *_idx: Joint indices

    Returns:
        Dictionary with individual and combined losses
    """
    # Extract joint positions
    skel_thorax = skel_joints[:, thorax_idx, :]
    skel_acromial_r = skel_joints[:, acromial_r_idx, :]
    skel_acromial_l = skel_joints[:, acromial_l_idx, :]

    addb_back = addb_joints[:, addb_back_idx, :]
    addb_acromial_r = addb_joints[:, addb_acromial_r_idx, :]
    addb_acromial_l = addb_joints[:, addb_acromial_l_idx, :]

    # Compute losses
    l_acromial = loss_acromial_from_joints(
        skel_joints, addb_acromial_r, addb_acromial_l,
        acromial_r_idx, acromial_l_idx
    )

    l_width = loss_shoulder_width(
        skel_acromial_r, skel_acromial_l,
        addb_acromial_r, addb_acromial_l
    )

    l_direction = loss_shoulder_direction(
        skel_thorax, skel_acromial_r, skel_acromial_l,
        addb_back, addb_acromial_r, addb_acromial_l
    )

    l_total = (
        lambda_acromial * l_acromial
        + lambda_width * l_width
        + lambda_direction * l_direction
    )

    return {
        'loss_acromial': l_acromial,
        'loss_width': l_width,
        'loss_direction': l_direction,
        'loss_total': l_total,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def visualize_acromial_positions(
    skel_vertices: torch.Tensor,
    addb_acromial_r: torch.Tensor,
    addb_acromial_l: torch.Tensor,
    v_idx_r: Optional[List[int]] = None,
    v_idx_l: Optional[List[int]] = None,
    frame_idx: int = 0,
) -> Dict[str, torch.Tensor]:
    """
    Extract positions for visualization/debugging.

    Returns positions of virtual acromial and AddB acromial for a single frame.
    """
    virtual_r, virtual_l = compute_virtual_acromial(skel_vertices, v_idx_r, v_idx_l)

    return {
        'skel_acromial_r': virtual_r[frame_idx].detach().cpu(),
        'skel_acromial_l': virtual_l[frame_idx].detach().cpu(),
        'addb_acromial_r': addb_acromial_r[frame_idx].detach().cpu(),
        'addb_acromial_l': addb_acromial_l[frame_idx].detach().cpu(),
        'skel_width': torch.norm(virtual_r[frame_idx] - virtual_l[frame_idx]).item(),
        'addb_width': torch.norm(addb_acromial_r[frame_idx] - addb_acromial_l[frame_idx]).item(),
    }


if __name__ == '__main__':
    # Simple test
    print("Shoulder Correction Module")
    print("=" * 50)
    print(f"Left acromion vertices: {ACROMIAL_VERTEX_IDX['left']}")
    print(f"Right acromion vertices: {ACROMIAL_VERTEX_IDX['right']}")
    print()

    # Test with dummy data
    B = 10  # batch size
    V = 6890  # vertices

    vertices = torch.randn(B, V, 3)
    addb_acromial_r = torch.randn(B, 3)
    addb_acromial_l = torch.randn(B, 3)

    # Test compute_virtual_acromial
    vr, vl = compute_virtual_acromial(vertices)
    print(f"Virtual acromial shape: R={vr.shape}, L={vl.shape}")

    # Test loss_acromial
    loss = loss_acromial(vertices, addb_acromial_r, addb_acromial_l)
    print(f"Acromial loss: {loss.item():.4f}")

    # Test loss_shoulder_width
    loss_w = loss_shoulder_width(vr, vl, addb_acromial_r, addb_acromial_l)
    print(f"Width loss: {loss_w.item():.4f}")

    print("\nAll tests passed!")
