#!/usr/bin/env python3
"""
Loss functions for SMPL optimization.

All loss functions are vectorized and support both single-frame and batch inputs.
"""

import torch
import torch.nn.functional as F
from typing import Optional, List, Tuple


def position_loss(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    L2 position loss between predicted and target joints.

    Args:
        pred_joints: [..., J, 3] Predicted joint positions
        target_joints: [..., J, 3] Target joint positions
        mask: [..., J] Boolean mask for valid joints (True = valid)
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss (or per-joint if reduction="none")
    """
    # Compute squared differences
    diff = pred_joints - target_joints
    squared_diff = (diff ** 2).sum(dim=-1)  # [..., J]

    # Apply mask if provided
    if mask is not None:
        squared_diff = squared_diff * mask

    # Apply reduction
    if reduction == "mean":
        if mask is not None:
            return squared_diff.sum() / (mask.sum() + 1e-8)
        else:
            return squared_diff.mean()
    elif reduction == "sum":
        return squared_diff.sum()
    elif reduction == "none":
        return squared_diff
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def bone_length_loss(
    joints: torch.Tensor,
    bone_pairs: List[Tuple[int, int]],
    target_lengths: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Bone length consistency loss.

    If target_lengths is provided, penalizes deviation from target.
    Otherwise, penalizes variance in bone lengths across frames.

    Args:
        joints: [T, J, 3] or [J, 3] Joint positions
        bone_pairs: List of (parent_idx, child_idx) tuples
        target_lengths: [B] Target bone lengths (optional)
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss
    """
    is_sequence = joints.ndim == 3

    losses = []

    for i, (parent_idx, child_idx) in enumerate(bone_pairs):
        if is_sequence:
            # [T, 3]
            bone_vecs = joints[:, child_idx] - joints[:, parent_idx]
            lengths = torch.norm(bone_vecs, dim=-1)  # [T]

            if target_lengths is not None:
                # Penalize deviation from target
                loss = (lengths - target_lengths[i]) ** 2
            else:
                # Penalize variance across frames
                mean_length = lengths.mean()
                loss = ((lengths - mean_length) ** 2).mean()

            losses.append(loss if target_lengths is not None else loss.unsqueeze(0))
        else:
            # Single frame [3]
            bone_vec = joints[child_idx] - joints[parent_idx]
            length = torch.norm(bone_vec)

            if target_lengths is not None:
                loss = (length - target_lengths[i]) ** 2
                losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=joints.device)

    losses_tensor = torch.stack(losses) if not is_sequence or target_lengths is not None else torch.cat(losses)

    if reduction == "mean":
        return losses_tensor.mean()
    elif reduction == "sum":
        return losses_tensor.sum()
    elif reduction == "none":
        return losses_tensor
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def bone_direction_loss(
    pred_joints: torch.Tensor,
    target_joints: torch.Tensor,
    bone_pairs: List[Tuple[int, int, int, int]],
    weights: Optional[torch.Tensor] = None,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Bone direction matching loss using cosine similarity.

    Matches bone directions between pred and target, bypassing absolute position errors.

    Args:
        pred_joints: [J, 3] Predicted joint positions
        target_joints: [J, 3] Target joint positions
        bone_pairs: List of (pred_parent, pred_child, target_parent, target_child)
        weights: [B] Optional per-bone weights
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss (lower is better, 0 = perfect alignment)
    """
    losses = []

    for i, (pred_p, pred_c, tgt_p, tgt_c) in enumerate(bone_pairs):
        # Skip if indices out of bounds
        if (pred_p >= pred_joints.shape[0] or pred_c >= pred_joints.shape[0] or
            tgt_p >= target_joints.shape[0] or tgt_c >= target_joints.shape[0]):
            continue

        # Get joint positions
        pred_parent = pred_joints[pred_p]
        pred_child = pred_joints[pred_c]
        tgt_parent = target_joints[tgt_p]
        tgt_child = target_joints[tgt_c]

        # Skip if any NaN
        if (torch.isnan(pred_parent).any() or torch.isnan(pred_child).any() or
            torch.isnan(tgt_parent).any() or torch.isnan(tgt_child).any()):
            continue

        # Compute bone directions
        pred_dir = pred_child - pred_parent
        tgt_dir = tgt_child - tgt_parent

        # Normalize
        pred_norm = torch.norm(pred_dir)
        tgt_norm = torch.norm(tgt_dir)

        if pred_norm < 1e-6 or tgt_norm < 1e-6:
            continue  # Skip degenerate bones

        pred_dir = pred_dir / pred_norm
        tgt_dir = tgt_dir / tgt_norm

        # Cosine similarity: 1 - cos(theta)
        cos_sim = torch.dot(pred_dir, tgt_dir)
        loss = 1.0 - cos_sim

        # Apply weight if provided
        if weights is not None:
            loss = loss * weights[i]

        losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, device=pred_joints.device)

    losses_tensor = torch.stack(losses)

    if reduction == "mean":
        return losses_tensor.mean()
    elif reduction == "sum":
        return losses_tensor.sum()
    elif reduction == "none":
        return losses_tensor
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def smoothness_loss(
    sequence: torch.Tensor,
    order: int = 1,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Temporal smoothness loss (penalize high derivatives).

    Args:
        sequence: [T, ...] Temporal sequence
        order: Derivative order (1=velocity, 2=acceleration)
        reduction: "mean", "sum", or "none"

    Returns:
        Scalar loss
    """
    T = sequence.shape[0]

    if T < order + 1:
        return torch.tensor(0.0, device=sequence.device)

    # Compute derivatives
    diff = sequence
    for _ in range(order):
        diff = diff[1:] - diff[:-1]

    # L2 norm
    loss = (diff ** 2).sum(dim=tuple(range(1, diff.ndim)))  # [T-order]

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")


def get_standard_bone_pairs(smpl_indices: List[int]) -> List[Tuple[int, int]]:
    """
    Get standard bone pairs for bone length loss.

    Args:
        smpl_indices: List of available SMPL joint indices

    Returns:
        List of (parent_idx, child_idx) pairs
    """
    # Standard bone pairs (parent, child)
    all_pairs = [
        (0, 1),   # pelvis → left_hip
        (0, 2),   # pelvis → right_hip
        (1, 4),   # left_hip → left_knee
        (2, 5),   # right_hip → right_knee
        (4, 7),   # left_knee → left_ankle
        (5, 8),   # right_knee → right_ankle
        (7, 10),  # left_ankle → left_foot
        (8, 11),  # right_ankle → right_foot
        (0, 3),   # pelvis → spine1
        (3, 6),   # spine1 → spine2
        (6, 9),   # spine2 → spine3
        (9, 12),  # spine3 → neck
        (12, 15), # neck → head
        (9, 13),  # spine3 → left_collar
        (9, 14),  # spine3 → right_collar
        (13, 16), # left_collar → left_shoulder
        (14, 17), # right_collar → right_shoulder
        (16, 18), # left_shoulder → left_elbow
        (17, 19), # right_shoulder → right_elbow
        (18, 20), # left_elbow → left_wrist
        (19, 21), # right_elbow → right_wrist
        (20, 22), # left_wrist → left_hand
        (21, 23), # right_wrist → right_hand
    ]

    # Filter to only available joints
    smpl_set = set(smpl_indices)
    pairs = [(p, c) for p, c in all_pairs if p in smpl_set and c in smpl_set]

    return pairs
