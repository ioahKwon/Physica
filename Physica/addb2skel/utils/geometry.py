"""
Geometry utilities for bone length, direction, and rotation computations.
"""

from typing import List, Tuple, Union
import numpy as np
import torch


def compute_bone_lengths(
    joints: Union[np.ndarray, torch.Tensor],
    bone_pairs: List[Tuple[int, int]],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute bone lengths from joint positions.

    Args:
        joints: Joint positions [B, J, 3] or [J, 3].
        bone_pairs: List of (parent_idx, child_idx) tuples.

    Returns:
        Bone lengths [B, num_bones] or [num_bones].
    """
    is_numpy = isinstance(joints, np.ndarray)
    is_batched = joints.ndim == 3

    if not is_batched:
        joints = joints[np.newaxis] if is_numpy else joints.unsqueeze(0)

    num_bones = len(bone_pairs)
    if is_numpy:
        lengths = np.zeros((joints.shape[0], num_bones))
    else:
        lengths = torch.zeros(joints.shape[0], num_bones, device=joints.device)

    for i, (p, c) in enumerate(bone_pairs):
        parent = joints[:, p, :]
        child = joints[:, c, :]
        if is_numpy:
            lengths[:, i] = np.linalg.norm(child - parent, axis=-1)
        else:
            lengths[:, i] = torch.norm(child - parent, dim=-1)

    if not is_batched:
        lengths = lengths[0] if is_numpy else lengths.squeeze(0)

    return lengths


def compute_bone_directions(
    joints: Union[np.ndarray, torch.Tensor],
    bone_pairs: List[Tuple[int, int]],
    normalize: bool = True,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Compute bone direction vectors from joint positions.

    Args:
        joints: Joint positions [B, J, 3] or [J, 3].
        bone_pairs: List of (parent_idx, child_idx) tuples.
        normalize: Whether to normalize to unit vectors.

    Returns:
        Bone directions [B, num_bones, 3] or [num_bones, 3].
    """
    is_numpy = isinstance(joints, np.ndarray)
    is_batched = joints.ndim == 3

    if not is_batched:
        joints = joints[np.newaxis] if is_numpy else joints.unsqueeze(0)

    num_bones = len(bone_pairs)
    if is_numpy:
        directions = np.zeros((joints.shape[0], num_bones, 3))
    else:
        directions = torch.zeros(joints.shape[0], num_bones, 3, device=joints.device)

    for i, (p, c) in enumerate(bone_pairs):
        parent = joints[:, p, :]
        child = joints[:, c, :]
        diff = child - parent

        if normalize:
            if is_numpy:
                norm = np.linalg.norm(diff, axis=-1, keepdims=True)
                norm = np.maximum(norm, 1e-8)
                diff = diff / norm
            else:
                norm = torch.norm(diff, dim=-1, keepdim=True)
                norm = torch.clamp(norm, min=1e-8)
                diff = diff / norm

        directions[:, i, :] = diff

    if not is_batched:
        directions = directions[0] if is_numpy else directions.squeeze(0)

    return directions


def cosine_similarity_loss(
    dir1: torch.Tensor,
    dir2: torch.Tensor,
) -> torch.Tensor:
    """
    Compute cosine similarity loss (1 - cos_sim) between direction vectors.

    Args:
        dir1: Direction vectors [B, N, 3] or [N, 3].
        dir2: Direction vectors [B, N, 3] or [N, 3].

    Returns:
        Loss value (scalar or per-sample).
    """
    # Normalize
    dir1_norm = dir1 / (torch.norm(dir1, dim=-1, keepdim=True) + 1e-8)
    dir2_norm = dir2 / (torch.norm(dir2, dim=-1, keepdim=True) + 1e-8)

    # Cosine similarity
    cos_sim = (dir1_norm * dir2_norm).sum(dim=-1)

    # Loss = 1 - cos_sim (so parallel = 0, perpendicular = 1, opposite = 2)
    return (1 - cos_sim).mean()


def rotation_matrix_to_euler(
    R: Union[np.ndarray, torch.Tensor],
    order: str = 'XYZ',
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert rotation matrix to Euler angles.

    Args:
        R: Rotation matrix [3, 3] or [B, 3, 3].
        order: Euler angle order (e.g., 'XYZ', 'ZYX').

    Returns:
        Euler angles [3] or [B, 3] in radians.
    """
    is_numpy = isinstance(R, np.ndarray)
    is_batched = R.ndim == 3

    if not is_batched:
        R = R[np.newaxis] if is_numpy else R.unsqueeze(0)

    if is_numpy:
        euler = np.zeros((R.shape[0], 3))
    else:
        euler = torch.zeros(R.shape[0], 3, device=R.device)

    # XYZ order (Tait-Bryan angles)
    if order == 'XYZ':
        if is_numpy:
            euler[:, 1] = np.arcsin(np.clip(R[:, 0, 2], -1, 1))
            euler[:, 0] = np.arctan2(-R[:, 1, 2], R[:, 2, 2])
            euler[:, 2] = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
        else:
            euler[:, 1] = torch.asin(torch.clamp(R[:, 0, 2], -1, 1))
            euler[:, 0] = torch.atan2(-R[:, 1, 2], R[:, 2, 2])
            euler[:, 2] = torch.atan2(-R[:, 0, 1], R[:, 0, 0])
    else:
        raise NotImplementedError(f"Order {order} not implemented")

    if not is_batched:
        euler = euler[0] if is_numpy else euler.squeeze(0)

    return euler


def euler_to_rotation_matrix(
    euler: Union[np.ndarray, torch.Tensor],
    order: str = 'XYZ',
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert Euler angles to rotation matrix.

    Args:
        euler: Euler angles [3] or [B, 3] in radians.
        order: Euler angle order.

    Returns:
        Rotation matrix [3, 3] or [B, 3, 3].
    """
    is_numpy = isinstance(euler, np.ndarray)
    is_batched = euler.ndim == 2

    if not is_batched:
        euler = euler[np.newaxis] if is_numpy else euler.unsqueeze(0)

    B = euler.shape[0]

    if is_numpy:
        cos = np.cos
        sin = np.sin
        zeros = np.zeros
        ones = np.ones
        stack = np.stack
    else:
        cos = torch.cos
        sin = torch.sin
        zeros = lambda x: torch.zeros(x, device=euler.device)
        ones = lambda x: torch.ones(x, device=euler.device)
        stack = torch.stack

    if order == 'XYZ':
        x, y, z = euler[:, 0], euler[:, 1], euler[:, 2]

        Rx = stack([
            stack([ones(B), zeros(B), zeros(B)], dim=-1),
            stack([zeros(B), cos(x), -sin(x)], dim=-1),
            stack([zeros(B), sin(x), cos(x)], dim=-1),
        ], dim=-2)

        Ry = stack([
            stack([cos(y), zeros(B), sin(y)], dim=-1),
            stack([zeros(B), ones(B), zeros(B)], dim=-1),
            stack([-sin(y), zeros(B), cos(y)], dim=-1),
        ], dim=-2)

        Rz = stack([
            stack([cos(z), -sin(z), zeros(B)], dim=-1),
            stack([sin(z), cos(z), zeros(B)], dim=-1),
            stack([zeros(B), zeros(B), ones(B)], dim=-1),
        ], dim=-2)

        if is_numpy:
            R = Rz @ Ry @ Rx
        else:
            R = torch.bmm(torch.bmm(Rz, Ry), Rx)
    else:
        raise NotImplementedError(f"Order {order} not implemented")

    if not is_batched:
        R = R[0] if is_numpy else R.squeeze(0)

    return R


def convert_addb_to_skel_coords(
    addb_joints: np.ndarray,
) -> np.ndarray:
    """
    Convert AddB joint coordinates to SKEL coordinate system.

    AddB uses a different coordinate convention than SKEL.
    This function handles the transformation.

    Args:
        addb_joints: AddB joint positions [T, J, 3] or [J, 3].

    Returns:
        Converted joint positions in SKEL coordinate system.
    """
    # AddB: X=right, Y=up, Z=forward
    # SKEL: X=right, Y=up, Z=backward (facing -Z)
    # So we need to flip Z
    converted = addb_joints.copy()
    converted[..., 2] = -converted[..., 2]
    return converted


def project_point_onto_line(
    point: Union[np.ndarray, torch.Tensor],
    line_start: Union[np.ndarray, torch.Tensor],
    line_end: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Project a point onto the line defined by line_start and line_end.

    Args:
        point: Point to project [3] or [B, 3].
        line_start: Start of the line [3] or [B, 3].
        line_end: End of the line [3] or [B, 3].

    Returns:
        Projected point on the line [3] or [B, 3].
    """
    is_numpy = isinstance(point, np.ndarray)

    # Line direction
    line_vec = line_end - line_start
    if is_numpy:
        line_len = np.linalg.norm(line_vec, axis=-1, keepdims=True) + 1e-8
    else:
        line_len = torch.norm(line_vec, dim=-1, keepdim=True) + 1e-8
    line_dir = line_vec / line_len

    # Vector from line start to point
    to_point = point - line_start

    # Project onto line direction
    if is_numpy:
        t = np.sum(to_point * line_dir, axis=-1, keepdims=True)
    else:
        t = (to_point * line_dir).sum(dim=-1, keepdim=True)

    # Projected point
    projected = line_start + t * line_dir

    return projected


def postprocess_humerus_to_arm_line(
    skel_joints: Union[np.ndarray, torch.Tensor],
    addb_joints: Union[np.ndarray, torch.Tensor],
    blend_factor: float = 1.0,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Post-process SKEL joints to move humerus onto AddB arm line.

    This is a post-processing step that projects SKEL humerus joints
    onto the AddB acromial→elbow line for better visual alignment,
    without affecting the optimization loss.

    Args:
        skel_joints: SKEL joint positions [T, 24, 3].
        addb_joints: AddB joint positions [T, 20, 3].
        blend_factor: How much to move humerus toward the line (0=no change, 1=on line).

    Returns:
        Modified SKEL joints with humerus projected onto arm line.
    """
    is_numpy = isinstance(skel_joints, np.ndarray)

    # Clone/copy to avoid modifying original
    if is_numpy:
        result = skel_joints.copy()
    else:
        result = skel_joints.clone()

    # AddB joint indices
    ADDB_ACROMIAL_R = 12
    ADDB_ACROMIAL_L = 16
    ADDB_ELBOW_R = 13
    ADDB_ELBOW_L = 17

    # SKEL joint indices
    SKEL_HUMERUS_R = 15
    SKEL_HUMERUS_L = 20

    # Right arm: project humerus onto acromial→elbow line
    addb_acr_r = addb_joints[:, ADDB_ACROMIAL_R, :]
    addb_elbow_r = addb_joints[:, ADDB_ELBOW_R, :]
    skel_hum_r = skel_joints[:, SKEL_HUMERUS_R, :]

    projected_r = project_point_onto_line(skel_hum_r, addb_acr_r, addb_elbow_r)
    result[:, SKEL_HUMERUS_R, :] = (1 - blend_factor) * skel_hum_r + blend_factor * projected_r

    # Left arm: project humerus onto acromial→elbow line
    addb_acr_l = addb_joints[:, ADDB_ACROMIAL_L, :]
    addb_elbow_l = addb_joints[:, ADDB_ELBOW_L, :]
    skel_hum_l = skel_joints[:, SKEL_HUMERUS_L, :]

    projected_l = project_point_onto_line(skel_hum_l, addb_acr_l, addb_elbow_l)
    result[:, SKEL_HUMERUS_L, :] = (1 - blend_factor) * skel_hum_l + blend_factor * projected_l

    return result
