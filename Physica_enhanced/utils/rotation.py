#!/usr/bin/env python3
"""
Vectorized rotation utilities for efficient batch processing.

All functions support both single and batch inputs:
- Single: [3] or [4] (quaternion)
- Batch: [B, 3] or [B, 4]
- Sequence: [T, J, 3] for T frames and J joints

Coordinate convention:
- Euler angles: (roll, pitch, yaw) in radians, XYZ order
- Quaternions: (w, x, y, z) - scalar-first convention
- Axis-angle: (rx, ry, rz) where magnitude is rotation angle
"""

import torch
import torch.nn.functional as F
from typing import Optional


def normalize_vector(v: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize vector(s) along specified dimension.

    Args:
        v: Input tensor [..., D]
        dim: Dimension to normalize
        eps: Small epsilon for numerical stability

    Returns:
        Normalized tensor [..., D]
    """
    return v / (torch.norm(v, dim=dim, keepdim=True) + eps)


def euler_to_rotation_matrix(euler: torch.Tensor, order: str = "XYZ") -> torch.Tensor:
    """
    Convert Euler angles to rotation matrices (vectorized).

    Args:
        euler: [..., 3] Euler angles in radians
        order: Rotation order (default: XYZ)

    Returns:
        [..., 3, 3] Rotation matrices
    """
    shape = euler.shape[:-1]
    euler = euler.reshape(-1, 3)

    roll, pitch, yaw = euler[:, 0], euler[:, 1], euler[:, 2]

    # Compute sin/cos
    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    if order == "XYZ":
        # R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
        R = torch.stack([
            torch.stack([cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr], dim=-1),
            torch.stack([sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr], dim=-1),
            torch.stack([-sp, cp*sr, cp*cr], dim=-1),
        ], dim=-2)
    elif order == "ZYX":
        # R = Rx(roll) @ Ry(pitch) @ Rz(yaw)
        R = torch.stack([
            torch.stack([cy*cp, -sy*cr + cy*sp*sr, sy*sr + cy*sp*cr], dim=-1),
            torch.stack([sy*cp, cy*cr + sy*sp*sr, -cy*sr + sy*sp*cr], dim=-1),
            torch.stack([-sp, cp*sr, cp*cr], dim=-1),
        ], dim=-2)
    else:
        raise ValueError(f"Unsupported rotation order: {order}")

    return R.reshape(*shape, 3, 3)


def rotation_matrix_to_axis_angle(R: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Convert rotation matrix to axis-angle representation (vectorized).

    Uses the formula:
        angle = arccos((trace(R) - 1) / 2)
        axis = [R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]] / (2 * sin(angle))

    Args:
        R: [..., 3, 3] Rotation matrices
        eps: Small epsilon for numerical stability

    Returns:
        [..., 3] Axis-angle vectors
    """
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    batch_size = R.shape[0]

    # Compute angle from trace
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1 + eps, 1 - eps))

    # Compute axis
    axis = torch.stack([
        R[:, 2, 1] - R[:, 1, 2],
        R[:, 0, 2] - R[:, 2, 0],
        R[:, 1, 0] - R[:, 0, 1],
    ], dim=-1)

    # Normalize axis
    sin_angle = torch.sin(angle).unsqueeze(-1)
    axis = axis / (2 * sin_angle + eps)

    # Handle small angles (axis is undefined when angle ≈ 0)
    small_angle_mask = angle.abs() < eps
    axis[small_angle_mask] = 0.0

    # Axis-angle = axis * angle
    axis_angle = axis * angle.unsqueeze(-1)

    return axis_angle.reshape(*shape, 3)


def euler_to_axis_angle(euler: torch.Tensor, order: str = "XYZ") -> torch.Tensor:
    """
    Convert Euler angles to axis-angle representation (vectorized).

    Args:
        euler: [..., 3] Euler angles in radians
        order: Rotation order (default: XYZ)

    Returns:
        [..., 3] Axis-angle vectors
    """
    R = euler_to_rotation_matrix(euler, order=order)
    return rotation_matrix_to_axis_angle(R)


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to rotation matrix (vectorized).

    Args:
        quat: [..., 4] Quaternions (w, x, y, z)

    Returns:
        [..., 3, 3] Rotation matrices
    """
    shape = quat.shape[:-1]
    quat = quat.reshape(-1, 4)

    # Normalize quaternion
    quat = normalize_vector(quat, dim=-1)

    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    # Compute rotation matrix elements
    R = torch.stack([
        torch.stack([1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)], dim=-1),
        torch.stack([2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)], dim=-1),
        torch.stack([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)], dim=-1),
    ], dim=-2)

    return R.reshape(*shape, 3, 3)


def quat_to_axis_angle(quat: torch.Tensor) -> torch.Tensor:
    """
    Convert quaternion to axis-angle representation (vectorized).

    Args:
        quat: [..., 4] Quaternions (w, x, y, z)

    Returns:
        [..., 3] Axis-angle vectors
    """
    R = quat_to_rotation_matrix(quat)
    return rotation_matrix_to_axis_angle(R)


def axis_angle_to_rotation_matrix(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert axis-angle to rotation matrix using Rodrigues formula (vectorized).

    Args:
        axis_angle: [..., 3] Axis-angle vectors
        eps: Small epsilon for numerical stability

    Returns:
        [..., 3, 3] Rotation matrices
    """
    shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.reshape(-1, 3)
    batch_size = axis_angle.shape[0]

    # Compute angle and axis
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + eps)

    # Handle zero rotation
    angle = angle.squeeze(-1)
    zero_mask = angle < eps

    # Rodrigues formula: R = I + sin(θ)K + (1-cos(θ))K²
    # where K is the skew-symmetric matrix of the axis
    cos_angle = torch.cos(angle)
    sin_angle = torch.sin(angle)

    # Skew-symmetric matrix K
    K = torch.zeros(batch_size, 3, 3, device=axis_angle.device, dtype=axis_angle.dtype)
    K[:, 0, 1] = -axis[:, 2]
    K[:, 0, 2] = axis[:, 1]
    K[:, 1, 0] = axis[:, 2]
    K[:, 1, 2] = -axis[:, 0]
    K[:, 2, 0] = -axis[:, 1]
    K[:, 2, 1] = axis[:, 0]

    # K² = K @ K
    K2 = torch.bmm(K, K)

    # R = I + sin(θ)K + (1-cos(θ))K²
    I = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype).unsqueeze(0).expand(batch_size, -1, -1)
    R = I + sin_angle.unsqueeze(-1).unsqueeze(-1) * K + (1 - cos_angle).unsqueeze(-1).unsqueeze(-1) * K2

    # Set identity for zero rotations
    R[zero_mask] = I[0]

    return R.reshape(*shape, 3, 3)


def axis_angle_to_quat(axis_angle: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Convert axis-angle to quaternion (vectorized).

    Args:
        axis_angle: [..., 3] Axis-angle vectors
        eps: Small epsilon for numerical stability

    Returns:
        [..., 4] Quaternions (w, x, y, z)
    """
    shape = axis_angle.shape[:-1]
    axis_angle = axis_angle.reshape(-1, 3)

    # Compute angle and axis
    angle = torch.norm(axis_angle, dim=-1, keepdim=True)
    axis = axis_angle / (angle + eps)

    # Quaternion: q = [cos(θ/2), sin(θ/2) * axis]
    half_angle = angle / 2
    w = torch.cos(half_angle)
    xyz = torch.sin(half_angle) * axis

    quat = torch.cat([w, xyz], dim=-1)

    return quat.reshape(*shape, 4)


def slerp(q0: torch.Tensor, q1: torch.Tensor, t: float, eps: float = 1e-8) -> torch.Tensor:
    """
    Spherical linear interpolation between two quaternions.

    Args:
        q0: [..., 4] Start quaternion
        q1: [..., 4] End quaternion
        t: Interpolation parameter in [0, 1]
        eps: Small epsilon for numerical stability

    Returns:
        [..., 4] Interpolated quaternion
    """
    # Normalize quaternions
    q0 = normalize_vector(q0, dim=-1)
    q1 = normalize_vector(q1, dim=-1)

    # Compute dot product
    dot = (q0 * q1).sum(dim=-1, keepdim=True)

    # If dot < 0, negate q1 to take shorter path
    q1 = torch.where(dot < 0, -q1, q1)
    dot = torch.abs(dot)

    # If quaternions are very close, use linear interpolation
    DOT_THRESHOLD = 0.9995
    close_mask = dot > DOT_THRESHOLD

    # SLERP formula
    theta = torch.acos(torch.clamp(dot, -1 + eps, 1 - eps))
    sin_theta = torch.sin(theta)

    w0 = torch.sin((1 - t) * theta) / (sin_theta + eps)
    w1 = torch.sin(t * theta) / (sin_theta + eps)

    result = w0 * q0 + w1 * q1

    # Use linear interpolation for close quaternions
    linear_result = (1 - t) * q0 + t * q1
    result = torch.where(close_mask, linear_result, result)

    return normalize_vector(result, dim=-1)


def batch_slerp(q: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Batch spherical linear interpolation for sequences.

    Interpolates between consecutive quaternions in a sequence.

    Args:
        q: [T, ..., 4] Sequence of quaternions
        t: [T_out] Interpolation time points (must be sorted)

    Returns:
        [T_out, ..., 4] Interpolated quaternions
    """
    T = q.shape[0]
    T_out = t.shape[0]

    # Map t to segment indices
    # t should be in range [0, T-1]
    t_clamped = torch.clamp(t, 0, T - 1 - 1e-6)
    idx = t_clamped.long()
    alpha = t_clamped - idx.float()

    # Get start and end quaternions
    q0 = q[idx]  # [T_out, ..., 4]
    q1 = q[idx + 1]  # [T_out, ..., 4]

    # Apply slerp for each pair
    result = torch.stack([
        slerp(q0[i], q1[i], alpha[i].item())
        for i in range(T_out)
    ], dim=0)

    return result


def axis_angle_slerp(aa0: torch.Tensor, aa1: torch.Tensor, t: float) -> torch.Tensor:
    """
    SLERP for axis-angle representations.

    Converts to quaternion, performs SLERP, converts back.

    Args:
        aa0: [..., 3] Start axis-angle
        aa1: [..., 3] End axis-angle
        t: Interpolation parameter in [0, 1]

    Returns:
        [..., 3] Interpolated axis-angle
    """
    q0 = axis_angle_to_quat(aa0)
    q1 = axis_angle_to_quat(aa1)
    q_interp = slerp(q0, q1, t)
    return quat_to_axis_angle(q_interp)
