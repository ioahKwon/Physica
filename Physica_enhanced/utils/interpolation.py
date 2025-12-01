#!/usr/bin/env python3
"""
Keyframe interpolation utilities for efficient pose estimation.

Supports different interpolation methods:
- Linear: Fast but may cause discontinuities
- SLERP: Smooth rotation interpolation (recommended)
- Cubic: Smooth with acceleration continuity
"""

import torch
import numpy as np
from typing import Tuple
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from .rotation import axis_angle_to_quat, quat_to_axis_angle, slerp


def sample_keyframes(
    num_frames: int,
    keyframe_ratio: float = 0.2,
    min_keyframes: int = 10,
    max_keyframes: int = 100,
    method: Literal["uniform", "adaptive"] = "uniform"
) -> np.ndarray:
    """
    Sample keyframe indices from a sequence.

    Args:
        num_frames: Total number of frames
        keyframe_ratio: Ratio of frames to use as keyframes (0.2 = 20%)
        min_keyframes: Minimum number of keyframes
        max_keyframes: Maximum number of keyframes
        method: Sampling method ("uniform" or "adaptive")

    Returns:
        Array of keyframe indices
    """
    num_keyframes = int(num_frames * keyframe_ratio)
    num_keyframes = np.clip(num_keyframes, min_keyframes, max_keyframes)
    num_keyframes = min(num_keyframes, num_frames)

    if method == "uniform":
        # Uniformly sample keyframes
        if num_keyframes >= num_frames:
            return np.arange(num_frames)
        else:
            indices = np.linspace(0, num_frames - 1, num_keyframes, dtype=int)
            # Ensure first and last frames are included
            indices[0] = 0
            indices[-1] = num_frames - 1
            return indices

    elif method == "adaptive":
        # Adaptive sampling based on motion (placeholder - requires motion data)
        # For now, use uniform sampling
        return sample_keyframes(num_frames, keyframe_ratio, min_keyframes, max_keyframes, "uniform")

    else:
        raise ValueError(f"Unknown sampling method: {method}")


def interpolate_keyframes_linear(
    keyframe_values: torch.Tensor,
    keyframe_indices: np.ndarray,
    num_frames: int
) -> torch.Tensor:
    """
    Linear interpolation between keyframes.

    Args:
        keyframe_values: [K, ...] Keyframe values
        keyframe_indices: [K] Keyframe indices in [0, num_frames-1]
        num_frames: Total number of frames

    Returns:
        [num_frames, ...] Interpolated values
    """
    device = keyframe_values.device
    dtype = keyframe_values.dtype
    shape = keyframe_values.shape[1:]

    # Create output tensor
    output = torch.zeros(num_frames, *shape, device=device, dtype=dtype)

    # Set keyframe values
    output[keyframe_indices] = keyframe_values

    # Interpolate between keyframes
    for i in range(len(keyframe_indices) - 1):
        idx0 = keyframe_indices[i]
        idx1 = keyframe_indices[i + 1]

        if idx1 - idx0 <= 1:
            continue  # Adjacent keyframes, no interpolation needed

        # Linear interpolation
        v0 = keyframe_values[i]
        v1 = keyframe_values[i + 1]

        for t in range(idx0 + 1, idx1):
            alpha = (t - idx0) / (idx1 - idx0)
            output[t] = (1 - alpha) * v0 + alpha * v1

    return output


def interpolate_keyframes_slerp(
    keyframe_poses: torch.Tensor,
    keyframe_indices: np.ndarray,
    num_frames: int
) -> torch.Tensor:
    """
    SLERP interpolation for axis-angle pose parameters.

    This is the recommended method for smooth rotation interpolation.

    Args:
        keyframe_poses: [K, J, 3] Keyframe poses (axis-angle)
        keyframe_indices: [K] Keyframe indices
        num_frames: Total number of frames

    Returns:
        [num_frames, J, 3] Interpolated poses
    """
    device = keyframe_poses.device
    dtype = keyframe_poses.dtype
    K, J, _ = keyframe_poses.shape

    # Create output tensor
    output = torch.zeros(num_frames, J, 3, device=device, dtype=dtype)

    # Set keyframe values
    output[keyframe_indices] = keyframe_poses

    # Convert to quaternions for smooth interpolation
    keyframe_quats = axis_angle_to_quat(keyframe_poses)  # [K, J, 4]

    # Interpolate between keyframes
    for i in range(len(keyframe_indices) - 1):
        idx0 = keyframe_indices[i]
        idx1 = keyframe_indices[i + 1]

        if idx1 - idx0 <= 1:
            continue  # Adjacent keyframes

        q0 = keyframe_quats[i]  # [J, 4]
        q1 = keyframe_quats[i + 1]  # [J, 4]

        # SLERP for each joint
        for t in range(idx0 + 1, idx1):
            alpha = (t - idx0) / (idx1 - idx0)

            # Interpolate each joint independently
            q_interp = slerp(q0, q1, alpha)  # [J, 4]

            # Convert back to axis-angle
            output[t] = quat_to_axis_angle(q_interp)  # [J, 3]

    return output


def interpolate_keyframes_cubic(
    keyframe_values: torch.Tensor,
    keyframe_indices: np.ndarray,
    num_frames: int
) -> torch.Tensor:
    """
    Cubic spline interpolation between keyframes.

    Uses Catmull-Rom splines for smooth interpolation with continuous velocity.

    Args:
        keyframe_values: [K, ...] Keyframe values
        keyframe_indices: [K] Keyframe indices
        num_frames: Total number of frames

    Returns:
        [num_frames, ...] Interpolated values
    """
    device = keyframe_values.device
    dtype = keyframe_values.dtype
    shape = keyframe_values.shape[1:]
    K = len(keyframe_indices)

    # Create output tensor
    output = torch.zeros(num_frames, *shape, device=device, dtype=dtype)

    # Set keyframe values
    output[keyframe_indices] = keyframe_values

    # Catmull-Rom spline interpolation
    for i in range(len(keyframe_indices) - 1):
        idx0 = keyframe_indices[i]
        idx1 = keyframe_indices[i + 1]

        if idx1 - idx0 <= 1:
            continue  # Adjacent keyframes

        # Get control points (with boundary handling)
        p_m1 = keyframe_values[max(0, i - 1)]
        p_0 = keyframe_values[i]
        p_1 = keyframe_values[i + 1]
        p_2 = keyframe_values[min(K - 1, i + 2)]

        # Interpolate
        for t in range(idx0 + 1, idx1):
            alpha = (t - idx0) / (idx1 - idx0)

            # Catmull-Rom basis functions
            t2 = alpha * alpha
            t3 = t2 * alpha

            # Catmull-Rom interpolation
            value = (
                0.5 * (
                    (2 * p_0) +
                    (-p_m1 + p_1) * alpha +
                    (2 * p_m1 - 5 * p_0 + 4 * p_1 - p_2) * t2 +
                    (-p_m1 + 3 * p_0 - 3 * p_1 + p_2) * t3
                )
            )

            output[t] = value

    return output


def interpolate_translations_linear(
    keyframe_trans: torch.Tensor,
    keyframe_indices: np.ndarray,
    num_frames: int
) -> torch.Tensor:
    """
    Linear interpolation for translation parameters.

    Args:
        keyframe_trans: [K, 3] Keyframe translations
        keyframe_indices: [K] Keyframe indices
        num_frames: Total number of frames

    Returns:
        [num_frames, 3] Interpolated translations
    """
    return interpolate_keyframes_linear(keyframe_trans, keyframe_indices, num_frames)


def interpolate_full_sequence(
    keyframe_poses: torch.Tensor,
    keyframe_trans: torch.Tensor,
    keyframe_indices: np.ndarray,
    num_frames: int,
    method: Literal["linear", "slerp", "cubic"] = "slerp"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Interpolate full pose sequence from keyframes.

    Args:
        keyframe_poses: [K, J, 3] Keyframe poses
        keyframe_trans: [K, 3] Keyframe translations
        keyframe_indices: [K] Keyframe indices
        num_frames: Total number of frames
        method: Interpolation method

    Returns:
        poses: [num_frames, J, 3]
        trans: [num_frames, 3]
    """
    if method == "slerp":
        poses = interpolate_keyframes_slerp(keyframe_poses, keyframe_indices, num_frames)
    elif method == "linear":
        poses = interpolate_keyframes_linear(keyframe_poses, keyframe_indices, num_frames)
    elif method == "cubic":
        poses = interpolate_keyframes_cubic(keyframe_poses, keyframe_indices, num_frames)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    # Always use linear interpolation for translations (more stable)
    trans = interpolate_translations_linear(keyframe_trans, keyframe_indices, num_frames)

    return poses, trans
