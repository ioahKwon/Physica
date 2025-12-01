#!/usr/bin/env python3
"""
Coordinate system conversion utilities.

AddBiomechanics uses Z-up coordinate system (OpenSim convention):
- X: right
- Y: forward
- Z: up

SMPL uses Y-up coordinate system (graphics convention):
- X: right
- Y: up
- Z: forward
"""

import torch
import numpy as np
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


class CoordinateConverter:
    """Handle coordinate system conversions between Z-up and Y-up."""

    @staticmethod
    def z_up_to_y_up(
        positions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert from Z-up (OpenSim) to Y-up (SMPL).

        Transformation:
        - X_new = X_old (right stays right)
        - Y_new = Z_old (up in Z becomes up in Y)
        - Z_new = -Y_old (forward in Y becomes forward in Z, negated)

        Args:
            positions: [..., 3] positions in Z-up coordinates

        Returns:
            [..., 3] positions in Y-up coordinates
        """
        is_torch = isinstance(positions, torch.Tensor)

        if is_torch:
            x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
            return torch.stack([x, z, -y], dim=-1)
        else:
            x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
            return np.stack([x, z, -y], axis=-1)

    @staticmethod
    def y_up_to_z_up(
        positions: Union[np.ndarray, torch.Tensor]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Convert from Y-up (SMPL) to Z-up (OpenSim).

        Inverse transformation of z_up_to_y_up.

        Args:
            positions: [..., 3] positions in Y-up coordinates

        Returns:
            [..., 3] positions in Z-up coordinates
        """
        is_torch = isinstance(positions, torch.Tensor)

        if is_torch:
            x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
            return torch.stack([x, -z, y], dim=-1)
        else:
            x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
            return np.stack([x, -z, y], axis=-1)

    @staticmethod
    def apply_conversion(
        positions: Union[np.ndarray, torch.Tensor],
        source: Literal["z_up", "y_up"],
        target: Literal["z_up", "y_up"]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Apply coordinate conversion from source to target system.

        Args:
            positions: [..., 3] positions
            source: Source coordinate system
            target: Target coordinate system

        Returns:
            [..., 3] converted positions
        """
        if source == target:
            return positions

        if source == "z_up" and target == "y_up":
            return CoordinateConverter.z_up_to_y_up(positions)
        elif source == "y_up" and target == "z_up":
            return CoordinateConverter.y_up_to_z_up(positions)
        else:
            raise ValueError(f"Invalid conversion: {source} -> {target}")

    @staticmethod
    def get_rotation_matrix_z_to_y() -> np.ndarray:
        """
        Get rotation matrix for Z-up to Y-up conversion.

        Returns:
            [3, 3] rotation matrix
        """
        # R = [[1, 0, 0],
        #      [0, 0, 1],
        #      [0, -1, 0]]
        return np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ], dtype=np.float32)

    @staticmethod
    def get_rotation_matrix_y_to_z() -> np.ndarray:
        """
        Get rotation matrix for Y-up to Z-up conversion.

        Returns:
            [3, 3] rotation matrix
        """
        # R = [[1, 0, 0],
        #      [0, 0, -1],
        #      [0, 1, 0]]
        return np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ], dtype=np.float32)
