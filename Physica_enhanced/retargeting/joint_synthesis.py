#!/usr/bin/env python3
"""
Joint synthesis for missing SMPL joints.

When certain SMPL joints are not present in the source data (e.g., spine2, spine3, neck, collar, toe),
we synthesize them using kinematic rules and proportional relationships.

Synthesis rules:
- spine2: Midpoint between spine1 and estimated spine3
- spine3: Proportional extension from spine1 towards head/neck
- neck: Midpoint between head and spine3
- left_collar/right_collar: Lateral offset from spine3 towards shoulder
- left_foot/right_foot toe extension: Forward offset from ankle
"""

import torch
import numpy as np
from typing import Union, Dict, List, Optional
from core.smpl_model import SMPLModelWrapper


class JointSynthesizer:
    """
    Synthesizes missing SMPL joints based on available joints.

    Uses kinematic rules and proportional relationships.
    """

    # SMPL parent indices (from SMPLModelWrapper)
    PARENT_IDX = SMPLModelWrapper.PARENT_IDX

    def __init__(
        self,
        spine_ratio: float = 0.5,
        clavicle_offset: float = 0.1,
        toe_offset: float = 0.1,
        neck_ratio: float = 0.5
    ):
        """
        Initialize joint synthesizer.

        Args:
            spine_ratio: Ratio for spine2/spine3 synthesis (0.5 = midpoint)
            clavicle_offset: Lateral offset for clavicle synthesis (in meters)
            toe_offset: Forward offset for toe synthesis (in meters)
            neck_ratio: Ratio for neck synthesis between head and spine3
        """
        self.spine_ratio = spine_ratio
        self.clavicle_offset = clavicle_offset
        self.toe_offset = toe_offset
        self.neck_ratio = neck_ratio

    def synthesize_missing_joints(
        self,
        joints: Union[np.ndarray, torch.Tensor],
        available_joints: List[int]
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Synthesize missing SMPL joints.

        Args:
            joints: [..., 24, 3] SMPL joint positions (may have NaN for missing joints)
            available_joints: List of available joint indices

        Returns:
            [..., 24, 3] Joint positions with synthesized joints filled in
        """
        is_torch = isinstance(joints, torch.Tensor)

        # Create mask of available joints
        available_mask = np.zeros(24, dtype=bool)
        available_mask[available_joints] = True

        # Synthesize each missing joint
        result = joints.copy() if not is_torch else joints.clone()

        # Synthesize spine joints (3, 6, 9 = spine1, spine2, spine3)
        result = self._synthesize_spine(result, available_mask)

        # Synthesize neck (12)
        result = self._synthesize_neck(result, available_mask)

        # Synthesize clavicles (13, 14 = left_collar, right_collar)
        result = self._synthesize_clavicles(result, available_mask)

        # Synthesize foot toe extensions if needed (10, 11 = left_foot, right_foot)
        result = self._refine_feet(result, available_mask)

        return result

    def _synthesize_spine(
        self,
        joints: Union[np.ndarray, torch.Tensor],
        available_mask: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Synthesize spine2 (6) and spine3 (9) if missing.

        Strategy:
        - If spine1 (3) and head (15) are available:
          - spine3 = spine1 + 0.6 * (head - spine1)
          - spine2 = spine1 + 0.3 * (head - spine1)
        - If spine1 and neck (12) are available:
          - spine3 = spine1 + 0.7 * (neck - spine1)
          - spine2 = spine1 + 0.35 * (neck - spine1)
        """
        is_torch = isinstance(joints, torch.Tensor)

        spine1_idx, spine2_idx, spine3_idx = 3, 6, 9
        head_idx, neck_idx = 15, 12

        # Check if spine1 is available
        if not available_mask[spine1_idx]:
            return joints  # Cannot synthesize without spine1

        spine1 = joints[..., spine1_idx, :]

        # Synthesize spine3 first
        if not available_mask[spine3_idx]:
            if available_mask[head_idx]:
                head = joints[..., head_idx, :]
                spine3 = spine1 + 0.6 * (head - spine1)
                joints[..., spine3_idx, :] = spine3
                available_mask[spine3_idx] = True
            elif available_mask[neck_idx]:
                neck = joints[..., neck_idx, :]
                spine3 = spine1 + 0.7 * (neck - spine1)
                joints[..., spine3_idx, :] = spine3
                available_mask[spine3_idx] = True

        # Synthesize spine2
        if not available_mask[spine2_idx] and available_mask[spine3_idx]:
            spine3 = joints[..., spine3_idx, :]
            spine2 = spine1 + self.spine_ratio * (spine3 - spine1)
            joints[..., spine2_idx, :] = spine2
            available_mask[spine2_idx] = True

        return joints

    def _synthesize_neck(
        self,
        joints: Union[np.ndarray, torch.Tensor],
        available_mask: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Synthesize neck (12) if missing.

        Strategy:
        - If head (15) and spine3 (9) are available:
          - neck = spine3 + neck_ratio * (head - spine3)
        """
        neck_idx, head_idx, spine3_idx = 12, 15, 9

        if not available_mask[neck_idx]:
            if available_mask[head_idx] and available_mask[spine3_idx]:
                head = joints[..., head_idx, :]
                spine3 = joints[..., spine3_idx, :]
                neck = spine3 + self.neck_ratio * (head - spine3)
                joints[..., neck_idx, :] = neck
                available_mask[neck_idx] = True

        return joints

    def _synthesize_clavicles(
        self,
        joints: Union[np.ndarray, torch.Tensor],
        available_mask: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Synthesize left_collar (13) and right_collar (14) if missing.

        Strategy:
        - If spine3 (9) and shoulder (16/17) are available:
          - collar = spine3 + 0.3 * (shoulder - spine3)
        """
        is_torch = isinstance(joints, torch.Tensor)

        spine3_idx = 9
        left_collar_idx, right_collar_idx = 13, 14
        left_shoulder_idx, right_shoulder_idx = 16, 17

        # Left collar
        if not available_mask[left_collar_idx]:
            if available_mask[spine3_idx] and available_mask[left_shoulder_idx]:
                spine3 = joints[..., spine3_idx, :]
                left_shoulder = joints[..., left_shoulder_idx, :]
                left_collar = spine3 + 0.3 * (left_shoulder - spine3)
                joints[..., left_collar_idx, :] = left_collar
                available_mask[left_collar_idx] = True

        # Right collar
        if not available_mask[right_collar_idx]:
            if available_mask[spine3_idx] and available_mask[right_shoulder_idx]:
                spine3 = joints[..., spine3_idx, :]
                right_shoulder = joints[..., right_shoulder_idx, :]
                right_collar = spine3 + 0.3 * (right_shoulder - spine3)
                joints[..., right_collar_idx, :] = right_collar
                available_mask[right_collar_idx] = True

        return joints

    def _refine_feet(
        self,
        joints: Union[np.ndarray, torch.Tensor],
        available_mask: np.ndarray
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Refine foot toe positions if needed.

        If foot joints are missing but ankles are available, synthesize toe position
        as forward extension from ankle.

        Strategy:
        - foot = ankle + [toe_offset, 0, 0] (forward in x-direction)
        """
        is_torch = isinstance(joints, torch.Tensor)

        left_ankle_idx, right_ankle_idx = 7, 8
        left_foot_idx, right_foot_idx = 10, 11

        # Left foot
        if not available_mask[left_foot_idx] and available_mask[left_ankle_idx]:
            left_ankle = joints[..., left_ankle_idx, :]
            if is_torch:
                offset = torch.tensor([self.toe_offset, -0.02, 0], device=joints.device, dtype=joints.dtype)
            else:
                offset = np.array([self.toe_offset, -0.02, 0], dtype=joints.dtype)
            left_foot = left_ankle + offset
            joints[..., left_foot_idx, :] = left_foot
            available_mask[left_foot_idx] = True

        # Right foot
        if not available_mask[right_foot_idx] and available_mask[right_ankle_idx]:
            right_ankle = joints[..., right_ankle_idx, :]
            if is_torch:
                offset = torch.tensor([self.toe_offset, -0.02, 0], device=joints.device, dtype=joints.dtype)
            else:
                offset = np.array([self.toe_offset, -0.02, 0], dtype=joints.dtype)
            right_foot = right_ankle + offset
            joints[..., right_foot_idx, :] = right_foot
            available_mask[right_foot_idx] = True

        return joints

    def get_synthesis_info(self, available_joints: List[int]) -> Dict[str, List[str]]:
        """
        Get information about which joints can be synthesized.

        Args:
            available_joints: List of available joint indices

        Returns:
            Dictionary with 'synthesizable' and 'missing' joint names
        """
        available_mask = np.zeros(24, dtype=bool)
        available_mask[available_joints] = True

        synthesizable = []
        missing = []

        # Check spine2 (6)
        if not available_mask[6]:
            if available_mask[3]:  # spine1 available
                synthesizable.append('spine2')
            else:
                missing.append('spine2')

        # Check spine3 (9)
        if not available_mask[9]:
            if available_mask[3] and (available_mask[15] or available_mask[12]):
                synthesizable.append('spine3')
            else:
                missing.append('spine3')

        # Check neck (12)
        if not available_mask[12]:
            if available_mask[15] and available_mask[9]:
                synthesizable.append('neck')
            else:
                missing.append('neck')

        # Check collars (13, 14)
        if not available_mask[13]:
            if available_mask[9] and available_mask[16]:
                synthesizable.append('left_collar')
            else:
                missing.append('left_collar')

        if not available_mask[14]:
            if available_mask[9] and available_mask[17]:
                synthesizable.append('right_collar')
            else:
                missing.append('right_collar')

        # Check feet (10, 11)
        if not available_mask[10]:
            if available_mask[7]:
                synthesizable.append('left_foot')
            else:
                missing.append('left_foot')

        if not available_mask[11]:
            if available_mask[8]:
                synthesizable.append('right_foot')
            else:
                missing.append('right_foot')

        return {
            'synthesizable': synthesizable,
            'missing': missing
        }
