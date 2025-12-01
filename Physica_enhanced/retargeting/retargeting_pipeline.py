#!/usr/bin/env python3
"""
Complete retargeting pipeline from AddBiomechanics/OpenSim to SMPL.

Combines joint mapping and synthesis for full retargeting workflow.
"""

import torch
import numpy as np
from typing import Union, Optional, List
from dataclasses import dataclass

from .joint_mapping import JointMapper
from .joint_synthesis import JointSynthesizer
from core.config import RetargetConfig


@dataclass
class RetargetingResult:
    """Result of retargeting process."""
    smpl_joints: Union[np.ndarray, torch.Tensor]  # [T, 24, 3] or [24, 3]
    mapped_indices: List[int]  # Which SMPL joints were directly mapped
    synthesized_indices: List[int]  # Which SMPL joints were synthesized
    unmapped_source_joints: List[str]  # Source joints that couldn't be mapped
    joint_mapper: JointMapper


class RetargetingPipeline:
    """
    Complete pipeline for retargeting from source to SMPL skeleton.

    Handles:
    1. Joint mapping (source → SMPL)
    2. Missing joint synthesis
    3. Pre/post alignment (optional)
    """

    def __init__(
        self,
        config: Optional[RetargetConfig] = None,
        joint_mapper: Optional[JointMapper] = None,
        joint_synthesizer: Optional[JointSynthesizer] = None
    ):
        """
        Initialize retargeting pipeline.

        Args:
            config: Retargeting configuration
            joint_mapper: Pre-configured joint mapper (optional)
            joint_synthesizer: Pre-configured joint synthesizer (optional)
        """
        self.config = config or RetargetConfig()
        self.joint_mapper = joint_mapper
        self.joint_synthesizer = joint_synthesizer or JointSynthesizer(
            spine_ratio=self.config.spine_ratio,
            clavicle_offset=self.config.clavicle_offset,
            toe_offset=self.config.toe_offset,
            neck_ratio=self.config.neck_ratio
        )

    def setup_mapping(
        self,
        source_joint_names: List[str],
        mapping_overrides: Optional[dict] = None
    ) -> None:
        """
        Setup joint mapping.

        Args:
            source_joint_names: List of source joint names
            mapping_overrides: Optional manual mapping overrides
        """
        self.joint_mapper = JointMapper(source_joint_names, mapping_overrides)

    def retarget(
        self,
        source_joints: Union[np.ndarray, torch.Tensor],
        source_joint_names: Optional[List[str]] = None,
        mapping_overrides: Optional[dict] = None
    ) -> RetargetingResult:
        """
        Retarget source joints to SMPL skeleton.

        Args:
            source_joints: Source joint positions [T, J, 3] or [J, 3]
            source_joint_names: List of source joint names (required if mapper not setup)
            mapping_overrides: Optional manual mapping overrides

        Returns:
            RetargetingResult with retargeted SMPL joints
        """
        is_torch = isinstance(source_joints, torch.Tensor)
        single_frame = source_joints.ndim == 2

        # Setup mapping if source_joint_names provided or not already done
        if source_joint_names is not None:
            self.setup_mapping(source_joint_names, mapping_overrides)
        elif self.joint_mapper is None:
            raise ValueError("source_joint_names required when joint_mapper not setup")

        # Get mapped indices
        source_indices, smpl_indices = self.joint_mapper.get_mapped_indices()

        # Initialize SMPL joints with NaN
        if single_frame:
            shape = (24, 3)
        else:
            shape = (source_joints.shape[0], 24, 3)

        if is_torch:
            smpl_joints = torch.full(shape, float('nan'), device=source_joints.device, dtype=source_joints.dtype)
        else:
            smpl_joints = np.full(shape, float('nan'), dtype=source_joints.dtype)

        # Map available joints
        for src_idx, smpl_idx in zip(source_indices, smpl_indices):
            if single_frame:
                smpl_joints[smpl_idx] = source_joints[src_idx]
            else:
                smpl_joints[:, smpl_idx, :] = source_joints[:, src_idx, :]

        # Synthesize missing joints if enabled
        synthesized_indices = []
        if self.config.synthesize_missing_joints:
            synthesis_info = self.joint_synthesizer.get_synthesis_info(smpl_indices)

            if len(synthesis_info['synthesizable']) > 0:
                smpl_joints = self.joint_synthesizer.synthesize_missing_joints(
                    smpl_joints,
                    available_joints=smpl_indices
                )

                # Track which joints were synthesized
                from core.smpl_model import SMPLModelWrapper
                for joint_name in synthesis_info['synthesizable']:
                    synthesized_indices.append(SMPLModelWrapper.get_joint_idx(joint_name))

        return RetargetingResult(
            smpl_joints=smpl_joints,
            mapped_indices=smpl_indices,
            synthesized_indices=synthesized_indices,
            unmapped_source_joints=self.joint_mapper.get_unmapped_source_joints(),
            joint_mapper=self.joint_mapper
        )

    def get_mapping_info(self) -> dict:
        """Get information about current mapping."""
        if self.joint_mapper is None:
            return {"error": "Joint mapper not initialized"}

        return {
            "num_mapped": len(self.joint_mapper.mapping),
            "mapped_smpl_joints": [
                self.joint_mapper.SMPL_JOINT_NAMES[i]
                for i in sorted(self.joint_mapper.mapping.values())
            ],
            "unmapped_source": self.joint_mapper.get_unmapped_source_joints(),
            "unmapped_smpl": self.joint_mapper.get_unmapped_smpl_joints(),
        }
