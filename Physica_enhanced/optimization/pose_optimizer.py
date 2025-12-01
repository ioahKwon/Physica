#!/usr/bin/env python3
"""
Fast pose optimization with keyframe sampling + interpolation.

Strategy (≤10 iterations):
1. Sample keyframes (e.g., 20% of frames)
2. Optimize poses only for keyframes (fast!)
3. Interpolate poses for intermediate frames using SLERP
4. Optional: Quick refinement pass on interpolated frames

This is 5-10x faster than optimizing every frame individually.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from torch.cuda.amp import autocast, GradScaler

from core.smpl_model import SMPLModelWrapper
from core.config import PoseOptConfig
from utils.interpolation import (
    sample_keyframes,
    interpolate_full_sequence,
)
from .losses import (
    position_loss,
    bone_direction_loss,
    smoothness_loss,
)


class PoseOptimizer:
    """
    Fast pose optimizer using keyframe sampling + interpolation.

    Optimizes pose parameters (θ) and translations for keyframes only,
    then interpolates to full sequence.
    """

    def __init__(
        self,
        smpl_model: SMPLModelWrapper,
        config: PoseOptConfig,
        device: torch.device = torch.device('cpu'),
        verbose: bool = True
    ):
        """
        Initialize pose optimizer.

        Args:
            smpl_model: SMPL model wrapper
            config: Pose optimization configuration
            device: Device for computation
            verbose: Print progress messages
        """
        self.smpl = smpl_model
        self.config = config
        self.device = device
        self.verbose = verbose

        # Setup mixed precision if enabled
        self.use_amp = config.use_mixed_precision and device.type == 'cuda'
        if self.use_amp:
            self.scaler = GradScaler()

    def optimize(
        self,
        target_joints: torch.Tensor,
        mapped_indices: Tuple[list, list],
        betas: torch.Tensor,
        initial_poses: Optional[torch.Tensor] = None,
        initial_trans: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize pose parameters with keyframe sampling.

        Args:
            target_joints: [T, J_source, 3] Target joint positions
            mapped_indices: (source_indices, smpl_indices) mapping
            betas: [10] Shape parameters (fixed during pose optimization)
            initial_poses: [T, 24, 3] Initial pose estimates (optional)
            initial_trans: [T, 3] Initial translations (optional)

        Returns:
            poses: [T, 24, 3] Optimized poses
            trans: [T, 3] Optimized translations
        """
        T = target_joints.shape[0]
        source_indices, smpl_indices = mapped_indices

        if self.verbose:
            print(f"[PoseOptimizer] Optimizing poses for {T} frames")
            print(f"  Keyframe ratio: {self.config.keyframe_ratio:.1%}")
            print(f"  Max iterations: {self.config.max_iters}")

        # Sample keyframes
        keyframe_indices = sample_keyframes(
            T,
            self.config.keyframe_ratio,
            self.config.min_keyframes,
            self.config.max_keyframes,
            method="uniform"
        )
        keyframe_indices_tensor = torch.from_numpy(keyframe_indices).to(self.device)

        num_keyframes = len(keyframe_indices)

        if self.verbose:
            print(f"  Selected {num_keyframes} keyframes")

        # Initialize keyframe poses and translations
        if initial_poses is None:
            poses_kf = torch.zeros(num_keyframes, 24, 3, device=self.device, requires_grad=True)
        else:
            poses_kf = initial_poses[keyframe_indices_tensor].to(self.device).requires_grad_(True)

        if initial_trans is None:
            trans_kf = self._estimate_translations(
                target_joints[keyframe_indices_tensor],
                source_indices,
                smpl_indices
            ).requires_grad_(True)
        else:
            trans_kf = initial_trans[keyframe_indices_tensor].to(self.device).requires_grad_(True)

        # Setup optimizer
        optimizer = torch.optim.Adam([poses_kf, trans_kf], lr=self.config.lr)

        # Build bone pairs for direction loss
        bone_pairs = self._build_bone_pairs(source_indices, smpl_indices)

        # Optimize keyframes
        if self.verbose:
            print(f"  Optimizing {num_keyframes} keyframes...")

        for iteration in range(self.config.max_iters):
            loss = self._optimize_keyframes(
                poses_kf,
                trans_kf,
                betas,
                target_joints[keyframe_indices_tensor],
                source_indices,
                smpl_indices,
                bone_pairs,
                optimizer
            )

            if self.verbose and (iteration + 1) % 5 == 0:
                print(f"    Iter {iteration + 1}/{self.config.max_iters}: Loss = {loss:.6f}")

        # Interpolate full sequence
        if self.verbose:
            print(f"  Interpolating to full sequence using {self.config.interpolation_method}...")

        poses_full, trans_full = interpolate_full_sequence(
            poses_kf.detach(),
            trans_kf.detach(),
            keyframe_indices,
            T,
            method=self.config.interpolation_method
        )

        if self.verbose:
            print(f"  Pose optimization complete!")

        return poses_full, trans_full

    def _optimize_keyframes(
        self,
        poses_kf: torch.Tensor,
        trans_kf: torch.Tensor,
        betas: torch.Tensor,
        target_kf: torch.Tensor,
        source_indices: list,
        smpl_indices: list,
        bone_pairs: list,
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Optimize single iteration over all keyframes."""
        optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                loss = self._compute_loss(
                    poses_kf,
                    trans_kf,
                    betas,
                    target_kf,
                    source_indices,
                    smpl_indices,
                    bone_pairs
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(
                poses_kf,
                trans_kf,
                betas,
                target_kf,
                source_indices,
                smpl_indices,
                bone_pairs
            )

            loss.backward()
            optimizer.step()

        return loss.item()

    def _compute_loss(
        self,
        poses_kf: torch.Tensor,
        trans_kf: torch.Tensor,
        betas: torch.Tensor,
        target_kf: torch.Tensor,
        source_indices: list,
        smpl_indices: list,
        bone_pairs: list
    ) -> torch.Tensor:
        """Compute loss for keyframes."""
        K = poses_kf.shape[0]
        total_loss = 0.0

        for i in range(K):
            # Forward pass
            _, joints_pred = self.smpl(betas, poses_kf[i], trans_kf[i])

            # Position loss
            target_subset = target_kf[i, source_indices, :]
            pred_subset = joints_pred[smpl_indices, :]

            mask = ~torch.isnan(target_subset).any(dim=-1)

            if mask.sum() > 0:
                pos_loss = position_loss(
                    pred_subset.unsqueeze(0),
                    target_subset.unsqueeze(0),
                    mask.unsqueeze(0),
                    reduction="mean"
                )

                total_loss = total_loss + self.config.weight_position * pos_loss

                # Bone direction loss
                if len(bone_pairs) > 0:
                    dir_loss = bone_direction_loss(
                        joints_pred,
                        target_kf[i],
                        bone_pairs,
                        reduction="mean"
                    )

                    total_loss = total_loss + self.config.weight_bone_direction * dir_loss

        # Average over keyframes
        total_loss = total_loss / K

        # Smoothness loss (temporal coherence across keyframes)
        if K > 1:
            smooth_loss_poses = smoothness_loss(poses_kf, order=1, reduction="mean")
            smooth_loss_trans = smoothness_loss(trans_kf, order=1, reduction="mean")

            total_loss = total_loss + self.config.weight_smoothness * (smooth_loss_poses + smooth_loss_trans)

        return total_loss

    def _build_bone_pairs(
        self,
        source_indices: list,
        smpl_indices: list
    ) -> list:
        """
        Build bone pairs for direction loss.

        Returns:
            List of (pred_parent, pred_child, target_parent, target_child)
        """
        # Create mapping
        smpl_to_source = {smpl: src for src, smpl in zip(source_indices, smpl_indices)}

        # Standard bone pairs (SMPL parent, SMPL child)
        standard_pairs = [
            (0, 1),   # pelvis → left_hip
            (0, 2),   # pelvis → right_hip
            (1, 4),   # left_hip → left_knee
            (2, 5),   # right_hip → right_knee
            (4, 7),   # left_knee → left_ankle
            (5, 8),   # right_knee → right_ankle
            (7, 10),  # left_ankle → left_foot
            (8, 11),  # right_ankle → right_foot
        ]

        bone_pairs = []
        for smpl_parent, smpl_child in standard_pairs:
            if smpl_parent in smpl_to_source and smpl_child in smpl_to_source:
                src_parent = smpl_to_source[smpl_parent]
                src_child = smpl_to_source[smpl_child]
                bone_pairs.append((smpl_parent, smpl_child, src_parent, src_child))

        return bone_pairs

    def _estimate_translations(
        self,
        target_joints: torch.Tensor,
        source_indices: list,
        smpl_indices: list
    ) -> torch.Tensor:
        """Estimate initial translations from target pelvis."""
        K = target_joints.shape[0]

        if 0 in smpl_indices:
            pelvis_source_idx = source_indices[smpl_indices.index(0)]
            pelvis_positions = target_joints[:, pelvis_source_idx, :]
            return pelvis_positions.to(self.device)
        else:
            return torch.zeros(K, 3, device=self.device)
