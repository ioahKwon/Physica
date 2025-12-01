#!/usr/bin/env python3
"""
Shape-first optimization strategy.

Optimizes global shape parameters (β) over all frames before pose refinement.
This is much faster than joint pose+shape optimization.

Strategy:
1. Sample representative keyframes from the sequence
2. Use simple T-pose or initial pose estimates
3. Optimize β to minimize position error across all sampled frames
4. Use mini-batch SGD for memory efficiency
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple
from torch.cuda.amp import autocast, GradScaler

from core.smpl_model import SMPLModelWrapper
from core.config import ShapeOptConfig
from .losses import position_loss, bone_length_loss, get_standard_bone_pairs


class ShapeOptimizer:
    """
    Optimizes SMPL shape parameters (β) over entire sequence.

    Uses mini-batch SGD for efficient large-scale optimization.
    """

    def __init__(
        self,
        smpl_model: SMPLModelWrapper,
        config: ShapeOptConfig,
        device: torch.device = torch.device('cpu'),
        verbose: bool = True
    ):
        """
        Initialize shape optimizer.

        Args:
            smpl_model: SMPL model wrapper
            config: Shape optimization configuration
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
        initial_poses: Optional[torch.Tensor] = None,
        initial_trans: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Optimize shape parameters over all frames.

        Args:
            target_joints: [T, J_source, 3] Target joint positions
            mapped_indices: (source_indices, smpl_indices) mapping
            initial_poses: [T, 24, 3] Initial pose estimates (optional, uses T-pose if None)
            initial_trans: [T, 3] Initial translations (optional, uses target pelvis if None)

        Returns:
            [10] Optimized shape parameters (β)
        """
        T = target_joints.shape[0]
        source_indices, smpl_indices = mapped_indices

        if self.verbose:
            print(f"[ShapeOptimizer] Optimizing shape over {T} frames")
            print(f"  Sample frames: {self.config.sample_frames}")
            print(f"  Batch size: {self.config.batch_size}")
            print(f"  Max iterations: {self.config.max_iters}")

        # Sample keyframes
        keyframe_indices = self._sample_keyframes(T)
        keyframe_indices_tensor = torch.from_numpy(keyframe_indices).to(self.device)
        num_keyframes = len(keyframe_indices)

        if self.verbose:
            print(f"  Selected {num_keyframes} keyframes")

        # Initialize shape parameters
        betas = torch.zeros(self.smpl.NUM_BETAS, device=self.device, requires_grad=True)

        # Initialize poses and translations for keyframes
        if initial_poses is None:
            # Use T-pose (all zeros)
            poses_kf = torch.zeros(num_keyframes, 24, 3, device=self.device)
        else:
            poses_kf = initial_poses[keyframe_indices_tensor].to(self.device)

        if initial_trans is None:
            # Estimate from target pelvis position
            trans_kf = self._estimate_translations(target_joints[keyframe_indices_tensor], source_indices, smpl_indices)
        else:
            trans_kf = initial_trans[keyframe_indices_tensor].to(self.device)

        # Setup optimizer
        optimizer = torch.optim.Adam([betas], lr=self.config.lr)

        # Compute target bone lengths (for regularization)
        target_bone_lengths = self._compute_target_bone_lengths(
            target_joints[keyframe_indices_tensor],
            source_indices,
            smpl_indices
        )

        # Mini-batch optimization
        num_batches = int(np.ceil(num_keyframes / self.config.batch_size))

        if self.verbose:
            print(f"  Starting optimization with {num_batches} batches per iteration...")

        best_loss = float('inf')
        best_betas = betas.clone().detach()
        patience_counter = 0
        patience = 10

        for iteration in range(self.config.max_iters):
            # Shuffle keyframes for SGD
            perm = torch.randperm(num_keyframes, device=self.device)

            epoch_loss = 0.0
            epoch_batches = 0

            for batch_idx in range(num_batches):
                start_idx = batch_idx * self.config.batch_size
                end_idx = min((batch_idx + 1) * self.config.batch_size, num_keyframes)
                batch_perm = perm[start_idx:end_idx]

                # Get batch data
                batch_poses = poses_kf[batch_perm]
                batch_trans = trans_kf[batch_perm]
                batch_targets = target_joints[keyframe_indices_tensor[batch_perm]]

                # Optimize batch
                loss = self._optimize_batch(
                    betas,
                    batch_poses,
                    batch_trans,
                    batch_targets,
                    source_indices,
                    smpl_indices,
                    target_bone_lengths,
                    optimizer
                )

                epoch_loss += loss
                epoch_batches += 1

            # Average loss for this epoch
            avg_loss = epoch_loss / epoch_batches

            # Early stopping check
            if avg_loss < best_loss - self.config.tolerance:
                best_loss = avg_loss
                best_betas = betas.clone().detach()
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"    Iter {iteration + 1}/{self.config.max_iters}: Loss = {avg_loss:.6f}")

            # Early stopping
            if patience_counter >= patience:
                if self.verbose:
                    print(f"  Early stopping at iteration {iteration + 1}")
                break

        if self.verbose:
            print(f"  Optimization complete! Final loss: {best_loss:.6f}")

        return best_betas

    def _optimize_batch(
        self,
        betas: torch.Tensor,
        batch_poses: torch.Tensor,
        batch_trans: torch.Tensor,
        batch_targets: torch.Tensor,
        source_indices: list,
        smpl_indices: list,
        target_bone_lengths: Optional[torch.Tensor],
        optimizer: torch.optim.Optimizer
    ) -> float:
        """Optimize single batch."""
        optimizer.zero_grad()

        if self.use_amp:
            with autocast():
                loss = self._compute_loss(
                    betas,
                    batch_poses,
                    batch_trans,
                    batch_targets,
                    source_indices,
                    smpl_indices,
                    target_bone_lengths
                )

            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = self._compute_loss(
                betas,
                batch_poses,
                batch_trans,
                batch_targets,
                source_indices,
                smpl_indices,
                target_bone_lengths
            )

            loss.backward()
            optimizer.step()

        return loss.item()

    def _compute_loss(
        self,
        betas: torch.Tensor,
        batch_poses: torch.Tensor,
        batch_trans: torch.Tensor,
        batch_targets: torch.Tensor,
        source_indices: list,
        smpl_indices: list,
        target_bone_lengths: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute loss for batch."""
        batch_size = batch_poses.shape[0]

        # Expand betas for batch
        betas_batch = betas.unsqueeze(0).expand(batch_size, -1)

        # Forward pass through SMPL (batch)
        total_loss = 0.0

        for i in range(batch_size):
            _, joints_pred = self.smpl(betas_batch[i], batch_poses[i], batch_trans[i])

            # Position loss
            target_subset = batch_targets[i, source_indices, :]
            pred_subset = joints_pred[smpl_indices, :]

            # Create mask for valid joints (non-NaN)
            mask = ~torch.isnan(target_subset).any(dim=-1)

            if mask.sum() > 0:
                pos_loss = position_loss(
                    pred_subset.unsqueeze(0),
                    target_subset.unsqueeze(0),
                    mask.unsqueeze(0),
                    reduction="mean"
                )

                total_loss = total_loss + self.config.weight_position * pos_loss

        # Average over batch
        total_loss = total_loss / batch_size

        # Bone length regularization (use first frame in batch)
        if target_bone_lengths is not None:
            _, joints_first = self.smpl(betas, batch_poses[0], batch_trans[0])

            bone_pairs = get_standard_bone_pairs(smpl_indices)
            bone_loss = bone_length_loss(
                joints_first,
                bone_pairs,
                target_bone_lengths,
                reduction="mean"
            )

            total_loss = total_loss + self.config.weight_bone_length * bone_loss

        return total_loss

    def _sample_keyframes(self, num_frames: int) -> np.ndarray:
        """Sample keyframe indices."""
        num_samples = min(self.config.sample_frames, num_frames)

        if num_samples >= num_frames:
            return np.arange(num_frames)

        # Uniform sampling
        indices = np.linspace(0, num_frames - 1, num_samples, dtype=int)

        # Ensure first and last frames are included
        indices[0] = 0
        indices[-1] = num_frames - 1

        return indices

    def _estimate_translations(
        self,
        target_joints: torch.Tensor,
        source_indices: list,
        smpl_indices: list
    ) -> torch.Tensor:
        """
        Estimate initial translations from target pelvis positions.

        Args:
            target_joints: [K, J_source, 3]
            source_indices: Source joint indices
            smpl_indices: SMPL joint indices

        Returns:
            [K, 3] Estimated translations
        """
        K = target_joints.shape[0]

        # Find pelvis index (SMPL joint 0)
        if 0 in smpl_indices:
            pelvis_source_idx = source_indices[smpl_indices.index(0)]
            pelvis_positions = target_joints[:, pelvis_source_idx, :]

            # Use pelvis position as initial translation
            return pelvis_positions.to(self.device)
        else:
            # Fallback: use mean of available joints
            return torch.zeros(K, 3, device=self.device)

    def _compute_target_bone_lengths(
        self,
        target_joints: torch.Tensor,
        source_indices: list,
        smpl_indices: list
    ) -> Optional[torch.Tensor]:
        """
        Compute average bone lengths from target data.

        Args:
            target_joints: [K, J_source, 3]
            source_indices: Source joint indices
            smpl_indices: SMPL joint indices

        Returns:
            [B] Average bone lengths (or None if not enough data)
        """
        bone_pairs = get_standard_bone_pairs(smpl_indices)

        if len(bone_pairs) == 0:
            return None

        # Map to source indices
        smpl_to_source = {smpl: src for src, smpl in zip(source_indices, smpl_indices)}

        lengths_per_bone = []

        for parent_smpl, child_smpl in bone_pairs:
            if parent_smpl not in smpl_to_source or child_smpl not in smpl_to_source:
                lengths_per_bone.append(None)
                continue

            parent_src = smpl_to_source[parent_smpl]
            child_src = smpl_to_source[child_smpl]

            # Compute lengths across all keyframes
            bone_vecs = target_joints[:, child_src, :] - target_joints[:, parent_src, :]
            lengths = torch.norm(bone_vecs, dim=-1)

            # Filter out NaN values
            valid_lengths = lengths[~torch.isnan(lengths)]

            if len(valid_lengths) > 0:
                avg_length = valid_lengths.mean()
                lengths_per_bone.append(avg_length)
            else:
                lengths_per_bone.append(None)

        # Filter out None values
        valid_lengths = [l for l in lengths_per_bone if l is not None]

        if len(valid_lengths) == 0:
            return None

        return torch.stack(valid_lengths)
