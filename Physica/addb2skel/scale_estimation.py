"""
Subject-specific scale estimation module.

Stage 1 of the 2-stage optimization pipeline (following AddBiomechanics approach):
1. Estimate subject scale/proportions first (global over all frames)
2. Then optimize pose per frame

This module estimates SKEL beta parameters to match AddB bone lengths.
"""

from typing import Tuple, Optional, Dict, List
import numpy as np
import torch
import torch.nn.functional as F

from .config import OptimizationConfig, SKEL_NUM_BETAS, SKEL_NUM_POSE_DOF
from .skel_interface import SKELInterface
from .joint_definitions import (
    RELIABLE_BONE_PAIRS_ADDB,
    RELIABLE_BONE_PAIRS_SKEL,
    ADDB_JOINT_TO_IDX,
    SKEL_JOINT_TO_IDX,
    get_bone_indices,
)
from .utils.geometry import compute_bone_lengths


class ScaleEstimator:
    """
    Estimates SKEL shape parameters (beta) to match AddB subject proportions.

    Uses reliable bone lengths (legs, forearms) that are not affected by
    shoulder/scapula uncertainty.
    """

    def __init__(
        self,
        skel_interface: SKELInterface,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize scale estimator.

        Args:
            skel_interface: SKEL model interface.
            config: Optimization configuration.
        """
        self.skel = skel_interface
        self.config = config or OptimizationConfig()
        self.device = self.config.get_device()

        # Build bone pair indices
        self.addb_bone_indices = get_bone_indices(
            RELIABLE_BONE_PAIRS_ADDB, ADDB_JOINT_TO_IDX
        )
        self.skel_bone_indices = get_bone_indices(
            RELIABLE_BONE_PAIRS_SKEL, SKEL_JOINT_TO_IDX
        )

    def estimate_from_bone_lengths(
        self,
        addb_joints: np.ndarray,
        initial_betas: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate SKEL betas by matching reliable bone lengths.

        Args:
            addb_joints: AddB joint positions [T, 20, 3] in meters.
            initial_betas: Initial beta values [10]. Default: zeros.
            verbose: Print progress.

        Returns:
            betas: Estimated shape parameters [10].
            stats: Dictionary with estimation statistics.
        """
        # Compute target bone lengths from AddB (average over frames)
        addb_joints_t = torch.from_numpy(addb_joints).float().to(self.device)
        target_lengths = compute_bone_lengths(
            addb_joints_t, self.addb_bone_indices
        ).mean(dim=0)  # [num_bones]

        if verbose:
            print(f"Target bone lengths (mm): {target_lengths.cpu().numpy() * 1000}")

        # Initialize betas
        if initial_betas is None:
            betas = torch.zeros(SKEL_NUM_BETAS, device=self.device)
        else:
            betas = initial_betas.clone().to(self.device)

        betas.requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([betas], lr=self.config.scale_lr)

        # T-pose for scale estimation
        poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
        trans = torch.zeros(1, 3, device=self.device)

        best_loss = float('inf')
        best_betas = betas.clone()

        for it in range(self.config.scale_iters):
            optimizer.zero_grad()

            # Forward through SKEL
            _, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0), poses, trans
            )

            # Compute SKEL bone lengths
            skel_lengths = compute_bone_lengths(
                skel_joints, self.skel_bone_indices
            )[0]  # [num_bones]

            # Bone length loss
            length_loss = F.mse_loss(skel_lengths, target_lengths)

            # Beta regularization (prefer smaller betas)
            reg_loss = 0.001 * (betas ** 2).mean()

            loss = length_loss + reg_loss

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_betas = betas.clone().detach()

            if verbose and (it + 1) % 50 == 0:
                with torch.no_grad():
                    length_err = (skel_lengths - target_lengths).abs().mean() * 1000
                print(f"  Iter {it+1}/{self.config.scale_iters}: "
                      f"Loss={loss.item():.6f}, LenErr={length_err:.2f}mm")

        # Final statistics
        with torch.no_grad():
            _, skel_joints, _ = self.skel.forward(
                best_betas.unsqueeze(0), poses, trans
            )
            skel_lengths = compute_bone_lengths(
                skel_joints, self.skel_bone_indices
            )[0]
            length_errors = (skel_lengths - target_lengths).abs() * 1000

        stats = {
            'final_loss': best_loss,
            'bone_length_errors_mm': length_errors.cpu().numpy(),
            'mean_length_error_mm': length_errors.mean().item(),
            'target_lengths_mm': target_lengths.cpu().numpy() * 1000,
            'fitted_lengths_mm': skel_lengths.cpu().numpy() * 1000,
        }

        return best_betas, stats

    def estimate_from_height_width(
        self,
        height_m: float,
        shoulder_width_m: float,
        gender: str = 'male',
    ) -> torch.Tensor:
        """
        Estimate betas from height and shoulder width.

        This is a simpler initialization method based on anthropometric data.

        Args:
            height_m: Subject height in meters.
            shoulder_width_m: Shoulder width in meters.
            gender: 'male' or 'female'.

        Returns:
            betas: Estimated shape parameters [10].
        """
        # Baseline values for male SKEL model (from T-pose)
        if gender == 'male':
            baseline_height = 1.58  # meters
            baseline_shoulder = 0.35  # meters
        else:
            baseline_height = 1.52
            baseline_shoulder = 0.32

        # Beta[0] primarily affects height
        # Beta[1] affects shoulder width
        height_ratio = height_m / baseline_height
        shoulder_ratio = shoulder_width_m / baseline_shoulder

        # Empirical scaling (approximate)
        beta0 = -3.5 * (height_ratio - 1.0)  # Height adjustment
        beta1 = 2.0 * (shoulder_ratio - 1.0)  # Shoulder adjustment

        betas = torch.zeros(SKEL_NUM_BETAS, device=self.device)
        betas[0] = beta0
        betas[1] = beta1

        return betas


def estimate_subject_scale(
    addb_joints: np.ndarray,
    skel_interface: SKELInterface,
    config: Optional[OptimizationConfig] = None,
    height_m: Optional[float] = None,
    shoulder_width_m: Optional[float] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, Dict]:
    """
    Convenience function to estimate subject scale.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        height_m: Optional known height for initialization.
        shoulder_width_m: Optional known shoulder width for initialization.
        verbose: Print progress.

    Returns:
        betas: Estimated shape parameters [10].
        stats: Estimation statistics.
    """
    estimator = ScaleEstimator(skel_interface, config)

    # Get initial betas from height/width if available
    initial_betas = None
    if height_m is not None and shoulder_width_m is not None:
        initial_betas = estimator.estimate_from_height_width(
            height_m, shoulder_width_m
        )
        if verbose:
            print(f"Initial betas from height/width: {initial_betas[:3].cpu().numpy()}")

    # Refine using bone lengths
    betas, stats = estimator.estimate_from_bone_lengths(
        addb_joints, initial_betas, verbose
    )

    return betas, stats
