"""
Per-frame pose optimization module.

Stage 2 of the 2-stage optimization pipeline:
1. Scale estimation (done in scale_estimation.py)
2. Pose optimization per frame with temporal smoothing

Implements IK-style gradient descent optimization following:
- HSMR SKELify approach for 2D keypoint fitting
- AddBiomechanics kinematics pass for marker fitting
"""

from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .config import (
    OptimizationConfig,
    SKEL_NUM_POSE_DOF,
    SPINE_DOF_INDICES,
)
from .skel_interface import SKELInterface
from .scapula_handler import ScapulaHandler, compute_shoulder_losses
from .joint_definitions import (
    build_direct_joint_mapping,
    get_bone_indices,
    ADDB_BONE_PAIRS,
    SKEL_BONE_PAIRS,
    ADDB_JOINT_TO_IDX,
    SKEL_JOINT_TO_IDX,
    ADDB_ACROMIAL_R_IDX,
    ADDB_ACROMIAL_L_IDX,
)
from .utils.geometry import (
    compute_bone_lengths,
    compute_bone_directions,
    cosine_similarity_loss,
)


class PoseOptimizer:
    """
    Optimizes SKEL pose parameters to match AddB joint positions.

    Uses a multi-term loss function:
    - Joint position loss (primary)
    - Bone direction loss
    - Bone length loss
    - Virtual acromial loss (for shoulder)
    - Shoulder width loss
    - Pose regularization
    - Temporal smoothness (optional)
    """

    def __init__(
        self,
        skel_interface: SKELInterface,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize pose optimizer.

        Args:
            skel_interface: SKEL model interface.
            config: Optimization configuration.
        """
        self.skel = skel_interface
        self.config = config or OptimizationConfig()
        self.device = self.config.get_device()

        # Build joint mapping
        self.addb_indices, self.skel_indices = build_direct_joint_mapping()

        # Exclude acromial from direct joint loss (use virtual acromial instead)
        self.addb_indices_no_acr = [
            i for i in self.addb_indices
            if i not in [ADDB_ACROMIAL_R_IDX, ADDB_ACROMIAL_L_IDX]
        ]
        self.skel_indices_no_acr = [
            self.skel_indices[j] for j, i in enumerate(self.addb_indices)
            if i not in [ADDB_ACROMIAL_R_IDX, ADDB_ACROMIAL_L_IDX]
        ]

        # Build bone pair indices
        self.addb_bone_indices = get_bone_indices(ADDB_BONE_PAIRS, ADDB_JOINT_TO_IDX)
        self.skel_bone_indices = get_bone_indices(SKEL_BONE_PAIRS, SKEL_JOINT_TO_IDX)

        # Scapula handler
        self.scapula_handler = ScapulaHandler(skel_interface, config)

    def optimize_single_frame(
        self,
        addb_joints: torch.Tensor,
        betas: torch.Tensor,
        initial_poses: Optional[torch.Tensor] = None,
        initial_trans: Optional[torch.Tensor] = None,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Optimize pose for a single frame.

        Args:
            addb_joints: AddB joint positions [20, 3] in meters.
            betas: SKEL shape parameters [10].
            initial_poses: Initial pose [46]. Default: zeros.
            initial_trans: Initial translation [3]. Default: pelvis position.
            verbose: Print progress.

        Returns:
            poses: Optimized pose parameters [46].
            trans: Optimized translation [3].
            stats: Optimization statistics.
        """
        # Ensure batch dimension
        if addb_joints.dim() == 2:
            addb_joints = addb_joints.unsqueeze(0)

        # Initialize pose
        if initial_poses is None:
            poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
        else:
            poses = initial_poses.clone().unsqueeze(0) if initial_poses.dim() == 1 else initial_poses.clone()

        # Initialize translation from pelvis
        if initial_trans is None:
            trans = addb_joints[0, 0, :].clone().unsqueeze(0)
        else:
            trans = initial_trans.clone().unsqueeze(0) if initial_trans.dim() == 1 else initial_trans.clone()

        poses.requires_grad_(True)
        trans.requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([poses, trans], lr=self.config.pose_lr)

        # Target bone lengths and directions
        target_bone_lengths = compute_bone_lengths(addb_joints, self.addb_bone_indices)
        target_bone_dirs = compute_bone_directions(addb_joints, self.addb_bone_indices)

        # Target shoulder width
        target_width = torch.norm(
            addb_joints[:, ADDB_ACROMIAL_R_IDX, :] - addb_joints[:, ADDB_ACROMIAL_L_IDX, :],
            dim=-1
        )

        best_loss = float('inf')
        best_poses = poses.clone()
        best_trans = trans.clone()

        for it in range(self.config.pose_iters):
            optimizer.zero_grad()

            # Forward through SKEL
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0), poses, trans, return_skeleton=False
            )

            # --- Loss computation ---

            # 1. Joint position loss (exclude acromial)
            pred_joints_mapped = skel_joints[:, self.skel_indices_no_acr, :]
            target_joints = addb_joints[:, self.addb_indices_no_acr, :]
            joint_loss = F.mse_loss(pred_joints_mapped, target_joints)

            # 2. Bone direction loss
            pred_bone_dirs = compute_bone_directions(skel_joints, self.skel_bone_indices)
            bone_dir_loss = cosine_similarity_loss(pred_bone_dirs, target_bone_dirs)

            # 3. Bone length loss
            pred_bone_lengths = compute_bone_lengths(skel_joints, self.skel_bone_indices)
            bone_len_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)

            # 4. Shoulder width loss
            pred_width = self.skel.get_shoulder_width(skel_joints)
            width_loss = F.mse_loss(pred_width, target_width)

            # 5. Shoulder/scapula losses
            shoulder_losses = compute_shoulder_losses(
                skel_verts, skel_joints, poses, addb_joints,
                self.scapula_handler, self.config
            )

            # 6. Pose regularization
            pose_reg = self.config.weight_pose_reg * (poses ** 2).mean()

            # 7. Spine regularization
            spine_dofs = poses[:, SPINE_DOF_INDICES]
            spine_reg = self.config.weight_spine_reg * (spine_dofs ** 2).mean()

            # Combine losses
            loss = (
                self.config.weight_joint * joint_loss +
                self.config.weight_bone_dir * bone_dir_loss +
                self.config.weight_bone_len * bone_len_loss +
                self.config.weight_width * width_loss +
                shoulder_losses['acromial'] +
                shoulder_losses['humerus_align'] +
                shoulder_losses['scapula_reg'] +
                shoulder_losses['humerus_reg'] +
                pose_reg +
                spine_reg
            )

            loss.backward()
            optimizer.step()

            # Clamp scapula DOFs
            with torch.no_grad():
                poses.data = self.scapula_handler.clamp_scapula_dofs(poses.data)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_poses = poses.clone().detach()
                best_trans = trans.clone().detach()

            if verbose and ((it + 1) % 50 == 0 or it == 0):
                with torch.no_grad():
                    mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                print(f"  Iter {it+1}/{self.config.pose_iters}: "
                      f"Loss={loss.item():.4f}, MPJPE={mpjpe:.1f}mm")

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0), best_poses, best_trans
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)

        stats = {
            'final_loss': best_loss,
            'mpjpe_mm': mpjpe,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(best_poses),
        }

        return best_poses.squeeze(0), best_trans.squeeze(0), stats

    def optimize_sequence(
        self,
        addb_joints: torch.Tensor,
        betas: torch.Tensor,
        use_temporal: bool = True,
        verbose: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Optimize poses for a sequence of frames.

        Args:
            addb_joints: AddB joint positions [T, 20, 3] in meters.
            betas: SKEL shape parameters [10].
            use_temporal: Use temporal smoothness regularization.
            verbose: Print progress.

        Returns:
            poses: Optimized pose parameters [T, 46].
            trans: Optimized translations [T, 3].
            stats: Optimization statistics.
        """
        T = addb_joints.shape[0]

        if verbose:
            print(f"Optimizing {T} frames...")

        # Initialize all poses and translations
        poses = torch.zeros(T, SKEL_NUM_POSE_DOF, device=self.device)
        trans = addb_joints[:, 0, :].clone()  # Initialize from pelvis

        poses.requires_grad_(True)
        trans.requires_grad_(True)

        # Optimizer
        optimizer = torch.optim.Adam([poses, trans], lr=self.config.pose_lr)

        # Precompute targets
        target_bone_lengths = compute_bone_lengths(addb_joints, self.addb_bone_indices)
        target_bone_dirs = compute_bone_directions(addb_joints, self.addb_bone_indices)
        target_width = torch.norm(
            addb_joints[:, ADDB_ACROMIAL_R_IDX, :] - addb_joints[:, ADDB_ACROMIAL_L_IDX, :],
            dim=-1
        )

        best_loss = float('inf')
        best_poses = poses.clone()
        best_trans = trans.clone()

        pbar = tqdm(range(self.config.pose_iters), disable=not verbose)
        for it in pbar:
            optimizer.zero_grad()

            # Forward through SKEL
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0).expand(T, -1), poses, trans
            )

            # --- Loss computation ---

            # 1. Joint position loss
            pred_joints_mapped = skel_joints[:, self.skel_indices_no_acr, :]
            target_joints = addb_joints[:, self.addb_indices_no_acr, :]
            joint_loss = F.mse_loss(pred_joints_mapped, target_joints)

            # 2. Bone direction loss
            pred_bone_dirs = compute_bone_directions(skel_joints, self.skel_bone_indices)
            bone_dir_loss = cosine_similarity_loss(pred_bone_dirs, target_bone_dirs)

            # 3. Bone length loss
            pred_bone_lengths = compute_bone_lengths(skel_joints, self.skel_bone_indices)
            bone_len_loss = F.mse_loss(pred_bone_lengths, target_bone_lengths)

            # 4. Shoulder width loss
            pred_width = self.skel.get_shoulder_width(skel_joints)
            width_loss = F.mse_loss(pred_width, target_width)

            # 5. Shoulder/scapula losses
            shoulder_losses = compute_shoulder_losses(
                skel_verts, skel_joints, poses, addb_joints,
                self.scapula_handler, self.config
            )

            # 6. Pose regularization
            pose_reg = self.config.weight_pose_reg * (poses ** 2).mean()

            # 7. Spine regularization
            spine_dofs = poses[:, SPINE_DOF_INDICES]
            spine_reg = self.config.weight_spine_reg * (spine_dofs ** 2).mean()

            # 8. Temporal smoothness
            temporal_loss = torch.tensor(0.0, device=self.device)
            if use_temporal and T > 1:
                pose_diff = poses[1:] - poses[:-1]
                trans_diff = trans[1:] - trans[:-1]
                temporal_loss = (
                    self.config.weight_temporal * (pose_diff ** 2).mean() +
                    self.config.weight_temporal * (trans_diff ** 2).mean()
                )

            # Combine losses
            loss = (
                self.config.weight_joint * joint_loss +
                self.config.weight_bone_dir * bone_dir_loss +
                self.config.weight_bone_len * bone_len_loss +
                self.config.weight_width * width_loss +
                shoulder_losses['acromial'] +
                shoulder_losses['humerus_align'] +
                shoulder_losses['scapula_reg'] +
                shoulder_losses['humerus_reg'] +
                pose_reg +
                spine_reg +
                temporal_loss
            )

            loss.backward()
            optimizer.step()

            # Clamp scapula DOFs
            with torch.no_grad():
                poses.data = self.scapula_handler.clamp_scapula_dofs(poses.data)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_poses = poses.clone().detach()
                best_trans = trans.clone().detach()

            if verbose:
                with torch.no_grad():
                    mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'mpjpe': f'{mpjpe:.1f}mm'})

        # Final statistics
        with torch.no_grad():
            skel_verts, skel_joints, _ = self.skel.forward(
                betas.unsqueeze(0).expand(T, -1), best_poses, best_trans
            )
            mpjpe = self._compute_mpjpe(skel_joints, addb_joints)
            per_joint_error = self._compute_per_joint_error(skel_joints, addb_joints)

        stats = {
            'final_loss': best_loss,
            'mpjpe_mm': mpjpe,
            'per_joint_error_mm': per_joint_error,
            'scapula_dofs': self.scapula_handler.get_scapula_dof_values(best_poses.mean(dim=0)),
        }

        return best_poses, best_trans, stats

    def _compute_mpjpe(
        self,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> float:
        """Compute mean per-joint position error in mm."""
        pred = skel_joints[:, self.skel_indices, :]
        target = addb_joints[:, self.addb_indices, :]
        error = torch.norm(pred - target, dim=-1)
        return error.mean().item() * 1000  # Convert to mm

    def _compute_per_joint_error(
        self,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> Dict[str, float]:
        """Compute per-joint error in mm."""
        from .joint_definitions import ADDB_JOINTS

        errors = {}
        for i, (addb_idx, skel_idx) in enumerate(zip(self.addb_indices, self.skel_indices)):
            pred = skel_joints[:, skel_idx, :]
            target = addb_joints[:, addb_idx, :]
            error = torch.norm(pred - target, dim=-1).mean().item() * 1000
            errors[ADDB_JOINTS[addb_idx]] = error

        return errors


def optimize_poses(
    addb_joints: np.ndarray,
    betas: torch.Tensor,
    skel_interface: SKELInterface,
    config: Optional[OptimizationConfig] = None,
    verbose: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """
    Convenience function to optimize poses for a sequence.

    Args:
        addb_joints: AddB joint positions [T, 20, 3] in meters.
        betas: SKEL shape parameters [10].
        skel_interface: SKEL model interface.
        config: Optimization configuration.
        verbose: Print progress.

    Returns:
        poses: Optimized pose parameters [T, 46].
        trans: Optimized translations [T, 3].
        stats: Optimization statistics.
    """
    optimizer = PoseOptimizer(skel_interface, config)

    device = config.get_device() if config else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    addb_joints_t = torch.from_numpy(addb_joints).float().to(device)

    return optimizer.optimize_sequence(addb_joints_t, betas, verbose=verbose)
