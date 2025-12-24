"""
Scapula DOF handling module.

Handles the complex shoulder/scapula mapping between AddB and SKEL:
- AddB has acromial_* (surface landmarks)
- SKEL has scapula_* (joint) and humerus_* (glenohumeral center)

Key approach:
1. Use virtual acromial computed from SKEL mesh to match AddB acromial_*
2. Regularize scapula DOFs toward zero with bounded constraints
3. Ensure humerus_* (glenohumeral) aligns with arm chain
"""

from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    OptimizationConfig,
    SCAPULA_DOF_INDICES,
    HUMERUS_DOF_INDICES,
    SCAPULA_DOF_BOUNDS,
    SKEL_NUM_POSE_DOF,
)
from .skel_interface import SKELInterface
from .joint_definitions import (
    ADDB_ACROMIAL_R_IDX,
    ADDB_ACROMIAL_L_IDX,
    SKEL_SCAPULA_R_IDX,
    SKEL_SCAPULA_L_IDX,
    SKEL_HUMERUS_R_IDX,
    SKEL_HUMERUS_L_IDX,
)


class ScapulaHandler:
    """
    Handles scapula DOF optimization and virtual acromial computation.
    """

    def __init__(
        self,
        skel_interface: SKELInterface,
        config: Optional[OptimizationConfig] = None,
    ):
        """
        Initialize scapula handler.

        Args:
            skel_interface: SKEL model interface.
            config: Optimization configuration.
        """
        self.skel = skel_interface
        self.config = config or OptimizationConfig()
        self.device = self.config.get_device()

        # DOF indices
        self.scapula_dof_all = self._get_all_scapula_dof_indices()
        self.humerus_dof_all = self._get_all_humerus_dof_indices()

    def _get_all_scapula_dof_indices(self) -> List[int]:
        """Get all scapula DOF indices (both sides)."""
        indices = []
        for side in ['right', 'left']:
            for dof_type in ['abduction', 'elevation', 'upward_rot']:
                indices.append(SCAPULA_DOF_INDICES[side][dof_type])
        return indices

    def _get_all_humerus_dof_indices(self) -> List[int]:
        """Get all humerus DOF indices (both sides)."""
        indices = []
        indices.extend(HUMERUS_DOF_INDICES['right'])
        indices.extend(HUMERUS_DOF_INDICES['left'])
        return indices

    def compute_virtual_acromial(
        self,
        vertices: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute virtual acromial positions from mesh vertices.

        Args:
            vertices: Mesh vertices [B, V, 3].

        Returns:
            Dictionary with 'right' and 'left' acromial positions [B, 3].
        """
        return self.skel.get_virtual_acromial(vertices, side='both')

    def compute_acromial_loss(
        self,
        skel_vertices: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss between virtual acromial and AddB acromial landmarks.

        Args:
            skel_vertices: SKEL mesh vertices [B, V, 3].
            addb_joints: AddB joint positions [B, 20, 3].

        Returns:
            Loss value (scalar).
        """
        # Get virtual acromial from SKEL mesh
        virtual_acr = self.compute_virtual_acromial(skel_vertices)

        # Get AddB acromial targets
        addb_acr_r = addb_joints[:, ADDB_ACROMIAL_R_IDX, :]
        addb_acr_l = addb_joints[:, ADDB_ACROMIAL_L_IDX, :]

        # Compute L2 loss
        loss_r = F.mse_loss(virtual_acr['right'], addb_acr_r)
        loss_l = F.mse_loss(virtual_acr['left'], addb_acr_l)

        return loss_r + loss_l

    def compute_humerus_alignment_loss(
        self,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute loss for humerus (glenohumeral) alignment with arm chain.

        The humerus joint should be positioned such that the upper arm
        (humerus to elbow) aligns with AddB (acromial to elbow).

        Args:
            skel_joints: SKEL joint positions [B, 24, 3].
            addb_joints: AddB joint positions [B, 20, 3].

        Returns:
            Loss value (scalar).
        """
        # SKEL upper arm vectors (humerus to ulna)
        skel_upperarm_r = skel_joints[:, 16, :] - skel_joints[:, SKEL_HUMERUS_R_IDX, :]
        skel_upperarm_l = skel_joints[:, 21, :] - skel_joints[:, SKEL_HUMERUS_L_IDX, :]

        # AddB upper arm vectors (acromial to elbow)
        addb_upperarm_r = addb_joints[:, 13, :] - addb_joints[:, ADDB_ACROMIAL_R_IDX, :]
        addb_upperarm_l = addb_joints[:, 17, :] - addb_joints[:, ADDB_ACROMIAL_L_IDX, :]

        # Normalize to get directions
        def normalize(v):
            return v / (torch.norm(v, dim=-1, keepdim=True) + 1e-8)

        # Cosine similarity loss (1 - cos_sim)
        cos_r = (normalize(skel_upperarm_r) * normalize(addb_upperarm_r)).sum(dim=-1)
        cos_l = (normalize(skel_upperarm_l) * normalize(addb_upperarm_l)).sum(dim=-1)

        return (2 - cos_r - cos_l).mean()

    def compute_scapula_regularization(
        self,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute regularization loss for scapula DOFs.

        Encourages scapula DOFs to stay near zero (neutral position).

        Args:
            poses: SKEL pose parameters [B, 46].

        Returns:
            Regularization loss (scalar).
        """
        scapula_dofs = poses[:, self.scapula_dof_all]
        return (scapula_dofs ** 2).mean()

    def compute_humerus_regularization(
        self,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute regularization loss for humerus DOFs.

        Prevents extreme shoulder rotations.

        Args:
            poses: SKEL pose parameters [B, 46].

        Returns:
            Regularization loss (scalar).
        """
        humerus_dofs = poses[:, self.humerus_dof_all]
        return (humerus_dofs ** 2).mean()

    def clamp_scapula_dofs(
        self,
        poses: torch.Tensor,
    ) -> torch.Tensor:
        """
        Clamp scapula DOFs to valid range.

        Args:
            poses: SKEL pose parameters [B, 46].

        Returns:
            Clamped poses.
        """
        poses = poses.clone()
        for idx in self.scapula_dof_all:
            poses[:, idx] = torch.clamp(
                poses[:, idx],
                min=SCAPULA_DOF_BOUNDS[0],
                max=SCAPULA_DOF_BOUNDS[1],
            )
        return poses

    def get_scapula_dof_values(
        self,
        poses: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Extract scapula DOF values for diagnostics.

        Args:
            poses: SKEL pose parameters [46] or [B, 46].

        Returns:
            Dictionary with named DOF values.
        """
        if poses.dim() == 1:
            poses = poses.unsqueeze(0)

        poses = poses[0].cpu().numpy()

        return {
            'abduction_r': poses[SCAPULA_DOF_INDICES['right']['abduction']],
            'elevation_r': poses[SCAPULA_DOF_INDICES['right']['elevation']],
            'upward_rot_r': poses[SCAPULA_DOF_INDICES['right']['upward_rot']],
            'abduction_l': poses[SCAPULA_DOF_INDICES['left']['abduction']],
            'elevation_l': poses[SCAPULA_DOF_INDICES['left']['elevation']],
            'upward_rot_l': poses[SCAPULA_DOF_INDICES['left']['upward_rot']],
        }

    def initialize_scapula_from_acromial(
        self,
        addb_joints: torch.Tensor,
        skel_joints_tpose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Initialize scapula DOFs based on AddB acromial positions.

        This provides a better starting point for optimization by
        estimating initial scapula DOFs from the relative position
        of AddB acromial compared to SKEL T-pose.

        Args:
            addb_joints: AddB joint positions [B, 20, 3].
            skel_joints_tpose: SKEL T-pose joints [24, 3].

        Returns:
            Initial scapula DOF values [6] (3 right + 3 left).
        """
        # This is a simplified heuristic
        # In practice, would need proper inverse kinematics

        # Compare acromial positions to estimate elevation/abduction
        addb_acr_r = addb_joints[0, ADDB_ACROMIAL_R_IDX, :].cpu().numpy()
        addb_acr_l = addb_joints[0, ADDB_ACROMIAL_L_IDX, :].cpu().numpy()

        skel_hum_r = skel_joints_tpose[SKEL_HUMERUS_R_IDX, :].cpu().numpy()
        skel_hum_l = skel_joints_tpose[SKEL_HUMERUS_L_IDX, :].cpu().numpy()

        # Vertical difference suggests elevation
        elev_r = (addb_acr_r[1] - skel_hum_r[1]) * 2.0  # Rough scaling
        elev_l = (addb_acr_l[1] - skel_hum_l[1]) * 2.0

        # Lateral difference suggests abduction
        abd_r = (addb_acr_r[0] - skel_hum_r[0]) * 1.5
        abd_l = -(addb_acr_l[0] - skel_hum_l[0]) * 1.5  # Negate for left side

        # Clamp to valid range
        def clamp(v):
            return np.clip(v, SCAPULA_DOF_BOUNDS[0], SCAPULA_DOF_BOUNDS[1])

        return torch.tensor([
            clamp(abd_r), clamp(elev_r), 0,  # Right: abd, elev, upward_rot
            clamp(abd_l), clamp(elev_l), 0,  # Left
        ], device=self.device)


def compute_shoulder_losses(
    skel_vertices: torch.Tensor,
    skel_joints: torch.Tensor,
    skel_poses: torch.Tensor,
    addb_joints: torch.Tensor,
    scapula_handler: ScapulaHandler,
    config: OptimizationConfig,
) -> Dict[str, torch.Tensor]:
    """
    Compute all shoulder-related losses.

    Args:
        skel_vertices: SKEL mesh vertices [B, V, 3].
        skel_joints: SKEL joint positions [B, 24, 3].
        skel_poses: SKEL pose parameters [B, 46].
        addb_joints: AddB joint positions [B, 20, 3].
        scapula_handler: Scapula handler instance.
        config: Optimization configuration.

    Returns:
        Dictionary of loss components.
    """
    losses = {}

    # Virtual acromial loss
    losses['acromial'] = config.weight_shoulder * scapula_handler.compute_acromial_loss(
        skel_vertices, addb_joints
    )

    # Humerus alignment loss
    losses['humerus_align'] = 0.5 * scapula_handler.compute_humerus_alignment_loss(
        skel_joints, addb_joints
    )

    # Scapula regularization
    losses['scapula_reg'] = config.weight_scapula_reg * scapula_handler.compute_scapula_regularization(
        skel_poses
    )

    # Humerus regularization
    losses['humerus_reg'] = 0.05 * scapula_handler.compute_humerus_regularization(
        skel_poses
    )

    return losses
