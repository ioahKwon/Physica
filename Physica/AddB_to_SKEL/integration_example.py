"""
Integration Example: Shoulder Correction in AddB → SKEL Optimization

This file demonstrates how to integrate the shoulder correction losses
into your existing SKEL optimization pipeline.

Two approaches are shown:
1. Step 1 (Virtual Joint): Using vertex-based virtual acromial (no model modification)
2. Step 2 (Extended Regressor): Using expanded J_regressor with 26 joints
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional

# Import shoulder correction module
from shoulder_correction import (
    ACROMIAL_VERTEX_IDX,
    ADDB_JOINT_IDX,
    SKEL_JOINT_IDX,
    compute_virtual_acromial,
    compute_shoulder_losses,
    compute_shoulder_losses_extended,
    loss_acromial,
    loss_shoulder_width,
    loss_shoulder_direction,
)


# =============================================================================
# Approach 1: Step 1 - Virtual Joint (No Model Modification)
# =============================================================================

class ShoulderCorrectionStep1:
    """
    Shoulder correction using virtual acromial joints computed from vertices.

    This approach does NOT modify the SKEL model - it computes virtual acromial
    positions on-the-fly from mesh vertices during optimization.

    Use this for initial experiments before committing to regressor expansion.
    """

    def __init__(
        self,
        lambda_acromial: float = 1.0,
        lambda_width: float = 1.0,
        lambda_direction: float = 0.5,
        v_idx_r: Optional[list] = None,
        v_idx_l: Optional[list] = None,
    ):
        """
        Args:
            lambda_acromial: Weight for acromial position matching loss
            lambda_width: Weight for shoulder width matching loss
            lambda_direction: Weight for shoulder direction matching loss
            v_idx_r: Right acromial vertex indices (default: standard SMPL/SKEL)
            v_idx_l: Left acromial vertex indices (default: standard SMPL/SKEL)
        """
        self.lambda_acromial = lambda_acromial
        self.lambda_width = lambda_width
        self.lambda_direction = lambda_direction
        self.v_idx_r = v_idx_r or ACROMIAL_VERTEX_IDX['right']
        self.v_idx_l = v_idx_l or ACROMIAL_VERTEX_IDX['left']

    def compute_loss(
        self,
        skel_vertices: torch.Tensor,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all shoulder correction losses.

        Args:
            skel_vertices: SKEL mesh vertices [B, 6890, 3]
            skel_joints: SKEL joints [B, 24, 3]
            addb_joints: AddBiomechanics joints [B, 20, 3]

        Returns:
            Dictionary containing:
                - loss_acromial: Acromial position matching loss
                - loss_width: Shoulder width matching loss
                - loss_direction: Shoulder direction matching loss
                - loss_total: Weighted sum of all losses
                - virtual_acromial_r: Computed right acromial position [B, 3]
                - virtual_acromial_l: Computed left acromial position [B, 3]
        """
        return compute_shoulder_losses(
            skel_vertices=skel_vertices,
            skel_joints=skel_joints,
            addb_joints=addb_joints,
            v_idx_r=self.v_idx_r,
            v_idx_l=self.v_idx_l,
            lambda_acromial=self.lambda_acromial,
            lambda_width=self.lambda_width,
            lambda_direction=self.lambda_direction,
        )


# =============================================================================
# Approach 2: Step 2 - Extended Regressor (26 Joints)
# =============================================================================

class ShoulderCorrectionStep2:
    """
    Shoulder correction using extended SKEL joint regressor (26 joints).

    This approach requires running create_acromial_regressor.py first to generate
    the extended J_regressor with acromial joints at indices 24 and 25.

    Use this after validating Step 1 works correctly.
    """

    def __init__(
        self,
        lambda_acromial: float = 1.0,
        lambda_width: float = 1.0,
        lambda_direction: float = 0.5,
        acromial_r_idx: int = 24,
        acromial_l_idx: int = 25,
    ):
        """
        Args:
            lambda_acromial: Weight for acromial position matching loss
            lambda_width: Weight for shoulder width matching loss
            lambda_direction: Weight for shoulder direction matching loss
            acromial_r_idx: Index of right acromial in extended joints
            acromial_l_idx: Index of left acromial in extended joints
        """
        self.lambda_acromial = lambda_acromial
        self.lambda_width = lambda_width
        self.lambda_direction = lambda_direction
        self.acromial_r_idx = acromial_r_idx
        self.acromial_l_idx = acromial_l_idx

    def compute_loss(
        self,
        skel_joints: torch.Tensor,
        addb_joints: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all shoulder correction losses.

        Args:
            skel_joints: Extended SKEL joints [B, 26, 3]
            addb_joints: AddBiomechanics joints [B, 20, 3]

        Returns:
            Dictionary containing individual and total losses
        """
        return compute_shoulder_losses_extended(
            skel_joints=skel_joints,
            addb_joints=addb_joints,
            lambda_acromial=self.lambda_acromial,
            lambda_width=self.lambda_width,
            lambda_direction=self.lambda_direction,
            acromial_r_idx=self.acromial_r_idx,
            acromial_l_idx=self.acromial_l_idx,
        )


# =============================================================================
# Integration Template: Full Optimization Loop
# =============================================================================

def example_optimization_loop_step1():
    """
    Example showing how to integrate shoulder correction (Step 1) into
    an existing optimization loop.
    """
    print("=" * 60)
    print("Example: Integration with Step 1 (Virtual Joint)")
    print("=" * 60)

    # =========================================================================
    # 1. Setup (replace with your actual SKEL model)
    # =========================================================================
    # from skel import SKEL
    # skel_model = SKEL(model_path='/path/to/skel_models')

    # Dummy data for demonstration
    B = 32  # batch size (frames)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simulated SKEL output
    skel_vertices = torch.randn(B, 6890, 3, device=device)
    skel_joints = torch.randn(B, 24, 3, device=device)

    # Simulated AddB ground truth
    addb_joints = torch.randn(B, 20, 3, device=device)

    # =========================================================================
    # 2. Initialize shoulder correction
    # =========================================================================
    shoulder_correction = ShoulderCorrectionStep1(
        lambda_acromial=1.0,
        lambda_width=1.0,
        lambda_direction=0.5,
    )

    # =========================================================================
    # 3. In your optimization loop
    # =========================================================================
    # optimizer = torch.optim.Adam([poses, betas, trans], lr=1e-3)

    # for iteration in range(num_iterations):
    #     optimizer.zero_grad()

    #     # Forward pass
    #     skel_vertices, skel_joints = skel_model(betas, poses, trans)

    #     # Compute existing losses
    #     loss_joint_position = ...
    #     loss_bone_length = ...

    # Compute shoulder correction losses
    shoulder_losses = shoulder_correction.compute_loss(
        skel_vertices=skel_vertices,
        skel_joints=skel_joints,
        addb_joints=addb_joints,
    )

    # Combine with existing losses
    loss_total = (
        # loss_joint_position
        # + loss_bone_length
        + shoulder_losses['loss_total']  # Add shoulder correction
    )

    print(f"Loss acromial:   {shoulder_losses['loss_acromial'].item():.4f}")
    print(f"Loss width:      {shoulder_losses['loss_width'].item():.4f}")
    print(f"Loss direction:  {shoulder_losses['loss_direction'].item():.4f}")
    print(f"Loss total:      {shoulder_losses['loss_total'].item():.4f}")

    # Virtual acromial positions (for visualization)
    print(f"\nVirtual acromial R shape: {shoulder_losses['virtual_acromial_r'].shape}")
    print(f"Virtual acromial L shape: {shoulder_losses['virtual_acromial_l'].shape}")

    #     # Backward pass
    #     loss_total.backward()
    #     optimizer.step()


def example_optimization_loop_step2():
    """
    Example showing how to integrate shoulder correction (Step 2) into
    an existing optimization loop with extended regressor.
    """
    print("\n" + "=" * 60)
    print("Example: Integration with Step 2 (Extended Regressor)")
    print("=" * 60)

    # =========================================================================
    # 1. Setup with extended regressor
    # =========================================================================
    # from skel import SKEL

    # # Load SKEL with extended regressor
    # skel_model = SKEL(
    #     model_path='/path/to/skel_models',
    #     custom_joint_reg_path='/path/to/J_regressor_osim_acromial.pkl'
    # )

    # Dummy data for demonstration
    B = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Simulated SKEL output (now 26 joints!)
    skel_joints = torch.randn(B, 26, 3, device=device)
    addb_joints = torch.randn(B, 20, 3, device=device)

    # =========================================================================
    # 2. Initialize shoulder correction
    # =========================================================================
    shoulder_correction = ShoulderCorrectionStep2(
        lambda_acromial=1.0,
        lambda_width=1.0,
        lambda_direction=0.5,
    )

    # =========================================================================
    # 3. Compute losses
    # =========================================================================
    shoulder_losses = shoulder_correction.compute_loss(
        skel_joints=skel_joints,
        addb_joints=addb_joints,
    )

    print(f"Loss acromial:   {shoulder_losses['loss_acromial'].item():.4f}")
    print(f"Loss width:      {shoulder_losses['loss_width'].item():.4f}")
    print(f"Loss direction:  {shoulder_losses['loss_direction'].item():.4f}")
    print(f"Loss total:      {shoulder_losses['loss_total'].item():.4f}")


# =============================================================================
# Full Integration Code Template
# =============================================================================

INTEGRATION_TEMPLATE = """
# =============================================================================
# Integration Template for Your Optimization Pipeline
# =============================================================================

# 1. Import
from shoulder_correction import (
    compute_shoulder_losses,      # For Step 1
    compute_shoulder_losses_extended,  # For Step 2
    ACROMIAL_VERTEX_IDX,
)

# 2. In your optimizer initialization
lambda_acromial = 1.0
lambda_width = 1.0
lambda_direction = 0.5

# 3. In your optimization loop (Step 1 - Virtual Joint)
def compute_total_loss_step1(skel_vertices, skel_joints, addb_joints, ...):
    # Existing losses
    loss_joint = ...
    loss_bone_length = ...
    loss_temporal = ...

    # Shoulder correction losses
    shoulder_losses = compute_shoulder_losses(
        skel_vertices=skel_vertices,
        skel_joints=skel_joints,
        addb_joints=addb_joints,
        lambda_acromial=lambda_acromial,
        lambda_width=lambda_width,
        lambda_direction=lambda_direction,
    )

    # Total loss
    loss_total = (
        loss_joint
        + loss_bone_length
        + loss_temporal
        + shoulder_losses['loss_total']  # Add this
    )

    return loss_total, shoulder_losses

# 4. For Step 2 (Extended Regressor)
# First run: python create_acromial_regressor.py --verify
# Then load SKEL with custom_joint_reg_path argument

def compute_total_loss_step2(skel_joints_26, addb_joints, ...):
    # Note: skel_joints_26 has shape [B, 26, 3]

    shoulder_losses = compute_shoulder_losses_extended(
        skel_joints=skel_joints_26,
        addb_joints=addb_joints,
        lambda_acromial=lambda_acromial,
        lambda_width=lambda_width,
        lambda_direction=lambda_direction,
    )

    loss_total = (
        ... + shoulder_losses['loss_total']
    )

    return loss_total, shoulder_losses
"""


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Shoulder Correction Integration Examples")
    print("=" * 60 + "\n")

    # Run examples
    example_optimization_loop_step1()
    example_optimization_loop_step2()

    print("\n" + "=" * 60)
    print("Integration Template")
    print("=" * 60)
    print(INTEGRATION_TEMPLATE)
