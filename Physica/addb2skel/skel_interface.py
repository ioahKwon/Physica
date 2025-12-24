"""
SKEL Model Interface Wrapper.

Provides a clean interface to the SKEL body model with:
- Forward kinematics
- Virtual acromial computation
- Skeleton mesh access
"""

import os
import sys
from typing import Optional, Tuple, Dict, List
import numpy as np
import torch
import torch.nn as nn

# Add SKEL to path
SKEL_REPO_PATH = '/egr/research-zijunlab/kwonjoon/01_Code/SKEL'
if SKEL_REPO_PATH not in sys.path:
    sys.path.insert(0, SKEL_REPO_PATH)

from skel.skel_model import SKEL

from .config import SKEL_MODEL_PATH, SKEL_NUM_JOINTS, SKEL_NUM_BETAS, SKEL_NUM_POSE_DOF
from .joint_definitions import SKEL_JOINTS, SKEL_JOINT_TO_IDX


class SKELInterface:
    """
    Clean interface to the SKEL body model.

    Provides:
    - forward(): Compute vertices and joints from pose/shape parameters
    - forward_kinematics(): Compute joints only (faster)
    - get_virtual_acromial(): Compute virtual acromial points from mesh
    - get_skeleton_vertices(): Get skeleton mesh vertices
    """

    def __init__(
        self,
        model_path: str = SKEL_MODEL_PATH,
        gender: str = 'male',
        device: Optional[torch.device] = None,
    ):
        """
        Initialize SKEL model.

        Args:
            model_path: Path to SKEL model files.
            gender: 'male' or 'female'.
            device: Torch device.
        """
        self.model_path = model_path
        self.gender = gender
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load SKEL model
        self._load_model()

        # Cache for virtual acromial vertex indices
        self._acromial_vertex_indices: Optional[Dict[str, List[int]]] = None

    def _load_model(self):
        """Load the SKEL model."""
        self.model = SKEL(
            model_path=self.model_path,
            gender=self.gender,
        ).to(self.device)

        # Get faces for mesh export
        self.faces = self.model.faces.astype(np.int32)

        # Get skeleton faces if available
        if hasattr(self.model, 'skel_f'):
            self.skel_faces = self.model.skel_f.astype(np.int32)
        else:
            self.skel_faces = None

        # Get parent indices for skeleton
        if hasattr(self.model, 'parents'):
            self.parents = self.model.parents.cpu().numpy().tolist()
        else:
            # Default SKEL kinematic tree
            self.parents = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 12, 14, 15, 16, 17, 12, 19, 20, 21, 22]

        # Number of vertices
        self.num_vertices = self.model.v_template.shape[0]

    def forward(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: torch.Tensor,
        return_skeleton: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through SKEL model.

        Args:
            betas: Shape parameters [B, 10] or [10].
            poses: Pose parameters [B, 46] or [46].
            trans: Translation [B, 3] or [3].
            return_skeleton: Whether to return skeleton mesh vertices.

        Returns:
            vertices: Skin mesh vertices [B, V, 3].
            joints: Joint positions [B, 24, 3].
            skel_vertices: Skeleton mesh vertices [B, V_skel, 3] if return_skeleton.
        """
        # Ensure batch dimension
        if betas.dim() == 1:
            betas = betas.unsqueeze(0)
        if poses.dim() == 1:
            poses = poses.unsqueeze(0)
        if trans.dim() == 1:
            trans = trans.unsqueeze(0)

        B = poses.shape[0]

        # Expand betas if needed
        if betas.shape[0] == 1 and B > 1:
            betas = betas.expand(B, -1)

        # Forward through SKEL
        output = self.model(
            betas=betas,
            poses=poses,
            trans=trans,
            return_skel=return_skeleton,
        )

        vertices = output.skin_verts
        joints = output.joints

        if return_skeleton and hasattr(output, 'skel_verts'):
            skel_vertices = output.skel_verts
            return vertices, joints, skel_vertices
        else:
            return vertices, joints, None

    def forward_kinematics(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute joint positions only (faster than full forward).

        Args:
            betas: Shape parameters [B, 10] or [10].
            poses: Pose parameters [B, 46] or [46].
            trans: Translation [B, 3] or [3].

        Returns:
            joints: Joint positions [B, 24, 3].
        """
        _, joints, _ = self.forward(betas, poses, trans, return_skeleton=False)
        return joints

    def get_virtual_acromial(
        self,
        vertices: torch.Tensor,
        side: str = 'both',
    ) -> Dict[str, torch.Tensor]:
        """
        Compute virtual acromial points from mesh vertices.

        The acromial is a bony landmark on the scapula that can be approximated
        from the mesh surface. We use a weighted average of shoulder region vertices.

        Args:
            vertices: Mesh vertices [B, V, 3].
            side: 'right', 'left', or 'both'.

        Returns:
            Dictionary with 'right' and/or 'left' acromial positions [B, 3].
        """
        if self._acromial_vertex_indices is None:
            self._find_acromial_vertices()

        result = {}

        if side in ['right', 'both']:
            indices = self._acromial_vertex_indices['right']
            result['right'] = vertices[:, indices, :].mean(dim=1)

        if side in ['left', 'both']:
            indices = self._acromial_vertex_indices['left']
            result['left'] = vertices[:, indices, :].mean(dim=1)

        return result

    def _find_acromial_vertices(self):
        """
        Find vertex indices for virtual acromial computation.

        These are vertices on the shoulder region of the mesh that best
        approximate the acromial landmark position.
        """
        # Get T-pose vertices
        betas = torch.zeros(1, SKEL_NUM_BETAS, device=self.device)
        poses = torch.zeros(1, SKEL_NUM_POSE_DOF, device=self.device)
        trans = torch.zeros(1, 3, device=self.device)

        with torch.no_grad():
            vertices, joints, _ = self.forward(betas, poses, trans)
            vertices = vertices[0].cpu().numpy()
            joints = joints[0].cpu().numpy()

        # Get shoulder joint positions
        humerus_r = joints[SKEL_JOINT_TO_IDX['humerus_r']]
        humerus_l = joints[SKEL_JOINT_TO_IDX['humerus_l']]
        scapula_r = joints[SKEL_JOINT_TO_IDX['scapula_r']]
        scapula_l = joints[SKEL_JOINT_TO_IDX['scapula_l']]

        # Acromial is slightly lateral and superior to glenohumeral center
        # Find vertices near the expected acromial position
        acromial_r_approx = humerus_r + np.array([0.03, 0.02, 0])  # Lateral + up
        acromial_l_approx = humerus_l + np.array([-0.03, 0.02, 0])

        # Find nearest vertices
        def find_nearest_vertices(target, vertices, k=20):
            distances = np.linalg.norm(vertices - target, axis=1)
            return np.argsort(distances)[:k].tolist()

        self._acromial_vertex_indices = {
            'right': find_nearest_vertices(acromial_r_approx, vertices),
            'left': find_nearest_vertices(acromial_l_approx, vertices),
        }

    def get_shoulder_width(
        self,
        joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute shoulder width from humerus joints.

        Args:
            joints: Joint positions [B, 24, 3].

        Returns:
            Shoulder width [B] in meters.
        """
        humerus_r = joints[:, SKEL_JOINT_TO_IDX['humerus_r'], :]
        humerus_l = joints[:, SKEL_JOINT_TO_IDX['humerus_l'], :]
        return torch.norm(humerus_r - humerus_l, dim=-1)

    def get_height(
        self,
        joints: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate body height from joints.

        Args:
            joints: Joint positions [B, 24, 3].

        Returns:
            Height estimate [B] in meters.
        """
        # Use head to feet distance
        head = joints[:, SKEL_JOINT_TO_IDX['head'], :]
        # Average of toes
        toes_r = joints[:, SKEL_JOINT_TO_IDX['toes_r'], :]
        toes_l = joints[:, SKEL_JOINT_TO_IDX['toes_l'], :]
        feet = (toes_r + toes_l) / 2

        # Vertical distance (Y axis typically up)
        return (head[:, 1] - feet[:, 1]).abs()

    def to(self, device: torch.device) -> 'SKELInterface':
        """Move model to device."""
        self.device = device
        self.model = self.model.to(device)
        return self

    @property
    def joint_names(self) -> List[str]:
        """Get joint names."""
        return SKEL_JOINTS

    def __repr__(self) -> str:
        return f"SKELInterface(gender={self.gender}, device={self.device})"


def create_skel_interface(
    gender: str = 'male',
    device: Optional[str] = None,
) -> SKELInterface:
    """
    Factory function to create SKEL interface.

    Args:
        gender: 'male' or 'female'.
        device: 'cuda' or 'cpu'. Auto-detect if None.

    Returns:
        SKELInterface instance.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    return SKELInterface(
        model_path=SKEL_MODEL_PATH,
        gender=gender,
        device=torch.device(device),
    )
