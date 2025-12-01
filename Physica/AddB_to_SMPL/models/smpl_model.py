import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch

SMPL_NUM_BETAS = 10
SMPL_NUM_JOINTS = 24
SMPL_PARENTS = torch.tensor([
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 9, 9, 12, 13, 14, 16, 17,
    18, 19, 20, 21
], dtype=torch.long)


def _to_numpy(array_like) -> np.ndarray:
    if isinstance(array_like, np.ndarray):
        return array_like
    try:
        import chumpy  # type: ignore

        if isinstance(array_like, chumpy.Ch):
            return np.asarray(array_like)
    except ImportError:
        pass
    return np.asarray(array_like)


def _batch_rodrigues(theta: torch.Tensor) -> torch.Tensor:
    batch_size = theta.shape[0]
    device = theta.device
    theta_norm = torch.norm(theta, dim=1, keepdim=True).clamp(min=1e-8)
    theta_normalized = theta / theta_norm

    angle = theta_norm.unsqueeze(-1)
    cos = torch.cos(angle)
    sin = torch.sin(angle)

    rx, ry, rz = theta_normalized[:, 0], theta_normalized[:, 1], theta_normalized[:, 2]
    zeros = torch.zeros_like(rx, device=device)

    K = torch.stack([
        zeros, -rz, ry,
        rz, zeros, -rx,
        -ry, rx, zeros
    ], dim=1).reshape(batch_size, 3, 3)

    eye = torch.eye(3, device=device).unsqueeze(0)
    rot_mat = eye + sin * K + (1 - cos) * torch.matmul(K, K)
    return rot_mat



def _blend_shapes(betas: torch.Tensor, shapedirs: torch.Tensor) -> torch.Tensor:
    return torch.einsum('bl,vcl->bvc', betas, shapedirs)


def _vertices2joints(J_regressor: torch.Tensor, vertices: torch.Tensor) -> torch.Tensor:
    return torch.einsum('iv,bvd->bid', J_regressor, vertices)


def _with_zeros(mat: torch.Tensor) -> torch.Tensor:
    filler = torch.tensor([0, 0, 0, 1], dtype=mat.dtype, device=mat.device)
    filler = filler.view(1, 1, 4).expand(mat.shape[0], -1, -1)
    return torch.cat([mat, filler], dim=1)


def _batch_rigid_transform(rot_mats: torch.Tensor,
                           joints: torch.Tensor,
                           parents: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply batch rigid transformations to joints using forward kinematics.

    This follows the standard SMPL/SMPL-X LBS implementation:
    1. Build local transforms for each joint
    2. Chain them through the kinematic hierarchy
    3. Compute relative transforms for skinning

    For LBS, each vertex v near joint j is transformed as:
        v_posed = R_j @ (v - J_j_rest) + J_j_posed
                = R_j @ v + (J_j_posed - R_j @ J_j_rest)

    So the relative transform has:
        - Rotation: R_j (unchanged from world transform)
        - Translation: J_j_posed - R_j @ J_j_rest
    """
    batch_size = rot_mats.shape[0]
    num_joints = rot_mats.shape[1]
    device = rot_mats.device

    # Save rest pose joints before modification
    joints_rest = joints.clone()  # [B, J, 3]

    joints = joints.unsqueeze(-1)  # [B, J, 3, 1]
    rel_joints = joints.clone()
    rel_joints[:, 1:] = joints[:, 1:] - joints[:, parents[1:]]

    # Build local transform matrices [R | t]
    transforms_mat = torch.cat([rot_mats, rel_joints], dim=-1)  # [B, J, 3, 4]

    # Add [0, 0, 0, 1] row to make 4x4
    padding = torch.zeros(batch_size, num_joints, 1, 4, device=device)
    padding[:, :, :, 3] = 1
    transforms_mat = torch.cat([transforms_mat, padding], dim=2)  # [B, J, 4, 4]

    # Forward kinematics: chain transforms through the kinematic tree
    transforms = [transforms_mat[:, 0]]
    for i in range(1, num_joints):
        parent = parents[i].item()
        transforms.append(torch.matmul(transforms[parent], transforms_mat[:, i]))
    transforms = torch.stack(transforms, dim=1)  # [B, J, 4, 4]

    # Extract posed joint positions from world transforms
    posed_joints = transforms[:, :, :3, 3]  # [B, J, 3]

    # Compute relative transforms for LBS skinning
    # For each joint j: new_translation = J_j_posed - R_j @ J_j_rest
    rot_part = transforms[:, :, :3, :3]  # [B, J, 3, 3]
    rotated_rest_joints = torch.matmul(rot_part, joints_rest.unsqueeze(-1)).squeeze(-1)  # [B, J, 3]
    new_trans = posed_joints - rotated_rest_joints  # [B, J, 3]

    # Build relative transforms: keep rotation, update translation
    rel_transforms = transforms.clone()
    rel_transforms[:, :, :3, 3] = new_trans

    return rel_transforms, posed_joints


def _lbs(betas: torch.Tensor,
         pose: torch.Tensor,
         v_template: torch.Tensor,
         shapedirs: torch.Tensor,
         posedirs: torch.Tensor,
         J_regressor: torch.Tensor,
         parents: torch.Tensor,
         lbs_weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = betas.shape[0]
    num_vertices = v_template.shape[0]
    device = betas.device

    v_shaped = v_template.unsqueeze(0) + _blend_shapes(betas, shapedirs)
    joints = _vertices2joints(J_regressor, v_shaped)

    rot_mats = _batch_rodrigues(pose.view(-1, 3)).view(batch_size, SMPL_NUM_JOINTS, 3, 3)
    ident = torch.eye(3, device=device).unsqueeze(0)
    pose_feature = (rot_mats[:, 1:] - ident).view(batch_size, -1)
    pose_offsets = torch.matmul(pose_feature, posedirs.T)
    pose_offsets = pose_offsets.view(batch_size, num_vertices, 3)

    v_posed = v_shaped + pose_offsets

    transforms_rel, joints_global = _batch_rigid_transform(rot_mats, joints, parents)

    weights = lbs_weights.unsqueeze(0).expand(batch_size, -1, -1)
    transforms_rel = transforms_rel.view(batch_size, SMPL_NUM_JOINTS, 16)
    T = torch.matmul(weights, transforms_rel).view(batch_size, num_vertices, 4, 4)

    v_posed_homo = torch.cat([v_posed, torch.ones(batch_size, num_vertices, 1, device=device)], dim=2)
    vertices = torch.matmul(T, v_posed_homo.unsqueeze(-1))[:, :, :3, 0]

    return vertices, joints_global


# SMPL model paths for different genders
SMPL_MODEL_DIR = '/egr/research-zijunlab/kwonjoon/01_Code/SMPL_python_v.1.1.0/smpl/models'
SMPL_MODEL_FILES = {
    'male': 'basicmodel_m_lbs_10_207_0_v1.1.0.pkl',
    'female': 'basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
    'neutral': 'basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl',
}


@dataclass
class SMPLModel:
    model_path: str = None
    gender: str = 'neutral'
    device: torch.device = torch.device('cpu')

    def __post_init__(self) -> None:
        # If model_path not provided, use gender to select model
        if self.model_path is None:
            if self.gender not in SMPL_MODEL_FILES:
                raise ValueError(f"Invalid gender: {self.gender}. Must be 'male', 'female', or 'neutral'")
            self.model_path = os.path.join(SMPL_MODEL_DIR, SMPL_MODEL_FILES[self.gender])

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SMPL model file not found: {self.model_path}")

        with open(self.model_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')

        self.v_template = torch.tensor(_to_numpy(data['v_template']), dtype=torch.float32, device=self.device)

        shapedirs = _to_numpy(data['shapedirs'])
        if shapedirs.ndim == 2:
            shapedirs = shapedirs.reshape(-1, 3, shapedirs.shape[-1])
        self.shapedirs = torch.tensor(shapedirs[:, :, :SMPL_NUM_BETAS], dtype=torch.float32, device=self.device)

        posedirs = _to_numpy(data['posedirs'])
        self.posedirs = torch.tensor(posedirs.reshape(-1, posedirs.shape[-1]), dtype=torch.float32, device=self.device)

        J_regressor = data['J_regressor']
        if hasattr(J_regressor, 'toarray'):
            J_regressor = J_regressor.toarray()
        self.J_regressor = torch.tensor(_to_numpy(J_regressor), dtype=torch.float32, device=self.device)

        self.parents = SMPL_PARENTS.to(self.device)

        self.lbs_weights = torch.tensor(_to_numpy(data['weights']), dtype=torch.float32, device=self.device)
        self.faces = torch.tensor(_to_numpy(data['f']).astype(np.int32)) if 'f' in data else None

    def forward(self,
                betas: torch.Tensor,
                poses: torch.Tensor,
                trans: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        single = betas.ndim == 1 and poses.ndim == 2 and (trans is None or trans.ndim == 1)

        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
        if poses.ndim == 2:
            poses = poses.unsqueeze(0)
        if trans is not None and trans.ndim == 1:
            trans = trans.unsqueeze(0)

        betas = betas.to(self.device)
        poses = poses.to(self.device)
        if trans is None:
            trans = torch.zeros(betas.shape[0], 3, device=self.device)
        else:
            trans = trans.to(self.device)

        vertices, joints = _lbs(
            betas,
            poses,
            self.v_template,
            self.shapedirs,
            self.posedirs,
            self.J_regressor,
            self.parents,
            self.lbs_weights
        )
        vertices = vertices + trans.unsqueeze(1)
        joints = joints + trans.unsqueeze(1)

        if single:
            vertices = vertices[0]
            joints = joints[0]

        return vertices, joints

    def joints(self,
               betas: torch.Tensor,
               poses: torch.Tensor,
               trans: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward(betas, poses, trans)[1]
