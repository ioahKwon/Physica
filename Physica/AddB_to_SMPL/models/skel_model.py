"""
SKELModelWrapper: SKEL 모델을 SMPLModel과 동일한 인터페이스로 래핑

SKEL (Skin to Skeleton) 모델 사양:
- Pose: 46 DOF (Euler angles, radians)
- Shape (betas): 10 (SMPL과 동일)
- Vertices: 6,890 (skin mesh)
- Joints: 24

SKEL Joint Names (24개):
    0: pelvis, 1: femur_r, 2: tibia_r, 3: talus_r, 4: calcn_r, 5: toes_r,
    6: femur_l, 7: tibia_l, 8: talus_l, 9: calcn_l, 10: toes_l,
    11: lumbar_body, 12: thorax, 13: head,
    14: scapula_r, 15: humerus_r, 16: ulna_r, 17: radius_r, 18: hand_r,
    19: scapula_l, 20: humerus_l, 21: ulna_l, 22: radius_l, 23: hand_l

SKEL Pose Parameters (46개):
    0-2: pelvis_tilt, pelvis_list, pelvis_rotation
    3-5: hip_flexion_r, hip_adduction_r, hip_rotation_r
    6: knee_angle_r
    7-9: ankle_angle_r, subtalar_angle_r, mtp_angle_r
    10-12: hip_flexion_l, hip_adduction_l, hip_rotation_l
    13: knee_angle_l
    14-16: ankle_angle_l, subtalar_angle_l, mtp_angle_l
    17-19: lumbar_bending, lumbar_extension, lumbar_twist
    20-22: thorax_bending, thorax_extension, thorax_twist
    23-25: head_bending, head_extension, head_twist
    26-28: scapula_abduction_r, scapula_elevation_r, scapula_upward_rot_r
    29-31: shoulder_r_x, shoulder_r_y, shoulder_r_z
    32-33: elbow_flexion_r, pro_sup_r
    34-35: wrist_flexion_r, wrist_deviation_r
    36-38: scapula_abduction_l, scapula_elevation_l, scapula_upward_rot_l
    39-41: shoulder_l_x, shoulder_l_y, shoulder_l_z
    42-43: elbow_flexion_l, pro_sup_l
    44-45: wrist_flexion_l, wrist_deviation_l

Usage:
    skel = SKELModelWrapper(model_path='/path/to/skel_models_v1.1', gender='male')
    vertices, joints = skel.forward(betas, poses, trans)

    # betas: [10] or [B, 10]
    # poses: [46] or [B, 46] or [T, 46] or [B, T, 46]
    # trans: [3] or [B, 3] or [T, 3] or [B, T, 3]
    #
    # Returns:
    #   vertices: [6890, 3] or [B*T, 6890, 3]
    #   joints: [24, 3] or [B*T, 24, 3]

Reference:
    - SKEL Paper: "From Skin to Skeleton: Towards Biomechanically Accurate 3D Digital Humans" (SIGGRAPH Asia 2023)
    - GitHub: https://github.com/MarilynKeller/SKEL
"""

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import torch
import numpy as np

# SKEL 모델 import
from skel.skel_model import SKEL


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SKEL_NUM_BETAS = 10
SKEL_NUM_POSE_DOF = 46
SKEL_NUM_JOINTS = 24
SKEL_NUM_JOINTS_WITH_ACROMIAL = 26  # 24 + acromial_r, acromial_l
SKEL_NUM_VERTICES = 6890

# Acromial vertex indices for Joint Regressor extension
# These vertices are on the outer shoulder surface, near humerus joint
ACROMIAL_R_VERTICES = [4816, 4819, 4873, 4917, 4916, 4912, 4888, 5009, 4259, 4817]
ACROMIAL_L_VERTICES = [1340, 1341, 1399, 1444, 1445, 1441, 770, 1438, 1415, 1343]

# SKEL Joint Names (for reference)
SKEL_JOINT_NAMES = [
    'pelvis',       # 0
    'femur_r',      # 1  (right hip)
    'tibia_r',      # 2  (right knee)
    'talus_r',      # 3  (right ankle)
    'calcn_r',      # 4  (right heel/calcaneus)
    'toes_r',       # 5  (right toes)
    'femur_l',      # 6  (left hip)
    'tibia_l',      # 7  (left knee)
    'talus_l',      # 8  (left ankle)
    'calcn_l',      # 9  (left heel/calcaneus)
    'toes_l',       # 10 (left toes)
    'lumbar_body',  # 11 (lumbar spine)
    'thorax',       # 12 (thoracic spine)
    'head',         # 13
    'scapula_r',    # 14 (right shoulder blade)
    'humerus_r',    # 15 (right upper arm / shoulder)
    'ulna_r',       # 16 (right elbow)
    'radius_r',     # 17 (right forearm)
    'hand_r',       # 18 (right wrist/hand)
    'scapula_l',    # 19 (left shoulder blade)
    'humerus_l',    # 20 (left upper arm / shoulder)
    'ulna_l',       # 21 (left elbow)
    'radius_l',     # 22 (left forearm)
    'hand_l',       # 23 (left wrist/hand)
]

# Extended joint names with acromial (for use with add_acromial_joints=True)
SKEL_JOINT_NAMES_WITH_ACROMIAL = SKEL_JOINT_NAMES + [
    'acromial_r',   # 24 (right acromial - shoulder tip)
    'acromial_l',   # 25 (left acromial - shoulder tip)
]

# SKEL Pose Parameter Names (46개)
SKEL_POSE_PARAM_NAMES = [
    'pelvis_tilt',           # 0
    'pelvis_list',           # 1
    'pelvis_rotation',       # 2
    'hip_flexion_r',         # 3
    'hip_adduction_r',       # 4
    'hip_rotation_r',        # 5
    'knee_angle_r',          # 6
    'ankle_angle_r',         # 7
    'subtalar_angle_r',      # 8
    'mtp_angle_r',           # 9
    'hip_flexion_l',         # 10
    'hip_adduction_l',       # 11
    'hip_rotation_l',        # 12
    'knee_angle_l',          # 13
    'ankle_angle_l',         # 14
    'subtalar_angle_l',      # 15
    'mtp_angle_l',           # 16
    'lumbar_bending',        # 17
    'lumbar_extension',      # 18
    'lumbar_twist',          # 19
    'thorax_bending',        # 20
    'thorax_extension',      # 21
    'thorax_twist',          # 22
    'head_bending',          # 23
    'head_extension',        # 24
    'head_twist',            # 25
    'scapula_abduction_r',   # 26
    'scapula_elevation_r',   # 27
    'scapula_upward_rot_r',  # 28
    'shoulder_r_x',          # 29
    'shoulder_r_y',          # 30
    'shoulder_r_z',          # 31
    'elbow_flexion_r',       # 32
    'pro_sup_r',             # 33
    'wrist_flexion_r',       # 34
    'wrist_deviation_r',     # 35
    'scapula_abduction_l',   # 36
    'scapula_elevation_l',   # 37
    'scapula_upward_rot_l',  # 38
    'shoulder_l_x',          # 39
    'shoulder_l_y',          # 40
    'shoulder_l_z',          # 41
    'elbow_flexion_l',       # 42
    'pro_sup_l',             # 43
    'wrist_flexion_l',       # 44
    'wrist_deviation_l',     # 45
]


# ---------------------------------------------------------------------------
# AddBiomechanics ↔ SKEL Joint Mapping
# ---------------------------------------------------------------------------

# AddBiomechanics joint name → SKEL joint name 매핑
# NOTE: AddB와 SKEL은 동일한 좌표계 사용
# AddB _r → SKEL _r, AddB _l → SKEL _l (직접 매핑)
AUTO_JOINT_NAME_MAP_SKEL: Dict[str, str] = {
    # Pelvis
    'ground_pelvis': 'pelvis',
    'pelvis': 'pelvis',
    'root': 'pelvis',

    # AddB _r → SKEL _r (직접 매핑)
    'hip_r': 'femur_r',
    'hip_right': 'femur_r',
    'walker_knee_r': 'tibia_r',
    'knee_r': 'tibia_r',
    'ankle_r': 'talus_r',
    'ankle_right': 'talus_r',
    'subtalar_r': 'calcn_r',
    'mtp_r': 'toes_r',
    'toe_r': 'toes_r',
    'right_toe': 'toes_r',

    # AddB _l → SKEL _l (직접 매핑)
    'hip_l': 'femur_l',
    'hip_left': 'femur_l',
    'walker_knee_l': 'tibia_l',
    'knee_l': 'tibia_l',
    'ankle_l': 'talus_l',
    'ankle_left': 'talus_l',
    'subtalar_l': 'calcn_l',
    'mtp_l': 'toes_l',
    'toe_l': 'toes_l',
    'left_toe': 'toes_l',

    # Spine
    'back': 'lumbar_body',
    'lumbar': 'lumbar_body',
    'spine': 'lumbar_body',
    'torso': 'thorax',
    'thorax': 'thorax',

    # Head/Neck
    'neck': 'head',               # SKEL doesn't have separate neck, use head
    'cervical': 'head',
    'head': 'head',
    'skull': 'head',

    # AddB _r arm → SKEL _r arm (직접 매핑)
    # acromial (어깨 끝) → scapula_r for wider shoulders (best result was with scapula mapping)
    'acromial_r': 'scapula_r',
    'shoulder_r': 'scapula_r',
    'elbow_r': 'ulna_r',
    'radioulnar_r': 'radius_r',
    'wrist_r': 'hand_r',
    'radius_hand_r': 'hand_r',
    'hand_r': 'hand_r',

    # AddB _l arm → SKEL _l arm (직접 매핑)
    'acromial_l': 'scapula_l',
    'shoulder_l': 'scapula_l',
    'elbow_l': 'ulna_l',
    'radioulnar_l': 'radius_l',
    'wrist_l': 'hand_l',
    'radius_hand_l': 'hand_l',
    'hand_l': 'hand_l',
}


def get_skel_joint_index(joint_name: str, use_acromial_joints: bool = False) -> int:
    """
    SKEL joint name → index 변환

    Args:
        joint_name: Joint name to look up
        use_acromial_joints: If True, use 26-joint list (includes acromial_r, acromial_l)

    Returns:
        Joint index, or -1 if not found
    """
    joint_list = SKEL_JOINT_NAMES_WITH_ACROMIAL if use_acromial_joints else SKEL_JOINT_NAMES
    try:
        return joint_list.index(joint_name)
    except ValueError:
        return -1


def build_addb_to_skel_mapping(
    addb_joint_names: List[str],
    use_acromial_joints: bool = False
) -> Tuple[List[int], List[int]]:
    """
    AddBiomechanics joint names를 SKEL joint indices로 매핑

    Args:
        addb_joint_names: AddBiomechanics joint names 리스트
        use_acromial_joints: If True, use 26-joint list (maps acromial to virtual joints 24, 25)

    Returns:
        addb_indices: 매핑된 AddB joint indices
        skel_indices: 대응하는 SKEL joint indices
    """
    addb_indices = []
    skel_indices = []

    for i, name in enumerate(addb_joint_names):
        name_lower = name.lower()
        if name_lower in AUTO_JOINT_NAME_MAP_SKEL:
            skel_name = AUTO_JOINT_NAME_MAP_SKEL[name_lower]
            skel_idx = get_skel_joint_index(skel_name, use_acromial_joints)
            if skel_idx >= 0:
                addb_indices.append(i)
                skel_indices.append(skel_idx)

    return addb_indices, skel_indices


# ---------------------------------------------------------------------------
# SKELModelWrapper Class
# ---------------------------------------------------------------------------

@dataclass
class SKELModelWrapper:
    """
    SKEL 모델을 SMPLModel과 동일한 인터페이스로 래핑하는 클래스

    기존 SMPLModel 코드에서 최소한의 수정으로 SKEL을 사용할 수 있도록 함.

    주요 차이점:
        - SMPL: poses shape = [B, 24, 3] (axis-angle, 72 params)
        - SKEL: poses shape = [B, 46] (Euler angles, 46 params)

    Args:
        model_path: SKEL 모델 파일이 있는 디렉토리 (skel_male.pkl, skel_female.pkl 포함)
        gender: 'male' 또는 'female'
        device: torch device
        add_acromial_joints: If True, add acromial joints (24→26 joints) via vertex averaging
    """
    model_path: str
    gender: str = 'male'
    device: torch.device = torch.device('cpu')
    add_acromial_joints: bool = False

    def __post_init__(self) -> None:
        """모델 초기화"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"SKEL model directory not found: {self.model_path}")

        # SKEL 모델 로드
        self.model = SKEL(
            gender=self.gender,
            model_path=self.model_path
        ).to(self.device)

        # 모델을 evaluation 모드로 설정
        self.model.eval()

        # Faces 저장 (mesh rendering용)
        self.faces = self.model.skin_f.cpu().numpy() if hasattr(self.model, 'skin_f') else None

        # Skeleton faces (optional)
        self.skel_faces = self.model.skel_f.cpu().numpy() if hasattr(self.model, 'skel_f') else None

        # 상수 저장
        self.num_betas = SKEL_NUM_BETAS
        self.num_pose_dof = SKEL_NUM_POSE_DOF

        # Joint count depends on whether acromial joints are added
        if self.add_acromial_joints:
            self.num_joints = SKEL_NUM_JOINTS_WITH_ACROMIAL
            self.joint_names = SKEL_JOINT_NAMES_WITH_ACROMIAL
            # Extend J_regressor_osim with acromial joint rows (24 → 26)
            # This modifies the regressor matrix directly instead of runtime computation
            self._extend_joint_regressor_with_acromial()
        else:
            self.num_joints = SKEL_NUM_JOINTS
            self.joint_names = SKEL_JOINT_NAMES

    def _extend_joint_regressor_with_acromial(self) -> None:
        """
        Create acromial joint regressor rows following the professor's recommended approach.

        Instead of modifying SKEL's internal J_regressor_osim (which would break kinematic chain),
        we create a separate acromial regressor [2, 6890] that uses the same einsum operation
        as the original joint regressor.

        This provides the same benefits as extending the regressor matrix:
        - Same einsum computation as original joints
        - Weights can be adjusted (uniform, distance-based, etc.)
        - Consistent with SKEL's joint computation approach
        """
        # Get dtype from SKEL model's regressor for consistency
        J_reg = self.model.J_regressor_osim  # [24, 6890]

        # Create acromial regressor [2, 6890] - same format as J_regressor_osim rows
        self.acromial_regressor = torch.zeros(2, SKEL_NUM_VERTICES, device=self.device, dtype=J_reg.dtype)

        # Right acromial (row 0): weighted average of shoulder surface vertices
        k_r = len(ACROMIAL_R_VERTICES)
        self.acromial_regressor[0, ACROMIAL_R_VERTICES] = 1.0 / k_r

        # Left acromial (row 1): weighted average of shoulder surface vertices
        k_l = len(ACROMIAL_L_VERTICES)
        self.acromial_regressor[1, ACROMIAL_L_VERTICES] = 1.0 / k_l

        print(f"[SKELModelWrapper] Created acromial_regressor: {self.acromial_regressor.shape}")

    def _compute_acromial_joints_via_regressor(self, vertices: torch.Tensor) -> torch.Tensor:
        """
        Compute acromial joint positions using the regressor matrix.

        Uses the same einsum operation as SKEL's internal joint computation:
            J = einsum('bik,ji->bjk', vertices, regressor)

        Args:
            vertices: [B, 6890, 3] mesh vertices

        Returns:
            acromial_joints: [B, 2, 3] acromial joint positions (right, left)
        """
        # Same einsum as SKEL.forward() line 324
        # acromial_joints[b, j, k] = sum_i(vertices[b, i, k] * regressor[j, i])
        acromial_joints = torch.einsum('bik,ji->bjk', vertices, self.acromial_regressor)
        return acromial_joints

    def forward(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: Optional[torch.Tensor] = None,
        return_skeleton: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        SKEL forward pass - SMPLModel과 동일한 인터페이스

        Args:
            betas: Shape parameters
                - [10] → single shape
                - [B, 10] → batched shapes
            poses: Pose parameters (46 DOF Euler angles)
                - [46] → single pose
                - [B, 46] → batched poses (batch)
                - [T, 46] → sequence (frames)
                - [B, T, 46] → batched sequences
            trans: Translation
                - [3] → single translation
                - [B, 3] or [T, 3] → batched/sequenced translations
                - [B, T, 3] → batched sequences
            return_skeleton: If True, also return skeleton vertices (default: False)

        Returns:
            vertices: Skin mesh vertices
                - [6890, 3] if single input
                - [B*T, 6890, 3] if batched/sequenced
            joints: Joint positions
                - [24, 3] if single input
                - [B*T, 24, 3] if batched/sequenced

        Note:
            SKEL은 내부적으로 poses_type='skel'을 사용하며,
            poses는 46개의 Euler angle 파라미터입니다.
        """
        # 입력 텐서를 device로 이동
        betas = betas.to(self.device)
        poses = poses.to(self.device)

        # 입력 차원 분석 및 정규화
        single_input = False
        original_betas_shape = betas.shape
        original_poses_shape = poses.shape

        # Betas 정규화: [10] → [1, 10]
        if betas.ndim == 1:
            betas = betas.unsqueeze(0)
            single_input = True

        # Poses 정규화
        if poses.ndim == 1:
            # [46] → [1, 46]
            poses = poses.unsqueeze(0)
            single_input = True
        elif poses.ndim == 2:
            # [B, 46] or [T, 46] → 그대로 사용
            pass
        elif poses.ndim == 3:
            # [B, T, 46] → [B*T, 46]
            B, T, D = poses.shape
            poses = poses.view(B * T, D)
            # betas를 T번 반복
            betas = betas.unsqueeze(1).expand(-1, T, -1).reshape(B * T, -1)
            single_input = False

        # Batch size 결정
        batch_size = poses.shape[0]

        # Betas가 poses보다 작은 경우 확장
        if betas.shape[0] == 1 and batch_size > 1:
            betas = betas.expand(batch_size, -1)
        elif betas.shape[0] != batch_size:
            # Betas를 batch_size에 맞게 조정
            betas = betas[:1].expand(batch_size, -1)

        # Translation 처리
        if trans is None:
            trans = torch.zeros(batch_size, 3, device=self.device)
        else:
            trans = trans.to(self.device)

            if trans.ndim == 1:
                # [3] → [batch_size, 3]
                trans = trans.unsqueeze(0).expand(batch_size, -1)
            elif trans.ndim == 2:
                # [B, 3] or [T, 3]
                if trans.shape[0] != batch_size:
                    trans = trans[:1].expand(batch_size, -1)
            elif trans.ndim == 3:
                # [B, T, 3] → [B*T, 3]
                B, T, _ = trans.shape
                trans = trans.view(B * T, 3)

        # SKEL forward 호출
        # Note: gradient 계산을 위해 torch.no_grad() 사용하지 않음
        output = self.model(
            poses=poses,
            betas=betas,
            trans=trans,
            poses_type='skel',
            skelmesh=return_skeleton,
            pose_dep_bs=True
        )

        # 출력 추출
        vertices = output.skin_verts  # [B, 6890, 3]
        joints = output.joints        # [B, 24, 3]

        # Add acromial joints if enabled (24 → 26 joints)
        # Acromial positions are computed via joint regressor (weighted vertex average)
        if self.add_acromial_joints:
            acromial_joints = self._compute_acromial_joints_via_regressor(vertices)  # [B, 2, 3]
            joints = torch.cat([joints, acromial_joints], dim=1)  # [B, 26, 3]

        # Single input이면 batch dimension 제거
        if single_input and vertices.shape[0] == 1:
            vertices = vertices[0]
            joints = joints[0]

        if return_skeleton:
            skel_verts = output.skel_verts
            if single_input and skel_verts.shape[0] == 1:
                skel_verts = skel_verts[0]
            return vertices, joints, skel_verts

        return vertices, joints

    def joints(
        self,
        betas: torch.Tensor,
        poses: torch.Tensor,
        trans: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Joint positions만 반환하는 convenience 메서드

        Args:
            betas: Shape parameters [10] or [B, 10]
            poses: Pose parameters [46] or [B, 46]
            trans: Translation [3] or [B, 3]

        Returns:
            joints: [24, 3] or [B, 24, 3]
        """
        return self.forward(betas, poses, trans)[1]

    def get_joint_names(self) -> List[str]:
        """SKEL joint names 반환 (acromial joints 포함 여부에 따라 24 or 26개)"""
        return self.joint_names.copy()

    def get_pose_param_names(self) -> List[str]:
        """SKEL pose parameter names 반환"""
        return SKEL_POSE_PARAM_NAMES.copy()

    @property
    def parents(self) -> torch.Tensor:
        """
        SKEL 모델의 parent joint indices 반환

        SKEL model.parent는 23개 (joint 1부터 시작, joint 0은 root)
        앞에 -1을 추가해서 24개로 만들어 반환

        Returns:
            parents: [24] tensor, parents[i] = parent joint index of joint i
                     parents[0] = -1 (root has no parent)
        """
        # SKEL 모델에서 parent buffer 가져오기
        # model.parent는 23개 element (joint 1~23의 parent)
        parent_buffer = self.model.parent  # [23]

        # 앞에 -1 추가 (root joint의 parent)
        root_parent = torch.tensor([-1], dtype=parent_buffer.dtype, device=parent_buffer.device)
        parents = torch.cat([root_parent, parent_buffer])  # [24]

        return parents


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def create_zero_pose(device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Zero pose (T-pose) 생성"""
    return torch.zeros(SKEL_NUM_POSE_DOF, device=device)


def create_zero_betas(device: torch.device = torch.device('cpu')) -> torch.Tensor:
    """Zero shape parameters 생성"""
    return torch.zeros(SKEL_NUM_BETAS, device=device)


# ---------------------------------------------------------------------------
# Test / Example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # 테스트 코드
    import sys

    model_path = '/egr/research-zijunlab/kwonjoon/02_Dataset/skel_models_v1.1'

    if not os.path.exists(model_path):
        print(f"Model path not found: {model_path}")
        sys.exit(1)

    print("Loading SKEL model...")
    skel = SKELModelWrapper(model_path=model_path, gender='male', device=torch.device('cpu'))

    print(f"Model loaded successfully!")
    print(f"  - num_joints: {skel.num_joints}")
    print(f"  - num_pose_dof: {skel.num_pose_dof}")
    print(f"  - num_betas: {skel.num_betas}")

    # 테스트: single input
    print("\n--- Test: Single Input ---")
    betas = torch.zeros(10)
    poses = torch.zeros(46)
    trans = torch.zeros(3)

    vertices, joints = skel.forward(betas, poses, trans)
    print(f"  vertices shape: {vertices.shape}")  # Expected: [6890, 3]
    print(f"  joints shape: {joints.shape}")      # Expected: [24, 3]

    # 테스트: batched input
    print("\n--- Test: Batched Input ---")
    B = 4
    betas = torch.zeros(B, 10)
    poses = torch.zeros(B, 46)
    trans = torch.zeros(B, 3)

    vertices, joints = skel.forward(betas, poses, trans)
    print(f"  vertices shape: {vertices.shape}")  # Expected: [4, 6890, 3]
    print(f"  joints shape: {joints.shape}")      # Expected: [4, 24, 3]

    # 테스트: sequence input [B, T, 46]
    print("\n--- Test: Sequence Input ---")
    B, T = 2, 10
    betas = torch.zeros(B, 10)
    poses = torch.zeros(B, T, 46)
    trans = torch.zeros(B, T, 3)

    vertices, joints = skel.forward(betas, poses, trans)
    print(f"  vertices shape: {vertices.shape}")  # Expected: [20, 6890, 3]
    print(f"  joints shape: {joints.shape}")      # Expected: [20, 24, 3]

    print("\n--- All tests passed! ---")
