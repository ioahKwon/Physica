#!/usr/bin/env python3
"""
Test each SMPL2AddBiomechanics improvement individually to identify which one causes issues
"""

import os
import sys
import argparse
import torch
import numpy as np

# Import from v2
sys.path.insert(0, os.path.dirname(__file__))
from addbiomechanics_to_smpl_v2 import (
    load_b3d_sequence,
    convert_addb_to_smpl_coords,
    infer_addb_joint_names,
    auto_map_addb_to_smpl,
    center_on_root_numpy,
    center_on_root_tensor,
    SMPLModel,
    SMPL_NUM_JOINTS,
    SMPL_JOINT_NAMES,
    AUTO_JOINT_NAME_MAP
)

# Enhanced bone-segment weights from SMPL2AddBiomechanics
BONE_SEGMENT_WEIGHTS = {
    # Lower limbs (high accuracy for gait analysis)
    'right_ankle': 3.5,
    'left_ankle': 3.5,
    'right_knee': 3.0,
    'left_knee': 3.0,
    'right_hip': 2.5,
    'left_hip': 2.5,
    'right_foot': 2.0,
    'left_foot': 2.0,
    # Pelvis and core
    'pelvis': 2.5,
    'spine1': 2.0,
    'spine2': 2.0,
    'spine3': 2.0,
    # Upper limbs (lower weight)
    'left_shoulder': 1.5,
    'right_shoulder': 1.5,
    'left_elbow': 1.5,
    'right_elbow': 1.5,
    'left_wrist': 1.2,
    'right_wrist': 1.2,
    # Head and neck
    'neck': 1.5,
    'head': 1.5,
    # Hands (lowest weight)
    'left_hand': 1.0,
    'right_hand': 1.0,
}

def estimate_height_from_betas(betas: torch.Tensor, smpl_model: SMPLModel, device) -> torch.Tensor:
    """Estimate height from SMPL betas"""
    zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=device)
    zero_trans = torch.zeros(3, device=device)
    joints = smpl_model.joints(betas.unsqueeze(0), zero_pose, zero_trans)

    head_idx = SMPL_JOINT_NAMES.index('head')
    left_foot_idx = SMPL_JOINT_NAMES.index('left_foot')
    right_foot_idx = SMPL_JOINT_NAMES.index('right_foot')

    head_y = joints[0, head_idx, 1]
    foot_y = (joints[0, left_foot_idx, 1] + joints[0, right_foot_idx, 1]) / 2
    height = head_y - foot_y
    return height


class IndividualImprovementTester:
    def __init__(self, b3d_file, smpl_model, device, target_height=None):
        self.device = device
        self.smpl = smpl_model
        self.target_height = target_height

        # Load AddBiomechanics data
        print("\n[1/3] Loading AddBiomechanics data...")
        target_joints, dt = load_b3d_sequence(b3d_file)
        target_joints = convert_addb_to_smpl_coords(target_joints)

        self.n_frames = target_joints.shape[0]
        print(f"  Frames : {self.n_frames}")
        print(f"  Joints : {target_joints.shape[1]}")
        if target_height is not None:
            print(f"  Height : {target_height:.3f} m")

        # Setup joint mapping
        print("\n[2/3] Resolving joint correspondences...")
        addb_joint_names = infer_addb_joint_names(b3d_file)
        joint_mapping = auto_map_addb_to_smpl(addb_joint_names)

        self.joint_mapping = joint_mapping
        self.root_addb_idx = addb_joint_names.index('ground_pelvis') if 'ground_pelvis' in addb_joint_names else None

        # Convert to tensors
        self.target_joints = torch.tensor(
            target_joints, dtype=torch.float32, device=device
        )

        # Setup indices for v2-style optimization
        self.addb_indices = torch.tensor(list(joint_mapping.keys()), dtype=torch.long, device=device)
        self.smpl_indices = torch.tensor(list(joint_mapping.values()), dtype=torch.long, device=device)

        # v2 hyperparameters
        self.pose_reg_weight = 1e-3
        self.trans_reg_weight = 1e-3
        self.ankle_weight = 3.0
        self.spine_weight = 2.0

        print(f"  Using {len(joint_mapping)} correspondences")

    def compute_mpjpe(self, pred_joints, target_joints):
        """Compute MPJPE using v2's correct method"""
        errors = []
        for pred_frame, target_frame in zip(pred_joints, target_joints):
            pred_centered, target_centered = center_on_root_tensor(
                pred_frame, target_frame, self.root_addb_idx
            )

            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx >= target_centered.shape[0]:
                    continue
                tgt = target_centered[addb_idx]
                if torch.isnan(tgt).any():
                    continue

                diff = pred_centered[smpl_idx] - tgt
                error_m = torch.norm(diff)
                errors.append(error_m.item() * 1000.0)  # Convert to mm

        return np.mean(errors) if errors else float('nan')

    def test_baseline(self):
        """Test baseline v2 method (no improvements)"""
        print("\n" + "="*80)
        print("TEST 1: BASELINE (v2 - no improvements)")
        print("="*80)

        # Stage 1: Shape optimization
        print("\nStage 1: Shape optimization (baseline)")
        betas = torch.zeros(10, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=0.005)

        sample_indices = np.linspace(0, self.n_frames - 1, min(10, self.n_frames), dtype=int)

        for i in range(150):
            optimizer.zero_grad()
            loss = 0.0

            for t in sample_indices:
                target_frame = self.target_joints[t]
                zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                zero_trans = torch.zeros(3, device=self.device)

                joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose, zero_trans)[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 20 == 0 or i == 149:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Stage 2: Pose optimization
        print("\nStage 2: Pose optimization (baseline)")
        betas_fixed = betas.detach()
        poses = torch.zeros(self.n_frames, SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device, requires_grad=True)
        trans = torch.zeros(self.n_frames, 3, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([poses, trans], lr=0.01)

        for i in range(100):
            optimizer.zero_grad()
            loss = 0.0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 10 == 0 or i == 99:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Compute MPJPE
        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Baseline MPJPE: {mpjpe:.2f} mm")
        return mpjpe

    def test_height_constraint(self):
        """Test with only height constraint (Priority 1)"""
        print("\n" + "="*80)
        print("TEST 2: PRIORITY 1 - Height Constraint Only")
        print("="*80)

        if self.target_height is None:
            print("⚠ No target height provided, skipping")
            return None

        # Stage 1: Shape optimization WITH height constraint
        print("\nStage 1: Shape optimization (with height constraint)")
        betas = torch.zeros(10, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=0.005)

        sample_indices = np.linspace(0, self.n_frames - 1, min(10, self.n_frames), dtype=int)
        height_weight = 0.5

        for i in range(150):
            optimizer.zero_grad()
            loss = 0.0

            for t in sample_indices:
                target_frame = self.target_joints[t]
                zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                zero_trans = torch.zeros(3, device=self.device)

                joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose, zero_trans)[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            # Add height constraint
            pred_height = estimate_height_from_betas(betas, self.smpl, self.device)
            height_diff = (pred_height - self.target_height) ** 2
            loss = loss + height_weight * height_diff

            loss.backward()
            optimizer.step()

            if i % 20 == 0 or i == 149:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}, height_pred={pred_height.item():.3f}, height_target={self.target_height:.3f}")

        # Stage 2: Pose optimization (baseline)
        print("\nStage 2: Pose optimization (baseline)")
        betas_fixed = betas.detach()
        poses = torch.zeros(self.n_frames, SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device, requires_grad=True)
        trans = torch.zeros(self.n_frames, 3, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([poses, trans], lr=0.01)

        for i in range(100):
            optimizer.zero_grad()
            loss = 0.0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 10 == 0 or i == 99:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Compute MPJPE
        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Height Constraint MPJPE: {mpjpe:.2f} mm")
        return mpjpe

    def test_bone_weights(self):
        """Test with only enhanced bone-segment weights (Priority 2)"""
        print("\n" + "="*80)
        print("TEST 3: PRIORITY 2 - Enhanced Bone-Segment Weights Only")
        print("="*80)

        # Build weight tensor
        joint_weights = torch.ones(len(SMPL_JOINT_NAMES), dtype=torch.float32, device=self.device)
        for joint_name, weight in BONE_SEGMENT_WEIGHTS.items():
            if joint_name in SMPL_JOINT_NAMES:
                idx = SMPL_JOINT_NAMES.index(joint_name)
                joint_weights[idx] = weight

        print(f"  Joint weights: min={joint_weights.min():.2f}, max={joint_weights.max():.2f}, mean={joint_weights.mean():.2f}")

        # Stage 1: Shape optimization (baseline)
        print("\nStage 1: Shape optimization (baseline)")
        betas = torch.zeros(10, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=0.005)

        sample_indices = np.linspace(0, self.n_frames - 1, min(10, self.n_frames), dtype=int)

        for i in range(150):
            optimizer.zero_grad()
            loss = 0.0

            for t in sample_indices:
                target_frame = self.target_joints[t]
                zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                zero_trans = torch.zeros(3, device=self.device)

                joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose, zero_trans)[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 20 == 0 or i == 149:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Stage 2: Pose optimization WITH enhanced weights
        print("\nStage 2: Pose optimization (with enhanced bone-segment weights)")
        betas_fixed = betas.detach()
        poses = torch.zeros(self.n_frames, SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device, requires_grad=True)
        trans = torch.zeros(self.n_frames, 3, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([poses, trans], lr=0.01)

        for i in range(100):
            optimizer.zero_grad()
            loss = 0.0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    # Apply joint-specific weight
                    weight = joint_weights[smpl_idx]
                    loss = loss + weight * torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 10 == 0 or i == 99:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Compute MPJPE
        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Bone Weights MPJPE: {mpjpe:.2f} mm")
        return mpjpe

    def test_joint_offsets(self):
        """Test with only per-joint learnable offsets (Priority 3)"""
        print("\n" + "="*80)
        print("TEST 4: PRIORITY 3 - Per-Joint Learnable Offsets Only")
        print("="*80)

        # Stage 1: Shape optimization (baseline)
        print("\nStage 1: Shape optimization (baseline)")
        betas = torch.zeros(10, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=0.005)

        sample_indices = np.linspace(0, self.n_frames - 1, min(10, self.n_frames), dtype=int)

        for i in range(150):
            optimizer.zero_grad()
            loss = 0.0

            for t in sample_indices:
                target_frame = self.target_joints[t]
                zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                zero_trans = torch.zeros(3, device=self.device)

                joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose, zero_trans)[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 20 == 0 or i == 149:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Stage 2: Pose optimization (baseline)
        print("\nStage 2: Pose optimization (baseline)")
        betas_fixed = betas.detach()
        poses = torch.zeros(self.n_frames, SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device, requires_grad=True)
        trans = torch.zeros(self.n_frames, 3, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = torch.optim.Adam([poses, trans], lr=0.01)

        for i in range(100):
            optimizer.zero_grad()
            loss = 0.0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_fixed.unsqueeze(0), poses[t], trans[t])[0]
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            if i % 10 == 0 or i == 99:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        # Stage 3: Per-joint offset refinement
        print("\nStage 3: Per-joint offset refinement")
        poses_fixed = poses.detach()
        trans_fixed = trans.detach()
        betas_t = betas_fixed.unsqueeze(0)

        joint_offsets = torch.zeros(SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([joint_offsets], lr=0.001)

        for i in range(50):
            optimizer.zero_grad()
            loss = 0.0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_t[0], poses_fixed[t], trans_fixed[t])
                joints_pred = joints_pred + joint_offsets  # Apply learnable offsets

                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                for addb_idx, smpl_idx in self.joint_mapping.items():
                    if addb_idx >= target_c.shape[0]:
                        continue
                    tgt = target_c[addb_idx]
                    if torch.isnan(tgt).any():
                        continue
                    diff = pred_c[smpl_idx] - tgt
                    loss = loss + torch.sum(diff ** 2)

            loss.backward()
            optimizer.step()

            offset_norm = torch.norm(joint_offsets).item()
            if i % 10 == 0 or i == 49:
                print(f"  Iter {i:3d}: loss={loss.item():.6f}, offset_norm={offset_norm:.4f}")

        # Compute MPJPE
        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas_t[0], poses_fixed[t], trans_fixed[t])
                joints = joints + joint_offsets
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Joint Offsets MPJPE: {mpjpe:.2f} mm")
        return mpjpe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', required=True, help='Path to .b3d file')
    parser.add_argument('--smpl_model', required=True, help='Path to SMPL model')
    parser.add_argument('--device', default='cuda', help='Device to use')
    args = parser.parse_args()

    device = torch.device(args.device)

    print("="*80)
    print("Testing Individual SMPL2AddBiomechanics Improvements")
    print("="*80)
    print(f"B3D file: {args.b3d}")
    print(f"Device  : {args.device}")
    print("="*80)

    # Load SMPL model
    print("\n[3/3] Initializing SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)

    # Get target height from b3d
    import nimblephysics as nimble
    target_height = None
    try:
        subj = nimble.biomechanics.SubjectOnDisk(args.b3d)
        target_height = float(subj.getHeightM())
    except:
        pass

    # Run tests
    tester = IndividualImprovementTester(args.b3d, smpl, device, target_height)

    results = {}
    results['baseline'] = tester.test_baseline()
    results['height_constraint'] = tester.test_height_constraint()
    results['bone_weights'] = tester.test_bone_weights()
    results['joint_offsets'] = tester.test_joint_offsets()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline (v2):                {results['baseline']:.2f} mm")

    if results['height_constraint'] is not None:
        diff = results['height_constraint'] - results['baseline']
        sign = "+" if diff > 0 else ""
        print(f"+ Height Constraint:          {results['height_constraint']:.2f} mm ({sign}{diff:.2f} mm)")

    diff = results['bone_weights'] - results['baseline']
    sign = "+" if diff > 0 else ""
    print(f"+ Enhanced Bone Weights:      {results['bone_weights']:.2f} mm ({sign}{diff:.2f} mm)")

    diff = results['joint_offsets'] - results['baseline']
    sign = "+" if diff > 0 else ""
    print(f"+ Per-Joint Offsets:          {results['joint_offsets']:.2f} mm ({sign}{diff:.2f} mm)")
    print("="*80)


if __name__ == '__main__':
    main()
