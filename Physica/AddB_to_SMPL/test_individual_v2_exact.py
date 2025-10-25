#!/usr/bin/env python3
"""
Test individual improvements with EXACT v2 baseline
Key difference from previous test: Frame-by-frame optimization + regularization
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from addbiomechanics_to_smpl_v2 import (
    load_b3d_sequence,
    convert_addb_to_smpl_coords,
    infer_addb_joint_names,
    auto_map_addb_to_smpl,
    center_on_root_tensor,
    SMPLModel,
    SMPL_NUM_JOINTS,
    SMPL_JOINT_NAMES,
)

# Enhanced bone-segment weights from SMPL2AddBiomechanics
BONE_SEGMENT_WEIGHTS = {
    'right_ankle': 3.5, 'left_ankle': 3.5,
    'right_knee': 3.0, 'left_knee': 3.0,
    'right_hip': 2.5, 'left_hip': 2.5,
    'right_foot': 2.0, 'left_foot': 2.0,
    'pelvis': 3.0, 'spine1': 2.5, 'spine2': 2.0, 'spine3': 1.8,
    'right_shoulder': 1.5, 'left_shoulder': 1.5,
    'right_elbow': 1.2, 'left_elbow': 1.2,
    'right_wrist': 1.0, 'left_wrist': 1.0,
    'neck': 1.5, 'head': 1.2,
}

def estimate_height_from_betas(betas, smpl_model, device):
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


class V2ExactTester:
    def __init__(self, b3d_file, smpl_model, device, target_height=None):
        self.device = device
        self.smpl = smpl_model
        self.target_height = target_height

        # Load data
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

        self.target_joints = torch.tensor(target_joints, dtype=torch.float32, device=device)

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
                errors.append(error_m.item() * 1000.0)

        return np.mean(errors) if errors else float('nan')

    def optimize_shape_v2(self, use_height_constraint=False):
        """v2-exact shape optimization"""
        print("\nStage 1: Shape optimization" + (" (with height constraint)" if use_height_constraint else " (baseline)"))
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

            # Height constraint
            if use_height_constraint and self.target_height is not None:
                pred_height = estimate_height_from_betas(betas, self.smpl, self.device)
                height_diff = (pred_height - self.target_height) ** 2
                loss = loss + 0.5 * height_diff

            loss.backward()
            optimizer.step()

            if i % 20 == 0 or i == 149:
                if use_height_constraint and self.target_height is not None:
                    pred_h = estimate_height_from_betas(betas, self.smpl, self.device).item()
                    print(f"  Iter {i:3d}: loss={loss.item():.6f}, height_pred={pred_h:.3f}, height_target={self.target_height:.3f}")
                else:
                    print(f"  Iter {i:3d}: loss={loss.item():.6f}")

        return betas.detach()

    def optimize_pose_v2(self, betas, use_enhanced_weights=False):
        """v2-exact pose optimization (frame-by-frame)"""
        print("\nStage 2: Pose optimization" + (" (with enhanced bone-segment weights)" if use_enhanced_weights else " (baseline)"))
        betas_fixed = betas.detach()
        poses = torch.zeros(self.n_frames, SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device)
        trans = torch.zeros(self.n_frames, 3, dtype=torch.float32, device=self.device)

        pose_lr = 0.01
        pose_iters = 80

        # Build weight tensor for enhanced weights
        if use_enhanced_weights:
            joint_weights_map = {}
            for joint_name, weight in BONE_SEGMENT_WEIGHTS.items():
                if joint_name in SMPL_JOINT_NAMES:
                    idx = SMPL_JOINT_NAMES.index(joint_name)
                    joint_weights_map[idx] = weight
            print(f"  Enhanced weights: min={min(BONE_SEGMENT_WEIGHTS.values()):.2f}, max={max(BONE_SEGMENT_WEIGHTS.values()):.2f}")

        # Frame-by-frame optimization (v2 style)
        for t in range(self.n_frames):
            target_frame = self.target_joints[t]
            if torch.isnan(target_frame).all():
                continue

            pose_param = torch.nn.Parameter(poses[t].clone())
            trans_param = torch.nn.Parameter(trans[t].clone())
            optimizer = torch.optim.Adam([pose_param, trans_param], lr=pose_lr)

            for _ in range(pose_iters):
                optimizer.zero_grad()
                joints_pred = self.smpl.joints(betas_fixed.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0)).squeeze(0)
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    break

                diff = pred_subset[mask] - target_subset[mask]

                # Per-joint weighting
                weights = torch.ones(diff.shape[0], device=self.device)
                for i, smpl_idx in enumerate(self.smpl_indices[mask]):
                    smpl_idx_val = smpl_idx.item()
                    if use_enhanced_weights and smpl_idx_val in joint_weights_map:
                        weights[i] = joint_weights_map[smpl_idx_val]
                    elif smpl_idx_val in [7, 8]:  # ankles (v2 baseline)
                        weights[i] = self.ankle_weight
                    elif smpl_idx_val in [3, 6, 9]:  # spine joints (v2 baseline)
                        weights[i] = self.spine_weight

                weighted_diff = diff * weights.unsqueeze(-1)
                loss = (weighted_diff ** 2).mean()

                # Pose regularization (v2)
                loss = loss + self.pose_reg_weight * (pose_param ** 2).mean()

                # Translation regularization (v2)
                if self.root_addb_idx is not None:
                    root_target = target_frame[self.root_addb_idx]
                    if not torch.isnan(root_target).any():
                        loss = loss + self.trans_reg_weight * ((trans_param - root_target) ** 2).mean()

                loss.backward()
                optimizer.step()

            poses[t] = pose_param.detach()
            trans[t] = trans_param.detach()

            if t % 200 == 0 or t == self.n_frames - 1:
                print(f"  Frame {t:4d}/{self.n_frames}: optimized")

        return poses, trans

    def refine_with_offsets_v2(self, betas, poses, trans):
        """v2-exact offset refinement"""
        print("\nStage 3: Per-joint offset refinement")
        poses_fixed = poses.clone()
        trans_fixed = trans.clone()
        betas_t = betas.unsqueeze(0)

        joint_offsets = torch.zeros(SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([joint_offsets], lr=0.001)

        for i in range(50):
            optimizer.zero_grad()
            loss = 0.0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_t[0], poses_fixed[t], trans_fixed[t])
                joints_pred = joints_pred + joint_offsets

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

        return joint_offsets.detach()

    def test_baseline(self):
        """TEST 1: Baseline (v2-exact)"""
        print("\n" + "="*80)
        print("TEST 1: BASELINE (v2-exact with regularization)")
        print("="*80)

        betas = self.optimize_shape_v2(use_height_constraint=False)
        poses, trans = self.optimize_pose_v2(betas, use_enhanced_weights=False)

        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas.unsqueeze(0), poses[t], trans[t])[0]
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Baseline MPJPE: {mpjpe:.2f} mm")
        return mpjpe

    def test_height_constraint(self):
        """TEST 2: Height constraint only"""
        print("\n" + "="*80)
        print("TEST 2: PRIORITY 1 - Height Constraint Only")
        print("="*80)

        if self.target_height is None:
            print("⚠ No target height provided, skipping")
            return None

        betas = self.optimize_shape_v2(use_height_constraint=True)
        poses, trans = self.optimize_pose_v2(betas, use_enhanced_weights=False)

        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas.unsqueeze(0), poses[t], trans[t])[0]
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Height Constraint MPJPE: {mpjpe:.2f} mm")
        return mpjpe

    def test_bone_weights(self):
        """TEST 3: Enhanced bone weights only"""
        print("\n" + "="*80)
        print("TEST 3: PRIORITY 2 - Enhanced Bone-Segment Weights Only")
        print("="*80)

        betas = self.optimize_shape_v2(use_height_constraint=False)
        poses, trans = self.optimize_pose_v2(betas, use_enhanced_weights=True)

        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas.unsqueeze(0), poses[t], trans[t])[0]
                pred_all.append(joints)
            pred_all = torch.stack(pred_all)

        mpjpe = self.compute_mpjpe(pred_all, self.target_joints)
        print(f"\n✓ Bone Weights MPJPE: {mpjpe:.2f} mm")
        return mpjpe

    def test_joint_offsets(self):
        """TEST 4: Per-joint offsets only"""
        print("\n" + "="*80)
        print("TEST 4: PRIORITY 3 - Per-Joint Learnable Offsets Only")
        print("="*80)

        betas = self.optimize_shape_v2(use_height_constraint=False)
        poses, trans = self.optimize_pose_v2(betas, use_enhanced_weights=False)
        offsets = self.refine_with_offsets_v2(betas, poses, trans)

        with torch.no_grad():
            pred_all = []
            for t in range(self.n_frames):
                joints = self.smpl.joints(betas.unsqueeze(0), poses[t], trans[t])[0]
                joints = joints + offsets
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
    print("Testing Individual Improvements (v2-EXACT baseline)")
    print("="*80)
    print(f"B3D file: {args.b3d}")
    print(f"Device  : {args.device}")
    print("="*80)

    # Load SMPL model
    print("\n[3/3] Initializing SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)

    # Get target height
    import nimblephysics as nimble
    target_height = None
    try:
        subj = nimble.biomechanics.SubjectOnDisk(args.b3d)
        target_height = float(subj.getHeightM())
    except:
        pass

    # Run tests
    tester = V2ExactTester(args.b3d, smpl, device, target_height)

    results = {}
    results['baseline'] = tester.test_baseline()
    results['height_constraint'] = tester.test_height_constraint()
    results['bone_weights'] = tester.test_bone_weights()
    results['joint_offsets'] = tester.test_joint_offsets()

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline (v2-exact):          {results['baseline']:.2f} mm")

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
