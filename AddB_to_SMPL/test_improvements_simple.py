#!/usr/bin/env python3
"""
Simple test of 3 improvements from SMPL2AddBiomechanics paper
Based on working v2 code with minimal changes
"""

import sys
import os
import argparse
import numpy as np
import torch
from typing import Dict, List, Tuple
from dataclasses import dataclass

# Import from v2
sys.path.insert(0, os.path.dirname(__file__))
from addbiomechanics_to_smpl_v2 import (
    load_b3d_sequence,
    convert_addb_to_smpl_coords,
    infer_addb_joint_names,
    auto_map_addb_to_smpl,
    center_on_root_numpy,
    center_on_root_tensor,  # ✅ ADD THIS for gradient flow
    SMPLModel,
    SMPL_NUM_JOINTS,
    SMPL_JOINT_NAMES,
    AUTO_JOINT_NAME_MAP
)

import nimblephysics as nimble


@dataclass
class ImprovedConfig:
    shape_lr: float = 5e-3
    shape_iters: int = 150
    pose_lr: float = 1e-2
    pose_iters: int = 80

    # NEW: Anthropometric constraint weight
    height_weight: float = 0.5

    # NEW: Per-joint offset refinement
    offset_lr: float = 0.001
    offset_iters: int = 50

    # Existing
    pose_reg_weight: float = 1e-3
    trans_reg_weight: float = 1e-3
    bone_length_weight: float = 10.0


# NEW: Enhanced bone-segment weights (Priority 2)
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

    # Spine & Pelvis (medium accuracy)
    'pelvis': 3.0,
    'spine1': 2.5,
    'spine2': 2.0,
    'spine3': 1.8,

    # Upper limbs (lower accuracy acceptable)
    'right_shoulder': 1.5,
    'left_shoulder': 1.5,
    'right_elbow': 1.2,
    'left_elbow': 1.2,
    'right_wrist': 1.0,
    'left_wrist': 1.0,

    # Head/neck
    'neck': 1.5,
    'head': 1.2,
}


def estimate_height_from_betas(betas: torch.Tensor, smpl_model: SMPLModel, device) -> torch.Tensor:
    """Estimate height from SMPL betas (Priority 1)"""
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


class ImprovedFitter:
    def __init__(self, smpl_model, target_joints, joint_mapping, dt, config, device, target_height=None):
        self.smpl = smpl_model
        self.config = config
        self.device = device
        self.target_height = target_height

        self.target_joints_np = target_joints
        self.target_joints = torch.from_numpy(target_joints).float().to(device)
        self.dt = dt
        self.joint_mapping = joint_mapping

        # Find root index
        self.root_addb_idx = None
        for addb_idx, smpl_idx in joint_mapping.items():
            if smpl_idx == 0:  # pelvis
                self.root_addb_idx = addb_idx
                break

        # Setup indices for vectorized operations
        self.addb_indices = []
        self.smpl_indices = []
        for addb_idx, smpl_idx in sorted(joint_mapping.items()):
            self.addb_indices.append(addb_idx)
            self.smpl_indices.append(smpl_idx)
        self.addb_indices = torch.tensor(self.addb_indices, dtype=torch.long, device=device)
        self.smpl_indices = torch.tensor(self.smpl_indices, dtype=torch.long, device=device)

        # NEW: Build per-joint weights based on BONE_SEGMENT_WEIGHTS (Priority 2)
        self.joint_weights = torch.ones(len(self.smpl_indices), device=device)
        for i, smpl_idx in enumerate(self.smpl_indices):
            joint_name = SMPL_JOINT_NAMES[smpl_idx]
            if joint_name in BONE_SEGMENT_WEIGHTS:
                self.joint_weights[i] = BONE_SEGMENT_WEIGHTS[joint_name]

        print(f"  Joint weights applied: min={self.joint_weights.min().item():.2f}, " +
              f"max={self.joint_weights.max().item():.2f}, mean={self.joint_weights.mean().item():.2f}")

    def optimize_shape(self):
        """Stage 1: Shape optimization with anthropometric constraint (Priority 1)"""
        print("Stage 1: Shape optimization (with height constraint)")

        n_frames = self.target_joints.shape[0]
        sample_size = min(self.config.shape_iters, n_frames)
        sample_indices = torch.randperm(n_frames, device=self.device)[:sample_size]

        betas = torch.nn.Parameter(torch.zeros(10, device=self.device))
        optimizer = torch.optim.Adam([betas], lr=self.config.shape_lr)

        for iteration in range(self.config.shape_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            valid_count = 0

            for t in sample_indices:
                target_frame = self.target_joints[t]
                zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                zero_trans = torch.zeros(3, device=self.device)

                joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose, zero_trans)[0]

                # ✅ FIX: Use center_on_root_tensor to maintain gradient flow!
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]

                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                # Apply per-joint weights (Priority 2)
                diff = pred_subset[mask] - target_subset[mask]
                weighted_diff = diff * self.joint_weights[mask].unsqueeze(1)
                total_loss += (weighted_diff ** 2).sum()
                valid_count += mask.sum().item()

            if valid_count == 0:
                break

            loss = total_loss / valid_count

            # NEW: Add anthropometric height constraint (Priority 1)
            if self.target_height is not None:
                pred_height = estimate_height_from_betas(betas, self.smpl, self.device)
                height_diff = (pred_height - self.target_height) ** 2
                loss = loss + self.config.height_weight * height_diff

            # Regularization
            loss = loss + 1e-4 * (betas ** 2).mean()

            loss.backward()
            optimizer.step()

            if iteration % 20 == 0:
                if self.target_height is not None:
                    print(f"  Iter {iteration:3d}: loss={loss.item():.6f}, " +
                          f"height_pred={pred_height.item():.3f}, height_target={self.target_height:.3f}")
                else:
                    print(f"  Iter {iteration:3d}: loss={loss.item():.6f}")

        return betas.detach()

    def optimize_pose_sequence(self, betas):
        """Stage 2: Pose optimization (with enhanced bone weights - Priority 2)"""
        print("Stage 2: Pose optimization (with enhanced bone-segment weights)")

        n_frames = self.target_joints.shape[0]
        poses = torch.nn.Parameter(torch.zeros(n_frames, SMPL_NUM_JOINTS, 3, device=self.device))
        trans = torch.nn.Parameter(torch.zeros(n_frames, 3, device=self.device))

        optimizer = torch.optim.Adam([poses, trans], lr=self.config.pose_lr)
        betas_t = betas.unsqueeze(0)

        for iteration in range(self.config.pose_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            valid_count = 0

            for t in range(n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_t[0], poses[t], trans[t])

                # ✅ FIX: Use center_on_root_tensor to maintain gradient flow!
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]

                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                # Apply per-joint weights (Priority 2)
                diff = pred_subset[mask] - target_subset[mask]
                weighted_diff = diff * self.joint_weights[mask].unsqueeze(1)
                total_loss += (weighted_diff ** 2).sum()
                valid_count += mask.sum().item()

            if valid_count == 0:
                break

            loss = total_loss / valid_count
            loss = loss + self.config.pose_reg_weight * (poses ** 2).mean()
            loss = loss + self.config.trans_reg_weight * (trans ** 2).mean()

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: loss={loss.item():.6f}")

        return poses.detach().cpu().numpy(), trans.detach().cpu().numpy()

    def refine_with_offsets(self, betas, poses, trans):
        """Stage 3: Per-joint offset refinement (Priority 3)"""
        print("Stage 3: Per-joint offset refinement")

        self.joint_offsets = torch.nn.Parameter(torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device))
        optimizer = torch.optim.Adam([self.joint_offsets], lr=self.config.offset_lr)

        betas_t = betas.unsqueeze(0)
        poses_t = torch.from_numpy(poses).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        n_frames = self.target_joints.shape[0]

        for iteration in range(self.config.offset_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            valid_count = 0

            for t in range(n_frames):
                target_frame = self.target_joints[t]
                joints_pred = self.smpl.joints(betas_t[0], poses_t[t], trans_t[t])
                joints_pred = joints_pred + self.joint_offsets  # Apply learnable offsets

                # ✅ FIX: Use center_on_root_tensor to maintain gradient flow!
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]

                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                # Apply per-joint weights
                diff = pred_subset[mask] - target_subset[mask]
                weighted_diff = diff * self.joint_weights[mask].unsqueeze(1)
                total_loss += (weighted_diff ** 2).sum()
                valid_count += mask.sum().item()

            if valid_count == 0:
                break

            loss = total_loss / valid_count
            # Regularize offsets (keep them small)
            loss = loss + 0.01 * (self.joint_offsets ** 2).mean()

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"  Iter {iteration:3d}: loss={loss.item():.6f}, " +
                      f"offset_norm={self.joint_offsets.norm().item():.4f}")

        return self.joint_offsets.detach().cpu().numpy()

    def evaluate(self, betas, poses, trans, offsets=None):
        """Evaluate MPJPE"""
        betas_t = betas.unsqueeze(0)
        poses_t = torch.from_numpy(poses).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        if offsets is not None:
            offsets_t = torch.from_numpy(offsets).float().to(self.device)
        else:
            offsets_t = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)

        joints_pred = []
        for t in range(poses_t.shape[0]):
            jp = self.smpl.joints(betas_t[0], poses_t[t], trans_t[t])
            jp = jp + offsets_t
            joints_pred.append(jp.cpu().numpy())
        joints_pred = np.stack(joints_pred, axis=0)

        errors = []
        for pred_frame, target_frame in zip(joints_pred, self.target_joints_np):
            pred_c, target_c = center_on_root_numpy(pred_frame, target_frame, self.root_addb_idx)

            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx >= target_c.shape[0]:
                    continue
                tgt = target_c[addb_idx]
                if np.any(np.isnan(tgt)):
                    continue
                diff = pred_c[smpl_idx] - tgt
                error_m = np.linalg.norm(diff)
                errors.append(error_m * 1000.0)  # Convert to mm

        mpjpe = float(np.mean(errors)) if errors else float('nan')
        return {'MPJPE': mpjpe}, joints_pred

    def run(self):
        """Run all 3 stages"""
        betas = self.optimize_shape()
        poses, trans = self.optimize_pose_sequence(betas)
        offsets = self.refine_with_offsets(betas, poses, trans)
        metrics, pred_joints = self.evaluate(betas, poses, trans, offsets)
        return betas, poses, trans, offsets, metrics, pred_joints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', required=True)
    parser.add_argument('--smpl_model', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--height_weight', type=float, default=0.5)
    parser.add_argument('--offset_lr', type=float, default=0.001)
    parser.add_argument('--offset_iters', type=int, default=50)
    args = parser.parse_args()

    print("=" * 80)
    print("Testing 3 SMPL2AddBiomechanics Improvements")
    print("=" * 80)
    print(f"B3D file: {args.b3d}")
    print(f"Output  : {args.out_dir}")
    print(f"Device  : {args.device}")
    print("=" * 80)

    # Load data
    print("\n[1/5] Loading AddBiomechanics data...")
    target_joints, dt = load_b3d_sequence(args.b3d)
    target_joints = convert_addb_to_smpl_coords(target_joints)
    print(f"  Frames : {target_joints.shape[0]}")
    print(f"  Joints : {target_joints.shape[1]}")

    # Get anthropometric data (Priority 1)
    target_height = None
    try:
        subj = nimble.biomechanics.SubjectOnDisk(args.b3d)
        target_height = float(subj.getHeightM())
        print(f"  Height : {target_height:.3f} m")
    except:
        print("  (No height data available)")

    # Map joints
    print("\n[2/5] Resolving joint correspondences...")
    addb_joint_names = infer_addb_joint_names(args.b3d)
    joint_mapping = auto_map_addb_to_smpl(addb_joint_names)
    print(f"  Using {len(joint_mapping)} correspondences")

    # Initialize SMPL
    print("\n[3/5] Initializing SMPL model...")
    device = torch.device(args.device)
    smpl_model = SMPLModel(args.smpl_model, device=device)

    # Run improved fitter
    print("\n[4/5] Running improved SMPL fitter...")
    config = ImprovedConfig(
        height_weight=args.height_weight,
        offset_lr=args.offset_lr,
        offset_iters=args.offset_iters
    )
    fitter = ImprovedFitter(smpl_model, target_joints, joint_mapping, dt, config, device, target_height)
    betas, poses, trans, offsets, metrics, pred_joints = fitter.run()

    # Evaluate
    print("\n[5/5] Results:")
    print(f"  MPJPE: {metrics['MPJPE']:.2f} mm")

    # Save
    os.makedirs(args.out_dir, exist_ok=True)
    np.savez(os.path.join(args.out_dir, 'smpl_params.npz'),
             betas=betas.cpu().numpy(),
             poses=poses,
             trans=trans,
             offsets=offsets)
    np.save(os.path.join(args.out_dir, 'pred_joints.npy'), pred_joints)
    np.save(os.path.join(args.out_dir, 'target_joints.npy'), target_joints)
    print(f"\n  Saved to {args.out_dir}")


if __name__ == '__main__':
    main()
