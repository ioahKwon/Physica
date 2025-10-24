#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved AddBiomechanics (.b3d) → SMPL parameter fitting.

Improvements inspired by SMPL2AddBiomechanics:
1. Anthropometric constraints (height/mass)
2. Enhanced bone-segment weights
3. Per-joint learnable offsets
4. Additional metrics (PA-MPJPE, per-segment errors)
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.smpl_model import SMPLModel, SMPL_NUM_BETAS, SMPL_NUM_JOINTS

try:
    import nimblephysics as nimble
except ImportError as exc:
    raise RuntimeError("nimblephysics is required") from exc


# Enhanced bone-segment weights (inspired by SMPL2AddBiomechanics OSSO)
BONE_SEGMENT_WEIGHTS = {
    # Lower limbs (high accuracy needed for gait analysis)
    'right_ankle': 3.5,
    'left_ankle': 3.5,
    'right_knee': 3.0,
    'left_knee': 3.0,
    'right_hip': 2.5,
    'left_hip': 2.5,
    'right_foot': 2.0,
    'left_foot': 2.0,

    # Spine & Pelvis (medium accuracy)
    'pelvis': 3.0,  # root is very important
    'spine1': 2.5,
    'spine2': 2.0,
    'spine3': 1.8,

    # Upper limbs (lower accuracy)
    'right_shoulder': 1.5,
    'left_shoulder': 1.5,
    'right_elbow': 1.2,
    'left_elbow': 1.2,
    'right_wrist': 1.0,
    'left_wrist': 1.0,
    'right_hand': 0.8,
    'left_hand': 0.8,

    # Neck & Head
    'neck': 1.5,
    'head': 1.0,
}

# Joint name mapping
AUTO_JOINT_NAME_MAP = {
    'ground_pelvis': 'pelvis', 'pelvis': 'pelvis', 'root': 'pelvis',
    'hip_r': 'right_hip', 'hip_right': 'right_hip',
    'hip_l': 'left_hip', 'hip_left': 'left_hip',
    'walker_knee_r': 'right_knee', 'knee_r': 'right_knee',
    'walker_knee_l': 'left_knee', 'knee_l': 'left_knee',
    'ankle_r': 'right_ankle', 'ankle_right': 'right_ankle',
    'ankle_l': 'left_ankle', 'ankle_left': 'left_ankle',
    'subtalar_r': 'right_foot', 'mtp_r': 'right_foot',
    'subtalar_l': 'left_foot', 'mtp_l': 'left_foot',
    'back': 'spine1', 'torso': 'spine1', 'spine': 'spine1',
    'acromial_r': 'right_shoulder', 'shoulder_r': 'right_shoulder',
    'acromial_l': 'left_shoulder', 'shoulder_l': 'left_shoulder',
    'elbow_r': 'right_elbow', 'elbow_l': 'left_elbow',
    'wrist_r': 'right_wrist', 'radius_hand_r': 'right_wrist',
    'wrist_l': 'left_wrist', 'radius_hand_l': 'left_wrist',
    'hand_r': 'right_hand', 'hand_l': 'left_hand',
}

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]


def load_b3d_sequence(b3d_path: str,
                      trial: int = 0,
                      processing_pass: int = 0,
                      start: int = 0,
                      num_frames: int = -1) -> Tuple[np.ndarray, float]:
    """Load AddBiomechanics sequence with correct API"""
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
    total = subj.getTrialLength(trial)
    count = total - start if num_frames < 0 else min(num_frames, total - start)

    frames = subj.readFrames(
        trial=trial,
        startFrame=start,
        numFramesToRead=count,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )
    if len(frames) == 0:
        raise RuntimeError("Failed to read frames from the .b3d file")

    def frame_joint_centers(frame):
        if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
            idx = min(processing_pass, len(frame.processingPasses) - 1)
            return np.asarray(frame.processingPasses[idx].jointCenters, dtype=np.float32)
        return np.asarray(frame.jointCenters, dtype=np.float32)

    first = frame_joint_centers(frames[0])
    if first.ndim != 1 or first.size % 3 != 0:
        raise ValueError("Unexpected joint center layout in .b3d file")

    num_joints = first.size // 3
    joints = np.zeros((len(frames), num_joints, 3), dtype=np.float32)
    for i, frame in enumerate(frames):
        data = frame_joint_centers(frame)
        joints[i] = data.reshape(-1, 3)[:num_joints]

    dt = float(subj.getTrialTimestep(trial))
    return joints, dt


def convert_addb_to_smpl_coords(joints: np.ndarray) -> np.ndarray:
    """Convert AddBiomechanics coordinates to SMPL coordinates"""
    return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)


def infer_addb_joint_names(b3d_path: str,
                           trial: int = 0,
                           processing_pass: int = 0) -> List[str]:
    """Infer joint names from skeleton using v2 method"""
    subj = nimble.biomechanics.SubjectOnDisk(b3d_path)
    frames = subj.readFrames(
        trial=trial,
        startFrame=0,
        numFramesToRead=1,
        includeProcessingPasses=True,
        includeSensorData=False,
        stride=1,
        contactThreshold=1.0
    )
    if len(frames) == 0:
        raise RuntimeError("Unable to infer joint names — no frames were read")

    pp_idx = min(processing_pass, len(frames[0].processingPasses) - 1)
    pp = frames[0].processingPasses[pp_idx]

    skel = subj.readSkel(trial)
    pos = np.asarray(pp.pos, dtype=np.float32)
    if pos.size == skel.getNumDofs():
        skel.setPositions(pos)

    world_joints = []
    for i in range(skel.getNumJoints()):
        joint = skel.getJoint(i)
        world = joint.getChildBodyNode().getWorldTransform().translation()
        world_joints.append((joint.getName(), world))

    joint_centers = np.asarray(pp.jointCenters, dtype=np.float32).reshape(-1, 3)
    names: List[str] = []
    for center in joint_centers:
        dists = [np.linalg.norm(center - w) for _, w in world_joints]
        best = int(np.argmin(dists))
        names.append(world_joints[best][0])
    return names


def auto_map_addb_to_smpl(addb_joint_names: List[str]) -> Dict[int, int]:
    """Automatically map AddB joints to SMPL joints (v2 method)"""
    mapping: Dict[int, int] = {}
    used_smpl: Set[int] = set()
    for idx, name in enumerate(addb_joint_names):
        key = name.lower().replace(' ', '')
        if key in AUTO_JOINT_NAME_MAP:
            smpl_name = AUTO_JOINT_NAME_MAP[key]
            if smpl_name in SMPL_JOINT_NAMES:
                smpl_idx = SMPL_JOINT_NAMES.index(smpl_name)
                if smpl_idx in used_smpl:
                    continue
                mapping[idx] = smpl_idx
                used_smpl.add(smpl_idx)
    return mapping


def center_on_root_numpy(pred, targ, root_idx):
    """Center joints on root (matches v2 logic exactly)"""
    pred_centered = pred.copy()
    targ_centered = targ.copy()
    if root_idx is not None and root_idx < targ.shape[0]:
        root_target = targ[root_idx]
        if not np.any(np.isnan(root_target)):
            pred_centered -= pred[0]  # ✅ CRITICAL: Center pred on SMPL pelvis (index 0)
            targ_centered -= root_target  # Center target on AddB root
    return pred_centered, targ_centered


def center_on_root_tensor(pred, targ, root_idx):
    """Center joints on root (tensor version, matches v2 logic exactly)"""
    pred_centered = pred.clone()
    targ_centered = targ.clone()
    if root_idx is not None and root_idx < targ.shape[0]:
        root_target = targ[root_idx]
        if not torch.isnan(root_target).any():
            pred_centered = pred_centered - pred[0]  # ✅ CRITICAL: Center pred on SMPL pelvis (index 0)
            targ_centered = targ_centered - root_target  # Center target on AddB root
    return pred_centered, targ_centered


def estimate_height_from_betas(betas: torch.Tensor, smpl_model: SMPLModel) -> torch.Tensor:
    """Estimate height from SMPL betas (approximate)"""
    # Use neutral pose to estimate height
    zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=betas.device)
    zero_trans = torch.zeros(3, device=betas.device)
    joints = smpl_model.joints(betas.unsqueeze(0), zero_pose, zero_trans)

    # Height approximation: head to foot distance
    head_idx = SMPL_JOINT_NAMES.index('head')
    left_foot_idx = SMPL_JOINT_NAMES.index('left_foot')
    right_foot_idx = SMPL_JOINT_NAMES.index('right_foot')

    head_y = joints[0, head_idx, 1]
    foot_y = (joints[0, left_foot_idx, 1] + joints[0, right_foot_idx, 1]) / 2
    height = head_y - foot_y

    return height


def procrustes_alignment(pred, target):
    """Procrustes alignment (removes rotation/translation)"""
    # Center
    pred_centered = pred - pred.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    # Find rotation using SVD
    H = pred_centered.T @ target_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Apply rotation
    pred_aligned = pred_centered @ R

    return pred_aligned, target_centered


@dataclass
class FitterConfig:
    """Configuration for SMPL fitter"""
    shape_lr: float = 0.005
    shape_iters: int = 150
    shape_sample_frames: int = 80
    pose_lr: float = 0.01
    pose_iters: int = 100
    pose_reg_weight: float = 0.001
    trans_reg_weight: float = 0.01
    bone_length_weight: float = 0.1
    height_weight: float = 0.5  # NEW: anthropometric constraint
    offset_lr: float = 0.001  # NEW: per-joint offset learning rate
    offset_iters: int = 50  # NEW: iterations for offset refinement


class ImprovedSMPLFitter:
    """Improved SMPL fitter with anthropometric constraints and offsets"""

    def __init__(self, smpl_path, target_joints, joint_mapping, dt,
                 addb_subject=None, config=None, device='cuda'):
        self.smpl = SMPLModel(smpl_path, device=device)
        self.device = device
        self.config = config or FitterConfig()

        self.target_joints_np = target_joints
        self.target_joints = torch.from_numpy(target_joints).float().to(device)
        self.joint_mapping = joint_mapping
        self.dt = dt
        self.addb_subject = addb_subject

        # Extract anthropometric data
        self.target_height = None
        self.target_mass = None
        if addb_subject is not None:
            try:
                self.target_mass = addb_subject.getMassKg()
                self.target_height = addb_subject.getHeightM()
            except:
                pass

        # Setup indices
        self.addb_indices = torch.tensor(list(joint_mapping.keys()), dtype=torch.long, device=device)
        self.smpl_indices = torch.tensor(list(joint_mapping.values()), dtype=torch.long, device=device)

        # Find root
        self.root_addb_idx = None
        for addb_idx, smpl_idx in joint_mapping.items():
            if smpl_idx == 0:  # pelvis
                self.root_addb_idx = addb_idx
                break

        # Per-joint offsets (learnable)
        self.joint_offsets = torch.zeros(SMPL_NUM_JOINTS, 3, device=device)

        # Get bone segment weights
        self.get_joint_weights()

    def get_joint_weights(self):
        """Get per-joint weights based on bone segments"""
        weights = []
        for smpl_idx in self.smpl_indices:
            joint_name = SMPL_JOINT_NAMES[smpl_idx.item()]
            weight = BONE_SEGMENT_WEIGHTS.get(joint_name, 1.0)
            weights.append(weight)
        self.joint_weights = torch.tensor(weights, device=self.device)

    def optimise_shape(self):
        """Shape optimization with anthropometric constraints"""
        print("\n[4/5] Optimising SMPL parameters...")
        print("  Stage 1: Shape optimization")

        betas = torch.zeros(SMPL_NUM_BETAS, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=self.config.shape_lr)

        T = self.target_joints.shape[0]
        sample_size = min(self.config.shape_sample_frames, T)
        sample_indices = torch.linspace(0, T-1, sample_size, dtype=torch.long, device=self.device)

        for iteration in range(self.config.shape_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            valid_frames = 0

            for t_idx in sample_indices:
                target_frame = self.target_joints[t_idx]
                if torch.isnan(target_frame).all():
                    continue

                zero_pose = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                zero_trans = torch.zeros(3, device=self.device)
                joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose, zero_trans).squeeze(0)

                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)
                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]

                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]
                weights = self.joint_weights[mask].unsqueeze(-1)
                weighted_diff = diff * weights
                total_loss += (weighted_diff ** 2).sum()
                valid_frames += mask.sum().item()

            if valid_frames == 0:
                break

            loss = total_loss / valid_frames

            # Anthropometric constraint: height
            if self.target_height is not None:
                pred_height = estimate_height_from_betas(betas, self.smpl)
                height_diff = (pred_height - self.target_height) ** 2
                loss = loss + self.config.height_weight * height_diff

            # Beta regularization
            loss = loss + 0.001 * (betas ** 2).mean()

            loss.backward()
            optimizer.step()

            if iteration % 30 == 0:
                print(f"    Iter {iteration:3d}: loss = {loss.item():.6f}")

        return betas.detach()

    def optimise_pose_sequence(self, betas):
        """Pose optimization with enhanced weights"""
        print("  Stage 2: Pose optimization (per-frame)")

        betas = betas.detach()
        T = self.target_joints.shape[0]

        poses = torch.zeros(T, SMPL_NUM_JOINTS, 3, device=self.device)
        trans = torch.zeros(T, 3, device=self.device)

        for t in range(T):
            if t % 100 == 0:
                print(f"    Frame {t}/{T}")

            target_frame = self.target_joints[t]
            if torch.isnan(target_frame).all():
                continue

            pose_param = torch.nn.Parameter(poses[t].clone() if t > 0 else poses[t])
            trans_param = torch.nn.Parameter(trans[t].clone() if t > 0 else trans[t])
            optimiser = torch.optim.Adam([pose_param, trans_param], lr=self.config.pose_lr)

            for _ in range(self.config.pose_iters):
                optimiser.zero_grad()
                joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_param.unsqueeze(0),
                                              trans_param.unsqueeze(0)).squeeze(0)
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)

                if mask.sum() == 0:
                    break

                diff = pred_subset[mask] - target_subset[mask]
                weights = self.joint_weights[mask].unsqueeze(-1)
                weighted_diff = diff * weights
                loss = (weighted_diff ** 2).mean()

                # Regularization
                loss = loss + self.config.pose_reg_weight * (pose_param ** 2).mean()
                if self.root_addb_idx is not None:
                    root_target = target_frame[self.root_addb_idx]
                    if not torch.isnan(root_target).any():
                        loss = loss + self.config.trans_reg_weight * ((trans_param - root_target) ** 2).mean()

                loss.backward()
                optimiser.step()

            poses[t] = pose_param.detach()
            trans[t] = trans_param.detach()

        return poses.cpu().numpy(), trans.cpu().numpy()

    def refine_with_offsets(self, betas, poses, trans):
        """Refine using per-joint learnable offsets"""
        print("  Stage 3: Offset refinement")

        betas_t = betas.unsqueeze(0)
        poses_t = torch.from_numpy(poses).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        # Make offsets learnable
        self.joint_offsets = torch.nn.Parameter(torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device))
        optimizer = torch.optim.Adam([self.joint_offsets], lr=self.config.offset_lr)

        for iteration in range(self.config.offset_iters):
            optimizer.zero_grad()
            total_loss = 0.0
            valid_count = 0

            # Sample frames for efficiency
            sample_size = min(80, poses_t.shape[0])
            sample_indices = torch.linspace(0, poses_t.shape[0]-1, sample_size, dtype=torch.long)

            for t in sample_indices:
                target_frame = self.target_joints[t]
                if torch.isnan(target_frame).all():
                    continue

                joints_pred = self.smpl.joints(betas_t[0], poses_t[t], trans_t[t])
                # Apply learnable offsets
                joints_pred = joints_pred + self.joint_offsets

                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)
                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]

                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]
                total_loss += (diff ** 2).sum()
                valid_count += mask.sum().item()

            if valid_count == 0:
                break

            loss = total_loss / valid_count
            # Regularize offsets (keep them small)
            loss = loss + 0.01 * (self.joint_offsets ** 2).mean()

            loss.backward()
            optimizer.step()

            if iteration % 10 == 0:
                print(f"    Iter {iteration:3d}: loss = {loss.item():.6f}, " +
                      f"offset_norm = {self.joint_offsets.norm().item():.4f}")

        return self.joint_offsets.detach().cpu().numpy()

    def evaluate(self, betas, poses, trans, offsets=None):
        """Evaluate with additional metrics"""
        print("\n[5/5] Computing evaluation metrics...")

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
        pa_errors = []  # Procrustes-aligned errors
        segment_errors = {'lower_limb': [], 'upper_limb': [], 'torso': []}

        thresholds = [0.05, 0.10, 0.15]
        pck_counts = np.zeros(len(thresholds), dtype=np.int64)

        lower_limb_joints = {'right_hip', 'left_hip', 'right_knee', 'left_knee',
                            'right_ankle', 'left_ankle', 'right_foot', 'left_foot'}
        upper_limb_joints = {'right_shoulder', 'left_shoulder', 'right_elbow', 'left_elbow',
                            'right_wrist', 'left_wrist', 'right_hand', 'left_hand'}
        torso_joints = {'pelvis', 'spine1', 'spine2', 'spine3', 'neck'}

        for pred_frame, target_frame in zip(joints_pred, self.target_joints_np):
            pred_c, target_c = center_on_root_numpy(pred_frame, target_frame, self.root_addb_idx)

            # Extract only mapped joints for Procrustes alignment
            pred_mapped = []
            target_mapped = []
            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx >= target_c.shape[0]:
                    continue
                tgt = target_c[addb_idx]
                if np.any(np.isnan(tgt)):
                    continue
                pred_mapped.append(pred_c[smpl_idx])
                target_mapped.append(tgt)

            if len(pred_mapped) == 0:
                continue

            pred_mapped = np.array(pred_mapped)
            target_mapped = np.array(target_mapped)

            # Procrustes alignment (on mapped joints only)
            pred_aligned, target_aligned = procrustes_alignment(pred_mapped, target_mapped)

            # Compute errors
            idx = 0
            for addb_idx, smpl_idx in self.joint_mapping.items():
                if addb_idx >= target_c.shape[0]:
                    continue
                tgt = target_c[addb_idx]
                if np.any(np.isnan(tgt)):
                    continue

                # Regular error
                diff = pred_c[smpl_idx] - tgt
                error_m = np.linalg.norm(diff)
                errors.append(error_m * 1000.0)

                # PA error
                diff_pa = pred_aligned[idx] - target_aligned[idx]
                idx += 1
                pa_error_m = np.linalg.norm(diff_pa)
                pa_errors.append(pa_error_m * 1000.0)

                # Segment errors
                joint_name = SMPL_JOINT_NAMES[smpl_idx]
                if joint_name in lower_limb_joints:
                    segment_errors['lower_limb'].append(error_m * 1000.0)
                elif joint_name in upper_limb_joints:
                    segment_errors['upper_limb'].append(error_m * 1000.0)
                elif joint_name in torso_joints:
                    segment_errors['torso'].append(error_m * 1000.0)

                # PCK
                for i, thr in enumerate(thresholds):
                    if error_m < thr:
                        pck_counts[i] += 1

        mpjpe = float(np.mean(errors)) if errors else float('nan')
        pa_mpjpe = float(np.mean(pa_errors)) if pa_errors else float('nan')
        total = max(len(errors), 1)

        metrics = {
            'MPJPE': mpjpe,
            'PA-MPJPE': pa_mpjpe,
            'num_comparisons': float(len(errors)),
            'MPJPE_per_segment': {
                'lower_limb': float(np.mean(segment_errors['lower_limb'])) if segment_errors['lower_limb'] else float('nan'),
                'upper_limb': float(np.mean(segment_errors['upper_limb'])) if segment_errors['upper_limb'] else float('nan'),
                'torso': float(np.mean(segment_errors['torso'])) if segment_errors['torso'] else float('nan'),
            },
        }

        for thr, count in zip(thresholds, pck_counts):
            metrics[f'PCK@{thr:.2f}m'] = float(100.0 * count / total)

        if self.target_height is not None:
            pred_height = estimate_height_from_betas(betas, self.smpl).item()
            metrics['height_error_m'] = abs(pred_height - self.target_height)

        return metrics, joints_pred

    def run(self):
        """Full pipeline with all improvements"""
        betas = self.optimise_shape()
        poses_np, trans_np = self.optimise_pose_sequence(betas)
        offsets_np = self.refine_with_offsets(betas, poses_np, trans_np)
        metrics, pred_joints = self.evaluate(betas, poses_np, trans_np, offsets_np)

        return betas, poses_np, trans_np, offsets_np, metrics, pred_joints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', required=True, help='Path to .b3d file')
    parser.add_argument('--smpl_model', required=True, help='Path to SMPL model')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--shape_lr', type=float, default=0.005)
    parser.add_argument('--shape_iters', type=int, default=150)
    parser.add_argument('--pose_lr', type=float, default=0.01)
    parser.add_argument('--pose_iters', type=int, default=100)
    parser.add_argument('--height_weight', type=float, default=0.5,
                       help='Weight for anthropometric height constraint')
    parser.add_argument('--offset_lr', type=float, default=0.001,
                       help='Learning rate for per-joint offset refinement')
    parser.add_argument('--offset_iters', type=int, default=50,
                       help='Iterations for per-joint offset refinement')
    args = parser.parse_args()

    print("="*80)
    print("Improved AddBiomechanics → SMPL Fitter")
    print("="*80)
    print(f"B3D file: {args.b3d}")
    print(f"Output  : {args.out_dir}")
    print(f"Device  : {args.device}")
    print("="*80)
    print()

    # Load data
    print("[1/5] Loading AddBiomechanics data...")
    target_joints, dt = load_b3d_sequence(args.b3d)

    # Convert AddB coordinates to SMPL coordinates (same as v2)
    target_joints = convert_addb_to_smpl_coords(target_joints)

    # Load subject metadata separately
    addb_subject = nimble.biomechanics.SubjectOnDisk(args.b3d)

    print(f"  Frames : {target_joints.shape[0]}")
    print(f"  Joints : {target_joints.shape[1]}")
    print(f"  dt     : {dt:.6f} s ({1/dt:.2f} fps)")

    # Get anthropometric data if available
    target_height = None
    try:
        mass = addb_subject.getMassKg()
        target_height = addb_subject.getHeightM()
        print(f"  Mass   : {mass:.2f} kg")
        print(f"  Height : {target_height:.3f} m")
    except:
        print("  (No anthropometric data available)")

    # Map joints using v2 method
    print("\n[2/5] Resolving joint correspondences...")
    addb_joint_names = infer_addb_joint_names(args.b3d)
    print(f"  DEBUG: Found {len(addb_joint_names)} AddB joint names")
    print(f"  DEBUG: First few names: {addb_joint_names[:5] if len(addb_joint_names) > 0 else 'NONE'}")
    joint_mapping = auto_map_addb_to_smpl(addb_joint_names)
    print(f"  DEBUG: Joint mapping dict has {len(joint_mapping)} entries")

    print(f"  Using {len(joint_mapping)} correspondences")
    for addb_idx, smpl_idx in sorted(joint_mapping.items())[:10]:
        print(f"    {addb_joint_names[addb_idx]:20s} → {SMPL_JOINT_NAMES[smpl_idx]}")

    unmapped_indices = set(range(len(addb_joint_names))) - set(joint_mapping.keys())
    unmapped = []
    if unmapped_indices:
        unmapped = [addb_joint_names[i] for i in sorted(unmapped_indices)]
        print(f"  Unmapped AddB joints ({len(unmapped)}): {', '.join(unmapped[:5])}")

    # Initialize fitter
    print("\n[3/5] Initialising SMPL model...")
    config = FitterConfig(
        shape_lr=args.shape_lr,
        shape_iters=args.shape_iters,
        pose_lr=args.pose_lr,
        pose_iters=args.pose_iters,
        height_weight=args.height_weight,
        offset_lr=args.offset_lr,
        offset_iters=args.offset_iters,
    )

    fitter = ImprovedSMPLFitter(
        args.smpl_model, target_joints, joint_mapping, dt,
        addb_subject=addb_subject, config=config, device=args.device
    )

    # Run optimization
    betas, poses, trans, offsets, metrics, pred_joints = fitter.run()

    # Print results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"MPJPE           : {metrics['MPJPE']:.2f} mm")
    print(f"PA-MPJPE        : {metrics['PA-MPJPE']:.2f} mm")
    print(f"Lower limb MPJPE: {metrics['MPJPE_per_segment']['lower_limb']:.2f} mm")
    print(f"Upper limb MPJPE: {metrics['MPJPE_per_segment']['upper_limb']:.2f} mm")
    print(f"Torso MPJPE     : {metrics['MPJPE_per_segment']['torso']:.2f} mm")
    if 'height_error_m' in metrics:
        print(f"Height error    : {metrics['height_error_m']*100:.2f} cm")
    print("="*80)

    # Save outputs
    os.makedirs(args.out_dir, exist_ok=True)

    run_name = os.path.splitext(os.path.basename(args.b3d))[0]
    output_name = f"improved_{run_name}"
    output_path = os.path.join(args.out_dir, output_name)
    os.makedirs(output_path, exist_ok=True)

    # Save SMPL params
    np.savez(
        os.path.join(output_path, 'smpl_params.npz'),
        betas=betas.cpu().numpy(),
        poses=poses,
        trans=trans,
        offsets=offsets,
    )

    # Save joints
    np.save(os.path.join(output_path, 'pred_joints.npy'), pred_joints)
    np.save(os.path.join(output_path, 'target_joints.npy'), target_joints)

    # Save metadata
    meta = {
        'b3d': args.b3d,
        'run_name': output_name,
        'dt': float(dt),
        'num_frames': int(target_joints.shape[0]),
        'joint_mapping': {int(k): int(v) for k, v in joint_mapping.items()},
        'unmapped_addb_joints': list(unmapped),
        'improvements': ['anthropometric_constraints', 'bone_segment_weights', 'joint_offsets'],
        'config': {
            'shape_lr': args.shape_lr,
            'shape_iters': args.shape_iters,
            'pose_lr': args.pose_lr,
            'pose_iters': args.pose_iters,
            'height_weight': args.height_weight,
        },
        'metrics': metrics,
    }

    with open(os.path.join(output_path, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutputs saved to: {output_path}")
    print("  - smpl_params.npz (includes offsets)")
    print("  - pred_joints.npy")
    print("  - target_joints.npy")
    print("  - meta.json")


if __name__ == '__main__':
    main()
