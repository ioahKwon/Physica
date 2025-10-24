#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AddBiomechanics (.b3d) → SMPL parameter fitting.

This script builds a fresh optimisation pipeline that mirrors the ideas from
SMPL2AddBiomechanics, but in reverse. Given an AddBiomechanics sequence we:
    1. infer the ordered AddB joint list and auto-map it to SMPL joints,
    2. optimise SMPL betas (shape) on a subsampled set of frames,
    3. optimise pose + translation sequentially with warm starts,
    4. optionally repeat the refinement until a target error tolerance is met.

Outputs include SMPL parameters, reconstructed joints, evaluation metrics,
metadata and visualisations (via the companion visualise_results.py script).
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import torch
import torch.nn.functional as F

from models.smpl_model import SMPLModel, SMPL_NUM_BETAS, SMPL_NUM_JOINTS

try:
    import nimblephysics as nimble
except ImportError as exc:  # pragma: no cover - nimble is an external dep
    raise RuntimeError(
        "nimblephysics is required to read AddBiomechanics .b3d files. "
        "Install nimblephysics inside the current environment."
    ) from exc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

AUTO_JOINT_NAME_MAP: Dict[str, str] = {
    'ground_pelvis': 'pelvis',
    'pelvis': 'pelvis',
    'root': 'pelvis',
    'hip_r': 'right_hip',
    'hip_right': 'right_hip',
    'hip_l': 'left_hip',
    'hip_left': 'left_hip',
    'walker_knee_r': 'right_knee',
    'knee_r': 'right_knee',
    'walker_knee_l': 'left_knee',
    'knee_l': 'left_knee',
    'ankle_r': 'right_ankle',
    'ankle_right': 'right_ankle',
    'ankle_l': 'left_ankle',
    'ankle_left': 'left_ankle',
    'subtalar_r': 'right_foot',
    'mtp_r': 'right_foot',
    'subtalar_l': 'left_foot',
    'mtp_l': 'left_foot',
    'back': 'spine1',
    'torso': 'spine1',
    'spine': 'spine1',
    'acromial_r': 'right_shoulder',
    'shoulder_r': 'right_shoulder',
    'acromial_l': 'left_shoulder',
    'shoulder_l': 'left_shoulder',
    'elbow_r': 'right_elbow',
    'elbow_l': 'left_elbow',
    # Skip radioulnar - it's forearm rotation, not wrist
    # 'radioulnar_r': 'right_wrist',  # REMOVED - intermediate joint
    'wrist_r': 'right_wrist',
    # 'radioulnar_l': 'left_wrist',  # REMOVED - intermediate joint
    'wrist_l': 'left_wrist',
    # Map radius_hand to wrist (it's the actual wrist position in AddB)
    'radius_hand_r': 'right_wrist',
    'hand_r': 'right_hand',
    'radius_hand_l': 'left_wrist',
    'hand_l': 'left_hand',
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
    return np.stack([joints[..., 0], joints[..., 2], -joints[..., 1]], axis=-1)


def infer_addb_joint_names(b3d_path: str,
                           trial: int = 0,
                           processing_pass: int = 0) -> List[str]:
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


def load_mapping_overrides(map_json: Optional[str],
                           addb_names: List[str]) -> Dict[int, int]:
    if map_json is None:
        return {}
    if not os.path.exists(map_json):
        raise FileNotFoundError(f"Mapping JSON not found: {map_json}")

    with open(map_json, 'r') as f:
        data = json.load(f)

    overrides_raw = data.get('addb2smpl', {})
    overrides: Dict[int, int] = {}
    for key, value in overrides_raw.items():
        if isinstance(key, str) and key.strip().isdigit():
            addb_idx = int(key)
        elif isinstance(key, str) and key in addb_names:
            addb_idx = addb_names.index(key)
        elif isinstance(key, int):
            addb_idx = key
        else:
            raise ValueError(f"Unknown AddB joint identifier: {key}")

        if isinstance(value, str) and value.strip().isdigit():
            smpl_idx = int(value)
        elif isinstance(value, str) and value in SMPL_JOINT_NAMES:
            smpl_idx = SMPL_JOINT_NAMES.index(value)
        elif isinstance(value, int):
            smpl_idx = value
        else:
            raise ValueError(f"Unknown SMPL joint identifier: {value}")

        overrides[addb_idx] = smpl_idx
    return overrides


def resolve_mapping(addb_names: List[str],
                    auto_map: Dict[int, int],
                    overrides: Dict[int, int]) -> Dict[int, int]:
    mapping = auto_map.copy()
    mapping.update(overrides)
    if not mapping:
        raise ValueError("No joint correspondences were resolved. "
                         "Provide a mapping JSON or extend the auto-map table.")
    return mapping


def center_on_root_tensor(pred: torch.Tensor,
                          target: torch.Tensor,
                          root_idx: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred_centered = pred.clone()
    target_centered = target.clone()
    if root_idx is not None and root_idx < target.shape[0]:
        root_target = target[root_idx]
        if not torch.isnan(root_target).any():
            pred_centered = pred_centered - pred[0]
            target_centered = target_centered - root_target
    return pred_centered, target_centered


def center_on_root_numpy(pred: np.ndarray,
                         target: np.ndarray,
                         root_idx: Optional[int]) -> Tuple[np.ndarray, np.ndarray]:
    pred_centered = pred.copy()
    target_centered = target.copy()
    if root_idx is not None and root_idx < target.shape[0]:
        root_target = target[root_idx]
        if not np.any(np.isnan(root_target)):
            pred_centered -= pred[0]
            target_centered -= root_target
    return pred_centered, target_centered


# ---------------------------------------------------------------------------
# Optimisation configuration and core fitter
# ---------------------------------------------------------------------------

@dataclass
class OptimisationConfig:
    shape_lr: float = 5e-3
    shape_iters: int = 150
    shape_sample_frames: int = 80

    pose_lr: float = 1e-2
    pose_iters: int = 80

    tolerance_mm: float = 20.0
    max_passes: int = 3

    pose_reg_weight: float = 1e-3
    trans_reg_weight: float = 1e-3
    bone_length_weight: float = 10.0  # Weight for bone length constraint
    temporal_smooth_weight: float = 0.1  # Weight for temporal smoothness
    ankle_weight: float = 3.0  # Extra weight for ankle joints (problematic)
    spine_weight: float = 2.0  # Extra weight for spine joints


class AddBToSMPLFitter:
    def __init__(self,
                 smpl_model: SMPLModel,
                 target_joints: np.ndarray,
                 joint_mapping: Dict[int, int],
                 addb_joint_names: List[str],
                 dt: float,
                 config: OptimisationConfig,
                 device: torch.device = torch.device('cpu')) -> None:
        self.smpl = smpl_model
        self.config = config
        self.device = device

        self.target_joints_np = target_joints
        self.target_joints = torch.from_numpy(target_joints).float().to(device)
        self.dt = dt

        self.joint_mapping = joint_mapping
        self.addb_joint_names = addb_joint_names

        self.addb_indices = torch.tensor(list(joint_mapping.keys()), dtype=torch.long, device=device)
        self.smpl_indices = torch.tensor(list(joint_mapping.values()), dtype=torch.long, device=device)

        self.root_addb_idx = None
        for addb_idx, smpl_idx in joint_mapping.items():
            if smpl_idx == 0:
                self.root_addb_idx = addb_idx
                break

    # --------------------------- Shape optimisation -----------------------
    def _compute_bone_lengths(self, joints: torch.Tensor) -> Dict[str, float]:
        """Compute bone lengths from joint positions."""
        # Define kinematic chains for major bones
        bone_pairs = [
            (1, 4, 'left_femur'),    # left_hip to left_knee
            (2, 5, 'right_femur'),   # right_hip to right_knee
            (4, 7, 'left_tibia'),    # left_knee to left_ankle
            (5, 8, 'right_tibia'),   # right_knee to right_ankle
            (16, 18, 'left_humerus'), # left_shoulder to left_elbow
            (17, 19, 'right_humerus'),# right_shoulder to right_elbow
            (18, 20, 'left_radius'),  # left_elbow to left_wrist
            (19, 21, 'right_radius'), # right_elbow to right_wrist
        ]

        lengths = {}
        for parent_idx, child_idx, name in bone_pairs:
            if parent_idx < joints.shape[0] and child_idx < joints.shape[0]:
                bone_vec = joints[child_idx] - joints[parent_idx]
                lengths[name] = torch.norm(bone_vec)
        return lengths

    def optimise_shape(self) -> torch.Tensor:
        betas = torch.zeros(SMPL_NUM_BETAS, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=self.config.shape_lr)

        num_frames = self.target_joints.shape[0]
        sample_count = min(self.config.shape_sample_frames, num_frames)
        frame_indices = np.linspace(0, num_frames - 1, sample_count, dtype=int)

        # Compute target bone lengths from AddB data (average across frames)
        target_bone_lengths = {}
        valid_count = {}
        for idx in frame_indices:
            target = self.target_joints[idx]
            if not torch.isnan(target).all():
                target_c = target - target[self.root_addb_idx] if self.root_addb_idx is not None else target
                lengths = self._compute_bone_lengths(target_c)
                for name, length in lengths.items():
                    if name not in target_bone_lengths:
                        target_bone_lengths[name] = 0.0
                        valid_count[name] = 0
                    target_bone_lengths[name] += length.item()
                    valid_count[name] += 1

        # Average bone lengths
        for name in target_bone_lengths:
            if valid_count[name] > 0:
                target_bone_lengths[name] /= valid_count[name]

        zero_pose = torch.zeros(1, SMPL_NUM_JOINTS, 3, device=self.device)
        zero_trans = torch.zeros(1, 3, device=self.device)

        for it in range(self.config.shape_iters):
            optimizer.zero_grad()

            joints_pred = self.smpl.joints(betas.unsqueeze(0), zero_pose[0], zero_trans[0]).squeeze(0)

            # Joint position loss
            total_loss = 0.0
            valid_frames = 0
            for idx in frame_indices:
                target = self.target_joints[idx]
                if torch.isnan(target).all():
                    continue
                pred_c, target_c = center_on_root_tensor(joints_pred, target, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]

                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]
                total_loss = total_loss + (diff ** 2).mean()
                valid_frames += 1

            if valid_frames == 0:
                break

            loss = total_loss / valid_frames

            # Bone length constraint loss
            pred_bone_lengths = self._compute_bone_lengths(joints_pred)
            bone_loss = 0.0
            bone_count = 0
            for name, target_len in target_bone_lengths.items():
                if name in pred_bone_lengths:
                    diff_len = pred_bone_lengths[name] - target_len
                    bone_loss += diff_len ** 2
                    bone_count += 1

            if bone_count > 0:
                bone_loss = bone_loss / bone_count
                loss = loss + self.config.bone_length_weight * bone_loss

            loss.backward()
            optimizer.step()

        return betas.detach()

    # --------------------------- Pose optimisation ------------------------
    def optimise_pose_sequence(self,
                               betas: torch.Tensor,
                               init_poses: Optional[torch.Tensor] = None,
                               init_trans: Optional[torch.Tensor] = None,
                               pose_iters: Optional[int] = None,
                               pose_lr: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        betas = betas.detach()
        T = self.target_joints.shape[0]

        poses = torch.zeros(T, SMPL_NUM_JOINTS, 3, device=self.device)
        trans = torch.zeros(T, 3, device=self.device)

        if init_poses is not None:
            poses.copy_(init_poses)
        if init_trans is not None:
            trans.copy_(init_trans)

        num_iters = pose_iters or self.config.pose_iters
        lr = pose_lr or self.config.pose_lr

        for t in range(T):
            target_frame = self.target_joints[t]
            if torch.isnan(target_frame).all():
                continue

            pose_param = torch.nn.Parameter(poses[t].clone())
            trans_param = torch.nn.Parameter(trans[t].clone())
            optimiser = torch.optim.Adam([pose_param, trans_param], lr=lr)

            for _ in range(num_iters):
                optimiser.zero_grad()
                joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_param.unsqueeze(0), trans_param.unsqueeze(0)).squeeze(0)
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)
                if mask.sum() == 0:
                    break

                diff = pred_subset[mask] - target_subset[mask]

                # Per-joint weighting for problematic joints
                weights = torch.ones(diff.shape[0], device=self.device)
                for i, smpl_idx in enumerate(self.smpl_indices[mask]):
                    if smpl_idx in [7, 8]:  # ankles
                        weights[i] = self.config.ankle_weight
                    elif smpl_idx in [3, 6, 9]:  # spine joints
                        weights[i] = self.config.spine_weight

                weighted_diff = diff * weights.unsqueeze(-1)
                loss = (weighted_diff ** 2).mean()

                # Pose regularization
                loss = loss + self.config.pose_reg_weight * (pose_param ** 2).mean()

                # Translation regularization
                if self.root_addb_idx is not None:
                    root_target = target_frame[self.root_addb_idx]
                    if not torch.isnan(root_target).any():
                        loss = loss + self.config.trans_reg_weight * ((trans_param - root_target) ** 2).mean()

                # Temporal smoothness (disabled - original optimization works best)
                # if self.config.temporal_smooth_weight > 0 and t > 0:
                #     pose_diff = pose_param - poses[t-1]
                #     trans_diff = trans_param - trans[t-1]
                #     temporal_loss = (pose_diff ** 2).mean() + (trans_diff ** 2).mean()
                #     loss = loss + self.config.temporal_smooth_weight * temporal_loss

                loss.backward()
                optimiser.step()

            poses[t] = pose_param.detach()
            trans[t] = trans_param.detach()

        return poses.cpu().numpy(), trans.cpu().numpy()

    # --------------------------- Evaluation --------------------------------
    def evaluate(self,
                 betas: torch.Tensor,
                 poses: np.ndarray,
                 trans: np.ndarray) -> Tuple[Dict[str, float], np.ndarray]:
        betas_t = betas.unsqueeze(0)
        poses_t = torch.from_numpy(poses).float().to(self.device)
        trans_t = torch.from_numpy(trans).float().to(self.device)

        joints_pred = []
        for t in range(poses_t.shape[0]):
            jp = self.smpl.joints(betas_t[0], poses_t[t], trans_t[t]).cpu().numpy()
            joints_pred.append(jp)
        joints_pred = np.stack(joints_pred, axis=0)

        errors = []
        thresholds = [0.05, 0.10, 0.15]
        pck_counts = np.zeros(len(thresholds), dtype=np.int64)

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
                errors.append(error_m * 1000.0)  # Convert to millimeters
                for i, thr in enumerate(thresholds):
                    if error_m < thr:
                        pck_counts[i] += 1

        mpjpe = float(np.mean(errors)) if errors else float('nan')  # Now in mm
        total = max(len(errors), 1)
        metrics = {
            'MPJPE': mpjpe,
            'num_comparisons': float(len(errors)),
        }
        for thr, count in zip(thresholds, pck_counts):
            metrics[f'PCK@{thr:.2f}m'] = float(100.0 * count / total)
        return metrics, joints_pred

    # --------------------------- Full fitting pipeline ---------------------
    def run(self) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Dict[str, float], np.ndarray]:
        betas = self.optimise_shape()

        poses_np, trans_np = self.optimise_pose_sequence(betas)
        metrics, pred_joints = self.evaluate(betas, poses_np, trans_np)

        tolerance = self.config.tolerance_mm / 1000.0
        passes = 1

        while (
            passes < self.config.max_passes
            and not np.isnan(metrics['MPJPE'])
            and metrics['MPJPE'] > tolerance
        ):
            poses_init = torch.from_numpy(poses_np).float().to(self.device)
            trans_init = torch.from_numpy(trans_np).float().to(self.device)
            poses_np, trans_np = self.optimise_pose_sequence(
                betas,
                init_poses=poses_init,
                init_trans=trans_init,
                pose_iters=max(40, self.config.pose_iters // 2),
                pose_lr=self.config.pose_lr * 0.5
            )
            metrics, pred_joints = self.evaluate(betas, poses_np, trans_np)
            passes += 1

        return betas, poses_np, trans_np, metrics, pred_joints


# ---------------------------------------------------------------------------
# Command-line interface
# ---------------------------------------------------------------------------

def derive_run_name(b3d_path: str) -> str:
    p_lower = b3d_path.lower()
    if "with_arm" in p_lower:
        arm = "with_arm"
    elif "no_arm" in p_lower:
        arm = "no_arm"
    else:
        arm = "unknown"

    import re
    matches = re.findall(r"/([A-Za-z]+20\d{2})(?:[_/])", b3d_path)
    study = matches[-1].lower() if matches else "unknown"

    base = os.path.splitext(os.path.basename(b3d_path))[0]
    m = re.match(r"([A-Za-z]*\d+)", base)
    subj = m.group(1).lower() if m else base.lower()
    return f"{arm}_{study}_{subj}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Convert AddBiomechanics motion to SMPL parameters')
    parser.add_argument('--b3d', required=True, help='Path to the .b3d file')
    parser.add_argument('--out_dir', required=True, help='Output root directory')
    parser.add_argument('--map_json', default=None,
                        help='Optional JSON file overriding AddB→SMPL joint mapping')
    parser.add_argument('--smpl_model', required=True, help='Path to SMPL model .pkl file')

    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--processing_pass', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=-1)

    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--shape_lr', type=float, default=5e-3)
    parser.add_argument('--shape_iters', type=int, default=150)
    parser.add_argument('--shape_sample_frames', type=int, default=80)
    parser.add_argument('--pose_lr', type=float, default=1e-2)
    parser.add_argument('--pose_iters', type=int, default=80)
    parser.add_argument('--tolerance_mm', type=float, default=20.0,
                        help='Early-stop once MPJPE drops below this (in millimetres)')
    parser.add_argument('--max_passes', type=int, default=3,
                        help='Maximum refinement passes after the first fitting')
    parser.add_argument('--pose_reg', type=float, default=1e-3)
    parser.add_argument('--trans_reg', type=float, default=1e-3)

    parser.add_argument('--verbose', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = torch.device('cuda' if args.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[WARN] CUDA not available, falling back to CPU')

    run_name = derive_run_name(args.b3d)
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)

    print('\n' + '=' * 80)
    print('AddBiomechanics → SMPL Fitter')
    print('=' * 80)
    print(f'Run name : {run_name}')
    print(f'Output   : {out_dir}')
    print(f'Device   : {device}')
    print('=' * 80 + '\n')

    print('[1/5] Loading AddBiomechanics data...')
    addb_joints, dt = load_b3d_sequence(
        args.b3d,
        trial=args.trial,
        processing_pass=args.processing_pass,
        start=args.start,
        num_frames=args.num_frames
    )
    joint_names = infer_addb_joint_names(
        args.b3d,
        trial=args.trial,
        processing_pass=args.processing_pass
    )
    addb_joints = convert_addb_to_smpl_coords(addb_joints)
    print(f'  Frames : {addb_joints.shape[0]}')
    print(f'  Joints : {addb_joints.shape[1]}')
    print(f'  dt     : {dt:.6f} s ({1.0 / dt:.2f} fps)')

    print('\n[2/5] Resolving joint correspondences...')
    auto_map = auto_map_addb_to_smpl(joint_names)
    overrides = load_mapping_overrides(args.map_json, joint_names)
    mapping = resolve_mapping(joint_names, auto_map, overrides)

    print(f'  Using {len(mapping)} correspondences')
    for idx, smpl_idx in sorted(mapping.items(), key=lambda kv: kv[0]):
        addb_name = joint_names[idx] if idx < len(joint_names) else str(idx)
        smpl_name = SMPL_JOINT_NAMES[smpl_idx]
        print(f'    {addb_name:20s} → {smpl_name}')

    unmapped = [name for i, name in enumerate(joint_names) if i not in mapping]
    if unmapped:
        print(f'  Unmapped AddB joints ({len(unmapped)}): {", ".join(unmapped)}')

    print('\n[3/5] Initialising SMPL model...')
    smpl = SMPLModel(args.smpl_model, device=device)

    cfg = OptimisationConfig(
        shape_lr=args.shape_lr,
        shape_iters=args.shape_iters,
        shape_sample_frames=args.shape_sample_frames,
        pose_lr=args.pose_lr,
        pose_iters=args.pose_iters,
        tolerance_mm=args.tolerance_mm,
        max_passes=args.max_passes,
        pose_reg_weight=args.pose_reg,
        trans_reg_weight=args.trans_reg
    )

    fitter = AddBToSMPLFitter(
        smpl_model=smpl,
        target_joints=addb_joints,
        joint_mapping=mapping,
        addb_joint_names=joint_names,
        dt=dt,
        config=cfg,
        device=device
    )

    print('\n[4/5] Optimising SMPL parameters...')
    betas, poses_np, trans_np, metrics, pred_joints = fitter.run()

    print('\n[5/5] Metrics:')
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.6f}')
        else:
            print(f'  {key}: {value}')

    print('\nSaving results...')
    np.savez(
        os.path.join(out_dir, 'smpl_params.npz'),
        betas=betas.cpu().numpy(),
        poses=poses_np,
        trans=trans_np
    )
    np.save(os.path.join(out_dir, 'pred_joints.npy'), pred_joints)
    np.save(os.path.join(out_dir, 'target_joints.npy'), addb_joints)

    meta = {
        'b3d': args.b3d,
        'run_name': run_name,
        'dt': float(dt),
        'num_frames': int(addb_joints.shape[0]),
        'num_joints_addb': int(addb_joints.shape[1]),
        'addb_joint_names': joint_names,
        'joint_mapping': mapping,
        'joint_mapping_named': {
            joint_names[idx]: SMPL_JOINT_NAMES[smpl_idx]
            for idx, smpl_idx in mapping.items()
        },
        'shape_lr': args.shape_lr,
        'shape_iters': args.shape_iters,
        'shape_sample_frames': args.shape_sample_frames,
        'pose_lr': args.pose_lr,
        'pose_iters': args.pose_iters,
        'tolerance_mm': args.tolerance_mm,
        'max_passes': args.max_passes,
        'metrics': metrics,
        'unmapped_addb_joints': unmapped,
    }

    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print('\n' + '=' * 80)
    print(f'DONE! Results saved to: {out_dir}')
    print('=' * 80 + '\n')


if __name__ == '__main__':
    main()
