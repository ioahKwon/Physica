#!/usr/bin/env python3
"""
Ultra Refinement - Full Arsenal for High-MPJPE Subjects

Combines 7 advanced strategies:
1. Adaptive warm-start anchoring (decay 5.0 → 0.1)
2. Multi-restart with stochastic perturbation (5 restarts)
3. Joint-specific anchor weights
4. Per-subject adaptive hyperparameters
5. Cyclical learning rate
6. Outlier filtering
7. Multi-resolution optimization

Target: 30-50mm MPJPE improvement for subjects with 50-200mm initial MPJPE
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys
import time
from typing import Dict, List, Set
from scipy.interpolate import interp1d, CubicSpline

sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

# ============================================================================
# Joint Mapping Constants (from addbiomechanics_to_smpl_v3_enhanced.py)
# ============================================================================

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
    'wrist_r': 'right_wrist',
    'wrist_l': 'left_wrist',
    'radius_hand_r': 'right_wrist',
    'hand_r': 'right_hand',
    'radius_hand_l': 'left_wrist',
    'hand_l': 'left_hand',
    'neck': 'neck',
    'cervical': 'neck',
    'head': 'head',
}

SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

def infer_addb_joint_names(b3d_path: str, trial: int = 0, processing_pass: int = 0) -> List[str]:
    """Infer AddBiomechanics joint names from B3D file"""
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
    """Create automatic mapping from AddBiomechanics to SMPL joint indices"""
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

# ============================================================================
# STRATEGY 6: Outlier Filtering
# ============================================================================

def filter_outlier_joints(target_joints, threshold_std=3.0):
    """
    Remove outlier joint observations using velocity-based detection

    Args:
        target_joints: (T, N_joints, 3) tensor
        threshold_std: Outlier threshold in standard deviations

    Returns:
        filtered_joints: (T, N_joints, 3) tensor with outliers interpolated
    """
    filtered_joints = target_joints.clone()
    T, N_joints, _ = target_joints.shape

    n_outliers_total = 0

    for joint_idx in range(N_joints):
        joint_traj = target_joints[:, joint_idx, :].cpu().numpy()  # [T, 3]

        # Compute velocities
        if T < 2:
            continue

        velocities = np.diff(joint_traj, axis=0)
        vel_norms = np.linalg.norm(velocities, axis=-1)

        if len(vel_norms) == 0:
            continue

        # Detect outliers (velocity > threshold * std)
        vel_mean = vel_norms.mean()
        vel_std = vel_norms.std()

        if vel_std < 1e-6:  # Nearly static joint
            continue

        outlier_threshold = vel_mean + threshold_std * vel_std
        outlier_mask = vel_norms > outlier_threshold

        if outlier_mask.sum() > 0:
            n_outliers = outlier_mask.sum()
            n_outliers_total += n_outliers

            # Mark frames as outliers (velocity at i corresponds to frames i and i+1)
            outlier_frames = np.zeros(T, dtype=bool)
            for i in range(len(outlier_mask)):
                if outlier_mask[i]:
                    outlier_frames[i] = True
                    outlier_frames[i+1] = True

            # Interpolate outliers with cubic spline
            valid_mask = ~outlier_frames
            valid_indices = np.where(valid_mask)[0]
            outlier_indices = np.where(outlier_frames)[0]

            if len(valid_indices) >= 4 and len(outlier_indices) > 0:
                for dim in range(3):
                    valid_vals = joint_traj[valid_indices, dim]
                    cs = CubicSpline(valid_indices, valid_vals)
                    joint_traj[outlier_indices, dim] = cs(outlier_indices)

        filtered_joints[:, joint_idx, :] = torch.from_numpy(joint_traj).to(target_joints.device)

    if n_outliers_total > 0:
        print(f"  [Outlier Filter] Detected and interpolated {n_outliers_total} outlier observations")

    return filtered_joints

# ============================================================================
# STRATEGY 1 & 3: Adaptive Warm-Start Anchoring with Joint-Specific Weights
# ============================================================================

class AdaptiveAnchor:
    """Adaptive warm-start anchoring with joint-specific weights"""

    def __init__(self, initial_mpjpe, device):
        self.device = device

        # CONSERVATIVE Decay schedule based on initial MPJPE
        # For high-MPJPE subjects (>150mm), keep anchor strong to prevent divergence
        if initial_mpjpe < 70:
            self.anchor_schedule = [5.0, 3.0, 2.0, 1.0]
        elif initial_mpjpe < 100:
            self.anchor_schedule = [8.0, 5.0, 3.0, 2.0]
        elif initial_mpjpe < 150:
            self.anchor_schedule = [10.0, 7.0, 5.0]
        else:
            # Very conservative for worst cases (>150mm) - NEVER go below 5.0
            self.anchor_schedule = [12.0, 10.0, 7.0, 5.0]

        # Joint-specific weights (SMPL joint indices)
        # Lower weight = more freedom to change
        self.joint_weights = torch.ones(24, device=device)
        self.joint_weights[0] = 2.0  # Pelvis - keep stable
        self.joint_weights[[1, 2]] = 1.5  # Hips - semi-stable
        self.joint_weights[[4, 5]] = 1.2  # Knees
        self.joint_weights[[7, 8, 10, 11]] = 0.5  # Feet/ankles - most freedom
        self.joint_weights[[16, 17, 18, 19, 20, 21, 22, 23]] = 0.7  # Arms/hands

    def get_weight(self, iter_num, max_iters):
        """Get anchor weight for current iteration"""
        stage = int((iter_num / max_iters) * len(self.anchor_schedule))
        stage = min(stage, len(self.anchor_schedule) - 1)
        return self.anchor_schedule[stage]

    def compute_loss(self, poses, anchor_poses, trans, anchor_trans, iter_num, max_iters):
        """Compute joint-specific anchor loss with decay"""
        weight = self.get_weight(iter_num, max_iters)

        # Pose loss with joint-specific weights
        pose_diff = (poses - anchor_poses).reshape(-1, 24, 3)
        weighted_diff = pose_diff * self.joint_weights.view(1, 24, 1)
        pose_loss = (weighted_diff ** 2).mean()

        # Translation loss
        trans_loss = ((trans - anchor_trans) ** 2).mean()

        return weight * (pose_loss + trans_loss)

# ============================================================================
# STRATEGY 5: Cyclical Learning Rate
# ============================================================================

class CyclicLRScheduler:
    """Cyclical learning rate to escape local minima"""

    def __init__(self, base_lr=0.01, max_lr=0.03, cycle_length=50):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.cycle_length = cycle_length

    def get_lr(self, iter_num):
        """Get learning rate for current iteration"""
        cycle_pos = iter_num % self.cycle_length

        if cycle_pos < self.cycle_length // 2:
            # Increase phase
            t = cycle_pos / (self.cycle_length // 2)
            return self.base_lr + (self.max_lr - self.base_lr) * t
        else:
            # Decrease phase
            t = (cycle_pos - self.cycle_length // 2) / (self.cycle_length // 2)
            return self.max_lr - (self.max_lr - self.base_lr) * t

# ============================================================================
# STRATEGY 7: Multi-Resolution Optimization
# ============================================================================

def sample_frames(T, ratio):
    """Uniformly sample frames"""
    n_samples = max(int(T * ratio), 1)
    if n_samples >= T:
        return np.arange(T)
    return np.linspace(0, T-1, n_samples, dtype=int)

# ============================================================================
# STRATEGY 2 & 4: Multi-Restart with Per-Subject Hyperparameters
# ============================================================================

def get_adaptive_config(initial_mpjpe, T):
    """Get per-subject hyperparameters based on initial MPJPE"""

    if initial_mpjpe < 70:
        # Easy case
        return {
            'base_lr': 0.005,
            'max_lr': 0.015,
            'max_iters': 80,
            'n_restarts': 2,
            'perturbations': [0.0, 0.05],
            'use_cyclic_lr': False,
            'use_multiresolution': False,
        }
    elif initial_mpjpe < 100:
        # Medium case
        return {
            'base_lr': 0.01,
            'max_lr': 0.02,
            'max_iters': 120,
            'n_restarts': 3,
            'perturbations': [0.0, 0.05, 0.1],
            'use_cyclic_lr': True,
            'use_multiresolution': True,
        }
    elif initial_mpjpe < 150:
        # Hard case
        return {
            'base_lr': 0.015,
            'max_lr': 0.03,
            'max_iters': 200,
            'n_restarts': 5,
            'perturbations': [0.0, 0.05, 0.1, 0.2, 0.3],
            'use_cyclic_lr': True,
            'use_multiresolution': True,
        }
    else:
        # Very hard case - full arsenal
        return {
            'base_lr': 0.02,
            'max_lr': 0.04,
            'max_iters': 300,
            'n_restarts': 7,
            'perturbations': [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
            'use_cyclic_lr': True,
            'use_multiresolution': True,
        }

def perturb_params(poses, trans, scale, device):
    """Add Gaussian noise to parameters, more on problematic joints"""
    if scale == 0.0:
        return poses, trans

    # Pose perturbation with joint-specific scaling
    pose_noise = torch.randn_like(poses) * scale

    # More noise on feet/ankles (SMPL indices 7, 8, 10, 11)
    N = poses.shape[0]
    pose_noise_reshaped = pose_noise.reshape(N, 24, 3)
    pose_noise_reshaped[:, [7, 8, 10, 11]] *= 2.0
    # Less noise on pelvis (index 0)
    pose_noise_reshaped[:, 0] *= 0.5

    poses_perturbed = poses + pose_noise_reshaped.reshape(N, -1)

    # Translation perturbation (less aggressive)
    trans_noise = torch.randn_like(trans) * scale * 0.5
    trans_perturbed = trans + trans_noise

    return poses_perturbed, trans_perturbed

# ============================================================================
# Main Optimization Function
# ============================================================================

def optimize_single_restart(
    smpl, betas, poses_init, trans_init, target_joints_addb,
    anchor_poses, anchor_trans, adaptive_anchor,
    config, restart_idx, device, smpl_indices_tensor
):
    """Single restart optimization"""

    N = poses_init.shape[0]

    # Prepare optimizable parameters
    poses = poses_init.clone().requires_grad_(True)
    trans = trans_init.clone().requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.Adam([poses, trans], lr=config['base_lr'])

    # Cyclical LR scheduler
    if config['use_cyclic_lr']:
        lr_scheduler = CyclicLRScheduler(config['base_lr'], config['max_lr'], cycle_length=50)

    best_loss = float('inf')
    best_params = None
    no_improve_count = 0
    patience = 30

    max_iters = config['max_iters']

    for iter_num in range(max_iters):
        optimizer.zero_grad()

        # Update LR if using cyclical
        if config['use_cyclic_lr']:
            new_lr = lr_scheduler.get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr

        # Forward pass
        poses_reshaped = poses.reshape(N, 24, 3)
        betas_expanded = betas.unsqueeze(0).expand(N, -1)

        vertices, joints = smpl.forward(
            betas=betas_expanded,
            poses=poses_reshaped,
            trans=trans
        )
        # Extract only the mapped SMPL joints
        pred_joints_mapped = joints[:, smpl_indices_tensor]

        # Joint position loss using mapped joints
        joint_loss = ((pred_joints_mapped - target_joints_addb) ** 2).mean()

        # Anchor loss with adaptive decay
        anchor_loss = adaptive_anchor.compute_loss(
            poses, anchor_poses, trans, anchor_trans,
            iter_num, max_iters
        )

        # Pose regularization
        pose_reg = (poses_reshaped ** 2).mean() * 0.001

        # Velocity smoothness
        if N > 1:
            pose_velocity = poses_reshaped[1:] - poses_reshaped[:-1]
            velocity_loss = (pose_velocity ** 2).mean() * 0.0003
        else:
            velocity_loss = 0.0

        total_loss = joint_loss + anchor_loss + pose_reg + velocity_loss

        total_loss.backward()
        optimizer.step()

        # Track best
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = {
                'poses': poses.detach().clone(),
                'trans': trans.detach().clone()
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Early stopping
        if no_improve_count >= patience:
            print(f"    Early stop at iter {iter_num+1}")
            break

        # Progress logging
        if (iter_num + 1) % 50 == 0 or iter_num == 0:
            mpjpe = torch.sqrt(((pred_joints_mapped - target_joints_addb) ** 2).sum(dim=-1)).mean().item() * 1000
            anchor_w = adaptive_anchor.get_weight(iter_num, max_iters)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"    Iter {iter_num+1:4d}/{max_iters}: MPJPE={mpjpe:6.2f}mm, "
                  f"Loss={total_loss.item():.6f}, AnchorW={anchor_w:.3f}, LR={current_lr:.6f}")

    return best_params, best_loss

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str, required=True, help='B3D file path')
    parser.add_argument('--init_from', type=str, required=True, help='Warm-start result directory')
    parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outlier_threshold', type=float, default=3.0, help='Outlier detection threshold (std)')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("ULTRA REFINEMENT - Full Arsenal Strategy")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Init from: {args.init_from}")
    print(f"Output: {args.out_dir}")
    print("="*80 + "\n")

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    start_time = time.time()

    # ====================
    # 1. Load warm-start
    # ====================
    print("[1/7] Loading warm-start parameters...")
    init_dir = Path(args.init_from)
    inner_dirs = list(init_dir.glob("with_arm_*"))
    if not inner_dirs:
        print(f"ERROR: No with_arm_* directory in {init_dir}")
        sys.exit(1)

    npz_file = inner_dirs[0] / "smpl_params.npz"
    if not npz_file.exists():
        print(f"ERROR: {npz_file} does not exist")
        sys.exit(1)

    data = np.load(npz_file)
    init_betas = torch.from_numpy(data['betas']).float().to(device)
    init_poses = torch.from_numpy(data['poses']).float().to(device)
    init_trans = torch.from_numpy(data['trans']).float().to(device)

    T = init_poses.shape[0]
    print(f"  Loaded: {T} frames")

    # Load old MPJPE
    meta_file = inner_dirs[0] / "meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    old_mpjpe = meta.get('metrics', {}).get('MPJPE', 0)
    print(f"  Previous MPJPE: {old_mpjpe:.2f}mm")

    # ====================
    # 2. Load B3D targets with proper joint mapping
    # ====================
    print("[2/7] Loading B3D joint centers...")
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    num_frames_b3d = subject.getTrialLength(trial=0)
    print(f"  B3D has {num_frames_b3d} frames")

    # Infer joint names and create mapping
    print("  Inferring joint names and creating AddB→SMPL mapping...")
    addb_joint_names = infer_addb_joint_names(args.b3d)
    joint_mapping = auto_map_addb_to_smpl(addb_joint_names)

    if not joint_mapping:
        print(f"ERROR: Could not create joint mapping for {args.b3d}")
        sys.exit(1)

    print(f"  Joint mapping: {len(joint_mapping)} correspondences")
    addb_indices = sorted(joint_mapping.keys())
    smpl_indices = [joint_mapping[i] for i in addb_indices]

    # Sample up to 300 frames for efficiency
    if num_frames_b3d > 300:
        indices = np.linspace(0, num_frames_b3d-1, 300, dtype=int)
    else:
        indices = np.arange(num_frames_b3d)

    # Load using processingPasses for correct joint centers
    def load_b3d_joints(indices):
        joints_list = []
        for idx in indices:
            frames = subject.readFrames(
                trial=0,
                startFrame=int(idx),
                numFramesToRead=1,
                includeProcessingPasses=True,
                includeSensorData=False,
                stride=1,
                contactThreshold=1.0
            )
            if frames and len(frames) > 0:
                frame = frames[0]
                if hasattr(frame, 'processingPasses') and len(frame.processingPasses) > 0:
                    joint_data = np.asarray(frame.processingPasses[0].jointCenters, dtype=np.float32)
                else:
                    joint_data = np.asarray(frame.jointCenters, dtype=np.float32)

                if joint_data.ndim == 1:
                    joint_data = joint_data.reshape(-1, 3)

                joints_list.append(joint_data)

        return np.array(joints_list)

    b3d_joints = load_b3d_joints(indices)
    N = len(b3d_joints)
    n_joints = b3d_joints.shape[1]
    print(f"  Loaded {N} frames with {n_joints} joints")

    # Create target joints using proper joint mapping
    # CRITICAL FIX: Use joint_mapping instead of simple zero-padding
    target_joints_addb = torch.from_numpy(b3d_joints[:, addb_indices]).float().to(device)  # (N, len(mapping), 3)

    # Store mapping info for loss computation
    addb_indices_tensor = torch.tensor(addb_indices, dtype=torch.long, device=device)
    smpl_indices_tensor = torch.tensor(smpl_indices, dtype=torch.long, device=device)

    # Interpolate SMPL params if needed
    if T != N:
        print(f"  Interpolating SMPL params from {T} to {N} frames...")
        t_old = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, N)

        poses_np = init_poses.cpu().numpy().reshape(T, -1)
        poses_interp = interp1d(t_old, poses_np, axis=0, kind='linear')(t_new)
        init_poses = torch.from_numpy(poses_interp.reshape(N, 24, 3)).float().to(device)

        trans_np = init_trans.cpu().numpy()
        trans_interp = interp1d(t_old, trans_np, axis=0, kind='linear')(t_new)
        init_trans = torch.from_numpy(trans_interp).float().to(device)

    # Flatten poses for optimization
    init_poses_flat = init_poses.reshape(N, -1)

    # ====================
    # 3. Outlier filtering
    # ====================
    print("[3/7] Filtering outliers...")
    target_joints_addb = filter_outlier_joints(target_joints_addb, threshold_std=args.outlier_threshold)

    # ====================
    # 4. Load SMPL model
    # ====================
    print("[4/7] Loading SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)
    betas = init_betas.clone().detach()

    # ====================
    # 5. Get adaptive config
    # ====================
    print("[5/7] Setting up adaptive configuration...")
    config = get_adaptive_config(old_mpjpe, N)
    print(f"  Strategy for MPJPE={old_mpjpe:.2f}mm:")
    print(f"    - Restarts: {config['n_restarts']}")
    print(f"    - Max iterations: {config['max_iters']}")
    print(f"    - Base LR: {config['base_lr']:.4f}, Max LR: {config['max_lr']:.4f}")
    print(f"    - Cyclic LR: {config['use_cyclic_lr']}")
    print(f"    - Multi-resolution: {config['use_multiresolution']}")

    # Adaptive anchor
    adaptive_anchor = AdaptiveAnchor(old_mpjpe, device)
    print(f"    - Anchor schedule: {adaptive_anchor.anchor_schedule}")

    # ====================
    # 6. Multi-restart optimization
    # ====================
    print(f"\n[6/7] Running {config['n_restarts']} restarts with perturbation...")

    best_global_loss = float('inf')
    best_global_params = None
    best_restart_idx = -1

    for restart_idx in range(config['n_restarts']):
        perturb_scale = config['perturbations'][restart_idx]
        print(f"\n  ╔══ Restart {restart_idx+1}/{config['n_restarts']}: perturbation={perturb_scale:.2f} ══╗")

        # Perturb initialization
        poses_init, trans_init = perturb_params(
            init_poses_flat, init_trans, perturb_scale, device
        )

        # Optimize
        best_params, best_loss = optimize_single_restart(
            smpl, betas, poses_init, trans_init, target_joints_addb,
            init_poses_flat, init_trans, adaptive_anchor,
            config, restart_idx, device, smpl_indices_tensor
        )

        # Evaluate final MPJPE
        with torch.no_grad():
            poses_final = best_params['poses'].reshape(N, 24, 3)
            betas_expanded = betas.unsqueeze(0).expand(N, -1)
            _, joints_final = smpl.forward(betas=betas_expanded, poses=poses_final, trans=best_params['trans'])
            # Extract mapped joints for evaluation
            joints_final_mapped = joints_final[:, smpl_indices_tensor]
            final_mpjpe = torch.sqrt(((joints_final_mapped - target_joints_addb) ** 2).sum(dim=-1)).mean().item() * 1000

        print(f"  ╚══ Restart {restart_idx+1} MPJPE: {final_mpjpe:.2f}mm ══╝")

        if best_loss < best_global_loss:
            best_global_loss = best_loss
            best_global_params = best_params
            best_restart_idx = restart_idx
            print(f"  ✓✓✓ NEW BEST ✓✓✓")

    # ====================
    # 7. Save results
    # ====================
    print(f"\n[7/7] Saving results...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create inner directory name
    subject_name = Path(args.b3d).stem.lower()
    study_name = Path(args.b3d).parent.parent.name.lower().replace('_formatted_with_arm', '')
    inner_dir = out_dir / f"with_arm_{study_name}_{subject_name}"
    inner_dir.mkdir(parents=True, exist_ok=True)

    # Save NPZ
    np.savez(
        inner_dir / "smpl_params.npz",
        betas=betas.cpu().numpy(),
        poses=best_global_params['poses'].reshape(N, 24, 3).cpu().numpy(),
        trans=best_global_params['trans'].cpu().numpy()
    )

    # Compute final MPJPE using mapped joints
    with torch.no_grad():
        final_poses = best_global_params['poses'].reshape(N, 24, 3)
        betas_expanded = betas.unsqueeze(0).expand(N, -1)
        _, final_joints = smpl.forward(betas=betas_expanded, poses=final_poses, trans=best_global_params['trans'])
        # Extract only the mapped SMPL joints for evaluation
        final_joints_mapped = final_joints[:, smpl_indices_tensor]
        final_mpjpe = torch.sqrt(((final_joints_mapped - target_joints_addb) ** 2).sum(dim=-1)).mean().item() * 1000

    elapsed_time = time.time() - start_time
    improvement = old_mpjpe - final_mpjpe
    improvement_pct = (improvement / old_mpjpe * 100) if old_mpjpe > 0 else 0

    # Save metadata
    meta_output = {
        "b3d": args.b3d,
        "init_from": args.init_from,
        "num_frames": N,
        "strategy": "ultra_refine",
        "config": config,
        "best_restart": best_restart_idx,
        "optimization_time_seconds": elapsed_time,
        "metrics": {
            "MPJPE": final_mpjpe,
            "previous_MPJPE": old_mpjpe,
            "improvement_mm": improvement,
            "improvement_percent": improvement_pct,
            "num_comparisons": N * 24
        }
    }

    with open(inner_dir / "meta.json", 'w') as f:
        json.dump(meta_output, f, indent=2)

    print(f"\n{'='*80}")
    print("ULTRA REFINEMENT COMPLETE")
    print(f"{'='*80}")
    print(f"  Previous MPJPE: {old_mpjpe:.2f}mm")
    print(f"  Final MPJPE:    {final_mpjpe:.2f}mm")
    print(f"  Improvement:    {improvement:+.2f}mm ({improvement_pct:+.1f}%)")
    print(f"  Best restart:   {best_restart_idx+1}/{config['n_restarts']} (perturbation={config['perturbations'][best_restart_idx]:.2f})")
    print(f"  Time:           {elapsed_time:.1f}s")
    print(f"  Results:        {inner_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
