#!/usr/bin/env python3
"""
Test v2 baseline (BATCH optimization) + per-joint offset refinement only

Uses batch optimization instead of frame-by-frame for better performance.
"""

import argparse
import numpy as np
import torch
import torch.nn as nn

from addbiomechanics_to_smpl_v2 import (
    load_b3d_sequence,
    convert_addb_to_smpl_coords,
    infer_addb_joint_names,
    auto_map_addb_to_smpl,
    center_on_root_tensor,
    SMPLModel,
    SMPL_NUM_JOINTS,
    SMPL_JOINT_NAMES
)


def compute_mpjpe(pred_joints, target_joints, joint_mapping, root_idx):
    """Compute MPJPE with proper centering (v2 style)"""
    errors = []
    addb_indices = sorted(joint_mapping.keys())
    smpl_indices = [joint_mapping[i] for i in addb_indices]

    for t in range(pred_joints.shape[0]):
        pred_frame = pred_joints[t]
        target_frame = target_joints[t]

        pred_c, target_c = center_on_root_tensor(pred_frame, target_frame, root_idx)

        for addb_idx, smpl_idx in zip(addb_indices, smpl_indices):
            if addb_idx >= target_c.shape[0]:
                continue
            tgt = target_c[addb_idx]
            if torch.isnan(tgt).any():
                continue
            pred_pt = pred_c[smpl_idx]
            diff = pred_pt - tgt
            error_m = torch.norm(diff).item()
            errors.append(error_m * 1000.0)

    return np.mean(errors) if errors else float('nan')


class V2BatchWithOffsetTester:
    def __init__(self, b3d_file, smpl_model, device):
        self.device = device
        self.smpl = smpl_model

        # Load AddBiomechanics data
        print("\n[1/2] Loading AddBiomechanics data...")
        target_joints, dt = load_b3d_sequence(b3d_file)
        target_joints = convert_addb_to_smpl_coords(target_joints)
        self.target_joints = torch.from_numpy(target_joints).to(device)

        self.n_frames = target_joints.shape[0]
        print(f"  Frames : {self.n_frames}")
        print(f"  Joints : {target_joints.shape[1]}")

        # Setup joint mapping
        print("\n[2/2] Resolving joint correspondences...")
        addb_joint_names = infer_addb_joint_names(b3d_file)
        joint_mapping = auto_map_addb_to_smpl(addb_joint_names)

        self.addb_indices = sorted(joint_mapping.keys())
        self.smpl_indices = [joint_mapping[i] for i in self.addb_indices]
        self.root_addb_idx = self.addb_indices[0] if self.addb_indices else None

        print(f"  Using {len(joint_mapping)} correspondences")

        # Batch optimization hyperparameters (simple, no regularization)
        self.shape_lr = 5e-3
        self.shape_iters = 150
        self.pose_lr = 1e-2
        self.pose_iters = 100

        # Offset refinement hyperparameters
        self.offset_lr = 0.01
        self.offset_iters = 50
        self.offset_reg_weight = 1e-4

    def optimize_shape_batch(self):
        """Batch shape optimization (simple baseline)"""
        betas = torch.zeros(10, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([betas], lr=self.shape_lr)

        print("\nStage 1: Shape optimization (batch baseline)")
        for it in range(self.shape_iters):
            optimizer.zero_grad()

            # Sample uniformly
            sample_frames = torch.linspace(0, self.n_frames - 1, 50, dtype=torch.long, device=self.device)

            total_loss = 0.0
            for t in sample_frames:
                target_frame = self.target_joints[t]
                if torch.isnan(target_frame).all():
                    continue

                # Zero pose for shape fitting
                pose_zero = torch.zeros(SMPL_NUM_JOINTS, 3, device=self.device)
                trans_zero = torch.zeros(3, device=self.device)

                joints_pred = self.smpl.joints(betas.unsqueeze(0), pose_zero.unsqueeze(0), trans_zero.unsqueeze(0)).squeeze(0)
                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)

                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]
                loss = (diff ** 2).mean()
                total_loss = total_loss + loss

            total_loss.backward()
            optimizer.step()

            if it % 20 == 0 or it == self.shape_iters - 1:
                print(f"  Iter {it:3d}: loss={total_loss.item():.6f}")

        return betas.detach()

    def optimize_pose_batch(self, betas):
        """Batch pose optimization (all frames at once)"""
        betas_fixed = betas.detach()
        poses = nn.Parameter(torch.zeros(self.n_frames, SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device))
        trans = nn.Parameter(torch.zeros(self.n_frames, 3, dtype=torch.float32, device=self.device))

        optimizer = torch.optim.Adam([poses, trans], lr=self.pose_lr)

        print("\nStage 2: Pose optimization (batch baseline)")
        for it in range(self.pose_iters):
            optimizer.zero_grad()

            total_loss = 0.0
            count = 0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                if torch.isnan(target_frame).all():
                    continue

                joints_pred = self.smpl.joints(
                    betas_fixed.unsqueeze(0),
                    poses[t].unsqueeze(0),
                    trans[t].unsqueeze(0)
                ).squeeze(0)

                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)

                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]
                loss = (diff ** 2).mean()
                total_loss = total_loss + loss
                count += 1

            if count > 0:
                total_loss = total_loss / count

            total_loss.backward()
            optimizer.step()

            if it % 10 == 0 or it == self.pose_iters - 1:
                print(f"  Iter {it:3d}: loss={total_loss.item():.6f}")

        return poses.detach(), trans.detach()

    def refine_with_offsets(self, betas, poses, trans):
        """Add per-joint learnable offsets"""
        betas_fixed = betas.detach()
        poses_fixed = poses.detach()
        trans_fixed = trans.detach()

        # Initialize offsets
        offsets = nn.Parameter(torch.zeros(SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=self.device))
        optimizer = torch.optim.Adam([offsets], lr=self.offset_lr)

        print("\nStage 3: Per-joint offset refinement")
        for it in range(self.offset_iters):
            optimizer.zero_grad()

            total_loss = 0.0
            count = 0

            for t in range(self.n_frames):
                target_frame = self.target_joints[t]
                if torch.isnan(target_frame).all():
                    continue

                # Forward pass with fixed params + offsets
                joints_pred = self.smpl.joints(
                    betas_fixed.unsqueeze(0),
                    poses_fixed[t].unsqueeze(0),
                    trans_fixed[t].unsqueeze(0)
                ).squeeze(0)

                # Add offsets
                joints_pred = joints_pred + offsets

                pred_c, target_c = center_on_root_tensor(joints_pred, target_frame, self.root_addb_idx)

                target_subset = target_c[self.addb_indices]
                pred_subset = pred_c[self.smpl_indices]
                mask = ~torch.isnan(target_subset).any(dim=1)

                if mask.sum() == 0:
                    continue

                diff = pred_subset[mask] - target_subset[mask]
                loss = (diff ** 2).mean()
                total_loss = total_loss + loss
                count += 1

            if count > 0:
                total_loss = total_loss / count

            # Regularize offsets
            offset_reg = (offsets ** 2).mean()
            total_loss = total_loss + self.offset_reg_weight * offset_reg

            total_loss.backward()
            optimizer.step()

            if it % 10 == 0 or it == self.offset_iters - 1:
                offset_norm = torch.norm(offsets).item()
                print(f"  Iter {it:3d}: loss={total_loss.item():.6f}, offset_norm={offset_norm:.4f}")

        return offsets.detach()

    def run_tests(self):
        print("=" * 80)
        print("Testing v2 BATCH baseline + Per-Joint Offset Only")
        print("=" * 80)

        # Test 1: v2 batch baseline
        print("\n" + "=" * 80)
        print("TEST 1: v2 BATCH BASELINE (no offsets)")
        print("=" * 80)

        betas = self.optimize_shape_batch()
        poses, trans = self.optimize_pose_batch(betas)

        # Compute baseline MPJPE
        all_pred_joints = []
        for t in range(self.n_frames):
            joints = self.smpl.joints(
                betas.unsqueeze(0),
                poses[t].unsqueeze(0),
                trans[t].unsqueeze(0)
            ).squeeze(0)
            all_pred_joints.append(joints.cpu().numpy())

        pred_joints = np.stack(all_pred_joints, axis=0)
        pred_joints_torch = torch.from_numpy(pred_joints).to(self.device)

        joint_mapping = {addb_idx: smpl_idx for addb_idx, smpl_idx in zip(self.addb_indices, self.smpl_indices)}
        baseline_mpjpe = compute_mpjpe(pred_joints_torch, self.target_joints, joint_mapping, self.root_addb_idx)

        print(f"\n✓ v2 Batch Baseline MPJPE: {baseline_mpjpe:.2f} mm")

        # Test 2: v2 batch + per-joint offsets
        print("\n" + "=" * 80)
        print("TEST 2: v2 BATCH + PER-JOINT OFFSETS")
        print("=" * 80)

        offsets = self.refine_with_offsets(betas, poses, trans)

        # Compute MPJPE with offsets
        all_pred_joints_offset = []
        for t in range(self.n_frames):
            joints = self.smpl.joints(
                betas.unsqueeze(0),
                poses[t].unsqueeze(0),
                trans[t].unsqueeze(0)
            ).squeeze(0)
            joints = joints + offsets
            all_pred_joints_offset.append(joints.cpu().numpy())

        pred_joints_offset = np.stack(all_pred_joints_offset, axis=0)
        pred_joints_offset_torch = torch.from_numpy(pred_joints_offset).to(self.device)

        offset_mpjpe = compute_mpjpe(pred_joints_offset_torch, self.target_joints, joint_mapping, self.root_addb_idx)

        print(f"\n✓ v2 Batch + Offsets MPJPE: {offset_mpjpe:.2f} mm")

        # Summary
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"v2 Batch Baseline:          {baseline_mpjpe:7.2f} mm")
        print(f"v2 Batch + Per-Joint Offsets: {offset_mpjpe:7.2f} mm ({offset_mpjpe - baseline_mpjpe:+.2f} mm, {(baseline_mpjpe - offset_mpjpe)/baseline_mpjpe*100:.1f}% improvement)")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Test v2 batch baseline + per-joint offset only")
    parser.add_argument('--b3d', type=str, required=True, help='Path to .b3d file')
    parser.add_argument('--smpl_model', type=str, required=True, help='Path to SMPL model pkl')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')

    args = parser.parse_args()

    print("=" * 80)
    print("Testing v2 BATCH Baseline + Per-Joint Offset Only")
    print("=" * 80)
    print(f"B3D file: {args.b3d}")
    print(f"Device  : {args.device}")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize SMPL model
    print("\n[3/3] Initializing SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)

    # Run tests
    tester = V2BatchWithOffsetTester(args.b3d, smpl, device)
    tester.run_tests()


if __name__ == '__main__':
    main()
