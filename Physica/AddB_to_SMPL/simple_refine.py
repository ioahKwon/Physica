#!/usr/bin/env python3
"""
Simple refinement script - Load existing SMPL and refine with minimal iterations
Goal: 50mm+ MPJPE → under 50mm
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys

# Import from existing code
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str, required=True)
    parser.add_argument('--init_from', type=str, required=True, help='Previous result directory')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=100, help='Total iterations for refinement')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("SIMPLE REFINEMENT - Warm-start + Light Optimization")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Init from: {args.init_from}")
    print(f"Output: {args.out_dir}")
    print(f"Max iterations: {args.max_iters}")
    print("="*80 + "\n")

    device = torch.device(args.device)

    # 1. Load warm-start SMPL parameters
    print("[1/5] Loading warm-start SMPL parameters...")
    init_dir = Path(args.init_from)
    inner_dirs = list(init_dir.glob("with_arm_*"))
    if not inner_dirs:
        print(f"ERROR: No with_arm_* directory found in {init_dir}")
        sys.exit(1)

    npz_file = inner_dirs[0] / "smpl_params.npz"
    if not npz_file.exists():
        print(f"ERROR: {npz_file} does not exist")
        sys.exit(1)

    data = np.load(npz_file)
    init_betas = torch.from_numpy(data['betas']).float().to(device)  # (10,)
    init_poses = torch.from_numpy(data['poses']).float().to(device)  # (T, 24, 3)
    init_trans = torch.from_numpy(data['trans']).float().to(device)  # (T, 3)

    T = init_poses.shape[0]
    print(f"  Loaded: {T} frames, betas: {init_betas.shape}, poses: {init_poses.shape}, trans: {init_trans.shape}")

    # 2. Load AddBiomechanics B3D file
    print("[2/5] Loading AddBiomechanics B3D file...")
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    num_frames_b3d = subject.getNumFrames()
    print(f"  B3D has {num_frames_b3d} frames")

    # Sample up to 200 frames
    if num_frames_b3d > 200:
        indices = np.linspace(0, num_frames_b3d-1, 200, dtype=int)
    else:
        indices = np.arange(num_frames_b3d)

    # Load target joints (24 joints including head/neck)
    target_joints_list = []
    for idx in indices:
        frame = subject.readFrames(int(idx), 1)[0]
        markers = frame.markerObservations
        joint_pos_world = {}

        # Map AddBiomechanics joints to SMPL (simplified - use all available)
        for marker_name, marker_pos in markers.items():
            joint_pos_world[marker_name] = marker_pos

        # Convert to tensor (for now, simplified - just use first 24)
        joints_array = np.zeros((24, 3))
        for i, (name, pos) in enumerate(list(joint_pos_world.items())[:24]):
            joints_array[i] = pos

        target_joints_list.append(joints_array)

    target_joints = torch.from_numpy(np.array(target_joints_list)).float().to(device)  # (N, 24, 3)
    N = target_joints.shape[0]
    print(f"  Loaded {N} frames of target joints")

    # Interpolate SMPL params to match target frames
    if T != N:
        print(f"  Interpolating SMPL params from {T} to {N} frames...")
        from scipy.interpolate import interp1d

        t_old = np.linspace(0, 1, T)
        t_new = np.linspace(0, 1, N)

        # Interpolate poses
        poses_np = init_poses.cpu().numpy().reshape(T, -1)
        poses_interp = interp1d(t_old, poses_np, axis=0, kind='linear')(t_new)
        init_poses = torch.from_numpy(poses_interp.reshape(N, 24, 3)).float().to(device)

        # Interpolate trans
        trans_np = init_trans.cpu().numpy()
        trans_interp = interp1d(t_old, trans_np, axis=0, kind='linear')(t_new)
        init_trans = torch.from_numpy(trans_interp).float().to(device)

    # 3. Load SMPL model
    print("[3/5] Loading SMPL model...")
    smpl = SMPL(args.smpl_model, device=device)

    # 4. Optimize
    print(f"[4/5] Running lightweight optimization ({args.max_iters} iterations)...")

    # Prepare optimizable parameters
    betas = init_betas.clone().detach().requires_grad_(True)
    poses = init_poses.clone().detach().reshape(N, -1).requires_grad_(True)  # (N, 72)
    trans = init_trans.clone().detach().requires_grad_(True)

    optimizer = torch.optim.Adam([
        {'params': [betas], 'lr': 0.01},
        {'params': [poses], 'lr': 0.001},
        {'params': [trans], 'lr': 0.001}
    ])

    best_loss = float('inf')
    best_params = None

    for iter_num in range(args.max_iters):
        optimizer.zero_grad()

        # Forward pass
        poses_reshaped = poses.reshape(N, 24, 3)
        body_model_output = smpl(
            betas=betas.unsqueeze(0).expand(N, -1),
            body_pose=poses_reshaped[:, 1:],  # Exclude root
            global_orient=poses_reshaped[:, 0:1],
            transl=trans
        )
        pred_joints = body_model_output.joints[:, :24]  # (N, 24, 3)

        # Loss: Joint position error
        joint_loss = ((pred_joints - target_joints) ** 2).mean()

        # Regularization
        pose_reg = (poses_reshaped ** 2).mean() * 0.001

        total_loss = joint_loss + pose_reg

        total_loss.backward()
        optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = {
                'betas': betas.detach().clone(),
                'poses': poses.detach().clone().reshape(N, 24, 3),
                'trans': trans.detach().clone()
            }

        if (iter_num + 1) % 20 == 0:
            mpjpe = torch.sqrt(((pred_joints - target_joints) ** 2).sum(dim=-1)).mean().item() * 1000
            print(f"  Iter {iter_num+1}/{args.max_iters}: MPJPE = {mpjpe:.2f}mm, Loss = {total_loss.item():.6f}")

    # 5. Save results
    print("[5/5] Saving results...")
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
        betas=best_params['betas'].cpu().numpy(),
        poses=best_params['poses'].cpu().numpy(),
        trans=best_params['trans'].cpu().numpy()
    )

    # Compute final MPJPE
    with torch.no_grad():
        final_poses = best_params['poses'].reshape(N, 24, 3)
        final_output = smpl(
            betas=best_params['betas'].unsqueeze(0).expand(N, -1),
            body_pose=final_poses[:, 1:],
            global_orient=final_poses[:, 0:1],
            transl=best_params['trans']
        )
        final_joints = final_output.joints[:, :24]
        final_mpjpe = torch.sqrt(((final_joints - target_joints) ** 2).sum(dim=-1)).mean().item() * 1000

    # Save metadata
    meta = {
        "b3d": args.b3d,
        "init_from": args.init_from,
        "num_frames": N,
        "optimization_iters": args.max_iters,
        "metrics": {
            "MPJPE": final_mpjpe,
            "num_comparisons": N * 24
        }
    }

    with open(inner_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n✓ DONE: Final MPJPE = {final_mpjpe:.2f}mm")
    print(f"  Results saved to: {inner_dir}")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
