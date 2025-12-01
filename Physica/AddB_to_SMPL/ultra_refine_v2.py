#!/usr/bin/env python3
"""
Robust refinement script - Addresses "optimization too fast" problem
Strategy: Slower learning rates + longer optimization + stronger regularization
Goal: 70-100mm MPJPE → under 50mm
"""

import torch
import numpy as np
import nimblephysics as nimble
from pathlib import Path
import argparse
import json
import sys
import time

# Import from existing code
sys.path.insert(0, str(Path(__file__).parent))
from models.smpl_model import SMPLModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str, required=True)
    parser.add_argument('--init_from', type=str, required=True, help='Previous result directory')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--smpl_model', type=str, default='models/smpl_model.pkl')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--max_iters', type=int, default=500, help='Total iterations for refinement (default: 500, 5× more than simple_refine)')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    args = parser.parse_args()

    print("\n" + "="*80)
    print("ROBUST REFINEMENT - Slow, Stable, Thorough Optimization")
    print("="*80)
    print(f"B3D: {args.b3d}")
    print(f"Init from: {args.init_from}")
    print(f"Output: {args.out_dir}")
    print(f"Max iterations: {args.max_iters} (vs 100 in simple_refine)")
    print(f"Early stopping patience: {args.patience} iters")
    print("="*80 + "\n")

    device = torch.device(args.device)
    start_time = time.time()

    # 1. Load warm-start SMPL parameters
    print("[1/6] Loading warm-start SMPL parameters...")
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

    # Load old MPJPE
    meta_file = inner_dirs[0] / "meta.json"
    with open(meta_file, 'r') as f:
        meta = json.load(f)
    old_mpjpe = meta.get('metrics', {}).get('MPJPE', 0)
    print(f"  Previous MPJPE: {old_mpjpe:.2f}mm")

    # 2. Load AddBiomechanics B3D file
    print("[2/6] Loading AddBiomechanics B3D file...")
    subject = nimble.biomechanics.SubjectOnDisk(args.b3d)
    num_frames_b3d = subject.getTrialLength(trial=0)
    print(f"  B3D has {num_frames_b3d} frames")

    # Sample up to 200 frames
    if num_frames_b3d > 200:
        indices = np.linspace(0, num_frames_b3d-1, 200, dtype=int)
    else:
        indices = np.arange(num_frames_b3d)

    # Load target markers
    print("  Loading marker observations...")
    markers_list = []
    for idx in indices:
        frames = subject.readFrames(
            trial=0,
            startFrame=int(idx),
            numFramesToRead=1,
            includeProcessingPasses=True,
            includeSensorData=False,
            stride=1
        )
        if frames:
            marker_obs = frames[0].markerObservations
            # markerObservations is a list of 3D positions
            if isinstance(marker_obs, list):
                markers = np.array(marker_obs)
                # Ensure it's (N, 3) shape
                if markers.ndim == 1:
                    markers = markers.reshape(-1, 3)
            else:
                # If it's a dict, extract values
                markers = np.array([pos for pos in marker_obs.values()])
            markers_list.append(markers)

    N = len(markers_list)
    print(f"  Loaded {N} frames of markers")

    # Convert to tensors - use markers directly as joint targets
    # (This is simplified - in production, you'd do proper marker-to-joint mapping)
    max_markers = max(m.shape[0] for m in markers_list)
    target_joints = torch.zeros(N, 24, 3).to(device)
    for i, markers in enumerate(markers_list):
        # Use first 24 markers as joint targets (simplified)
        n_use = min(markers.shape[0], 24)
        target_joints[i, :n_use] = torch.from_numpy(markers[:n_use]).float()

    print(f"  Target joints shape: {target_joints.shape}")

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
    print("[3/6] Loading SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)

    # 4. Setup optimizer with SLOWER learning rates
    print(f"[4/6] Setting up ROBUST optimizer (slower LR, longer schedule)...")

    # Prepare optimizable parameters
    betas = init_betas.clone().detach().requires_grad_(True)
    poses = init_poses.clone().detach().reshape(N, -1).requires_grad_(True)  # (N, 72)
    trans = init_trans.clone().detach().requires_grad_(True)

    # MUCH SLOWER learning rates than simple_refine (0.01/0.001/0.001)
    optimizer = torch.optim.Adam([
        {'params': [betas], 'lr': 0.002},    # 5× slower
        {'params': [poses], 'lr': 0.0002},   # 5× slower
        {'params': [trans], 'lr': 0.0002}    # 5× slower
    ])

    # Learning rate scheduler - reduce on plateau
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, verbose=True
    )

    best_loss = float('inf')
    best_params = None
    no_improve_count = 0

    # 5. Optimize with robust strategy
    print(f"[5/6] Running ROBUST optimization ({args.max_iters} max iterations)...")
    print("  Strategy:")
    print("    • 5× slower learning rates (0.002/0.0002/0.0002 vs 0.01/0.001/0.001)")
    print("    • 5× more iterations (500 vs 100)")
    print("    • Learning rate decay on plateau")
    print("    • Early stopping with patience=50")
    print("    • Stronger pose regularization")
    print("")

    for iter_num in range(args.max_iters):
        optimizer.zero_grad()

        # Forward pass
        poses_reshaped = poses.reshape(N, 24, 3)
        # Need to expand betas for each frame
        betas_expanded = betas.unsqueeze(0).expand(N, -1)
        vertices, joints = smpl.forward(
            betas=betas_expanded,
            poses=poses_reshaped,
            trans=trans
        )
        pred_joints = joints[:, :24]  # (N, 24, 3)

        # Loss: Joint position error
        joint_loss = ((pred_joints - target_joints) ** 2).mean()

        # STRONGER pose regularization (2× stronger than simple_refine)
        pose_reg = (poses_reshaped ** 2).mean() * 0.002

        # Velocity smoothness (new: penalize sudden changes between frames)
        if N > 1:
            pose_velocity = poses_reshaped[1:] - poses_reshaped[:-1]
            velocity_loss = (pose_velocity ** 2).mean() * 0.0005
        else:
            velocity_loss = 0.0

        total_loss = joint_loss + pose_reg + velocity_loss

        total_loss.backward()
        optimizer.step()

        # Track best
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_params = {
                'betas': betas.detach().clone(),
                'poses': poses.detach().clone().reshape(N, 24, 3),
                'trans': trans.detach().clone()
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        # Learning rate scheduling
        scheduler.step(total_loss)

        # Early stopping
        if no_improve_count >= args.patience:
            print(f"  Early stopping at iter {iter_num+1}: No improvement for {args.patience} iters")
            break

        # Progress logging
        if (iter_num + 1) % 50 == 0 or iter_num == 0:
            mpjpe = torch.sqrt(((pred_joints - target_joints) ** 2).sum(dim=-1)).mean().item() * 1000
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Iter {iter_num+1:4d}/{args.max_iters}: MPJPE = {mpjpe:6.2f}mm, Loss = {total_loss.item():.6f}, LR = {current_lr:.6f}")

    # 6. Save results
    print("[6/6] Saving results...")
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
        final_betas = best_params['betas'].unsqueeze(0).expand(N, -1)
        final_vertices, final_joints = smpl.forward(
            betas=final_betas,
            poses=final_poses,
            trans=best_params['trans']
        )
        final_joints_24 = final_joints[:, :24]
        final_mpjpe = torch.sqrt(((final_joints_24 - target_joints) ** 2).sum(dim=-1)).mean().item() * 1000

    elapsed_time = time.time() - start_time

    # Save metadata
    meta = {
        "b3d": args.b3d,
        "init_from": args.init_from,
        "num_frames": N,
        "optimization_iters": iter_num + 1,
        "optimization_time_seconds": elapsed_time,
        "early_stopped": no_improve_count >= args.patience,
        "strategy": "robust_refine",
        "learning_rates": {
            "betas": 0.002,
            "poses": 0.0002,
            "trans": 0.0002
        },
        "metrics": {
            "MPJPE": final_mpjpe,
            "previous_MPJPE": old_mpjpe,
            "improvement_mm": old_mpjpe - final_mpjpe,
            "improvement_percent": ((old_mpjpe - final_mpjpe) / old_mpjpe * 100) if old_mpjpe > 0 else 0,
            "num_comparisons": N * 24
        }
    }

    with open(inner_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    improvement = old_mpjpe - final_mpjpe
    improvement_pct = (improvement / old_mpjpe * 100) if old_mpjpe > 0 else 0

    print(f"\n{'='*80}")
    print("ROBUST REFINEMENT COMPLETE")
    print(f"{'='*80}")
    print(f"  Previous MPJPE: {old_mpjpe:.2f}mm")
    print(f"  Final MPJPE:    {final_mpjpe:.2f}mm")
    print(f"  Improvement:    {improvement:+.2f}mm ({improvement_pct:+.1f}%)")
    print(f"  Iterations:     {iter_num + 1}/{args.max_iters}")
    print(f"  Time:           {elapsed_time:.1f}s")
    print(f"  Results:        {inner_dir}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
