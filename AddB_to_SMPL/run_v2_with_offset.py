#!/usr/bin/env python3
"""
Run V2 baseline (99.74mm) with per-joint offset refinement added.
This takes the V2 results and adds Stage 3: Offset refinement.
"""
import numpy as np
import torch
import torch.nn as nn
import json
import os
import argparse
from addbiomechanics_to_smpl_v2 import (
    SMPLModel, load_b3d_sequence, infer_addb_joint_names,
    convert_addb_to_smpl_coords, auto_map_addb_to_smpl,
    load_mapping_overrides, resolve_mapping, SMPL_JOINT_NAMES,
    center_on_root_tensor, derive_run_name
)

SMPL_NUM_JOINTS = 24

def compute_mpjpe(pred_joints, target_joints, mapping):
    """Compute MPJPE between predicted and target joints"""
    errors = []
    for addb_idx, smpl_idx in mapping.items():
        pred = pred_joints[:, smpl_idx, :]
        target = target_joints[:, addb_idx, :]
        
        # Center on root
        pred_centered, target_centered = center_on_root_tensor(pred, target, 0)
        
        # Compute error
        diff = pred_centered - target_centered
        error = torch.sqrt(torch.sum(diff ** 2, dim=1))
        errors.append(error)
    
    all_errors = torch.cat(errors)
    mpjpe = torch.mean(all_errors) * 1000  # Convert to mm
    return mpjpe.item()

def refine_with_per_joint_offsets(smpl, betas, poses, trans, target_joints, mapping, device, args):
    """
    Stage 3: Per-joint offset refinement
    Add learnable offsets to each mapped joint
    """
    print("\n" + "="*80)
    print("STAGE 3: Per-Joint Offset Refinement")
    print("="*80)
    
    n_frames = poses.shape[0]
    
    # Initialize per-joint offsets (one offset vector per joint)
    offsets = nn.Parameter(torch.zeros(SMPL_NUM_JOINTS, 3, dtype=torch.float32, device=device))
    optimizer = torch.optim.Adam([offsets], lr=args.offset_lr)
    
    # Keep betas, poses, trans fixed
    betas_fixed = betas.detach()
    poses_fixed = poses.detach()
    trans_fixed = trans.detach()
    
    target_tensor = torch.from_numpy(target_joints).float().to(device)
    
    print(f"Offset learning rate: {args.offset_lr}")
    print(f"Offset iterations: {args.offset_iters}")
    print(f"Offset regularization weight: {args.offset_reg_weight}")
    
    for it in range(args.offset_iters):
        optimizer.zero_grad()
        total_loss = 0.0
        
        for t in range(n_frames):
            # Forward pass with fixed params
            joints_pred = smpl.joints(
                betas_fixed.unsqueeze(0),
                poses_fixed[t].unsqueeze(0),
                trans_fixed[t].unsqueeze(0)
            ).squeeze(0)  # (24, 3)
            
            # Add per-joint offsets
            joints_pred_with_offset = joints_pred + offsets
            
            # Compute loss only on mapped joints
            frame_loss = 0.0
            for addb_idx, smpl_idx in mapping.items():
                target_joint = target_tensor[t, addb_idx]
                if not torch.isnan(target_joint).any():
                    pred_centered, target_centered = center_on_root_tensor(
                        joints_pred_with_offset[smpl_idx],
                        target_joint,
                        None
                    )
                    diff = pred_centered - target_centered
                    frame_loss += torch.sum(diff ** 2)
            
            total_loss += frame_loss
        
        # Regularization: penalize large offsets
        offset_reg = args.offset_reg_weight * torch.sum(offsets ** 2)
        total_loss += offset_reg
        
        total_loss.backward()
        optimizer.step()
        
        if it % 10 == 0 or it == args.offset_iters - 1:
            offset_norm = torch.sqrt(torch.mean(offsets ** 2)).item()
            print(f"  Iter {it:3d}: loss={total_loss.item():.6f}, offset_norm={offset_norm:.4f}")
    
    # Apply offsets to get final joints
    final_joints = []
    with torch.no_grad():
        for t in range(n_frames):
            joints_pred = smpl.joints(
                betas_fixed.unsqueeze(0),
                poses_fixed[t].unsqueeze(0),
                trans_fixed[t].unsqueeze(0)
            ).squeeze(0)
            joints_with_offset = joints_pred + offsets
            final_joints.append(joints_with_offset.cpu().numpy())
    
    final_joints = np.array(final_joints)
    final_mpjpe = compute_mpjpe(torch.from_numpy(final_joints).to(device), target_tensor, mapping)
    
    print(f"\n✓ Final MPJPE with offsets: {final_mpjpe:.2f} mm")
    
    return final_joints, offsets.detach().cpu().numpy(), final_mpjpe

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', required=True)
    parser.add_argument('--smpl_model', required=True)
    parser.add_argument('--v2_results_dir', required=True, help='V2 baseline results directory')
    parser.add_argument('--out_dir', default='results_v2_with_offset')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--map_json', default=None)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--processing_pass', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--num_frames', type=int, default=-1)
    
    # Offset refinement hyperparameters
    parser.add_argument('--offset_lr', type=float, default=0.01)
    parser.add_argument('--offset_iters', type=int, default=50)
    parser.add_argument('--offset_reg_weight', type=float, default=1e-4)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("V2 Baseline + Per-Joint Offset Refinement")
    print("="*80)
    print(f"V2 results dir: {args.v2_results_dir}")
    print(f"Device: {device}")
    print("="*80)
    
    # Load V2 results
    print("\nLoading V2 baseline results...")
    v2_params = np.load(os.path.join(args.v2_results_dir, 'smpl_params.npz'))
    betas = torch.from_numpy(v2_params['betas']).float().to(device)
    poses = torch.from_numpy(v2_params['poses']).float().to(device)
    trans = torch.from_numpy(v2_params['trans']).float().to(device)
    
    with open(os.path.join(args.v2_results_dir, 'meta.json')) as f:
        meta = json.load(f)
    
    print(f"V2 baseline MPJPE: {meta['metrics']['MPJPE']:.2f} mm")
    
    # Load AddB data
    print("\nLoading AddBiomechanics data...")
    addb_joints, dt = load_b3d_sequence(args.b3d, args.trial, args.processing_pass, args.start, args.num_frames)
    joint_names = infer_addb_joint_names(args.b3d, args.trial, args.processing_pass)
    addb_joints = convert_addb_to_smpl_coords(addb_joints)
    
    # Resolve mapping
    auto_map = auto_map_addb_to_smpl(joint_names)
    overrides = load_mapping_overrides(args.map_json, joint_names)
    mapping = resolve_mapping(joint_names, auto_map, overrides)
    
    # Load SMPL model
    print("\nInitializing SMPL model...")
    smpl = SMPLModel(args.smpl_model, device=device)
    
    # Run offset refinement
    final_joints, offsets, final_mpjpe = refine_with_per_joint_offsets(
        smpl, betas, poses, trans, addb_joints, mapping, device, args
    )
    
    # Save results
    run_name = derive_run_name(args.b3d)
    out_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\nSaving results to: {out_dir}")
    
    np.savez(
        os.path.join(out_dir, 'smpl_params_with_offset.npz'),
        betas=betas.cpu().numpy(),
        poses=poses.cpu().numpy(),
        trans=trans.cpu().numpy(),
        offsets=offsets
    )
    
    np.save(os.path.join(out_dir, 'pred_joints_with_offset.npy'), final_joints)
    np.save(os.path.join(out_dir, 'target_joints.npy'), addb_joints)
    
    meta_with_offset = meta.copy()
    meta_with_offset['metrics_with_offset'] = {
        'MPJPE': final_mpjpe,
        'baseline_MPJPE': meta['metrics']['MPJPE'],
        'improvement_mm': meta['metrics']['MPJPE'] - final_mpjpe,
        'improvement_percent': (meta['metrics']['MPJPE'] - final_mpjpe) / meta['metrics']['MPJPE'] * 100
    }
    meta_with_offset['offset_hyperparameters'] = {
        'lr': args.offset_lr,
        'iters': args.offset_iters,
        'reg_weight': args.offset_reg_weight
    }
    
    with open(os.path.join(out_dir, 'meta_with_offset.json'), 'w') as f:
        json.dump(meta_with_offset, f, indent=2)
    
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    print(f"V2 Baseline MPJPE:     {meta['metrics']['MPJPE']:.2f} mm")
    print(f"With Offsets MPJPE:    {final_mpjpe:.2f} mm")
    print(f"Improvement:           {meta['metrics']['MPJPE'] - final_mpjpe:.2f} mm ({(meta['metrics']['MPJPE'] - final_mpjpe) / meta['metrics']['MPJPE'] * 100:.1f}%)")
    print("="*80)

if __name__ == '__main__':
    main()
