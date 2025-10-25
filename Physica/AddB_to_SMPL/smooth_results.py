#!/usr/bin/env python3
"""
Post-processing script to apply temporal smoothing to SMPL fitting results.
Uses Savitzky-Golay filter for smooth motion while preserving accuracy.
"""

import argparse
import json
import numpy as np
import os
import time
from scipy.signal import savgol_filter
from pathlib import Path

def smooth_poses_and_trans(poses, trans, window_length=5, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth poses and translations.

    Args:
        poses: (T, 24, 3) numpy array
        trans: (T, 3) numpy array
        window_length: int, must be odd
        polyorder: int, polynomial order for fitting

    Returns:
        smoothed_poses, smoothed_trans
    """
    T = poses.shape[0]

    # Adjust window length if sequence is too short
    if T < window_length:
        window_length = T if T % 2 == 1 else T - 1
        if window_length < 3:
            print(f"  Warning: Sequence too short ({T} frames), skipping smoothing")
            return poses, trans

    # Smooth poses (flatten, smooth, reshape)
    poses_flat = poses.reshape(T, -1)  # (T, 24*3)
    poses_smooth_flat = np.zeros_like(poses_flat)

    for i in range(poses_flat.shape[1]):
        poses_smooth_flat[:, i] = savgol_filter(poses_flat[:, i], window_length, polyorder)

    poses_smooth = poses_smooth_flat.reshape(T, 24, 3)

    # Smooth translations
    trans_smooth = np.zeros_like(trans)
    for i in range(3):
        trans_smooth[:, i] = savgol_filter(trans[:, i], window_length, polyorder)

    return poses_smooth, trans_smooth


def compute_mpjpe(pred_joints, target_joints, joint_mapping):
    """Compute MPJPE between predicted and target joints."""
    errors = []
    for pred_frame, target_frame in zip(pred_joints, target_joints):
        for addb_idx, smpl_idx in joint_mapping.items():
            if addb_idx >= target_frame.shape[0]:
                continue
            tgt = target_frame[addb_idx]
            if np.any(np.isnan(tgt)):
                continue
            diff = pred_frame[smpl_idx] - tgt
            errors.append(np.linalg.norm(diff))

    return np.mean(errors) * 1000  # Convert to mm


def main():
    parser = argparse.ArgumentParser(description="Apply temporal smoothing to SMPL fitting results")
    parser.add_argument("--input_dir", required=True, help="Input directory with results")
    parser.add_argument("--output_dir", required=True, help="Output directory for smoothed results")
    parser.add_argument("--window_length", type=int, default=5, help="Smoothing window length (must be odd)")
    parser.add_argument("--polyorder", type=int, default=2, help="Polynomial order for Savitzky-Golay filter")
    args = parser.parse_args()

    print("\n" + "="*80)
    print("TEMPORAL SMOOTHING POST-PROCESSING")
    print("="*80)
    print(f"Input  : {args.input_dir}")
    print(f"Output : {args.output_dir}")
    print(f"Savitzky-Golay filter: window={args.window_length}, polyorder={args.polyorder}")
    print("="*80 + "\n")

    start_time = time.time()

    # Load original results
    input_path = Path(args.input_dir)
    smpl_params = np.load(input_path / "smpl_params.npz")
    poses = smpl_params["poses"]  # (T, 24, 3)
    trans = smpl_params["trans"]  # (T, 3)
    betas = smpl_params["betas"]  # (10,)

    pred_joints = np.load(input_path / "pred_joints.npy")  # (T, 24, 3)
    target_joints = np.load(input_path / "target_joints.npy")  # (T, 12, 3)

    with open(input_path / "meta.json", "r") as f:
        meta = json.load(f)

    print(f"Loaded results: {poses.shape[0]} frames")
    print(f"Original MPJPE: {meta['metrics']['MPJPE']:.2f} mm\n")

    # Apply smoothing
    print("Applying Savitzky-Golay filter...")
    poses_smooth, trans_smooth = smooth_poses_and_trans(poses, trans, args.window_length, args.polyorder)

    # Recompute joints with SMPL model (simplified - just save smoothed params)
    # For accurate MPJPE, would need to rerun SMPL forward pass
    print("Saving smoothed results...")

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save smoothed parameters
    np.savez(output_path / "smpl_params.npz",
             poses=poses_smooth,
             trans=trans_smooth,
             betas=betas)

    # Copy other files
    np.save(output_path / "pred_joints.npy", pred_joints)  # Would need SMPL forward pass to update
    np.save(output_path / "target_joints.npy", target_joints)

    # Update metadata
    meta["smoothing"] = {
        "method": "savitzky_golay",
        "window_length": args.window_length,
        "polyorder": args.polyorder,
        "original_mpjpe": meta["metrics"]["MPJPE"]
    }

    with open(output_path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed_time = time.time() - start_time

    print(f"\nSmoothing completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_path}")
    print("\nNote: For accurate MPJPE after smoothing, rerun SMPL forward pass")
    print("      Current pred_joints.npy contains original (unsmoothed) joint positions")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
