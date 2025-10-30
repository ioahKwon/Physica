#!/usr/bin/env python3
"""Summarize pilot test results"""
import json
import numpy as np
from pathlib import Path

results_dir = Path('/egr/research-zijunlab/kwonjoon/Output/output_full_Physica/pilot_results')

print("=" * 80)
print("FULL-FRAME PILOT TEST - COMPREHENSIVE RESULTS")
print("=" * 80)
print()

print("VERIFICATION #1: Full Frames Used (NOT truncated to 64)")
print("-" * 80)
print(f"{'Subject':<25} | {'Frames':>10} | {'MPJPE (mm)':>12} | {'Time (s)':>10}")
print("-" * 80)

completed = []
for subj_dir in sorted(results_dir.iterdir()):
    meta_file = subj_dir / 'meta.json'
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)

        name = subj_dir.name
        frames = meta.get('num_frames', 0)
        mpjpe = meta.get('MPJPE', 0.0)
        time = meta.get('optimization_time_seconds', 0.0)

        print(f"{name:<25} | {frames:>10} | {mpjpe:>12.2f} | {time:>10.1f}")
        completed.append((name, frames, mpjpe, time))

print()
print("VERIFICATION #2: SMPL Parameters Saved Correctly")
print("-" * 80)

for subj_dir in sorted(results_dir.iterdir()):
    npz_file = subj_dir / 'smpl_params.npz'
    if npz_file.exists():
        params = np.load(npz_file)
        name = subj_dir.name
        print(f"{name:<25} | betas={params['betas'].shape}, poses={params['poses'].shape}, trans={params['trans'].shape}")

print()
print("VERIFICATION #3: Optimization Speed (Aggressive Config)")
print("-" * 80)
print("Configuration: shape_iters=80 (vs 150 conservative)")
print("               pose_iters=50 (vs 80 conservative)")
print("               adaptive shape sampling (15% of frames, max 200)")
print()

if completed:
    avg_time_per_frame = sum(t / f for _, f, _, t in completed) / len(completed)
    print(f"Average time per frame: {avg_time_per_frame:.2f} seconds/frame")
    print(f"Completed: {len(completed)}/6 subjects")

print()
print("=" * 80)
print(f"Status: {len(completed)}/6 subjects completed")
print("=" * 80)
