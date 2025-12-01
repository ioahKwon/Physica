#!/usr/bin/env python3
"""
Analyze and compare FIX 5 results
- Measure pelvis rotation magnitude
- Measure upper body forward lean (spine X position)
- Compare foot direction stability
"""

import numpy as np
import json
from pathlib import Path

def analyze_results(result_dir):
    """Analyze a single result directory"""
    result_path = Path(result_dir)

    # Load SMPL parameters
    smpl_file = result_path / "smpl_params.npz"
    if not smpl_file.exists():
        # Check subdirectories
        subdirs = list(result_path.glob("*/smpl_params.npz"))
        if subdirs:
            smpl_file = subdirs[0]
        else:
            print(f"No SMPL params found in {result_dir}")
            return None

    data = np.load(smpl_file)
    poses = data['poses']  # [T, 72] or [T, 24, 3]

    # Reshape if needed
    if len(poses.shape) == 2:
        poses = poses.reshape(-1, 24, 3)

    # Pelvis rotation (joint 0)
    pelvis_rot = poses[:, 0, :]  # [T, 3]
    pelvis_mag = np.linalg.norm(pelvis_rot, axis=1)  # [T]

    # Average pelvis rotation magnitude
    avg_pelvis_rot = np.mean(pelvis_mag)
    max_pelvis_rot = np.max(pelvis_mag)

    # Pelvis pitch (X-axis) and roll (Z-axis)
    avg_pitch = np.mean(np.abs(pelvis_rot[:, 0]))
    avg_roll = np.mean(np.abs(pelvis_rot[:, 2]))

    # Load meta.json for MPJPE
    meta_file = smpl_file.parent / "meta.json"
    mpjpe = None
    if meta_file.exists():
        with open(meta_file, 'r') as f:
            meta = json.load(f)
            mpjpe = meta.get('MPJPE', None)

    return {
        'path': str(result_dir),
        'mpjpe': mpjpe,
        'avg_pelvis_rotation_deg': np.degrees(avg_pelvis_rot),
        'max_pelvis_rotation_deg': np.degrees(max_pelvis_rot),
        'avg_pelvis_pitch_deg': np.degrees(avg_pitch),
        'avg_pelvis_roll_deg': np.degrees(avg_roll),
        'num_frames': len(poses)
    }

def main():
    base_dir = Path("/tmp/v5_comparison")

    options = {
        "Option 1: Pelvis Upright": base_dir / "v5_option1_pelvis_constraint",
        "Option 2: Spine Mapping": base_dir / "v5_option2_spine_mapping",
        "Option 3: Coord Fix": base_dir / "v5_option3_coord_fix",
        "Option 4: ALL FIXES": base_dir / "v5_option4_all_fixes",
    }

    print("=" * 80)
    print("FIX 5 (v5) Results Analysis")
    print("=" * 80)
    print()

    results = {}
    for name, path in options.items():
        print(f"Analyzing {name}...")
        result = analyze_results(path)
        if result:
            results[name] = result
            mpjpe_str = f"{result['mpjpe']:.2f}" if result['mpjpe'] is not None else "N/A"
            print(f"  MPJPE: {mpjpe_str} mm")
            print(f"  Avg Pelvis Rotation: {result['avg_pelvis_rotation_deg']:.1f}°")
            print(f"  Max Pelvis Rotation: {result['max_pelvis_rotation_deg']:.1f}°")
            print(f"  Avg Pelvis Pitch: {result['avg_pelvis_pitch_deg']:.1f}°")
            print(f"  Avg Pelvis Roll: {result['avg_pelvis_roll_deg']:.1f}°")
            print()

    print("=" * 80)
    print("Comparison Table")
    print("=" * 80)
    print()
    print(f"{'Option':<25} {'MPJPE':>10} {'Avg Pelvis':>12} {'Max Pelvis':>12} {'Pitch':>10} {'Roll':>10}")
    print(f"{'':25} {'(mm)':>10} {'Rot (°)':>12} {'Rot (°)':>12} {'(°)':>10} {'(°)':>10}")
    print("-" * 80)

    for name, result in results.items():
        mpjpe_str = f"{result['mpjpe']:>10.2f}" if result['mpjpe'] is not None else f"{'N/A':>10}"
        print(f"{name:<25} {mpjpe_str} {result['avg_pelvis_rotation_deg']:>12.1f} "
              f"{result['max_pelvis_rotation_deg']:>12.1f} {result['avg_pelvis_pitch_deg']:>10.1f} "
              f"{result['avg_pelvis_roll_deg']:>10.1f}")

    print()
    print("=" * 80)
    print("Key Observations:")
    print("=" * 80)

    # Find best MPJPE
    best_mpjpe = min(results.values(), key=lambda x: x['mpjpe'])
    print(f"\n✓ Best MPJPE: {list(results.keys())[list(results.values()).index(best_mpjpe)]} "
          f"({best_mpjpe['mpjpe']:.2f} mm)")

    # Find lowest pelvis rotation
    lowest_pelvis = min(results.values(), key=lambda x: x['avg_pelvis_rotation_deg'])
    print(f"✓ Lowest Pelvis Rotation: {list(results.keys())[list(results.values()).index(lowest_pelvis)]} "
          f"({lowest_pelvis['avg_pelvis_rotation_deg']:.1f}°)")

    # Find lowest pitch (forward lean indicator)
    lowest_pitch = min(results.values(), key=lambda x: x['avg_pelvis_pitch_deg'])
    print(f"✓ Lowest Forward Lean (Pitch): {list(results.keys())[list(results.values()).index(lowest_pitch)]} "
          f"({lowest_pitch['avg_pelvis_pitch_deg']:.1f}°)")

    print()

if __name__ == '__main__':
    main()
