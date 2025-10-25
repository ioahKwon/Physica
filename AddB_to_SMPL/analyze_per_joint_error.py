#!/usr/bin/env python3
"""
Analyze per-joint errors to identify which joints contribute most to MPJPE.
"""
import numpy as np
import json
import argparse
from pathlib import Path

# SMPL joint names
SMPL_JOINT_NAMES = [
    'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
    'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
    'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder',
    'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hand', 'right_hand'
]

def analyze_per_joint_errors(result_dir):
    """Analyze per-joint errors from a result directory."""
    result_path = Path(result_dir)

    # Load data
    pred_joints = np.load(result_path / "pred_joints.npy")  # (T, 24, 3)
    target_joints = np.load(result_path / "target_joints.npy")  # (T, num_joints, 3)

    with open(result_path / "meta.json", "r") as f:
        meta = json.load(f)

    joint_mapping = meta["joint_mapping"]
    joint_mapping_named = meta.get("joint_mapping_named", {})

    print(f"\n{'='*80}")
    print(f"Per-Joint Error Analysis: {result_path.name}")
    print(f"{'='*80}")
    print(f"Overall MPJPE: {meta['metrics']['MPJPE']:.2f} mm\n")

    # Convert joint_mapping keys to int
    joint_mapping_int = {int(k): v for k, v in joint_mapping.items()}

    # Compute per-joint errors
    joint_errors = {}
    joint_counts = {}

    for t in range(pred_joints.shape[0]):
        pred_frame = pred_joints[t]
        target_frame = target_joints[t]

        for addb_idx, smpl_idx in joint_mapping_int.items():
            if addb_idx >= target_frame.shape[0]:
                continue

            tgt = target_frame[addb_idx]
            if np.any(np.isnan(tgt)):
                continue

            pred = pred_frame[smpl_idx]
            error = np.linalg.norm(pred - tgt) * 1000  # Convert to mm

            joint_name = SMPL_JOINT_NAMES[smpl_idx]

            if joint_name not in joint_errors:
                joint_errors[joint_name] = []
                joint_counts[joint_name] = 0

            joint_errors[joint_name].append(error)
            joint_counts[joint_name] += 1

    # Compute statistics
    joint_stats = []
    for joint_name, errors in joint_errors.items():
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        min_error = np.min(errors)

        joint_stats.append({
            'name': joint_name,
            'mean': mean_error,
            'std': std_error,
            'max': max_error,
            'min': min_error,
            'count': joint_counts[joint_name]
        })

    # Sort by mean error (descending)
    joint_stats.sort(key=lambda x: x['mean'], reverse=True)

    # Print results
    print(f"{'Joint Name':<20} {'Mean (mm)':<12} {'Std (mm)':<12} {'Max (mm)':<12} {'Min (mm)':<12} {'Count':<8}")
    print(f"{'-'*88}")

    for stat in joint_stats:
        print(f"{stat['name']:<20} {stat['mean']:>10.2f}   {stat['std']:>10.2f}   "
              f"{stat['max']:>10.2f}   {stat['min']:>10.2f}   {stat['count']:>6}")

    print(f"\n{'='*80}")
    print(f"Joints with highest error:")
    for i, stat in enumerate(joint_stats[:3], 1):
        print(f"  {i}. {stat['name']}: {stat['mean']:.2f} mm (±{stat['std']:.2f})")

    print(f"\nJoints with lowest error:")
    for i, stat in enumerate(joint_stats[-3:], 1):
        print(f"  {i}. {stat['name']}: {stat['mean']:.2f} mm (±{stat['std']:.2f})")

    print(f"{'='*80}\n")

    return joint_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze per-joint errors")
    parser.add_argument("--result_dir", required=True, help="Result directory path")
    args = parser.parse_args()

    analyze_per_joint_errors(args.result_dir)


if __name__ == "__main__":
    main()
