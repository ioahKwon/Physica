#!/usr/bin/env python3
"""
Evaluate SMPL Body Proportions Accuracy

Compares body segment lengths between AddBiomechanics skeleton and SMPL mesh:
- Torso length (pelvis to spine3/neck)
- Leg length (hip to ankle)
- Foot height (ankle to foot tip)
- Arm length (shoulder to wrist) - With_Arm only
- Shoulder width (left shoulder to right shoulder)

Metrics:
- MAE for each body segment
- Relative error (percentage)
- Distribution of errors

Outputs:
- Body proportion accuracy statistics
- Comparison plots for No_Arm vs With_Arm
- JSON report with per-subject metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


# SMPL joint indices (from addbiomechanics_to_smpl_v3_enhanced.py)
SMPL_JOINT_IDS = {
    'pelvis': 0, 'left_hip': 1, 'right_hip': 2, 'spine1': 3,
    'left_knee': 4, 'right_knee': 5, 'spine2': 6,
    'left_ankle': 7, 'right_ankle': 8, 'spine3': 9,
    'left_foot': 10, 'right_foot': 11, 'neck': 12,
    'left_collar': 13, 'right_collar': 14, 'head': 15,
    'left_shoulder': 16, 'right_shoulder': 17,
    'left_elbow': 18, 'right_elbow': 19,
    'left_wrist': 20, 'right_wrist': 21,
    'left_hand': 22, 'right_hand': 23
}


def compute_segment_length(joints: np.ndarray, start_idx: int, end_idx: int) -> float:
    """
    Compute Euclidean distance between two joints.

    Args:
        joints: (N, 3) array of joint positions
        start_idx: Index of start joint
        end_idx: Index of end joint

    Returns:
        Length in meters
    """
    if start_idx >= len(joints) or end_idx >= len(joints):
        return None
    start = joints[start_idx]
    end = joints[end_idx]
    if np.any(np.isnan(start)) or np.any(np.isnan(end)):
        return None
    return np.linalg.norm(end - start)


def compute_body_proportions(joints: np.ndarray, include_arms: bool = True) -> Dict[str, float]:
    """
    Compute body segment lengths from joint positions.

    Args:
        joints: (24, 3) array of SMPL joint positions
        include_arms: Whether to compute arm measurements

    Returns:
        Dictionary of segment lengths in meters
    """
    proportions = {}

    # Torso length (pelvis to spine3)
    torso_length = compute_segment_length(joints,
                                         SMPL_JOINT_IDS['pelvis'],
                                         SMPL_JOINT_IDS['spine3'])
    if torso_length is not None:
        proportions['torso_length'] = torso_length

    # Left leg length (left_hip to left_ankle)
    left_leg = compute_segment_length(joints,
                                      SMPL_JOINT_IDS['left_hip'],
                                      SMPL_JOINT_IDS['left_ankle'])
    if left_leg is not None:
        proportions['left_leg_length'] = left_leg

    # Right leg length (right_hip to right_ankle)
    right_leg = compute_segment_length(joints,
                                       SMPL_JOINT_IDS['right_hip'],
                                       SMPL_JOINT_IDS['right_ankle'])
    if right_leg is not None:
        proportions['right_leg_length'] = right_leg

    # Average leg length
    if left_leg is not None and right_leg is not None:
        proportions['avg_leg_length'] = (left_leg + right_leg) / 2

    # Left foot height (left_ankle to left_foot)
    left_foot = compute_segment_length(joints,
                                       SMPL_JOINT_IDS['left_ankle'],
                                       SMPL_JOINT_IDS['left_foot'])
    if left_foot is not None:
        proportions['left_foot_height'] = left_foot

    # Right foot height (right_ankle to right_foot)
    right_foot = compute_segment_length(joints,
                                        SMPL_JOINT_IDS['right_ankle'],
                                        SMPL_JOINT_IDS['right_foot'])
    if right_foot is not None:
        proportions['right_foot_height'] = right_foot

    # Shoulder width (left_shoulder to right_shoulder)
    shoulder_width = compute_segment_length(joints,
                                            SMPL_JOINT_IDS['left_shoulder'],
                                            SMPL_JOINT_IDS['right_shoulder'])
    if shoulder_width is not None:
        proportions['shoulder_width'] = shoulder_width

    if include_arms:
        # Left arm length (left_shoulder to left_wrist)
        left_arm = compute_segment_length(joints,
                                          SMPL_JOINT_IDS['left_shoulder'],
                                          SMPL_JOINT_IDS['left_wrist'])
        if left_arm is not None:
            proportions['left_arm_length'] = left_arm

        # Right arm length (right_shoulder to right_wrist)
        right_arm = compute_segment_length(joints,
                                           SMPL_JOINT_IDS['right_shoulder'],
                                           SMPL_JOINT_IDS['right_wrist'])
        if right_arm is not None:
            proportions['right_arm_length'] = right_arm

        # Average arm length
        if left_arm is not None and right_arm is not None:
            proportions['avg_arm_length'] = (left_arm + right_arm) / 2

    return proportions


def evaluate_single_subject(subject_dir: Path, include_arms: bool = True) -> Dict:
    """
    Evaluate body proportions for a single subject.

    Args:
        subject_dir: Directory containing SMPL results
        include_arms: Whether to evaluate arm proportions

    Returns:
        Dictionary with evaluation metrics
    """
    # Load AddB original joints (ground truth)
    addb_joints_file = subject_dir / "original_joints.npy"
    if not addb_joints_file.exists():
        print(f"Warning: original_joints.npy not found in {subject_dir}")
        return None

    addb_joints = np.load(addb_joints_file)  # (T, N, 3)

    # Load SMPL predicted joints
    smpl_joints_file = subject_dir / "pred_joints.npy"
    if not smpl_joints_file.exists():
        print(f"Warning: pred_joints.npy not found in {subject_dir}")
        return None

    smpl_joints = np.load(smpl_joints_file)  # (T, 24, 3)

    # Compute proportions for each frame and average
    addb_proportions_list = []
    smpl_proportions_list = []

    for t in range(len(smpl_joints)):
        # AddB proportions (use same SMPL joint indices since they're mapped)
        addb_props = compute_body_proportions(addb_joints[t], include_arms=include_arms)
        smpl_props = compute_body_proportions(smpl_joints[t], include_arms=include_arms)

        if addb_props and smpl_props:
            addb_proportions_list.append(addb_props)
            smpl_proportions_list.append(smpl_props)

    if not addb_proportions_list:
        return None

    # Average across frames
    result = {'subject_name': subject_dir.name}

    # Get all proportion keys
    all_keys = set()
    for props in addb_proportions_list:
        all_keys.update(props.keys())

    # Compute mean and error for each proportion
    for key in all_keys:
        addb_values = [p[key] for p in addb_proportions_list if key in p]
        smpl_values = [p[key] for p in smpl_proportions_list if key in p]

        if addb_values and smpl_values:
            addb_mean = np.mean(addb_values)
            smpl_mean = np.mean(smpl_values)
            error = abs(smpl_mean - addb_mean)
            relative_error = (error / addb_mean) * 100 if addb_mean > 0 else 0

            result[f'addb_{key}'] = addb_mean
            result[f'smpl_{key}'] = smpl_mean
            result[f'{key}_error'] = error
            result[f'{key}_relative_error'] = relative_error

    return result


def evaluate_dataset(results_dir: Path, include_arms: bool = True) -> List[Dict]:
    """
    Evaluate body proportions for all subjects in a dataset.

    Args:
        results_dir: Directory containing SMPL results
        include_arms: Whether to evaluate arm proportions

    Returns:
        List of evaluation results
    """
    results = []

    for subject_dir in sorted(results_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        result = evaluate_single_subject(subject_dir, include_arms=include_arms)
        if result is not None:
            results.append(result)
            print(f"Processed {subject_dir.name}")

    return results


def compute_statistics(results: List[Dict], proportion_keys: List[str]) -> Dict:
    """
    Compute summary statistics for body proportions.

    Args:
        results: List of evaluation results
        proportion_keys: List of proportion names to analyze

    Returns:
        Dictionary with summary statistics
    """
    stats = {'count': len(results)}

    for key in proportion_keys:
        error_key = f'{key}_error'
        rel_error_key = f'{key}_relative_error'

        errors = [r[error_key] for r in results if error_key in r]
        rel_errors = [r[rel_error_key] for r in results if rel_error_key in r]

        if errors:
            stats[f'{key}_mae'] = np.mean(errors)
            stats[f'{key}_median_error'] = np.median(errors)
            stats[f'{key}_min_error'] = np.min(errors)
            stats[f'{key}_max_error'] = np.max(errors)
            stats[f'{key}_mean_relative_error'] = np.mean(rel_errors)

    return stats


def plot_comparison(no_arm_results: List[Dict],
                   with_arm_results: List[Dict],
                   output_path: str):
    """
    Create comparison plots for body proportions.

    Args:
        no_arm_results: Results for No_Arm dataset
        with_arm_results: Results for With_Arm dataset
        output_path: Path to save the plot
    """
    # Common proportions for both datasets
    common_props = ['torso_length', 'avg_leg_length', 'shoulder_width']
    # Additional proportions for With_Arm
    arm_props = ['avg_arm_length']

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('SMPL Body Proportions Accuracy Evaluation', fontsize=16, fontweight='bold')

    # 1. Torso length error
    ax = axes[0, 0]
    no_arm_errors = np.array([r['torso_length_error'] * 100 for r in no_arm_results if 'torso_length_error' in r])
    with_arm_errors = np.array([r['torso_length_error'] * 100 for r in with_arm_results if 'torso_length_error' in r])
    ax.hist(no_arm_errors, bins=20, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_errors, bins=20, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Torso Length Error (cm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Torso Length Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Leg length error
    ax = axes[0, 1]
    no_arm_errors = np.array([r['avg_leg_length_error'] * 100 for r in no_arm_results if 'avg_leg_length_error' in r])
    with_arm_errors = np.array([r['avg_leg_length_error'] * 100 for r in with_arm_results if 'avg_leg_length_error' in r])
    ax.hist(no_arm_errors, bins=20, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_errors, bins=20, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Leg Length Error (cm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Leg Length Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Shoulder width error
    ax = axes[0, 2]
    no_arm_errors = np.array([r['shoulder_width_error'] * 100 for r in no_arm_results if 'shoulder_width_error' in r])
    with_arm_errors = np.array([r['shoulder_width_error'] * 100 for r in with_arm_results if 'shoulder_width_error' in r])
    ax.hist(no_arm_errors, bins=20, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_errors, bins=20, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Shoulder Width Error (cm)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Shoulder Width Error', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Arm length error (With_Arm only)
    ax = axes[1, 0]
    with_arm_arm_errors = np.array([r['avg_arm_length_error'] * 100 for r in with_arm_results if 'avg_arm_length_error' in r])
    if len(with_arm_arm_errors) > 0:
        ax.hist(with_arm_arm_errors, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax.set_xlabel('Arm Length Error (cm)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Arm Length Error (With_Arm only)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No arm data\navailable', ha='center', va='center', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])

    # 5. Box plot comparison
    ax = axes[1, 1]
    no_arm_stats = compute_statistics(no_arm_results, common_props)
    with_arm_stats = compute_statistics(with_arm_results, common_props + arm_props)

    prop_names = ['Torso', 'Leg', 'Shoulder']
    no_arm_maes = [no_arm_stats.get(f'{p}_mae', 0) * 100 for p in ['torso_length', 'avg_leg_length', 'shoulder_width']]
    with_arm_maes = [with_arm_stats.get(f'{p}_mae', 0) * 100 for p in ['torso_length', 'avg_leg_length', 'shoulder_width']]

    x = np.arange(len(prop_names))
    width = 0.35
    ax.bar(x - width/2, no_arm_maes, width, label='No_Arm', alpha=0.7, color='blue', edgecolor='black')
    ax.bar(x + width/2, with_arm_maes, width, label='With_Arm', alpha=0.7, color='green', edgecolor='black')
    ax.set_ylabel('MAE (cm)', fontsize=11)
    ax.set_title('Mean Absolute Error by Segment', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(prop_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 6. Statistics table
    ax = axes[1, 2]
    ax.axis('off')

    table_data = [
        ['Segment', 'No_Arm MAE', 'With_Arm MAE'],
        ['Torso (cm)', f"{no_arm_stats.get('torso_length_mae', 0)*100:.2f}",
         f"{with_arm_stats.get('torso_length_mae', 0)*100:.2f}"],
        ['Leg (cm)', f"{no_arm_stats.get('avg_leg_length_mae', 0)*100:.2f}",
         f"{with_arm_stats.get('avg_leg_length_mae', 0)*100:.2f}"],
        ['Shoulder (cm)', f"{no_arm_stats.get('shoulder_width_mae', 0)*100:.2f}",
         f"{with_arm_stats.get('shoulder_width_mae', 0)*100:.2f}"],
        ['Arm (cm)', 'N/A', f"{with_arm_stats.get('avg_arm_length_mae', 0)*100:.2f}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.35, 0.325, 0.325])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SMPL body proportions accuracy')
    parser.add_argument('--results_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/filtered_results',
                       help='Directory containing filtered results')
    parser.add_argument('--output_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica',
                       help='Directory to save evaluation outputs')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("SMPL Body Proportions Accuracy Evaluation")
    print("=" * 80)

    # Evaluate No_Arm dataset
    print("\nEvaluating No_Arm dataset...")
    no_arm_results = evaluate_dataset(results_dir / "No_Arm", include_arms=False)

    common_props = ['torso_length', 'avg_leg_length', 'shoulder_width']
    no_arm_stats = compute_statistics(no_arm_results, common_props)

    print(f"\nNo_Arm Statistics:")
    print(f"  Count: {no_arm_stats['count']}")
    for prop in common_props:
        mae_key = f'{prop}_mae'
        if mae_key in no_arm_stats:
            print(f"  {prop} MAE: {no_arm_stats[mae_key]*100:.2f} cm")

    # Evaluate With_Arm dataset
    print("\nEvaluating With_Arm dataset...")
    with_arm_results = evaluate_dataset(results_dir / "With_Arm", include_arms=True)

    all_props = common_props + ['avg_arm_length']
    with_arm_stats = compute_statistics(with_arm_results, all_props)

    print(f"\nWith_Arm Statistics:")
    print(f"  Count: {with_arm_stats['count']}")
    for prop in all_props:
        mae_key = f'{prop}_mae'
        if mae_key in with_arm_stats:
            print(f"  {prop} MAE: {with_arm_stats[mae_key]*100:.2f} cm")

    # Save detailed results
    output_data = {
        'no_arm': {
            'statistics': no_arm_stats,
            'per_subject': no_arm_results
        },
        'with_arm': {
            'statistics': with_arm_stats,
            'per_subject': with_arm_results
        }
    }

    json_path = output_dir / "body_proportions_evaluation.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nDetailed results saved to: {json_path}")

    # Create comparison plot
    plot_path = output_dir / "body_proportions_comparison.png"
    plot_comparison(no_arm_results, with_arm_results, str(plot_path))

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
