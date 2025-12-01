#!/usr/bin/env python3
"""
Evaluate SMPL Shape Consistency across Frames

Measures temporal consistency of shape parameters (beta) for each subject.
A subject's body shape should remain constant across all frames.

Metrics:
- Per-subject std of beta across frames
- Mean std across all 10 beta dimensions
- Distribution of consistency scores

Outputs:
- Shape consistency statistics
- Comparison plots for No_Arm vs With_Arm
- JSON report with per-subject metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import argparse


def evaluate_shape_consistency(subject_dir: Path) -> Dict:
    """
    Evaluate shape consistency for a single subject.

    Args:
        subject_dir: Directory containing SMPL results

    Returns:
        Dictionary with consistency metrics
    """
    # Load shape parameters from smpl_params.npz
    smpl_params_file = subject_dir / "smpl_params.npz"
    if not smpl_params_file.exists():
        print(f"Warning: smpl_params.npz not found in {subject_dir}")
        return None

    try:
        smpl_params = np.load(smpl_params_file)
        betas = smpl_params['betas']  # (T, 10) or (10,)
    except Exception as e:
        print(f"Warning: Could not load betas from {smpl_params_file}: {e}")
        return None

    # Handle both cases: single beta or per-frame betas
    if betas.ndim == 1:
        # Single beta for all frames - perfect consistency
        return {
            'subject_name': subject_dir.name,
            'num_frames': 1,
            'mean_std': 0.0,
            'max_std': 0.0,
            'per_beta_std': [0.0] * 10,
            'perfect_consistency': True
        }

    num_frames = len(betas)

    # Compute std for each beta dimension
    per_beta_std = np.std(betas, axis=0)  # (10,)

    # Overall consistency metrics
    mean_std = np.mean(per_beta_std)
    max_std = np.max(per_beta_std)

    return {
        'subject_name': subject_dir.name,
        'num_frames': num_frames,
        'mean_std': float(mean_std),
        'max_std': float(max_std),
        'per_beta_std': per_beta_std.tolist(),
        'perfect_consistency': False
    }


def evaluate_dataset(results_dir: Path) -> List[Dict]:
    """
    Evaluate shape consistency for all subjects in a dataset.

    Args:
        results_dir: Directory containing SMPL results

    Returns:
        List of evaluation results
    """
    results = []

    for subject_dir in sorted(results_dir.iterdir()):
        if not subject_dir.is_dir():
            continue

        result = evaluate_shape_consistency(subject_dir)
        if result is not None:
            results.append(result)
            print(f"Processed {subject_dir.name}: "
                  f"Mean_std={result['mean_std']:.4f}, "
                  f"Max_std={result['max_std']:.4f}")

    return results


def compute_statistics(results: List[Dict]) -> Dict:
    """
    Compute summary statistics from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with summary statistics
    """
    if not results:
        return {
            'count': 0,
            'mean_of_mean_std': 0.0,
            'median_of_mean_std': 0.0,
            'min_of_mean_std': 0.0,
            'max_of_mean_std': 0.0,
            'mean_of_max_std': 0.0,
            'median_of_max_std': 0.0,
            'per_beta_mean_std': [0.0] * 10
        }

    mean_stds = [r['mean_std'] for r in results]
    max_stds = [r['max_std'] for r in results]

    # Compute per-beta statistics
    per_beta_means = []
    for beta_idx in range(10):
        beta_stds = [r['per_beta_std'][beta_idx] for r in results]
        per_beta_means.append(np.mean(beta_stds))

    return {
        'count': len(results),
        'mean_of_mean_std': np.mean(mean_stds),
        'median_of_mean_std': np.median(mean_stds),
        'min_of_mean_std': np.min(mean_stds),
        'max_of_mean_std': np.max(mean_stds),
        'mean_of_max_std': np.mean(max_stds),
        'median_of_max_std': np.median(max_stds),
        'per_beta_mean_std': per_beta_means
    }


def plot_comparison(no_arm_results: List[Dict],
                   with_arm_results: List[Dict],
                   output_path: str):
    """
    Create comparison plots for shape consistency.

    Args:
        no_arm_results: Results for No_Arm dataset
        with_arm_results: Results for With_Arm dataset
        output_path: Path to save the plot
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    fig.suptitle('SMPL Shape Consistency Evaluation', fontsize=16, fontweight='bold')

    # Extract consistency metrics
    no_arm_mean_stds = np.array([r['mean_std'] for r in no_arm_results])
    with_arm_mean_stds = np.array([r['mean_std'] for r in with_arm_results])

    no_arm_max_stds = np.array([r['max_std'] for r in no_arm_results])
    with_arm_max_stds = np.array([r['max_std'] for r in with_arm_results])

    # 1. Histogram of mean std
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(no_arm_mean_stds, bins=30, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_mean_stds, bins=30, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Mean Std of Beta', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Mean Beta Std Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Histogram of max std
    ax = fig.add_subplot(gs[0, 1])
    ax.hist(no_arm_max_stds, bins=30, alpha=0.6, label='No_Arm', color='blue', edgecolor='black')
    ax.hist(with_arm_max_stds, bins=30, alpha=0.6, label='With_Arm', color='green', edgecolor='black')
    ax.set_xlabel('Max Std of Beta', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Max Beta Std Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Box plot of mean std
    ax = fig.add_subplot(gs[0, 2])
    bp = ax.boxplot([no_arm_mean_stds, with_arm_mean_stds],
                     labels=['No_Arm', 'With_Arm'],
                     patch_artist=True,
                     showmeans=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    ax.set_ylabel('Mean Std of Beta', fontsize=11)
    ax.set_title('Mean Beta Std Box Plot', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. Per-beta dimension analysis (No_Arm)
    ax = fig.add_subplot(gs[1, :])
    no_arm_stats = compute_statistics(no_arm_results)
    with_arm_stats = compute_statistics(with_arm_results)

    beta_indices = np.arange(10)
    width = 0.35

    ax.bar(beta_indices - width/2, no_arm_stats['per_beta_mean_std'],
           width, label='No_Arm', alpha=0.7, color='blue', edgecolor='black')
    ax.bar(beta_indices + width/2, with_arm_stats['per_beta_mean_std'],
           width, label='With_Arm', alpha=0.7, color='green', edgecolor='black')

    ax.set_xlabel('Beta Dimension', fontsize=11)
    ax.set_ylabel('Mean Std across Subjects', fontsize=11)
    ax.set_title('Per-Beta Dimension Consistency', fontsize=12, fontweight='bold')
    ax.set_xticks(beta_indices)
    ax.set_xticklabels([f'β{i}' for i in range(10)])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # 5. Statistics table
    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')

    table_data = [
        ['Metric', 'No_Arm', 'With_Arm'],
        ['Count', f"{no_arm_stats['count']}", f"{with_arm_stats['count']}"],
        ['Mean of Mean Std', f"{no_arm_stats['mean_of_mean_std']:.4f}", f"{with_arm_stats['mean_of_mean_std']:.4f}"],
        ['Median of Mean Std', f"{no_arm_stats['median_of_mean_std']:.4f}", f"{with_arm_stats['median_of_mean_std']:.4f}"],
        ['Min of Mean Std', f"{no_arm_stats['min_of_mean_std']:.4f}", f"{with_arm_stats['min_of_mean_std']:.4f}"],
        ['Max of Mean Std', f"{no_arm_stats['max_of_mean_std']:.4f}", f"{with_arm_stats['max_of_mean_std']:.4f}"],
        ['Mean of Max Std', f"{no_arm_stats['mean_of_max_std']:.4f}", f"{with_arm_stats['mean_of_max_std']:.4f}"],
        ['Median of Max Std', f"{no_arm_stats['median_of_max_std']:.4f}", f"{with_arm_stats['median_of_max_std']:.4f}"],
    ]

    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.5, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
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

    # 6. Interpretation guide
    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')

    interpretation_text = (
        "Interpretation:\n\n"
        "• Lower std = Better consistency\n\n"
        "• Ideal: Mean std < 0.01\n"
        "  (shape stable across frames)\n\n"
        "• Acceptable: Mean std < 0.1\n"
        "  (minor variations)\n\n"
        "• Concerning: Mean std > 0.5\n"
        "  (shape fluctuates significantly)"
    )

    ax.text(0.1, 0.5, interpretation_text,
            verticalalignment='center',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate SMPL shape consistency')
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
    print("SMPL Shape Consistency Evaluation")
    print("=" * 80)

    # Evaluate No_Arm dataset
    print("\nEvaluating No_Arm dataset...")
    no_arm_results = evaluate_dataset(results_dir / "No_Arm")
    no_arm_stats = compute_statistics(no_arm_results)

    print(f"\nNo_Arm Statistics:")
    print(f"  Count: {no_arm_stats['count']}")
    print(f"  Mean of Mean Std: {no_arm_stats['mean_of_mean_std']:.4f}")
    print(f"  Median of Mean Std: {no_arm_stats['median_of_mean_std']:.4f}")
    print(f"  Mean of Max Std: {no_arm_stats['mean_of_max_std']:.4f}")

    # Evaluate With_Arm dataset
    print("\nEvaluating With_Arm dataset...")
    with_arm_results = evaluate_dataset(results_dir / "With_Arm")
    with_arm_stats = compute_statistics(with_arm_results)

    print(f"\nWith_Arm Statistics:")
    print(f"  Count: {with_arm_stats['count']}")
    print(f"  Mean of Mean Std: {with_arm_stats['mean_of_mean_std']:.4f}")
    print(f"  Median of Mean Std: {with_arm_stats['median_of_mean_std']:.4f}")
    print(f"  Mean of Max Std: {with_arm_stats['mean_of_max_std']:.4f}")

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

    # json_path = output_dir / "shape_consistency_evaluation.json"
    # with open(json_path, 'w') as f:
        # json.dump(output_data, f, indent=2)
    # print(f"\nDetailed results saved to: {json_path}")

    # Create comparison plot
    plot_path = output_dir / "shape_consistency_comparison.png"
    plot_comparison(no_arm_results, with_arm_results, str(plot_path))

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
