#!/usr/bin/env python3
"""
Analyze and visualize MPJPE distribution for No_Arm and With_Arm datasets
"""
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import argparse


def collect_mpjpe(result_dir):
    """Collect MPJPE values from all meta.json files"""
    result_dir = Path(result_dir)
    mpjpe_values = []

    for meta_file in result_dir.rglob("meta.json"):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'MPJPE' in meta:
                    mpjpe_values.append(meta['MPJPE'])
        except:
            continue

    return np.array(mpjpe_values)


def compute_statistics(mpjpe_values, name):
    """Compute and print statistics"""
    print(f"\n=== {name} Statistics ===")
    print(f"Total subjects: {len(mpjpe_values)}")
    print(f"Mean MPJPE: {np.mean(mpjpe_values):.2f}mm")
    print(f"Median MPJPE: {np.median(mpjpe_values):.2f}mm")
    print(f"Min MPJPE: {np.min(mpjpe_values):.2f}mm")
    print(f"Max MPJPE: {np.max(mpjpe_values):.2f}mm")
    print(f"Std Dev: {np.std(mpjpe_values):.2f}mm")
    print(f"25th percentile: {np.percentile(mpjpe_values, 25):.2f}mm")
    print(f"75th percentile: {np.percentile(mpjpe_values, 75):.2f}mm")

    return {
        'count': len(mpjpe_values),
        'mean': np.mean(mpjpe_values),
        'median': np.median(mpjpe_values),
        'min': np.min(mpjpe_values),
        'max': np.max(mpjpe_values),
        'std': np.std(mpjpe_values),
        'q25': np.percentile(mpjpe_values, 25),
        'q75': np.percentile(mpjpe_values, 75),
    }


def create_comparison_plot(no_arm_mpjpe, with_arm_mpjpe, no_arm_stats, with_arm_stats, output_path):
    """Create comparison visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Histogram comparison
    ax1 = axes[0, 0]
    bins = np.linspace(0, max(np.max(no_arm_mpjpe), np.max(with_arm_mpjpe)), 50)
    ax1.hist(no_arm_mpjpe, bins=bins, alpha=0.6, label='No_Arm (Lower Body Only)',
             color='skyblue', edgecolor='blue', linewidth=1.5)
    ax1.hist(with_arm_mpjpe, bins=bins, alpha=0.6, label='With_Arm (Full Body)',
             color='lightcoral', edgecolor='red', linewidth=1.5)
    ax1.axvline(no_arm_stats['mean'], color='blue', linestyle='--', linewidth=2,
                label=f"No_Arm Mean: {no_arm_stats['mean']:.2f}mm")
    ax1.axvline(with_arm_stats['mean'], color='red', linestyle='--', linewidth=2,
                label=f"With_Arm Mean: {with_arm_stats['mean']:.2f}mm")
    ax1.set_xlabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax1.set_title('MPJPE Distribution Comparison', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Box plot comparison
    ax2 = axes[0, 1]
    bp = ax2.boxplot([no_arm_mpjpe, with_arm_mpjpe],
                      labels=['No_Arm\n(Lower Body)', 'With_Arm\n(Full Body)'],
                      patch_artist=True,
                      widths=0.6)
    bp['boxes'][0].set_facecolor('skyblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    for box in bp['boxes']:
        box.set_linewidth(2)
    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(2.5)
        median.set_color('darkred')

    ax2.set_ylabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax2.set_title('Box Plot Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    no_arm_sorted = np.sort(no_arm_mpjpe)
    with_arm_sorted = np.sort(with_arm_mpjpe)
    no_arm_cumulative = np.arange(1, len(no_arm_sorted) + 1) / len(no_arm_sorted) * 100
    with_arm_cumulative = np.arange(1, len(with_arm_sorted) + 1) / len(with_arm_sorted) * 100

    ax3.plot(no_arm_sorted, no_arm_cumulative, color='blue', linewidth=2.5,
             label='No_Arm (Lower Body Only)', alpha=0.8)
    ax3.plot(with_arm_sorted, with_arm_cumulative, color='red', linewidth=2.5,
             label='With_Arm (Full Body)', alpha=0.8)
    ax3.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax3.set_xlabel('MPJPE (mm)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Cumulative Percentage (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Cumulative Distribution', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # 4. Statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    stats_data = [
        ['Metric', 'No_Arm\n(Lower Body)', 'With_Arm\n(Full Body)'],
        ['Count', f"{no_arm_stats['count']}", f"{with_arm_stats['count']}"],
        ['Mean', f"{no_arm_stats['mean']:.2f}mm", f"{with_arm_stats['mean']:.2f}mm"],
        ['Median', f"{no_arm_stats['median']:.2f}mm", f"{with_arm_stats['median']:.2f}mm"],
        ['Min', f"{no_arm_stats['min']:.2f}mm", f"{with_arm_stats['min']:.2f}mm"],
        ['Max', f"{no_arm_stats['max']:.2f}mm", f"{with_arm_stats['max']:.2f}mm"],
        ['Std Dev', f"{no_arm_stats['std']:.2f}mm", f"{with_arm_stats['std']:.2f}mm"],
        ['Q25', f"{no_arm_stats['q25']:.2f}mm", f"{with_arm_stats['q25']:.2f}mm"],
        ['Q75', f"{no_arm_stats['q75']:.2f}mm", f"{with_arm_stats['q75']:.2f}mm"],
    ]

    table = ax4.table(cellText=stats_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style data rows
    for i in range(1, len(stats_data)):
        table[(i, 0)].set_facecolor('#e8e8e8')
        table[(i, 0)].set_text_props(weight='bold')
        table[(i, 1)].set_facecolor('#e3f2fd')
        table[(i, 2)].set_facecolor('#ffebee')

    ax4.set_title('Statistics Summary', fontsize=14, fontweight='bold', pad=20)

    # Overall title
    fig.suptitle('MPJPE Distribution Analysis: No_Arm vs With_Arm',
                 fontsize=16, fontweight='bold', y=0.995)

    plt.tight_layout(rect=[0, 0, 1, 0.99])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Plot saved to: {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze MPJPE distribution')
    parser.add_argument('--no_arm_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_enhanced_minibatch_full',
                       help='No_Arm results directory')
    parser.add_argument('--with_arm_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_enhanced_minibatch_with_arm_fullbody',
                       help='With_Arm results directory')
    parser.add_argument('--output', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/mpjpe_distribution_comparison.png',
                       help='Output plot path')

    args = parser.parse_args()

    print("Collecting MPJPE values...")
    print(f"No_Arm directory: {args.no_arm_dir}")
    print(f"With_Arm directory: {args.with_arm_dir}")

    # Collect MPJPE values
    no_arm_mpjpe = collect_mpjpe(args.no_arm_dir)
    with_arm_mpjpe = collect_mpjpe(args.with_arm_dir)

    print(f"\nCollected {len(no_arm_mpjpe)} No_Arm results")
    print(f"Collected {len(with_arm_mpjpe)} With_Arm results")

    # Compute statistics
    no_arm_stats = compute_statistics(no_arm_mpjpe, "No_Arm (Lower Body Only)")
    with_arm_stats = compute_statistics(with_arm_mpjpe, "With_Arm (Full Body)")

    # Create comparison plot
    print("\nCreating comparison plot...")
    create_comparison_plot(no_arm_mpjpe, with_arm_mpjpe, no_arm_stats, with_arm_stats, args.output)

    # Print comparison
    print("\n=== Comparison ===")
    print(f"Mean difference: {with_arm_stats['mean'] - no_arm_stats['mean']:.2f}mm")
    print(f"With_Arm is {((with_arm_stats['mean'] - no_arm_stats['mean']) / no_arm_stats['mean'] * 100):.1f}% higher MPJPE")
    print(f"This is expected since With_Arm optimizes more joints (16 vs 9)")


if __name__ == '__main__':
    main()
