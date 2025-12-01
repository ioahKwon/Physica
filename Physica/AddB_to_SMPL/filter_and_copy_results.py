#!/usr/bin/env python3
"""
Filter results by MPJPE threshold and copy to new directory
"""
import json
import shutil
from pathlib import Path
import argparse


def filter_and_copy(source_dir, target_dir, threshold, dataset_name):
    """Filter results and copy to target directory"""
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    kept = []
    filtered = []

    print(f"\n=== Processing {dataset_name} ===")
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print(f"Threshold: {threshold}mm")

    # Iterate through all result directories
    for result_subdir in sorted(source_dir.iterdir()):
        if not result_subdir.is_dir():
            continue

        meta_file = result_subdir / "meta.json"
        if not meta_file.exists():
            continue

        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                mpjpe = meta.get('MPJPE', float('inf'))

            if mpjpe <= threshold:
                # Copy entire directory
                target_subdir = target_dir / result_subdir.name
                if target_subdir.exists():
                    shutil.rmtree(target_subdir)
                shutil.copytree(result_subdir, target_subdir)
                kept.append((result_subdir.name, mpjpe))
            else:
                filtered.append((result_subdir.name, mpjpe))

        except Exception as e:
            print(f"Error processing {result_subdir.name}: {e}")
            continue

    print(f"\n✓ Kept: {len(kept)} subjects")
    print(f"✗ Filtered: {len(filtered)} subjects")
    print(f"Kept percentage: {len(kept)/(len(kept)+len(filtered))*100:.1f}%")

    # Save filter report
    report_file = target_dir / "filter_report.txt"
    with open(report_file, 'w') as f:
        f.write(f"Filter Report: {dataset_name}\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Threshold: {threshold}mm\n")
        f.write(f"Total processed: {len(kept) + len(filtered)}\n")
        f.write(f"Kept: {len(kept)} ({len(kept)/(len(kept)+len(filtered))*100:.1f}%)\n")
        f.write(f"Filtered: {len(filtered)} ({len(filtered)/(len(kept)+len(filtered))*100:.1f}%)\n\n")

        if kept:
            import numpy as np
            mpjpe_values = [m for _, m in kept]
            f.write(f"Statistics of kept subjects:\n")
            f.write(f"  Mean MPJPE: {np.mean(mpjpe_values):.2f}mm\n")
            f.write(f"  Median MPJPE: {np.median(mpjpe_values):.2f}mm\n")
            f.write(f"  Min MPJPE: {np.min(mpjpe_values):.2f}mm\n")
            f.write(f"  Max MPJPE: {np.max(mpjpe_values):.2f}mm\n\n")

        f.write("Filtered subjects:\n")
        f.write("-" * 80 + "\n")
        for name, mpjpe in sorted(filtered, key=lambda x: x[1], reverse=True):
            f.write(f"{name}: {mpjpe:.2f}mm\n")

    print(f"✓ Report saved: {report_file}")

    return len(kept), len(filtered)


def main():
    parser = argparse.ArgumentParser(description='Filter results by MPJPE threshold')
    parser.add_argument('--no_arm_source', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_enhanced_minibatch_full',
                       help='No_Arm source directory')
    parser.add_argument('--with_arm_source', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_enhanced_minibatch_with_arm_fullbody',
                       help='With_Arm source directory')
    parser.add_argument('--output_dir', type=str,
                       default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/filtered_results',
                       help='Output directory for filtered results')
    parser.add_argument('--no_arm_threshold', type=float, default=60.0,
                       help='MPJPE threshold for No_Arm (default: 60mm)')
    parser.add_argument('--with_arm_threshold', type=float, default=70.0,
                       help='MPJPE threshold for With_Arm (default: 70mm)')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Filtering SMPL Results by MPJPE Threshold")
    print("=" * 80)

    # Filter No_Arm
    no_arm_target = output_dir / "No_Arm"
    no_arm_kept, no_arm_filtered = filter_and_copy(
        args.no_arm_source,
        no_arm_target,
        args.no_arm_threshold,
        "No_Arm (Lower Body Only)"
    )

    # Filter With_Arm
    with_arm_target = output_dir / "With_Arm"
    with_arm_kept, with_arm_filtered = filter_and_copy(
        args.with_arm_source,
        with_arm_target,
        args.with_arm_threshold,
        "With_Arm (Full Body)"
    )

    # Overall summary
    print("\n" + "=" * 80)
    print("OVERALL SUMMARY")
    print("=" * 80)
    print(f"No_Arm:   {no_arm_kept} kept, {no_arm_filtered} filtered (threshold: {args.no_arm_threshold}mm)")
    print(f"With_Arm: {with_arm_kept} kept, {with_arm_filtered} filtered (threshold: {args.with_arm_threshold}mm)")
    print(f"\nTotal kept: {no_arm_kept + with_arm_kept} subjects")
    print(f"Total filtered: {no_arm_filtered + with_arm_filtered} subjects")
    print(f"\nFiltered results saved to: {output_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
