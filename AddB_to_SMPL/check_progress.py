#!/usr/bin/env python3
"""
Check progress of batch processing
"""
import os
import json
import numpy as np
from pathlib import Path

def check_progress(output_dir):
    """Check how many files have been processed"""
    total_subjects = 0
    completed_subjects = 0
    failed_subjects = 0
    mpjpes = []

    results = []

    for root, dirs, files in os.walk(output_dir):
        if 'meta.json' in files:
            total_subjects += 1
            meta_path = os.path.join(root, 'meta.json')

            with open(meta_path, 'r') as f:
                meta = json.load(f)

            rel_path = root.replace(output_dir, '').lstrip('/')

            if 'metrics' in meta and 'MPJPE' in meta['metrics']:
                mpjpe = meta['metrics']['MPJPE']
                mpjpes.append(mpjpe)
                completed_subjects += 1
                results.append({
                    'path': rel_path,
                    'mpjpe': mpjpe,
                    'status': 'completed'
                })
            else:
                failed_subjects += 1
                results.append({
                    'path': rel_path,
                    'status': 'failed'
                })

    print("=" * 80)
    print("Batch Processing Progress")
    print("=" * 80)
    print(f"Total subjects found: {total_subjects}")
    print(f"Completed: {completed_subjects}")
    print(f"Failed: {failed_subjects}")
    print()

    if mpjpes:
        print("Statistics (MPJPE):")
        print(f"  Mean: {np.mean(mpjpes):.2f} mm")
        print(f"  Std:  {np.std(mpjpes):.2f} mm")
        print(f"  Min:  {np.min(mpjpes):.2f} mm")
        print(f"  Max:  {np.max(mpjpes):.2f} mm")
        print(f"  Median: {np.median(mpjpes):.2f} mm")
        print()

    # Show top 10 best and worst
    results_with_mpjpe = [r for r in results if r['status'] == 'completed']
    if results_with_mpjpe:
        results_with_mpjpe.sort(key=lambda x: x['mpjpe'])

        print("Top 10 Best Results:")
        for i, r in enumerate(results_with_mpjpe[:10], 1):
            print(f"  {i}. {r['path']}: {r['mpjpe']:.2f} mm")
        print()

        print("Top 10 Worst Results:")
        for i, r in enumerate(results_with_mpjpe[-10:][::-1], 1):
            print(f"  {i}. {r['path']}: {r['mpjpe']:.2f} mm")
        print()

    if failed_subjects > 0:
        print("Failed subjects:")
        for r in results:
            if r['status'] == 'failed':
                print(f"  - {r['path']}")

    print("=" * 80)

if __name__ == '__main__':
    import sys
    output_dir = sys.argv[1] if len(sys.argv) > 1 else '/egr/research-zijunlab/kwonjoon/out/AddB_SMPL_Full'
    check_progress(output_dir)
