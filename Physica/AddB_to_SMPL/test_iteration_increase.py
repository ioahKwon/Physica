#!/usr/bin/env python3
"""
Test iteration increase by +3 for each parameter.
Compare baseline vs increased iterations on 3 subjects.
"""

import subprocess
import time
import json
import os
from pathlib import Path

# Test subjects (use small ones for quick test)
TEST_SUBJECTS = [
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject7/Subject7.b3d",
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d",
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject19/Subject19.b3d",
]

# Baseline settings (current)
BASELINE = {
    "shape_iters": 100,
    "pose_iters": 40,
    "max_passes": 1,
}

# Increased settings (+3 each)
INCREASED = {
    "shape_iters": 103,
    "pose_iters": 43,
    "max_passes": 1,  # Keep at 1 (can't increase pass count by 3, would be too much)
}

OUTPUT_BASE = "/egr/research-zijunlab/kwonjoon/Output/test_iteration_increase"
SCRIPT_PATH = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py"

def run_test(b3d_path, config, config_name):
    """Run SMPL fitting with given configuration."""
    subject_name = Path(b3d_path).stem
    output_dir = f"{OUTPUT_BASE}/{config_name}/{subject_name}"
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "python", SCRIPT_PATH,
        "--b3d", b3d_path,
        "--output_dir", output_dir,
        "--shape_iters", str(config["shape_iters"]),
        "--pose_iters", str(config["pose_iters"]),
        "--max_passes", str(config["max_passes"]),
        "--max_frames", "500",  # Use 500-frame sampling for speed
    ]

    print(f"\n{'='*80}")
    print(f"Running: {config_name} - {subject_name}")
    print(f"Config: {config}")
    print(f"{'='*80}\n")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - start_time

    # Parse MPJPE from output
    mpjpe = None
    for line in result.stdout.split('\n'):
        if 'MPJPE' in line and 'mm' in line:
            try:
                mpjpe = float(line.split(':')[1].strip().split('mm')[0].strip())
            except:
                pass

    # Also check meta.json
    meta_path = f"{output_dir}/meta.json"
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
            if 'metrics' in meta and 'MPJPE' in meta['metrics']:
                mpjpe = meta['metrics']['MPJPE']

    return {
        'subject': subject_name,
        'config': config_name,
        'mpjpe': mpjpe,
        'time': elapsed,
        'success': result.returncode == 0
    }

def main():
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results = []

    print(f"\n{'#'*80}")
    print("ITERATION INCREASE TEST: +3 iterations")
    print(f"{'#'*80}\n")
    print(f"Baseline: {BASELINE}")
    print(f"Increased: {INCREASED}")
    print(f"Test subjects: {len(TEST_SUBJECTS)}")
    print(f"\n{'#'*80}\n")

    # Run baseline
    print("\n### BASELINE RUNS ###\n")
    for b3d_path in TEST_SUBJECTS:
        result = run_test(b3d_path, BASELINE, "baseline")
        results.append(result)

    # Run increased
    print("\n### INCREASED RUNS ###\n")
    for b3d_path in TEST_SUBJECTS:
        result = run_test(b3d_path, INCREASED, "increased")
        results.append(result)

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}\n")

    baseline_results = [r for r in results if r['config'] == 'baseline']
    increased_results = [r for r in results if r['config'] == 'increased']

    print(f"{'Subject':<15} {'Baseline MPJPE':<20} {'Increased MPJPE':<20} {'Improvement':<15} {'Time Diff'}")
    print("-" * 90)

    total_baseline_mpjpe = 0
    total_increased_mpjpe = 0
    total_baseline_time = 0
    total_increased_time = 0

    for i, subject in enumerate([Path(p).stem for p in TEST_SUBJECTS]):
        baseline = baseline_results[i]
        increased = increased_results[i]

        if baseline['mpjpe'] and increased['mpjpe']:
            improvement = ((baseline['mpjpe'] - increased['mpjpe']) / baseline['mpjpe']) * 100
            time_diff = increased['time'] - baseline['time']

            print(f"{subject:<15} {baseline['mpjpe']:>8.2f} mm         "
                  f"{increased['mpjpe']:>8.2f} mm         "
                  f"{improvement:>6.2f}%         "
                  f"{time_diff:>+6.1f}s")

            total_baseline_mpjpe += baseline['mpjpe']
            total_increased_mpjpe += increased['mpjpe']
            total_baseline_time += baseline['time']
            total_increased_time += increased['time']

    print("-" * 90)
    avg_baseline_mpjpe = total_baseline_mpjpe / len(TEST_SUBJECTS)
    avg_increased_mpjpe = total_increased_mpjpe / len(TEST_SUBJECTS)
    avg_improvement = ((avg_baseline_mpjpe - avg_increased_mpjpe) / avg_baseline_mpjpe) * 100
    avg_time_diff = (total_increased_time - total_baseline_time) / len(TEST_SUBJECTS)

    print(f"{'AVERAGE':<15} {avg_baseline_mpjpe:>8.2f} mm         "
          f"{avg_increased_mpjpe:>8.2f} mm         "
          f"{avg_improvement:>6.2f}%         "
          f"{avg_time_diff:>+6.1f}s")

    # Save results
    summary_path = f"{OUTPUT_BASE}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'baseline_config': BASELINE,
            'increased_config': INCREASED,
            'results': results,
            'summary': {
                'avg_baseline_mpjpe': avg_baseline_mpjpe,
                'avg_increased_mpjpe': avg_increased_mpjpe,
                'avg_improvement_pct': avg_improvement,
                'avg_time_increase_sec': avg_time_diff,
            }
        }, f, indent=2)

    print(f"\nResults saved to: {summary_path}")

if __name__ == "__main__":
    main()
