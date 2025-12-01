#!/usr/bin/env python3
"""
Test different pose_iters settings with GPU assignment.
- Baseline: shape_iters=100, pose_iters=40
- Config1: shape_iters=100, pose_iters=100
- Config2: shape_iters=100, pose_iters=300
- Config3: shape_iters=100, pose_iters=500

Each config runs on a different GPU (0, 1, 2, 3).
"""

import subprocess
import time
import json
import os
from pathlib import Path

# Test subjects (use same as before for comparison)
TEST_SUBJECTS = [
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject7/Subject7.b3d",
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject11/Subject11.b3d",
    "/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/With_Arm/Tiziana2019_Formatted_With_Arm/Subject19/Subject19.b3d",
]

# Test configurations
CONFIGS = {
    "baseline_40": {
        "shape_iters": 100,
        "pose_iters": 40,
        "max_passes": 1,
        "gpu": 0,
    },
    "pose_100": {
        "shape_iters": 100,
        "pose_iters": 100,
        "max_passes": 1,
        "gpu": 1,
    },
    "pose_300": {
        "shape_iters": 100,
        "pose_iters": 300,
        "max_passes": 1,
        "gpu": 2,
    },
    "pose_500": {
        "shape_iters": 100,
        "pose_iters": 500,
        "max_passes": 1,
        "gpu": 3,
    },
}

OUTPUT_BASE = "/egr/research-zijunlab/kwonjoon/Output/test_pose_iters_gpu"
SCRIPT_PATH = "/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/addbiomechanics_to_smpl_v11_with_physics.py"

def run_single_test(b3d_path, config, config_name, gpu_id):
    """Run SMPL fitting with given configuration on specific GPU."""
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

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"[GPU {gpu_id}] Starting: {config_name} - {subject_name}")
    print(f"  Config: shape_iters={config['shape_iters']}, pose_iters={config['pose_iters']}")

    start_time = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
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

    success = result.returncode == 0
    status = "✓" if success else "✗"
    print(f"[GPU {gpu_id}] {status} {config_name} - {subject_name}: MPJPE={mpjpe:.2f}mm, Time={elapsed:.1f}s")

    return {
        'subject': subject_name,
        'config': config_name,
        'shape_iters': config['shape_iters'],
        'pose_iters': config['pose_iters'],
        'mpjpe': mpjpe,
        'time': elapsed,
        'success': success,
        'gpu': gpu_id,
    }

def run_config_parallel(config_name, config, all_results):
    """Run all subjects for a single config sequentially on assigned GPU."""
    gpu_id = config['gpu']
    print(f"\n{'='*80}")
    print(f"Starting {config_name} on GPU {gpu_id}")
    print(f"Config: shape_iters={config['shape_iters']}, pose_iters={config['pose_iters']}")
    print(f"{'='*80}\n")

    for b3d_path in TEST_SUBJECTS:
        result = run_single_test(b3d_path, config, config_name, gpu_id)
        all_results.append(result)

    print(f"\n[GPU {gpu_id}] Completed {config_name}\n")

def main():
    import threading

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    all_results = []
    threads = []

    print(f"\n{'#'*80}")
    print("POSE ITERATION TEST WITH GPU PARALLELIZATION")
    print(f"{'#'*80}\n")
    print("Configurations:")
    for name, cfg in CONFIGS.items():
        print(f"  {name:15} GPU {cfg['gpu']}: shape_iters={cfg['shape_iters']}, pose_iters={cfg['pose_iters']}")
    print(f"\nTest subjects: {len(TEST_SUBJECTS)}")
    print(f"Total runs: {len(CONFIGS)} configs × {len(TEST_SUBJECTS)} subjects = {len(CONFIGS) * len(TEST_SUBJECTS)} runs")
    print(f"\n{'#'*80}\n")

    # Start all configs in parallel (each on different GPU)
    for config_name, config in CONFIGS.items():
        thread = threading.Thread(target=run_config_parallel, args=(config_name, config, all_results))
        thread.start()
        threads.append(thread)
        time.sleep(2)  # Small delay between thread starts

    # Wait for all to complete
    for thread in threads:
        thread.join()

    # Print summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}\n")

    # Group results by config
    config_summaries = {}
    for config_name in CONFIGS.keys():
        config_results = [r for r in all_results if r['config'] == config_name]
        if config_results:
            valid_mpjpe = [r['mpjpe'] for r in config_results if r['mpjpe'] is not None]
            if valid_mpjpe:
                avg_mpjpe = sum(valid_mpjpe) / len(valid_mpjpe)
                avg_time = sum(r['time'] for r in config_results) / len(config_results)
                config_summaries[config_name] = {
                    'pose_iters': config_results[0]['pose_iters'],
                    'avg_mpjpe': avg_mpjpe,
                    'avg_time': avg_time,
                    'results': config_results,
                }

    # Check if we have baseline results
    if 'baseline_40' not in config_summaries:
        print("ERROR: No baseline results available yet!")
        print(f"Available configs: {list(config_summaries.keys())}")
        # Save partial results
        summary_path = f"{OUTPUT_BASE}/summary_partial.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'configs': CONFIGS,
                'results': all_results,
                'summary': config_summaries,
            }, f, indent=2)
        print(f"Partial results saved to: {summary_path}")
        return

    # Print comparison table
    baseline_mpjpe = config_summaries['baseline_40']['avg_mpjpe']
    baseline_time = config_summaries['baseline_40']['avg_time']

    print(f"{'Config':<15} {'Pose Iters':<12} {'Avg MPJPE':<15} {'Improvement':<15} {'Avg Time':<12} {'Time Increase'}")
    print("-" * 95)

    for config_name in ["baseline_40", "pose_100", "pose_300", "pose_500"]:
        summary = config_summaries[config_name]
        improvement = ((baseline_mpjpe - summary['avg_mpjpe']) / baseline_mpjpe) * 100
        time_increase = ((summary['avg_time'] - baseline_time) / baseline_time) * 100

        print(f"{config_name:<15} {summary['pose_iters']:<12} "
              f"{summary['avg_mpjpe']:>8.2f} mm    "
              f"{improvement:>+7.2f}%        "
              f"{summary['avg_time']:>7.1f}s      "
              f"{time_increase:>+7.1f}%")

    # Detailed per-subject results
    print(f"\n{'='*80}")
    print("PER-SUBJECT RESULTS")
    print(f"{'='*80}\n")

    for subject in [Path(p).stem for p in TEST_SUBJECTS]:
        print(f"\n{subject}:")
        print(f"{'  Config':<17} {'MPJPE':<15} {'Time':<12} {'GPU'}")
        print("  " + "-" * 50)

        for config_name in ["baseline_40", "pose_100", "pose_300", "pose_500"]:
            result = next((r for r in all_results if r['config'] == config_name and r['subject'] == subject), None)
            if result and result['mpjpe']:
                print(f"  {config_name:<15} {result['mpjpe']:>8.2f} mm    {result['time']:>7.1f}s     GPU {result['gpu']}")

    # Save results
    summary_path = f"{OUTPUT_BASE}/summary.json"
    with open(summary_path, 'w') as f:
        json.dump({
            'configs': CONFIGS,
            'results': all_results,
            'summary': config_summaries,
            'comparison': {
                'baseline_mpjpe': baseline_mpjpe,
                'baseline_time': baseline_time,
            }
        }, f, indent=2)

    print(f"\n\nResults saved to: {summary_path}")
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
