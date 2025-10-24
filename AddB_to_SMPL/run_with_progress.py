#!/usr/bin/env python3
"""
Run V2 baseline + offset optimization with real-time progress display
"""
import subprocess
import sys
import time
import os
import argparse
from datetime import datetime

def print_header(text):
    """Print a formatted header"""
    width = 80
    print("\n" + "="*width)
    print(f" {text}")
    print("="*width + "\n")

def print_progress(stage, current, total, start_time):
    """Print progress bar with timing info"""
    elapsed = time.time() - start_time
    percent = (current / total) * 100 if total > 0 else 0
    bar_length = 40
    filled = int(bar_length * current / total) if total > 0 else 0
    bar = "█" * filled + "░" * (bar_length - filled)

    # Estimate time remaining
    if current > 0:
        time_per_item = elapsed / current
        remaining = time_per_item * (total - current)
        eta = f"ETA: {remaining:.0f}s"
    else:
        eta = "ETA: calculating..."

    print(f"\r{stage}: |{bar}| {current}/{total} ({percent:.1f}%) | Elapsed: {elapsed:.0f}s | {eta}", end="", flush=True)

def run_command_with_progress(cmd, description, cwd=None):
    """Run command and show progress"""
    print_header(description)
    print(f"Command: {' '.join(cmd)}\n")

    start_time = time.time()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=cwd
    )

    output_lines = []
    for line in process.stdout:
        output_lines.append(line)
        line = line.strip()

        # Print important lines
        if any(keyword in line.lower() for keyword in ['error', 'warning', 'stage', 'iteration', 'mpjpe', 'improvement']):
            print(line)

        # Parse progress from output
        if 'Stage' in line or 'iteration' in line.lower() or 'frame' in line.lower():
            sys.stdout.write('.')
            sys.stdout.flush()

    process.wait()
    elapsed = time.time() - start_time

    if process.returncode == 0:
        print(f"\n✓ Completed in {elapsed:.1f}s")
    else:
        print(f"\n✗ Failed with return code {process.returncode}")
        print("\n--- Last 20 lines of output ---")
        for line in output_lines[-20:]:
            print(line, end="")

    return process.returncode, elapsed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--b3d', type=str, required=True, nargs='+', help='B3D file paths')
    parser.add_argument('--smpl_model', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='results_random_samples')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--skip_baseline', action='store_true', help='Skip V2 baseline (if already computed)')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print_header(f"SMPL2AddBiomechanics Processing - {len(args.b3d)} samples")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {args.device}")
    print(f"\nSamples:")
    for i, b3d_path in enumerate(args.b3d, 1):
        print(f"  {i}. {os.path.basename(os.path.dirname(b3d_path))}/{os.path.basename(b3d_path)}")

    total_start = time.time()
    results = []

    for idx, b3d_path in enumerate(args.b3d, 1):
        sample_name = os.path.basename(os.path.dirname(b3d_path)) + "_" + os.path.basename(b3d_path).replace('.b3d', '')
        sample_out_dir = os.path.join(args.out_dir, sample_name)

        print_header(f"Sample {idx}/{len(args.b3d)}: {sample_name}")

        # Stage 1: V2 Baseline
        if not args.skip_baseline:
            baseline_out = os.path.join(sample_out_dir, 'v2_baseline')
            cmd_baseline = [
                'python', 'addbiomechanics_to_smpl_v2.py',
                '--b3d', b3d_path,
                '--smpl_model', args.smpl_model,
                '--out_dir', baseline_out,
                '--device', args.device
            ]

            ret, elapsed_baseline = run_command_with_progress(
                cmd_baseline,
                f"[{idx}/{len(args.b3d)}] Stage 1/2: V2 Baseline Optimization",
                cwd=os.path.dirname(os.path.abspath(__file__))
            )

            if ret != 0:
                print(f"✗ V2 baseline failed for {sample_name}")
                continue
        else:
            print("Skipping V2 baseline (already computed)")
            elapsed_baseline = 0

        # Stage 2: Offset Refinement
        offset_out = os.path.join(sample_out_dir, 'v2_with_offset')
        cmd_offset = [
            'python', 'run_v2_with_offset.py',
            '--b3d', b3d_path,
            '--smpl_model', args.smpl_model,
            '--out_dir', offset_out,
            '--device', args.device
        ]

        ret, elapsed_offset = run_command_with_progress(
            cmd_offset,
            f"[{idx}/{len(args.b3d)}] Stage 2/2: Per-Joint Offset Refinement",
            cwd=os.path.dirname(os.path.abspath(__file__))
        )

        if ret != 0:
            print(f"✗ Offset refinement failed for {sample_name}")
            continue

        # Read results
        import json
        meta_path = os.path.join(offset_out, sample_name, 'meta_with_offset.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)

            results.append({
                'sample': sample_name,
                'baseline_mpjpe': meta['metrics_with_offset']['baseline_MPJPE'],
                'offset_mpjpe': meta['metrics_with_offset']['MPJPE'],
                'improvement_mm': meta['metrics_with_offset']['improvement_mm'],
                'improvement_percent': meta['metrics_with_offset']['improvement_percent'],
                'time_baseline': elapsed_baseline,
                'time_offset': elapsed_offset
            })

    # Final summary
    total_elapsed = time.time() - total_start

    print_header("FINAL SUMMARY")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    print(f"Completed: {len(results)}/{len(args.b3d)} samples\n")

    if results:
        print("Results:")
        print("-" * 100)
        print(f"{'Sample':<35} {'Baseline':>12} {'Offset':>12} {'Improvement':>15} {'Time':>10}")
        print("-" * 100)

        for r in results:
            print(f"{r['sample']:<35} {r['baseline_mpjpe']:>10.2f}mm {r['offset_mpjpe']:>10.2f}mm "
                  f"{r['improvement_mm']:>8.2f}mm ({r['improvement_percent']:>5.1f}%) "
                  f"{r['time_baseline']+r['time_offset']:>8.1f}s")

        print("-" * 100)

        # Average improvement
        avg_improvement_mm = sum(r['improvement_mm'] for r in results) / len(results)
        avg_improvement_pct = sum(r['improvement_percent'] for r in results) / len(results)
        print(f"{'Average improvement:':<35} {'':<24} {avg_improvement_mm:>8.2f}mm ({avg_improvement_pct:>5.1f}%)")
        print("-" * 100)

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {args.out_dir}\n")

if __name__ == '__main__':
    main()
