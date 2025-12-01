#!/usr/bin/env python3
"""
Sequential batch processing with detailed progress tracking
Shows what's being loaded and current status
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict
import sys

def print_loading_status(message: str, spinner_char: str = "⏳"):
    """Print loading status with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {spinner_char} {message}", flush=True)

def process_single_subject(
    b3d_file: Path,
    output_dir: Path,
    smpl_model: Path,
    num_frames: int,
    device: str,
    python_path: str,
    script_path: Path,
) -> Dict:
    """Process a single subject with detailed progress."""

    subject_name = b3d_file.stem
    parent_name = b3d_file.parent.name

    # Determine split
    if "test" in str(b3d_file):
        split = "test"
    elif "train" in str(b3d_file):
        split = "train"
    else:
        split = "unknown"

    output_name = f"{split}_{parent_name}_{subject_name}"
    subject_output_dir = output_dir / output_name

    # Check if already processed
    meta_file = subject_output_dir / "meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'MPJPE' in meta:
                    print_loading_status(f"⊙ {output_name[:50]} - Already processed (MPJPE: {meta['MPJPE']:.2f}mm)", "⊙")
                    return {
                        'subject': output_name,
                        'status': 'skipped',
                        'mpjpe': meta['MPJPE'],
                        'time': 0,
                    }
        except:
            pass

    # Show what we're processing
    print()
    print("=" * 80)
    print_loading_status(f"Processing: {output_name}")
    print("=" * 80)

    # Process using subprocess
    cmd = [
        python_path,
        str(script_path),
        "--b3d", str(b3d_file),
        "--smpl_model", str(smpl_model),
        "--out_dir", str(output_dir),
        "--num_frames", str(num_frames),
        "--device", device
    ]

    start_time = time.time()

    print_loading_status("  [1/6] Starting Python subprocess...")
    print_loading_status("  [2/6] Loading torch & nimblephysics (takes ~2min)...")

    try:
        result_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - start_time

        if result_proc.returncode == 0:
            print_loading_status(f"  [6/6] Complete! Time: {elapsed:.1f}s", "✓")

            # Try to read the meta.json
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    mpjpe = meta.get('MPJPE', -1)
                    print_loading_status(f"  MPJPE: {mpjpe:.2f}mm", "✓")

                    return {
                        'subject': output_name,
                        'status': 'success',
                        'mpjpe': mpjpe,
                        'time': elapsed,
                    }
            else:
                print_loading_status(f"  ERROR: No meta.json found", "✗")
                return {
                    'subject': output_name,
                    'status': 'failed',
                    'error': 'No meta.json found',
                    'time': elapsed,
                }
        else:
            # Extract error from stderr
            print_loading_status(f"  ERROR: Process failed (exit code {result_proc.returncode})", "✗")
            error_lines = result_proc.stderr.split('\n')[-20:]  # Last 20 lines
            print("  Last error lines:")
            for line in error_lines:
                if line.strip():
                    print(f"    {line}")

            error_msg = next((line for line in error_lines if 'Error' in line or 'Exception' in line), 'Unknown error')

            return {
                'subject': output_name,
                'status': 'failed',
                'error': error_msg[:100],
                'time': elapsed,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print_loading_status(f"  TIMEOUT after {elapsed:.1f}s (30min limit)", "✗")
        return {
            'subject': output_name,
            'status': 'timeout',
            'time': elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print_loading_status(f"  EXCEPTION: {str(e)[:100]}", "✗")
        return {
            'subject': output_name,
            'status': 'error',
            'error': str(e)[:100],
            'time': elapsed,
        }


def print_summary(stats: Dict, total: int, elapsed_total: float):
    """Print summary statistics."""
    print()
    print("=" * 80)
    print("BATCH PROCESSING SUMMARY")
    print("=" * 80)
    print(f"Total time:     {elapsed_total/3600:.2f} hours ({elapsed_total/60:.1f} minutes)")
    print(f"Completed:      {stats['completed']}/{total}")
    print(f"  ✓ Success:    {stats['success']}")
    print(f"  ✗ Failed:     {stats['failed']}")
    print(f"  ⊙ Skipped:    {stats['skipped']}")

    if stats['success'] + stats['skipped'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        print(f"\nAverage MPJPE:  {avg_mpjpe:.2f}mm")

    if stats['success'] > 0:
        avg_time = stats['total_time'] / stats['success']
        print(f"Avg time/subj:  {avg_time:.1f}s ({avg_time/60:.1f} min)")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Sequential batch processing with detailed progress')
    parser.add_argument('--test_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/No_Arm',
                        help='Path to test dataset')
    parser.add_argument('--train_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm',
                        help='Path to train dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_sequential',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to process')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device (cpu or cuda)')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of subjects (for testing)')
    parser.add_argument('--python_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python',
                        help='Python interpreter path')

    args = parser.parse_args()

    script_path = Path(__file__).parent / "addbiomechanics_to_smpl_v3_enhanced.py"

    print("=" * 80)
    print("SEQUENTIAL BATCH PROCESSING - v3_enhanced")
    print("=" * 80)
    print()

    # Find all .b3d files
    print_loading_status("Finding .b3d files...")
    b3d_files = []

    test_path = Path(args.test_path)
    train_path = Path(args.train_path)

    if test_path.exists():
        b3d_files.extend(list(test_path.rglob("*.b3d")))
    if train_path.exists():
        b3d_files.extend(list(train_path.rglob("*.b3d")))

    # Apply limit if specified
    if args.limit:
        b3d_files = b3d_files[:args.limit]

    # Count test vs train
    test_count = sum(1 for f in b3d_files if "test" in str(f))
    train_count = len(b3d_files) - test_count

    print_loading_status(f"Found {len(b3d_files)} subjects (Test: {test_count}, Train: {train_count})", "✓")
    print()
    print(f"Output directory: {args.out_dir}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Create output directory
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        'completed': 0,
        'success': 0,
        'failed': 0,
        'skipped': 0,
        'total_mpjpe': 0.0,
        'total_time': 0.0,
    }

    # Process subjects sequentially
    batch_start_time = time.time()

    for idx, b3d_file in enumerate(b3d_files, 1):
        print()
        print(f"[{idx}/{len(b3d_files)}] Progress: {idx/len(b3d_files)*100:.1f}%")

        result = process_single_subject(
            b3d_file=b3d_file,
            output_dir=output_dir,
            smpl_model=Path(args.smpl_model),
            num_frames=args.num_frames,
            device=args.device,
            python_path=args.python_path,
            script_path=script_path,
        )

        # Update stats
        stats['completed'] += 1
        if result['status'] == 'success':
            stats['success'] += 1
            stats['total_mpjpe'] += result['mpjpe']
            stats['total_time'] += result['time']
        elif result['status'] == 'skipped':
            stats['skipped'] += 1
            stats['total_mpjpe'] += result['mpjpe']
        else:
            stats['failed'] += 1

        # Interim summary every 10 subjects
        if idx % 10 == 0 or idx == len(b3d_files):
            elapsed_so_far = time.time() - batch_start_time
            remaining = len(b3d_files) - idx
            if stats['success'] > 0:
                avg_time_per_subject = stats['total_time'] / stats['success']
                estimated_remaining = remaining * avg_time_per_subject
                print()
                print(f"  Interim stats: ✓{stats['success']} ✗{stats['failed']} ⊙{stats['skipped']}")
                print(f"  Est. remaining time: {estimated_remaining/3600:.1f}h ({estimated_remaining/60:.0f}min)")

    # Final summary
    batch_elapsed = time.time() - batch_start_time
    print_summary(stats, len(b3d_files), batch_elapsed)


if __name__ == "__main__":
    main()
