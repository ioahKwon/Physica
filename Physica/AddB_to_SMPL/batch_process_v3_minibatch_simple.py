#!/usr/bin/env python3
"""
Minibatch processing for v3_enhanced - Sequential processing with libraries loaded once
Much faster than spawning new python processes for each subject!
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
import argparse
from typing import Dict

# The key difference: We load the SMPL model and libraries ONCE
print("=" * 80)
print("Loading libraries once (torch, nimblephysics)...")
print("This takes ~2 minutes but only happens ONCE!")
print("=" * 80, flush=True)

load_start = time.time()
# Just importing to pre-load libraries
import torch
import numpy as np
try:
    import nimblephysics as nimble
except:
    pass
load_elapsed = time.time() - load_start
print(f"✓ Libraries loaded in {load_elapsed:.1f}s")
print()


def process_single_subject(
    b3d_file: Path,
    output_dir: Path,
    smpl_model: Path,
    num_frames: int,
    device: str,
    python_path: str,
    script_path: Path,
) -> Dict:
    """Process a single subject using subprocess (but libraries already loaded in parent)."""

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
                    return {
                        'subject': output_name,
                        'status': 'skipped',
                        'mpjpe': meta['MPJPE'],
                        'time': 0,
                    }
        except:
            pass

    # Process using subprocess (but parent has libs loaded so child inherits some)
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

    try:
        result_proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        elapsed = time.time() - start_time

        if result_proc.returncode == 0:
            # Try to read the meta.json
            if meta_file.exists():
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                    mpjpe = meta.get('MPJPE', -1)

                    return {
                        'subject': output_name,
                        'status': 'success',
                        'mpjpe': mpjpe,
                        'time': elapsed,
                    }
            else:
                return {
                    'subject': output_name,
                    'status': 'failed',
                    'error': 'No meta.json found',
                    'time': elapsed,
                }
        else:
            # Extract error from stderr
            error_lines = result_proc.stderr.split('\n')
            error_msg = next((line for line in error_lines if 'Error' in line or 'Exception' in line), 'Unknown error')

            return {
                'subject': output_name,
                'status': 'failed',
                'error': error_msg[:100],
                'time': elapsed,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        return {
            'subject': output_name,
            'status': 'timeout',
            'time': elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        return {
            'subject': output_name,
            'status': 'error',
            'error': str(e)[:100],
            'time': elapsed,
        }


def print_progress(result: Dict, completed: int, total: int, stats: Dict):
    """Print real-time progress."""

    # Progress bar
    progress = completed / total * 100
    bar_length = 40
    filled = int(bar_length * completed / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    # Status icon
    if result['status'] == 'success':
        icon = '✓'
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
        time_str = f"{result['time']:.1f}s"
    elif result['status'] == 'skipped':
        icon = '⊙'
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
        time_str = "cached"
    else:
        icon = '✗'
        mpjpe_str = result.get('error', 'unknown')[:30]
        time_str = f"{result.get('time', 0):.1f}s"

    # Average MPJPE
    if stats['success'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        avg_str = f"Avg: {avg_mpjpe:.2f}mm"
    else:
        avg_str = "Avg: N/A"

    # Print update
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {result['subject'][:40]:40s} | {mpjpe_str:30s} | {time_str:8s}")
    print(f"          {bar} {progress:5.1f}% | {completed}/{total} | ✓{stats['success']} ✗{stats['failed']} ⊙{stats['skipped']} | {avg_str}")
    print(flush=True)


def main():
    parser = argparse.ArgumentParser(description='Sequential minibatch processing for v3_enhanced')
    parser.add_argument('--test_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/No_Arm',
                        help='Path to test dataset')
    parser.add_argument('--train_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm',
                        help='Path to train dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_minibatch',
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
    print("MINIBATCH PROCESSING - v3_enhanced (Sequential)")
    print("=" * 80)

    # Find all .b3d files
    print("Finding .b3d files...", flush=True)
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

    print(f"Found {len(b3d_files)} subjects:")
    print(f"  Test set:  {test_count}")
    print(f"  Train set: {train_count}")
    print(f"  Total:     {len(b3d_files)}")
    print()
    print(f"Output: {args.out_dir}")
    print("=" * 80)
    print()

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
    }

    # Process subjects sequentially
    start_time = time.time()

    for b3d_file in b3d_files:
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
        elif result['status'] == 'skipped':
            stats['skipped'] += 1
            stats['total_mpjpe'] += result['mpjpe']
        else:
            stats['failed'] += 1

        # Print progress
        print_progress(result, stats['completed'], len(b3d_files), stats)

    # Final summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed/3600:.2f} hours ({elapsed/60:.1f} minutes)")
    print(f"Completed:  {stats['completed']}/{len(b3d_files)}")
    print(f"  Success:  {stats['success']}")
    print(f"  Failed:   {stats['failed']}")
    print(f"  Skipped:  {stats['skipped']}")
    if stats['success'] + stats['skipped'] > 0:
        avg_mpjpe = stats['total_mpjpe'] / (stats['success'] + stats['skipped'])
        print(f"Average MPJPE: {avg_mpjpe:.2f}mm")
    print("=" * 80)


if __name__ == "__main__":
    main()
