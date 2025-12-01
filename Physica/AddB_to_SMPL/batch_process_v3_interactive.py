#!/usr/bin/env python3
"""
Interactive batch processing for v3_enhanced SMPL fitting
Run in Jupyter notebook or terminal for real-time progress updates
"""

import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, Manager
import argparse

def process_subject(args):
    """Process a single subject and return results immediately."""
    b3d_file, output_dir, smpl_model, num_frames, script_path, shared_dict = args

    # Extract subject name from path
    subject_name = b3d_file.stem
    parent_name = b3d_file.parent.name

    # Determine if test or train
    if "test" in str(b3d_file):
        split = "test"
    elif "train" in str(b3d_file):
        split = "train"
    else:
        split = "unknown"

    # Create output directory name
    output_name = f"{split}_{parent_name}_{subject_name}"
    subject_output_dir = Path(output_dir) / output_name

    # Skip if already processed
    meta_file = subject_output_dir / "meta.json"
    if meta_file.exists():
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'MPJPE' in meta:
                    result = {
                        'subject': output_name,
                        'status': 'skipped',
                        'mpjpe': meta['MPJPE'],
                        'b3d_file': str(b3d_file)
                    }
                    # Update shared statistics
                    shared_dict['completed'] = shared_dict.get('completed', 0) + 1
                    shared_dict['skipped'] = shared_dict.get('skipped', 0) + 1
                    return result
        except:
            pass

    # Run the script
    cmd = [
        "/egr/research-zijunlab/kwonjoon/Miscellaneous/miniforge3/envs/physpt/bin/python",
        str(script_path),
        "--b3d", str(b3d_file),
        "--smpl_model", str(smpl_model),
        "--out_dir", str(output_dir),
        "--num_frames", str(num_frames),
        "--device", "cpu"
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

                    result = {
                        'subject': output_name,
                        'status': 'success',
                        'mpjpe': mpjpe,
                        'time': elapsed,
                        'b3d_file': str(b3d_file)
                    }

                    # Update shared statistics
                    shared_dict['completed'] = shared_dict.get('completed', 0) + 1
                    shared_dict['total_mpjpe'] = shared_dict.get('total_mpjpe', 0.0) + mpjpe
                    shared_dict['success'] = shared_dict.get('success', 0) + 1

                    return result
            else:
                result = {
                    'subject': output_name,
                    'status': 'failed',
                    'error': 'No meta.json found',
                    'time': elapsed,
                    'b3d_file': str(b3d_file)
                }
                shared_dict['completed'] = shared_dict.get('completed', 0) + 1
                shared_dict['failed'] = shared_dict.get('failed', 0) + 1
                return result
        else:
            result = {
                'subject': output_name,
                'status': 'failed',
                'error': f"Return code {result_proc.returncode}",
                'stderr': result_proc.stderr[:500],
                'time': elapsed,
                'b3d_file': str(b3d_file)
            }
            shared_dict['completed'] = shared_dict.get('completed', 0) + 1
            shared_dict['failed'] = shared_dict.get('failed', 0) + 1
            return result

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        result = {
            'subject': output_name,
            'status': 'timeout',
            'time': elapsed,
            'b3d_file': str(b3d_file)
        }
        shared_dict['completed'] = shared_dict.get('completed', 0) + 1
        shared_dict['failed'] = shared_dict.get('failed', 0) + 1
        return result
    except Exception as e:
        elapsed = time.time() - start_time
        result = {
            'subject': output_name,
            'status': 'error',
            'error': str(e),
            'time': elapsed,
            'b3d_file': str(b3d_file)
        }
        shared_dict['completed'] = shared_dict.get('completed', 0) + 1
        shared_dict['failed'] = shared_dict.get('failed', 0) + 1
        return result

def print_progress(result, shared_dict, total):
    """Print progress update for a completed subject."""
    completed = shared_dict.get('completed', 0)
    success = shared_dict.get('success', 0)
    failed = shared_dict.get('failed', 0)
    skipped = shared_dict.get('skipped', 0)

    # Progress bar
    progress = completed / total * 100
    bar_length = 40
    filled = int(bar_length * completed / total)
    bar = '█' * filled + '░' * (bar_length - filled)

    # Status icon
    if result['status'] == 'success':
        icon = '✓'
        time_str = f"{result['time']:.1f}s"
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
    elif result['status'] == 'skipped':
        icon = '⊙'
        time_str = "cached"
        mpjpe_str = f"{result['mpjpe']:.2f}mm"
    else:
        icon = '✗'
        time_str = f"{result.get('time', 0):.1f}s"
        mpjpe_str = result.get('error', 'unknown')[:30]

    # Average MPJPE
    if success > 0:
        avg_mpjpe = shared_dict.get('total_mpjpe', 0.0) / success
        avg_str = f"Avg: {avg_mpjpe:.2f}mm"
    else:
        avg_str = "Avg: N/A"

    # Print update
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {icon} {result['subject'][:40]:40s} | {mpjpe_str:30s} | {time_str:8s}")
    print(f"          {bar} {progress:5.1f}% | {completed}/{total} | ✓{success} ✗{failed} ⊙{skipped} | {avg_str}")
    print()

def callback_wrapper(shared_dict, total):
    """Create a callback function with access to shared state."""
    def callback(result):
        print_progress(result, shared_dict, total)
    return callback

def main():
    parser = argparse.ArgumentParser(description='Batch process v3_enhanced SMPL fitting with real-time updates')
    parser.add_argument('--test_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/No_Arm',
                        help='Path to test dataset')
    parser.add_argument('--train_path', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm',
                        help='Path to train dataset')
    parser.add_argument('--out_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_full_batch',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='Path to SMPL model')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to process')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of subjects (for testing)')

    args = parser.parse_args()

    # Script path
    script_dir = Path(__file__).parent
    script_path = script_dir / "addbiomechanics_to_smpl_v3_enhanced.py"

    # Find all .b3d files
    print("=" * 80)
    print("BATCH PROCESSING - v3_enhanced (Interactive Mode)")
    print("=" * 80)
    print("Finding .b3d files...")

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
    print(f"Using {args.num_workers} parallel workers")
    print(f"Output: {args.out_dir}")
    print("=" * 80)
    print()

    # Create output directory
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Shared dictionary for statistics
    manager = Manager()
    shared_dict = manager.dict()
    shared_dict['completed'] = 0
    shared_dict['success'] = 0
    shared_dict['failed'] = 0
    shared_dict['skipped'] = 0
    shared_dict['total_mpjpe'] = 0.0

    # Prepare arguments for each subject
    job_args = [
        (b3d_file, args.out_dir, args.smpl_model, args.num_frames, script_path, shared_dict)
        for b3d_file in b3d_files
    ]

    # Process with pool
    start_time = time.time()

    with Pool(processes=args.num_workers) as pool:
        for result in pool.imap_unordered(process_subject, job_args):
            print_progress(result, shared_dict, len(b3d_files))

    # Final summary
    elapsed = time.time() - start_time
    print()
    print("=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total time: {elapsed/3600:.2f} hours")
    print(f"Completed:  {shared_dict['completed']}/{len(b3d_files)}")
    print(f"  Success:  {shared_dict['success']}")
    print(f"  Failed:   {shared_dict['failed']}")
    print(f"  Skipped:  {shared_dict['skipped']}")
    if shared_dict['success'] > 0:
        avg_mpjpe = shared_dict['total_mpjpe'] / shared_dict['success']
        print(f"Average MPJPE: {avg_mpjpe:.2f}mm")
    print("=" * 80)

if __name__ == "__main__":
    main()
