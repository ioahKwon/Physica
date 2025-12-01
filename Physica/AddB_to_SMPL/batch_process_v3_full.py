#!/usr/bin/env python3
"""
Batch process all No_Arm datasets with v3_enhanced in parallel.
Processes both test and train sets.
"""
import argparse
import subprocess
import multiprocessing
from pathlib import Path
from datetime import datetime
import json
import time

def find_all_b3d_files(input_dirs):
    """Find all .b3d files in the given directories."""
    b3d_files = []
    for input_dir in input_dirs:
        input_path = Path(input_dir)
        for b3d_file in input_path.rglob("*.b3d"):
            b3d_files.append(b3d_file)
    return sorted(b3d_files)

def process_subject(args):
    """Process a single subject."""
    b3d_file, output_dir, smpl_model, num_frames, script_path = args

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
                    print(f"[SKIP] {output_name} already processed (MPJPE: {meta['MPJPE']:.2f}mm)")
                    return {
                        'subject': output_name,
                        'status': 'skipped',
                        'mpjpe': meta['MPJPE'],
                        'b3d_file': str(b3d_file)
                    }
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
    print(f"[START] {output_name}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per subject
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Extract MPJPE from output
            mpjpe = None
            for line in result.stdout.split('\n'):
                if 'MPJPE:' in line:
                    try:
                        mpjpe = float(line.split('MPJPE:')[1].strip())
                        break
                    except:
                        pass

            print(f"[SUCCESS] {output_name} - MPJPE: {mpjpe:.2f}mm ({elapsed:.1f}s)")
            return {
                'subject': output_name,
                'status': 'success',
                'mpjpe': mpjpe,
                'elapsed': elapsed,
                'b3d_file': str(b3d_file)
            }
        else:
            print(f"[FAILED] {output_name} - Error: {result.stderr[:200]}")
            return {
                'subject': output_name,
                'status': 'failed',
                'error': result.stderr[:500],
                'b3d_file': str(b3d_file)
            }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"[TIMEOUT] {output_name} ({elapsed:.1f}s)")
        return {
            'subject': output_name,
            'status': 'timeout',
            'elapsed': elapsed,
            'b3d_file': str(b3d_file)
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[ERROR] {output_name} - {str(e)}")
        return {
            'subject': output_name,
            'status': 'error',
            'error': str(e),
            'elapsed': elapsed,
            'b3d_file': str(b3d_file)
        }

def main():
    parser = argparse.ArgumentParser(description='Batch process No_Arm datasets with v3_enhanced')
    parser.add_argument('--test_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/test/No_Arm',
                        help='Test set directory')
    parser.add_argument('--train_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Dataset/AddB/train/No_Arm',
                        help='Train set directory')
    parser.add_argument('--output_dir', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Output/output_Physica/AddB_to_SMPL/results_v3_full_batch',
                        help='Output directory')
    parser.add_argument('--smpl_model', type=str,
                        default='/egr/research-zijunlab/kwonjoon/Code/Physica/AddB_to_SMPL/models/smpl_model.pkl',
                        help='SMPL model path')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of parallel workers')
    parser.add_argument('--num_frames', type=int, default=64,
                        help='Number of frames to process')
    parser.add_argument('--script', type=str,
                        default='addbiomechanics_to_smpl_v3_enhanced.py',
                        help='Script to run')

    args = parser.parse_args()

    # Get script path
    script_path = Path(__file__).parent / args.script
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return

    # Find all .b3d files
    print("="*80)
    print("BATCH PROCESSING - v3_enhanced")
    print("="*80)
    print(f"Finding .b3d files...")

    input_dirs = [args.test_dir, args.train_dir]
    b3d_files = find_all_b3d_files(input_dirs)

    print(f"Found {len(b3d_files)} subjects:")
    test_count = sum(1 for f in b3d_files if "test" in str(f))
    train_count = sum(1 for f in b3d_files if "train" in str(f))
    print(f"  Test set:  {test_count}")
    print(f"  Train set: {train_count}")
    print(f"  Total:     {len(b3d_files)}")
    print(f"\nUsing {args.num_workers} parallel workers")
    print(f"Estimated time: ~{len(b3d_files) * 90 / args.num_workers / 3600:.1f} hours")
    print("="*80)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Prepare arguments for multiprocessing
    process_args = [
        (b3d_file, args.output_dir, args.smpl_model, args.num_frames, script_path)
        for b3d_file in b3d_files
    ]

    # Process in parallel
    start_time = time.time()
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        results = pool.map(process_subject, process_args)

    total_elapsed = time.time() - start_time

    # Summarize results
    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)

    success = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    timeout = [r for r in results if r['status'] == 'timeout']
    error = [r for r in results if r['status'] == 'error']
    skipped = [r for r in results if r['status'] == 'skipped']

    print(f"\nTotal subjects: {len(results)}")
    print(f"  Success:  {len(success)}")
    print(f"  Skipped:  {len(skipped)} (already processed)")
    print(f"  Failed:   {len(failed)}")
    print(f"  Timeout:  {len(timeout)}")
    print(f"  Error:    {len(error)}")

    # Calculate average MPJPE
    mpjpe_values = [r['mpjpe'] for r in success + skipped if r['mpjpe'] is not None]
    if mpjpe_values:
        avg_mpjpe = sum(mpjpe_values) / len(mpjpe_values)
        print(f"\nAverage MPJPE: {avg_mpjpe:.2f}mm (n={len(mpjpe_values)})")

    print(f"\nTotal time: {total_elapsed/3600:.2f} hours")
    print(f"Average time per subject: {total_elapsed/len(results):.1f}s")

    # Save results
    results_file = Path(args.output_dir) / "batch_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'total_subjects': len(results),
            'success': len(success),
            'failed': len(failed),
            'timeout': len(timeout),
            'error': len(error),
            'skipped': len(skipped),
            'average_mpjpe': avg_mpjpe if mpjpe_values else None,
            'total_time_hours': total_elapsed / 3600,
            'results': results
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("="*80)

    # Print failed subjects
    if failed or timeout or error:
        print("\nFailed subjects:")
        for r in failed + timeout + error:
            print(f"  - {r['subject']}: {r['status']}")

if __name__ == '__main__':
    main()
